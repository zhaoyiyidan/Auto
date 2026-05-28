from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from researchclaw.llm.acp_client import ACPClient, ACPConfig
from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline.stage_impls._hypothesis_debate import (
    _build_claim_prompt,
    _build_fork_config,
    _build_judge_prompt,
    _build_synthesizer_prompt,
    _import_fork,
    _open_fork_or_fresh,
    _parse_judge_verdict,
    _strip_debate_metadata,
    run_acp_debate,
)
from researchclaw.pipeline.stage_impls._synthesis import _is_acp_provider


def test_parse_judge_verdict_valid_json() -> None:
    verdict = _parse_judge_verdict(
        '{"verdict": "pass", "criticisms": [], "fatal_flaws": [], '
        '"confidence": 0.82}'
    )

    assert verdict["verdict"] == "pass"
    assert verdict["confidence"] == 0.82
    assert verdict["parse_error"] is False


def test_parse_judge_verdict_malformed_json_uses_heuristics() -> None:
    verdict = _parse_judge_verdict(
        "Verdict: fail\n- prediction is not measurable\nConfidence: 0.4"
    )

    assert verdict["verdict"] == "fail"
    assert "prediction is not measurable" in verdict["criticisms"]
    assert verdict["confidence"] == 0.4
    assert verdict["parse_error"] is True


def test_parse_judge_verdict_empty_fails_safe() -> None:
    verdict = _parse_judge_verdict("")

    assert verdict["verdict"] == "fail"
    assert verdict["confidence"] == 0.0
    assert verdict["parse_error"] is True


def test_build_claim_prompt_round_one_vs_revision() -> None:
    role_prompts = {
        "system": "You are the {topic} specialist.",
        "user": "Use synthesis: {synthesis}",
    }

    first_messages, first_system = _build_claim_prompt(
        "innovator",
        role_prompts,
        "routing",
        "gap text",
        None,
        None,
    )
    second_messages, _ = _build_claim_prompt(
        "innovator",
        role_prompts,
        "routing",
        "gap text",
        "old claim",
        ["not measurable"],
    )

    assert first_system == "You are the routing specialist."
    assert "gap text" in first_messages[0]["content"]
    assert "Prior Candidate Claim" not in first_messages[0]["content"]
    assert "old claim" in second_messages[0]["content"]
    assert "not measurable" in second_messages[0]["content"]
    assert "hidden session" in second_messages[0]["content"]


def test_build_judge_prompt_includes_candidate_and_forbids_rewrites() -> None:
    messages, system = _build_judge_prompt("candidate claim", "synthesis", "topic")

    assert "candidate claim" in messages[0]["content"]
    assert "Return exactly this JSON shape" in messages[0]["content"]
    assert "Do not rewrite" in system


def test_build_synthesizer_prompt_excludes_debate_metadata() -> None:
    messages, system = _build_synthesizer_prompt(
        {"innovator": "claim one", "contrarian": "claim two"},
        "synthesis",
        "topic",
    )

    content = messages[0]["content"]
    assert "Do not mention agent names" in content
    assert "debate rounds" in content
    assert "judges" in content
    assert "Candidate Set 1" in content
    assert "Output only the hypotheses document" in system


def test_strip_debate_metadata_removes_process_but_keeps_research_terms() -> None:
    cleaned = _strip_debate_metadata(
        "\n".join([
            "## H1: Multi-agent routing improves robustness",
            "Agent: innovator",
            "This came from the innovator perspective.",
            "Required Baselines: compare multi-agent and single-agent systems",
            "Failure Condition: round-robin routing matches the proposed method",
        ]),
        role_names=["innovator"],
    )

    assert "Agent: innovator" not in cleaned
    assert "innovator perspective" not in cleaned
    assert "Multi-agent routing" in cleaned
    assert "single-agent systems" in cleaned
    assert "round-robin routing" in cleaned


def test_is_acp_provider_detection() -> None:
    assert _is_acp_provider(SimpleNamespace(llm=SimpleNamespace(provider="acp")))
    assert not _is_acp_provider(
        SimpleNamespace(llm=SimpleNamespace(provider="openai-compatible"))
    )


def test_acp_client_export_import_and_close_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("researchclaw.llm.acp_client.subprocess.run", fake_run)
    config = ACPConfig(
        agent="codex",
        cwd=str(tmp_path),
        acpx_command="acpx",
        session_name="main-session",
    )
    client = ACPClient(config)

    archive = tmp_path / "ancestor.archive"
    client.export_session(archive)
    fork = ACPClient.fork_from_archive(archive, "debate-role-claim-r1", config)
    client.close_session("debate-role-claim-r1")

    assert calls[0] == [
        "acpx", "--ttl", "0", "--cwd", str(tmp_path),
        "codex", "sessions", "export", "main-session",
        "--output", str(archive),
    ]
    assert calls[1] == [
        "acpx", "--ttl", "0", "--cwd", str(tmp_path),
        "codex", "sessions", "import", str(archive),
        "--name", "debate-role-claim-r1",
    ]
    assert calls[2] == [
        "acpx", "--ttl", "0", "--cwd", str(tmp_path),
        "codex", "sessions", "close", "debate-role-claim-r1",
    ]
    assert fork.config.session_name == "debate-role-claim-r1"


def test_acp_client_export_retries_after_locked_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[7] == "export" and len([c for c in calls if c[7] == "export"]) == 1:
            return subprocess.CompletedProcess(
                cmd,
                1,
                "",
                "session is currently locked by a running queue owner",
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("researchclaw.llm.acp_client.subprocess.run", fake_run)
    client = ACPClient(
        ACPConfig(
            agent="claude",
            cwd=str(tmp_path),
            acpx_command="acpx",
            session_name="main-session",
        )
    )
    client._session_ready = True

    client.export_session(tmp_path / "ancestor.archive")

    assert [cmd[7] for cmd in calls] == ["export", "close", "export"]
    assert client._session_ready is False


def test_build_fork_config_preserves_base_fields() -> None:
    base = ACPConfig(
        agent="codex",
        cwd="/tmp/project",
        acpx_command="/bin/acpx",
        session_name="main",
        timeout_sec=120,
        base_url="https://provider.example.com/v1",
        api_key_env="MY_KEY",
        model="gpt-5.5",
        debate_max_rounds=3,
        debate_confidence_min=0.75,
    )

    fork = _build_fork_config(base, "debate-innovator-claim-r1")

    assert fork.session_name == "debate-innovator-claim-r1"
    assert fork.agent == "codex"
    assert fork.cwd == "/tmp/project"
    assert fork.acpx_command == "/bin/acpx"
    assert fork.timeout_sec == 120
    assert fork.base_url == "https://provider.example.com/v1"
    assert fork.api_key_env == "MY_KEY"
    assert fork.model == "gpt-5.5"
    assert fork.debate_max_rounds == 3
    assert fork.debate_confidence_min == 0.75


def test_open_fork_or_fresh_uses_fresh_session_when_acpx_import_is_restore(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_import(*args: object, **kwargs: object) -> ACPClient:
        _ = args, kwargs
        raise RuntimeError(
            "A local session already uses this provider session id; prune first"
        )

    closed: list[str] = []
    monkeypatch.setattr(ACPClient, "fork_from_archive", staticmethod(fail_import))
    monkeypatch.setattr(
        ACPClient,
        "close_session",
        lambda self, name: closed.append(name),
    )

    client = _open_fork_or_fresh(
        tmp_path / "ancestor.archive",
        "debate-innovator-claim-r1",
        ACPConfig(
            agent="codex",
            cwd=str(tmp_path),
            acpx_command="acpx",
            session_name="main",
            base_url="https://provider.example.com/v1",
            api_key_env="MY_KEY",
            model="gpt-5.5",
        ),
    )

    assert client.config.session_name == "debate-innovator-claim-r1"
    assert client.config.base_url == "https://provider.example.com/v1"
    assert client.config.api_key_env == "MY_KEY"
    assert client.config.model == "gpt-5.5"
    assert closed == ["debate-innovator-claim-r1"]


def test_import_fork_command(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        "researchclaw.pipeline.stage_impls._hypothesis_debate.subprocess.run",
        fake_run,
    )

    archive = tmp_path / "ancestor.archive"
    _import_fork("codex", str(tmp_path), archive, "debate-fork", "acpx")

    assert calls == [[
        "acpx", "--ttl", "0", "--cwd", str(tmp_path.resolve()),
        "codex", "sessions", "import", str(archive),
        "--name", "debate-fork",
    ]]


class FakeACPClient(ACPClient):
    def __init__(
        self,
        session_name: str,
        responses: dict[str, str],
        fork_log: list[str] | None = None,
        close_log: list[str] | None = None,
    ) -> None:
        self.config = ACPConfig(
            agent="codex",
            cwd=".",
            acpx_command="acpx",
            session_name=session_name,
            debate_max_rounds=2,
            debate_confidence_min=0.6,
        )
        self.responses = responses
        self.fork_log = fork_log if fork_log is not None else []
        self.close_log = close_log if close_log is not None else []
        self.exported: list[Path] = []
        self.chat_calls: list[tuple[str, list[dict[str, str]]]] = []

    def export_session(self, output_path: Path) -> None:
        output_path.write_text("archive", encoding="utf-8")
        self.exported.append(output_path)

    def close_session(self, name: str) -> None:
        self.close_log.append(name)

    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> LLMResponse:
        self.chat_calls.append((self.config.session_name, messages))
        return LLMResponse(
            content=self.responses.get(self.config.session_name, "OK"),
            model="fake-acp",
        )


class TinyPromptManager:
    def debate_roles_hypothesis(self) -> dict[str, dict[str, str]]:
        return {
            "innovator": {
                "system": "system {topic}",
                "user": "user {synthesis}",
            }
        }


class ThreeRolePromptManager:
    def debate_roles_hypothesis(self) -> dict[str, dict[str, str]]:
        return {
            "innovator": {"system": "system {topic}", "user": "user {synthesis}"},
            "pragmatist": {"system": "system {topic}", "user": "user {synthesis}"},
            "contrarian": {"system": "system {topic}", "user": "user {synthesis}"},
        }


def test_run_acp_debate_runs_perspectives_in_parallel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    responses = {
        "debate-final-synth": (
            "## H1: Combined claim\n"
            "Hypothesis Statement: combined claim"
        ),
    }
    fork_log: list[str] = []
    close_log: list[str] = []
    final_clients: list[FakeACPClient] = []
    main = FakeACPClient("main", responses, fork_log, close_log)

    def fake_fork(
        archive_path: Path,
        fork_name: str,
        base_config: ACPConfig,
    ) -> FakeACPClient:
        _ = archive_path, base_config
        fork_log.append(fork_name)
        client = FakeACPClient(fork_name, responses, fork_log, close_log)
        if fork_name == "debate-final-synth":
            final_clients.append(client)
        return client

    barrier = threading.Barrier(3)
    lock = threading.Lock()
    entered: list[str] = []
    active = 0
    max_active = 0

    def fake_role_debate(**kwargs: Any) -> str:
        nonlocal active, max_active
        role_name = str(kwargs["role_name"])
        with lock:
            active += 1
            max_active = max(max_active, active)
            entered.append(role_name)
        try:
            barrier.wait(timeout=2)
            time.sleep(0.05)
            return f"claim from {role_name}"
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(ACPClient, "fork_from_archive", staticmethod(fake_fork))
    monkeypatch.setattr(
        "researchclaw.pipeline.stage_impls._hypothesis_debate._run_role_debate",
        fake_role_debate,
    )

    run_dir = tmp_path / "run"
    stage7 = run_dir / "stage-07"
    stage8 = run_dir / "stage-08"
    stage7.mkdir(parents=True)
    stage8.mkdir()
    (stage7 / "synthesis.md").write_text("## Synthesis\nGap.", encoding="utf-8")
    config = SimpleNamespace(
        research=SimpleNamespace(topic="routing"),
        llm=SimpleNamespace(acp=main.config),
    )

    result = run_acp_debate(run_dir, stage8, config, main, ThreeRolePromptManager())

    assert "Combined claim" in result
    assert set(entered) == {"innovator", "pragmatist", "contrarian"}
    assert max_active == 3
    assert fork_log == ["debate-final-synth"]
    final_prompt = final_clients[0].chat_calls[0][1][0]["content"]
    assert final_prompt.index("claim from innovator") < final_prompt.index(
        "claim from pragmatist"
    )
    assert final_prompt.index("claim from pragmatist") < final_prompt.index(
        "claim from contrarian"
    )


def test_run_acp_debate_uses_fresh_forks_for_revision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    responses = {
        "debate-innovator-claim-r1": "## H1\nweak claim",
        "debate-innovator-judge-r1": (
            '{"verdict": "fail", "criticisms": ["not measurable"], '
            '"fatal_flaws": [], "confidence": 0.2}'
        ),
        "debate-innovator-claim-r2": "## H1\nrevised measurable claim",
        "debate-innovator-judge-r2": (
            '{"verdict": "pass", "criticisms": [], "fatal_flaws": [], '
            '"confidence": 0.9}'
        ),
        "debate-final-synth": (
            "## H1: Revised measurable claim\n"
            "Hypothesis Statement: revised measurable claim"
        ),
    }
    fork_log: list[str] = []
    close_log: list[str] = []
    main = FakeACPClient("main", responses, fork_log, close_log)

    def fake_fork(
        archive_path: Path,
        fork_name: str,
        base_config: ACPConfig,
    ) -> FakeACPClient:
        _ = archive_path, base_config
        fork_log.append(fork_name)
        return FakeACPClient(fork_name, responses, fork_log, close_log)

    monkeypatch.setattr(ACPClient, "fork_from_archive", staticmethod(fake_fork))

    run_dir = tmp_path / "run"
    stage7 = run_dir / "stage-07"
    stage8 = run_dir / "stage-08"
    stage7.mkdir(parents=True)
    stage8.mkdir()
    (stage7 / "synthesis.md").write_text("## Synthesis\nGap.", encoding="utf-8")
    config = SimpleNamespace(
        research=SimpleNamespace(topic="routing"),
        llm=SimpleNamespace(acp=main.config),
    )

    result = run_acp_debate(run_dir, stage8, config, main, TinyPromptManager())

    assert "Revised measurable claim" in result
    assert fork_log == [
        "debate-innovator-claim-r1",
        "debate-innovator-judge-r1",
        "debate-innovator-claim-r2",
        "debate-innovator-judge-r2",
        "debate-final-synth",
    ]
    assert "debate-innovator-claim-r1" in close_log
    assert "debate-innovator-claim-r2" in close_log
    assert not (stage8 / "fork_debate" / "ancestor.archive").exists()


def test_run_acp_debate_stops_after_passing_first_judge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    responses = {
        "debate-innovator-claim-r1": "## H1\nstrong claim",
        "debate-innovator-judge-r1": (
            '{"verdict": "pass", "criticisms": [], "fatal_flaws": [], '
            '"confidence": 0.95}'
        ),
        "debate-final-synth": "## H1\nHypothesis Statement: strong claim",
    }
    fork_log: list[str] = []
    close_log: list[str] = []
    main = FakeACPClient("main", responses, fork_log, close_log)

    def fake_fork(
        archive_path: Path,
        fork_name: str,
        base_config: ACPConfig,
    ) -> FakeACPClient:
        _ = archive_path, base_config
        fork_log.append(fork_name)
        return FakeACPClient(fork_name, responses, fork_log, close_log)

    monkeypatch.setattr(ACPClient, "fork_from_archive", staticmethod(fake_fork))

    run_dir = tmp_path / "run"
    stage7 = run_dir / "stage-07"
    stage8 = run_dir / "stage-08"
    stage7.mkdir(parents=True)
    stage8.mkdir()
    (stage7 / "synthesis.md").write_text("## Synthesis\nGap.", encoding="utf-8")
    config = SimpleNamespace(
        research=SimpleNamespace(topic="routing"),
        llm=SimpleNamespace(acp=main.config),
    )

    run_acp_debate(run_dir, stage8, config, main, TinyPromptManager())

    assert "debate-innovator-claim-r2" not in fork_log
    assert fork_log == [
        "debate-innovator-claim-r1",
        "debate-innovator-judge-r1",
        "debate-final-synth",
    ]
