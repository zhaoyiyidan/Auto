from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from researchclaw.experiment.acp_workspace_session import AcpWorkspaceSession


def test_build_env_injects_codex_provider_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("WORKSPACE_CODEX_KEY", "secret-key")
    session = AcpWorkspaceSession(
        agent="codex",
        cwd=tmp_path,
        acpx_command="acpx",
        session_name="researchclaw-code-run-1",
        base_url="https://provider.example.com/v1",
        api_key_env="WORKSPACE_CODEX_KEY",
        model="gpt-5.2",
    )

    env = session._build_env()

    assert env["OPENAI_BASE_URL"] == "https://provider.example.com/v1"
    assert env["OPENAI_API_KEY"] == "secret-key"
    assert env["MODEL_PROVIDER"] == "custom-gateway"
    codex_config = json.loads(env["CODEX_CONFIG"])
    assert codex_config["model"] == "gpt-5.2"
    assert codex_config["model_providers"]["custom-gateway"]["env_key"] == "WORKSPACE_CODEX_KEY"
    assert "secret-key" not in env["CODEX_CONFIG"]


def test_acpx_base_args_use_workspace_cwd(tmp_path: Path) -> None:
    session = AcpWorkspaceSession(
        agent="claude",
        cwd=tmp_path,
        acpx_command="/bin/acpx",
        session_name="researchclaw-code-run-1",
    )

    assert session._acpx_base_args("/bin/acpx") == [
        "/bin/acpx",
        "--ttl",
        "0",
        "--cwd",
        str(tmp_path.resolve()),
        "claude",
    ]


def test_ensure_session_uses_named_session_without_warmup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    session = AcpWorkspaceSession(
        agent="claude",
        cwd=tmp_path,
        acpx_command="acpx",
        session_name="researchclaw-code-run-1",
    )

    session.ensure_session()

    assert calls == [
        [
            "acpx",
            "--ttl",
            "0",
            "--cwd",
            str(tmp_path.resolve()),
            "claude",
            "sessions",
            "ensure",
            "--name",
            "researchclaw-code-run-1",
        ]
    ]


def test_ensure_session_falls_back_to_new(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[-3:] == ["ensure", "--name", "researchclaw-code-run-1"]:
            return subprocess.CompletedProcess(cmd, 1, "", "missing")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    session = AcpWorkspaceSession(
        agent="claude",
        cwd=tmp_path,
        acpx_command="acpx",
        session_name="researchclaw-code-run-1",
    )

    session.ensure_session()

    assert calls[-1] == [
        "acpx",
        "--ttl",
        "0",
        "--cwd",
        str(tmp_path.resolve()),
        "claude",
        "sessions",
        "new",
        "--name",
        "researchclaw-code-run-1",
    ]


def test_run_task_uses_configured_max_turns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    session = AcpWorkspaceSession(
        agent="claude",
        cwd=tmp_path,
        acpx_command="acpx",
        session_name="researchclaw-code-run-1",
        max_turns=77,
    )
    session.ensure_session = lambda: None  # type: ignore[method-assign]

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["input_data"] = kwargs.get("input_data")
        return subprocess.CompletedProcess(cmd, 0, "agent log", "")

    session._run_acp_with_heartbeat = fake_run  # type: ignore[method-assign]

    assert session.run_task("modify workspace") == "agent log"
    assert captured["cmd"] == [
        "acpx",
        "--approve-all",
        "--max-turns",
        "77",
        "--ttl",
        "0",
        "--cwd",
        str(tmp_path.resolve()),
        "claude",
        "-s",
        "researchclaw-code-run-1",
        "modify workspace",
    ]
    assert captured["input_data"] is None


def test_run_task_uses_stdin_transport_for_large_prompts(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    session = AcpWorkspaceSession(
        agent="claude",
        cwd=tmp_path,
        acpx_command="acpx",
        session_name="researchclaw-code-run-1",
    )
    session._MAX_CLI_PROMPT_BYTES = 10  # type: ignore[attr-defined]
    session.ensure_session = lambda: None  # type: ignore[method-assign]

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["input_data"] = kwargs.get("input_data")
        return subprocess.CompletedProcess(cmd, 0, "done", "")

    session._run_acp_with_heartbeat = fake_run  # type: ignore[method-assign]

    session.run_task("x" * 11)

    assert captured["cmd"][-2:] == ["-f", "-"]
    assert captured["input_data"] == "x" * 11


def test_run_task_retries_on_transient_stdout_returncode_zero(tmp_path: Path) -> None:
    """FIX#2: an exit-0 disconnect banner in stdout is retried."""
    session = AcpWorkspaceSession(
        agent="claude",
        cwd=tmp_path,
        acpx_command="acpx",
        session_name="researchclaw-code-run-1",
        max_retries=3,
    )
    session.ensure_session = lambda: None  # type: ignore[method-assign]
    session.close = lambda: None  # type: ignore[method-assign]
    session._retry_sleep = lambda _s: None  # type: ignore[attr-defined]

    calls = {"n": 0}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        if calls["n"] == 1:
            return subprocess.CompletedProcess(
                cmd, 0, "Reconnecting... 5/5\nstream closed before response.completed\n", ""
            )
        return subprocess.CompletedProcess(cmd, 0, "agent finished work", "")

    session._run_acp_with_heartbeat = fake_run  # type: ignore[method-assign]

    assert session.run_task("modify workspace") == "agent finished work"
    assert calls["n"] == 2


def test_run_task_no_retry_on_real_failure(tmp_path: Path) -> None:
    """FIX#2: a non-zero exit without a transient signature is NOT retried."""
    session = AcpWorkspaceSession(
        agent="claude",
        cwd=tmp_path,
        acpx_command="acpx",
        session_name="researchclaw-code-run-1",
        max_retries=3,
    )
    session.ensure_session = lambda: None  # type: ignore[method-assign]
    session._retry_sleep = lambda _s: None  # type: ignore[attr-defined]

    calls = {"n": 0}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        return subprocess.CompletedProcess(cmd, 1, "", "ZeroDivisionError in train loop")

    session._run_acp_with_heartbeat = fake_run  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="ACP workspace task failed"):
        session.run_task("modify workspace")
    assert calls["n"] == 1


def test_run_task_respects_max_retries_config(tmp_path: Path) -> None:
    """FIX#2: retries are bounded by max_retries (1 + max_retries attempts)."""
    from researchclaw.llm.acp_retry import TransientAcpDisconnect

    session = AcpWorkspaceSession(
        agent="claude",
        cwd=tmp_path,
        acpx_command="acpx",
        session_name="researchclaw-code-run-1",
        max_retries=1,
    )
    session.ensure_session = lambda: None  # type: ignore[method-assign]
    session.close = lambda: None  # type: ignore[method-assign]
    session._retry_sleep = lambda _s: None  # type: ignore[attr-defined]

    calls = {"n": 0}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        return subprocess.CompletedProcess(
            cmd, 0, "stream disconnected before completion", ""
        )

    session._run_acp_with_heartbeat = fake_run  # type: ignore[method-assign]

    with pytest.raises(TransientAcpDisconnect):
        session.run_task("modify workspace")
    assert calls["n"] == 2  # 1 initial + 1 retry


def test_export_session_writes_requested_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    session = AcpWorkspaceSession(
        agent="claude",
        cwd=tmp_path,
        acpx_command="acpx",
        session_name="researchclaw-code-run-1",
    )
    archive = tmp_path / "ledger" / "session_export.tar.gz"

    session.export_session(archive)

    assert archive.parent.exists()
    assert calls[-1] == [
        "acpx",
        "--ttl",
        "0",
        "--cwd",
        str(tmp_path.resolve()),
        "claude",
        "sessions",
        "export",
        "researchclaw-code-run-1",
        "--output",
        str(archive),
    ]
