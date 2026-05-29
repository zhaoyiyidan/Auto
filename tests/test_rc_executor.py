# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline import executor as rc_executor
from researchclaw.pipeline.stages import Stage, StageStatus


class FakeLLMClient:
    def __init__(self, response_text: str = "mock response"):
        self.response_text: str = response_text
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: object):
        _ = kwargs
        self.calls.append(messages)
        from researchclaw.llm.client import LLMResponse

        return LLMResponse(content=self.response_text, model="fake-model")


class FakeLLMClientWithConfig(FakeLLMClient):
    def __init__(self, response_text: str = "mock response"):
        super().__init__(response_text=response_text)
        self.config: SimpleNamespace = SimpleNamespace(
            base_url="http://fake", api_key="fake-key"
        )


@pytest.fixture()
def rc_config(tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "rc-test", "mode": "docs-first"},
        "research": {
            "topic": "test-driven science",
            "domains": ["ml", "systems"],
            "daily_paper_count": 2,
            "quality_threshold": 8.2,
        },
        "runtime": {"timezone": "UTC"},
        "notifications": {
            "channel": "local",
            "on_stage_start": True,
            "on_stage_fail": False,
            "on_gate_required": True,
        },
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {"use_memory": True, "use_message": True},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline-test-key",
            "primary_model": "fake-model",
            "fallback_models": [],
        },
        "security": {"hitl_required_stages": [5, 9, 20]},
        "experiment": {"mode": "sandbox"},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


@pytest.fixture()
def adapters() -> AdapterBundle:
    return AdapterBundle()


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path


def _write_prior_artifact(
    run_dir: Path, stage_num: int, filename: str, content: str
) -> None:
    stage_dir = run_dir / f"stage-{stage_num:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / filename).write_text(content, encoding="utf-8")


def _stub_novelty_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_check_novelty(*args: object, **kwargs: object) -> dict[str, object]:
        _ = args, kwargs
        return {
            "novelty_score": 1.0,
            "assessment": "high",
            "recommendation": "proceed",
            "similar_papers_found": 0,
            "similar_papers": [],
            "search_coverage": "mocked",
            "total_papers_retrieved": 0,
        }

    monkeypatch.setattr(
        "researchclaw.literature.novelty.check_novelty",
        fake_check_novelty,
    )


def test_executor_map_has_23_entries() -> None:
    executor_map = getattr(rc_executor, "EXECUTOR_MAP", rc_executor._STAGE_EXECUTORS)
    assert len(executor_map) == 23


def test_every_stage_member_has_matching_executor() -> None:
    executor_map = getattr(rc_executor, "EXECUTOR_MAP", rc_executor._STAGE_EXECUTORS)
    assert set(executor_map.keys()) == set(Stage)


def test_stage_result_dataclass_fields() -> None:
    result = rc_executor.StageResult(
        stage=Stage.TOPIC_INIT, status=StageStatus.DONE, artifacts=("goal.md",)
    )
    assert result.stage == Stage.TOPIC_INIT
    assert result.status == StageStatus.DONE
    assert result.artifacts == ("goal.md",)
    assert result.error is None
    assert result.decision == "proceed"
    assert result.evidence_refs == ()


def test_utcnow_iso_returns_valid_iso_timestamp() -> None:
    ts = rc_executor._utcnow_iso()
    assert ts.endswith("+00:00")
    assert "T" in ts


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("before\n```yaml\na: 1\n```\nafter", "a: 1"),
        ("```yml\nkey: value\n```", "key: value"),
        ("```\nplain: true\n```", "plain: true"),
        ("  x: y  ", "x: y"),
    ],
)
def test_extract_yaml_block_variants(text: str, expected: str) -> None:
    assert rc_executor._extract_yaml_block(text) == expected


@pytest.mark.parametrize(
    ("payload", "default", "expected"),
    [
        ('{"ok": true}', {"fallback": True}, {"ok": True}),
        ("[1, 2, 3]", {"fallback": True}, [1, 2, 3]),
        ("not-json", {"fallback": True}, {"fallback": True}),
    ],
)
def test_safe_json_loads_valid_and_invalid(payload: str, default, expected) -> None:
    assert rc_executor._safe_json_loads(payload, default) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a/b", "a_b"),
        ("a\\b", "a_b"),
        ("../secret", "__secret"),
        ("name with spaces!.md", "name_with_spaces_.md"),
        ("", "unnamed"),
    ],
)
def test_safe_filename_sanitization(raw: str, expected: str) -> None:
    assert rc_executor._safe_filename(raw) == expected


def test_safe_filename_truncates_to_100_chars() -> None:
    raw = "x" * 120
    cleaned = rc_executor._safe_filename(raw)
    assert len(cleaned) == 100
    assert cleaned == "x" * 100


class TestWorkspaceAgentStageWiring:
    def test_stage9_emits_task_spec_template(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
    ) -> None:
        from researchclaw.experiment.workspace import TaskSpec

        cfg = _workspace_agent_rc_config(tmp_path)
        _write_prior_artifact(run_dir, 8, "hypotheses.md", "# Hypothesis\nImprove X")
        stage_dir = run_dir / "stage-09"
        stage_dir.mkdir()

        result = rc_executor._execute_experiment_design(
            stage_dir,
            run_dir,
            cfg,
            adapters,
            llm=None,
        )

        assert result.status is StageStatus.DONE
        assert result.artifacts == ("task_spec.yaml",)
        spec = TaskSpec.from_path(stage_dir / "task_spec.yaml")
        assert spec.workspace == cfg.experiment.workspace_agent.workspace_path
        assert spec.primary_metric == cfg.experiment.metric_key
        assert spec.metric_direction == cfg.experiment.metric_direction
        assert spec.objective
        assert spec.expected_outputs

    def test_stage9_uses_llm_task_spec_fields(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
    ) -> None:
        from researchclaw.experiment.workspace import TaskSpec

        cfg = _workspace_agent_rc_config(tmp_path)
        _write_prior_artifact(run_dir, 8, "hypotheses.md", "# Hypothesis\nImprove X")
        stage_dir = run_dir / "stage-09"
        stage_dir.mkdir()
        llm = FakeLLMClient(
            "\n".join(
                [
                    "workspace: /model/workspace",
                    "objective: model objective",
                    "constraints:",
                    "  - single GPU",
                    "primary_metric: f1",
                    "metric_direction: minimize",
                    "allowed_scope:",
                    "  - src/",
                    "forbidden_scope:",
                    "  - data/raw/",
                    "expected_outputs:",
                    "  - outputs/f1.json",
                ]
            )
        )

        result = rc_executor._execute_experiment_design(
            stage_dir,
            run_dir,
            cfg,
            adapters,
            llm=llm,
        )

        assert result.status is StageStatus.DONE
        spec = TaskSpec.from_path(stage_dir / "task_spec.yaml")
        assert spec.workspace == "/model/workspace"
        assert spec.objective == "model objective"
        assert spec.primary_metric == "f1"
        assert spec.expected_outputs == ["outputs/f1.json"]

    def test_workspace_codegen_prompt_has_agent_contract(self) -> None:
        from researchclaw.pipeline.stage_impls._code_generation import (
            _workspace_codegen_prompt,
        )

        prompt = _workspace_codegen_prompt(
            topic="topic",
            exp_plan="plan",
            metric="accuracy",
            pkg_hint="numpy",
            compute_budget="1 hour",
            extra_guidance="guidance",
            manifest_filename="run_manifest.json",
        )

        assert prompt.count("MUST") >= 6
        assert prompt.count("MUST NOT") >= 4
        assert "git commit" in prompt.lower()
        assert "Do not submit" in prompt
        assert "run_manifest.json" in prompt
        assert "main.py" not in prompt

    def test_workspace_refine_prompt_has_agent_contract(self) -> None:
        from researchclaw.pipeline.stage_impls._execution import (
            _workspace_refine_prompt,
        )

        prompt = _workspace_refine_prompt(
            topic="topic",
            metric_key="accuracy",
            metric_direction="maximize",
            exp_plan="plan",
            project_files=["train.py"],
            run_summaries=["previous run"],
            manifest_filename="run_manifest.json",
        )

        assert prompt.count("MUST") >= 6
        assert prompt.count("MUST NOT") >= 4
        assert "git commit" in prompt.lower()
        assert "Do not submit" in prompt
        assert "run_manifest.json" in prompt

    def test_stage10_drives_agent_implement_and_copies_manifest(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from researchclaw.experiment.workspace import LaunchCommand, RunManifest

        cfg = _workspace_agent_rc_config(tmp_path)
        calls: list[dict[str, Any]] = []
        agent = object()

        monkeypatch.setattr(
            "researchclaw.experiment.workspace_agent.create_workspace_agent",
            lambda *args, **kwargs: agent,
        )

        def fake_implement(**kwargs: Any):
            from researchclaw.experiment.workspace import WorkspaceAgentResult

            calls.append(kwargs)
            workspace = Path(cfg.experiment.workspace_agent.workspace_path)
            manifest = RunManifest(
                code_commit="head",
                launch=LaunchCommand(command="python train.py"),
                result_paths=["outputs/metrics.json"],
            )
            (workspace / "run_manifest.json").write_text(
                manifest.to_json(),
                encoding="utf-8",
            )
            return WorkspaceAgentResult(
                base_sha="base",
                agent_commit_sha="head",
                manifest_path="run_manifest.json",
                diff_stat=" train.py | 1 +",
                raw_log="done",
                provider_name="acp",
                elapsed_sec=0.1,
            )

        monkeypatch.setattr(
            "researchclaw.pipeline.workspace_orchestrator.run_workspace_agent_implement",
            fake_implement,
        )
        _write_prior_artifact(
            run_dir,
            9,
            "task_spec.yaml",
            "workspace: .\nobjective: improve\nconstraints: []\n"
            "primary_metric: accuracy\nmetric_direction: maximize\n"
            "allowed_scope: ['.']\nforbidden_scope: []\n"
            "expected_outputs: ['outputs/metrics.json']\n",
        )
        stage_dir = run_dir / "stage-10"
        stage_dir.mkdir()

        result = rc_executor._execute_code_generation(
            stage_dir,
            run_dir,
            cfg,
            adapters,
            llm=FakeLLMClient(),
        )

        assert result.status is StageStatus.DONE
        assert calls[0]["stage"] == 10
        assert calls[0]["agent"] is agent
        assert calls[0]["close_policy"] == "keep"
        assert "submitter" not in calls[0]
        assert (stage_dir / "run_manifest.json").is_file()
        assert (stage_dir / "stage-10-workspace-agent-result.json").is_file()

    def test_stage10_failed_agent_does_not_copy_manifest(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = _workspace_agent_rc_config(tmp_path)
        agent = object()

        monkeypatch.setattr(
            "researchclaw.experiment.workspace_agent.create_workspace_agent",
            lambda *args, **kwargs: agent,
        )

        def fake_implement(**kwargs: Any):
            from researchclaw.experiment.workspace import WorkspaceAgentResult

            return WorkspaceAgentResult(
                base_sha="base",
                agent_commit_sha=None,
                manifest_path=None,
                diff_stat="",
                raw_log="no commit",
                provider_name="acp",
                elapsed_sec=0.1,
                error="Agent did not create a new git commit",
            )

        monkeypatch.setattr(
            "researchclaw.pipeline.workspace_orchestrator.run_workspace_agent_implement",
            fake_implement,
        )
        _write_prior_artifact(
            run_dir,
            9,
            "task_spec.yaml",
            "workspace: .\nobjective: improve\nconstraints: []\n"
            "primary_metric: accuracy\nmetric_direction: maximize\n"
            "allowed_scope: ['.']\nforbidden_scope: []\n"
            "expected_outputs: ['outputs/metrics.json']\n",
        )
        stage_dir = run_dir / "stage-10"
        stage_dir.mkdir()

        result = rc_executor._execute_code_generation(
            stage_dir,
            run_dir,
            cfg,
            adapters,
            llm=FakeLLMClient(),
        )

        assert result.status is StageStatus.FAILED
        assert not (stage_dir / "run_manifest.json").exists()

    def test_stage11_validates_manifest_and_copies_it(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
    ) -> None:
        from researchclaw.experiment.workspace import LaunchCommand, RunManifest

        cfg = _workspace_agent_rc_config(tmp_path)
        workspace = Path(cfg.experiment.workspace_agent.workspace_path)
        head = _init_workspace_git(workspace)
        manifest = RunManifest(
            code_commit=head,
            launch=LaunchCommand(command="python train.py"),
            result_paths=["outputs/metrics.json"],
        )
        _write_prior_artifact(run_dir, 10, "run_manifest.json", manifest.to_json())
        stage_dir = run_dir / "stage-11"
        stage_dir.mkdir()

        result = rc_executor._execute_resource_planning(
            stage_dir,
            run_dir,
            cfg,
            adapters,
            llm=None,
        )

        assert result.status is StageStatus.DONE
        validation = json.loads(
            (stage_dir / "manifest_validation.json").read_text(encoding="utf-8")
        )
        assert validation["ok"] is True
        assert json.loads((stage_dir / "run_manifest.json").read_text(encoding="utf-8"))[
            "code_commit"
        ] == head

    def test_stage11_invalid_manifest_prompts_agent_to_fix(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from researchclaw.experiment.workspace import LaunchCommand, RunManifest

        cfg = _workspace_agent_rc_config(tmp_path)
        workspace = Path(cfg.experiment.workspace_agent.workspace_path)
        head = _init_workspace_git(workspace)
        bad_manifest = RunManifest(
            code_commit=head,
            launch=LaunchCommand(command="python train.py"),
            result_paths=[],
        )
        _write_prior_artifact(run_dir, 10, "run_manifest.json", bad_manifest.to_json())
        agent = object()
        prompts_seen: list[str] = []

        monkeypatch.setattr(
            "researchclaw.experiment.workspace_agent.create_workspace_agent",
            lambda *args, **kwargs: agent,
        )

        def fake_implement(**kwargs: Any):
            from researchclaw.experiment.workspace import WorkspaceAgentResult

            prompts_seen.append(kwargs["prompt"])
            fixed = RunManifest(
                code_commit=head,
                launch=LaunchCommand(command="python train.py"),
                result_paths=["outputs/metrics.json"],
            )
            (workspace / "run_manifest.json").write_text(
                fixed.to_json(),
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "run_manifest.json"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "commit", "-m", "fix manifest"],
                cwd=workspace,
                check=True,
                capture_output=True,
            )
            return WorkspaceAgentResult(
                base_sha=head,
                agent_commit_sha=_git_head(workspace),
                manifest_path="run_manifest.json",
                diff_stat="",
                raw_log="fixed manifest",
                provider_name="acp",
                elapsed_sec=0.1,
            )

        monkeypatch.setattr(
            "researchclaw.pipeline.workspace_orchestrator.run_workspace_agent_implement",
            fake_implement,
        )
        stage_dir = run_dir / "stage-11"
        stage_dir.mkdir()

        result = rc_executor._execute_resource_planning(
            stage_dir,
            run_dir,
            cfg,
            adapters,
            llm=None,
        )

        assert result.status is StageStatus.DONE
        assert prompts_seen
        assert "result_paths" in prompts_seen[0]
        validation = json.loads(
            (stage_dir / "manifest_validation.json").read_text(encoding="utf-8")
        )
        assert validation["ok"] is True

    def test_stage12_submits_waits_collects_hashed_results(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from researchclaw.experiment.workspace import (
            LaunchCommand,
            ManifestValidation,
            RunManifest,
        )
        from tests.test_workspace_orchestrator import DummySubmitter

        cfg = _workspace_agent_rc_config(tmp_path)
        workspace = Path(cfg.experiment.workspace_agent.workspace_path)
        head = _init_workspace_git(workspace)
        manifest = RunManifest(
            code_commit=head,
            launch=LaunchCommand(command="python train.py"),
            result_paths=["outputs/metrics.json"],
        )
        _write_prior_artifact(run_dir, 11, "run_manifest.json", manifest.to_json())
        _write_prior_artifact(
            run_dir,
            11,
            "manifest_validation.json",
            json.dumps(
                ManifestValidation(
                    ok=True,
                    schema_version=manifest.schema_version,
                    code_commit=head,
                    commit_exists=True,
                    workspace_dirty=False,
                    launch_command=manifest.launch.command,
                    launch_cwd=manifest.launch.cwd,
                    result_paths=manifest.result_paths,
                ).to_dict()
            ),
        )
        monkeypatch.setattr(
            "researchclaw.experiment.submitter.create_submitter",
            lambda *args, **kwargs: DummySubmitter(),
        )
        stage_dir = run_dir / "stage-12"
        stage_dir.mkdir()

        result = rc_executor._execute_experiment_run(
            stage_dir,
            run_dir,
            cfg,
            adapters,
            llm=None,
        )

        assert result.status is StageStatus.DONE
        assert result.artifacts == (
            "execution_record.json",
            "submit_result.json",
            "result_artifacts.json",
        )
        assert json.loads((stage_dir / "submit_result.json").read_text(encoding="utf-8"))[
            "job_id"
        ] == "job-1"
        assert json.loads(
            (stage_dir / "execution_record.json").read_text(encoding="utf-8")
        )["result_hashes"]["outputs/metrics.json"]
        artifacts = json.loads(
            (stage_dir / "result_artifacts.json").read_text(encoding="utf-8")
        )
        assert artifacts["artifacts"][0]["exists"] is True

    def test_stage12_submitter_exception_fails(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from researchclaw.experiment.workspace import (
            LaunchCommand,
            ManifestValidation,
            RunManifest,
        )

        cfg = _workspace_agent_rc_config(tmp_path)
        workspace = Path(cfg.experiment.workspace_agent.workspace_path)
        head = _init_workspace_git(workspace)
        manifest = RunManifest(
            code_commit=head,
            launch=LaunchCommand(command="python train.py"),
            result_paths=["outputs/metrics.json"],
        )
        _write_prior_artifact(run_dir, 11, "run_manifest.json", manifest.to_json())
        _write_prior_artifact(
            run_dir,
            11,
            "manifest_validation.json",
            json.dumps(
                ManifestValidation(
                    ok=True,
                    schema_version=manifest.schema_version,
                    code_commit=head,
                    commit_exists=True,
                    workspace_dirty=False,
                    launch_command=manifest.launch.command,
                    launch_cwd=manifest.launch.cwd,
                    result_paths=manifest.result_paths,
                ).to_dict()
            ),
        )

        class RaisingSubmitter:
            name = "raising"

            def submit(self, request: object) -> object:
                raise RuntimeError("cluster down")

        monkeypatch.setattr(
            "researchclaw.experiment.submitter.create_submitter",
            lambda *args, **kwargs: RaisingSubmitter(),
        )
        stage_dir = run_dir / "stage-12"
        stage_dir.mkdir()

        result = rc_executor._execute_experiment_run(
            stage_dir,
            run_dir,
            cfg,
            adapters,
            llm=None,
        )

        assert result.status is StageStatus.FAILED
        assert "E12_HARNESS_FAIL" in (result.error or "")

    def test_stage12_all_missing_results_fails(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from researchclaw.experiment.workspace import (
            LaunchCommand,
            ManifestValidation,
            RunManifest,
            SubmitResult,
        )

        cfg = _workspace_agent_rc_config(tmp_path)
        workspace = Path(cfg.experiment.workspace_agent.workspace_path)
        head = _init_workspace_git(workspace)
        manifest = RunManifest(
            code_commit=head,
            launch=LaunchCommand(command="python train.py"),
            result_paths=["outputs/missing.json"],
        )
        _write_prior_artifact(run_dir, 11, "run_manifest.json", manifest.to_json())
        _write_prior_artifact(
            run_dir,
            11,
            "manifest_validation.json",
            json.dumps(
                ManifestValidation(
                    ok=True,
                    schema_version=manifest.schema_version,
                    code_commit=head,
                    commit_exists=True,
                    workspace_dirty=False,
                    launch_command=manifest.launch.command,
                    launch_cwd=manifest.launch.cwd,
                    result_paths=manifest.result_paths,
                ).to_dict()
            ),
        )

        class NoopSubmitter:
            name = "noop"

            def submit(self, request: object) -> SubmitResult:
                return SubmitResult("job-1", self.name, "submitted")

        monkeypatch.setattr(
            "researchclaw.experiment.submitter.create_submitter",
            lambda *args, **kwargs: NoopSubmitter(),
        )
        stage_dir = run_dir / "stage-12"
        stage_dir.mkdir()

        result = rc_executor._execute_experiment_run(
            stage_dir,
            run_dir,
            cfg,
            adapters,
            llm=None,
        )

        assert result.status is StageStatus.FAILED
        assert "E12_HARNESS_FAIL" in (result.error or "")
        artifacts = json.loads(
            (stage_dir / "result_artifacts.json").read_text(encoding="utf-8")
        )
        assert artifacts["artifacts"][0]["exists"] is False

    def test_stage13_refines_from_real_results_and_writes_manifest(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from researchclaw.experiment.workspace import (
            ExecutionRecord,
            LaunchCommand,
            ResultArtifact,
            ResultArtifacts,
            RunManifest,
        )

        cfg = _workspace_agent_rc_config(tmp_path)
        workspace = Path(cfg.experiment.workspace_agent.workspace_path)
        head = _init_workspace_git(workspace)
        calls: list[dict[str, Any]] = []
        agent = object()

        execution = ExecutionRecord(
            stage=12,
            code_commit=head,
            submitter="local",
            job_id="job-1",
            submit_status="submitted",
            final_status="completed",
            log_path="logs/run.log",
            result_paths=["outputs/metrics.json"],
            result_hashes={"outputs/metrics.json": "sha"},
            metrics={"accuracy": 0.9},
            elapsed_sec=1.2,
            waited=True,
            recorded_at="2026-05-29T00:00:00Z",
        )
        artifacts = ResultArtifacts(
            code_commit=head,
            artifacts=[
                ResultArtifact(
                    path="outputs/metrics.json",
                    sha256="sha",
                    size_bytes=12,
                    exists=True,
                )
            ],
            collected_at="2026-05-29T00:00:00Z",
        )
        _write_prior_artifact(
            run_dir, 12, "execution_record.json", json.dumps(execution.to_dict())
        )
        _write_prior_artifact(
            run_dir, 12, "result_artifacts.json", json.dumps(artifacts.to_dict())
        )
        monkeypatch.setattr(
            "researchclaw.experiment.workspace_agent.create_workspace_agent",
            lambda *args, **kwargs: agent,
        )

        def fake_implement(**kwargs: Any):
            from researchclaw.experiment.workspace import WorkspaceAgentResult

            calls.append(kwargs)
            manifest = RunManifest(
                code_commit="refined",
                launch=LaunchCommand(command="python train.py --refined"),
                result_paths=["outputs/metrics.json"],
            )
            (workspace / "run_manifest.json").write_text(
                manifest.to_json(),
                encoding="utf-8",
            )
            return WorkspaceAgentResult(
                base_sha=head,
                agent_commit_sha="refined",
                manifest_path="run_manifest.json",
                diff_stat=" train.py | 2 +",
                raw_log="refined",
                provider_name="acp",
                elapsed_sec=0.2,
            )

        monkeypatch.setattr(
            "researchclaw.pipeline.workspace_orchestrator.run_workspace_agent_implement",
            fake_implement,
        )
        stage_dir = run_dir / "stage-13"
        stage_dir.mkdir()

        result = rc_executor._execute_iterative_refine(
            stage_dir,
            run_dir,
            cfg,
            adapters,
            llm=FakeLLMClient(),
        )

        assert result.status is StageStatus.DONE
        assert calls[0]["stage"] == 13
        assert calls[0]["agent"] is agent
        assert calls[0]["close_policy"] == "keep"
        assert "submitter" not in calls[0]
        assert "accuracy" in calls[0]["prompt"]
        assert "RESULT ARTIFACTS" in calls[0]["prompt"]
        assert (stage_dir / "refine_record.json").is_file()
        assert json.loads((stage_dir / "run_manifest.json").read_text(encoding="utf-8"))[
            "code_commit"
        ] == "refined"

    def test_stage14_analyzes_registry_execution_and_provenance(
        self,
        tmp_path: Path,
        run_dir: Path,
        adapters: AdapterBundle,
    ) -> None:
        from researchclaw.experiment.workspace import (
            ExecutionRecord,
            ExperimentRecord,
            ResultArtifact,
            ResultArtifacts,
        )

        cfg = _workspace_agent_rc_config(tmp_path)
        execution = ExecutionRecord(
            stage=12,
            code_commit="commit-1",
            submitter="local",
            job_id="job-1",
            submit_status="submitted",
            final_status="completed",
            log_path="logs/run.log",
            result_paths=["outputs/metrics.json"],
            result_hashes={"outputs/metrics.json": "sha"},
            metrics={"accuracy": 0.91},
            elapsed_sec=1.2,
            waited=True,
            recorded_at="2026-05-29T00:00:00Z",
        )
        artifacts = ResultArtifacts(
            code_commit="commit-1",
            artifacts=[
                ResultArtifact(
                    path="outputs/metrics.json",
                    sha256="sha",
                    size_bytes=12,
                    exists=True,
                )
            ],
            collected_at="2026-05-29T00:00:00Z",
        )
        record = ExperimentRecord(
            workspace="/workspace",
            stage=12,
            base_sha="base",
            agent_commit_sha="commit-1",
            provider="acp",
            agent_manifest="run_manifest.json",
            submitter="local",
            job_id="job-1",
            result_paths=["outputs/metrics.json"],
            result_hashes={"outputs/metrics.json": "sha"},
            recorded_at="2026-05-29T00:00:00Z",
            session_name="researchclaw-code-test",
        )
        _write_prior_artifact(
            run_dir, 12, "execution_record.json", json.dumps(execution.to_dict())
        )
        _write_prior_artifact(
            run_dir, 12, "result_artifacts.json", json.dumps(artifacts.to_dict())
        )
        s12 = run_dir / "stage-12"
        with (s12 / "workspace_experiment_registry.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record.to_dict()) + "\n")
        stage_dir = run_dir / "stage-14"
        stage_dir.mkdir()

        result = rc_executor._execute_result_analysis(
            stage_dir,
            run_dir,
            cfg,
            adapters,
            llm=None,
        )

        assert result.status is StageStatus.DONE
        assert result.artifacts == (
            "analysis.md",
            "experiment_summary.json",
            "provenance.json",
        )
        summary = json.loads(
            (stage_dir / "experiment_summary.json").read_text(encoding="utf-8")
        )
        assert summary["primary_metric"] == "accuracy"
        assert summary["metric_direction"] == "maximize"
        assert summary["best_metric"] == 0.91
        assert summary["best_commit"] == "commit-1"
        assert summary["n_runs"] == 1
        provenance = json.loads(
            (stage_dir / "provenance.json").read_text(encoding="utf-8")
        )
        assert provenance["base_sha"] == "base"
        assert provenance["commits"] == ["commit-1"]
        assert provenance["result_hashes"]["outputs/metrics.json"] == "sha"


def _workspace_agent_rc_config(tmp_path: Path) -> RCConfig:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    data = {
        "project": {"name": "rc-test", "mode": "docs-first"},
        "research": {"topic": "workspace topic", "domains": ["ml"]},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {"use_memory": True, "use_message": True},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline-test-key",
            "primary_model": "fake-model",
            "fallback_models": [],
        },
        "security": {"hitl_required_stages": [5, 9, 20]},
        "experiment": {
            "mode": "sandbox",
            "metric_key": "accuracy",
            "metric_direction": "maximize",
            "workspace_agent": {
                "enabled": True,
                "transport": "acp",
                "workspace_path": str(workspace),
                "session_name": "researchclaw-code-test",
                "agent": "claude",
                "timeout_sec": 300,
                "max_turns": 20,
                "close_policy": "close",
            },
            "submitter": {"type": "manual"},
        },
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def _init_workspace_git(workspace: Path) -> str:
    workspace.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=workspace,
        check=True,
    )
    (workspace / "README.md").write_text("# workspace\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _git_head(workspace: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _patch_workspace_runner(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, Any]],
    agent: object,
    submitter: object,
) -> None:
    from researchclaw.experiment.workspace import WorkspaceAgentResult

    def fake_task(**kwargs: Any) -> WorkspaceAgentResult:
        calls.append(kwargs)
        return WorkspaceAgentResult(
            base_sha="base",
            agent_commit_sha="head",
            manifest_path="run_manifest.json",
            diff_stat=" train.py | 1 +",
            raw_log="done",
            provider_name="acp",
            elapsed_sec=0.1,
        )

    monkeypatch.setattr(
        "researchclaw.experiment.workspace_agent.create_workspace_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        "researchclaw.experiment.submitter.create_submitter",
        lambda *args, **kwargs: submitter,
    )
    monkeypatch.setattr(
        "researchclaw.pipeline.workspace_orchestrator.run_workspace_agent_task",
        fake_task,
    )


def test_build_context_preamble_basic_fields(
    rc_config: RCConfig, run_dir: Path
) -> None:
    text = rc_executor._build_context_preamble(rc_config, run_dir)
    assert "## Research Context" in text
    assert "test-driven science" in text
    assert "ml, systems" in text


def test_build_context_preamble_includes_selected_prior_artifacts(
    rc_config: RCConfig, run_dir: Path
) -> None:
    _write_prior_artifact(run_dir, 1, "goal.md", "goal content")
    _write_prior_artifact(run_dir, 8, "hypotheses.md", "hyp content")
    _write_prior_artifact(run_dir, 7, "synthesis.md", "synth content")
    text = rc_executor._build_context_preamble(
        rc_config,
        run_dir,
        include_goal=True,
        include_hypotheses=True,
        include_synthesis=True,
    )
    assert "### Goal" in text
    assert "goal content" in text
    assert "### Hypotheses" in text
    assert "hyp content" in text
    assert "### Synthesis" in text
    assert "synth content" in text


def test_read_prior_artifact_finds_newest_file(run_dir: Path) -> None:
    _write_prior_artifact(run_dir, 1, "goal.md", "old")
    _write_prior_artifact(run_dir, 3, "goal.md", "new")
    found = rc_executor._read_prior_artifact(run_dir, "goal.md")
    assert found == "new"


def test_read_prior_artifact_finds_directory_path(run_dir: Path) -> None:
    cards_dir = run_dir / "stage-06" / "cards"
    cards_dir.mkdir(parents=True)
    (cards_dir / "card-1.json").write_text("{}", encoding="utf-8")
    found = rc_executor._read_prior_artifact(run_dir, "cards/")
    assert found == str(cards_dir)


def test_read_prior_artifact_returns_none_when_not_found(run_dir: Path) -> None:
    assert rc_executor._read_prior_artifact(run_dir, "missing.md") is None


def test_read_best_analysis_prefers_best_file(run_dir: Path) -> None:
    """BUG-225: _read_best_analysis prefers analysis_best.md at run root."""
    from researchclaw.pipeline._helpers import _read_best_analysis

    # Create degenerate analysis in stage-14 and best at run root
    s14 = run_dir / "stage-14"
    s14.mkdir(parents=True)
    (s14 / "analysis.md").write_text("Degenerate analysis", encoding="utf-8")
    (run_dir / "analysis_best.md").write_text("Best analysis", encoding="utf-8")

    result = _read_best_analysis(run_dir)
    assert result == "Best analysis"


def test_read_best_analysis_falls_back_to_prior_artifact(run_dir: Path) -> None:
    """BUG-225: Falls back to _read_prior_artifact when no analysis_best.md."""
    from researchclaw.pipeline._helpers import _read_best_analysis

    s14 = run_dir / "stage-14"
    s14.mkdir(parents=True)
    (s14 / "analysis.md").write_text("Only analysis", encoding="utf-8")

    result = _read_best_analysis(run_dir)
    assert result == "Only analysis"


def test_read_best_analysis_returns_empty_when_none(run_dir: Path) -> None:
    """BUG-225: Returns empty string when no analysis exists at all."""
    from researchclaw.pipeline._helpers import _read_best_analysis

    result = _read_best_analysis(run_dir)
    assert result == ""


def test_write_stage_meta_writes_expected_json(run_dir: Path) -> None:
    stage_dir = run_dir / "stage-01"
    stage_dir.mkdir()
    result = rc_executor.StageResult(
        stage=Stage.TOPIC_INIT,
        status=StageStatus.DONE,
        artifacts=("goal.md",),
        decision="proceed",
        evidence_refs=("stage-01/goal.md",),
    )
    rc_executor._write_stage_meta(stage_dir, Stage.TOPIC_INIT, "run-abc", result)
    payload = cast(
        dict[str, Any],
        json.loads((stage_dir / "decision.json").read_text(encoding="utf-8")),
    )
    assert payload["stage_id"] == "01-topic_init"
    assert payload["run_id"] == "run-abc"
    assert payload["status"] == "done"
    assert payload["decision"] == "proceed"
    assert payload["output_artifacts"] == ["goal.md"]
    assert payload["evidence_refs"] == ["stage-01/goal.md"]
    assert payload["next_stage"] == 2
    assert re.match(r"\d{4}-\d{2}-\d{2}T", payload["ts"])


def test_write_stage_meta_keeps_paused_stage_as_next_stage(run_dir: Path) -> None:
    stage_dir = run_dir / "stage-02"
    stage_dir.mkdir()
    result = rc_executor.StageResult(
        stage=Stage.PROBLEM_DECOMPOSE,
        status=StageStatus.PAUSED,
        artifacts=("refinement_log.json",),
        decision="resume",
        error="ACP prompt timed out after 1800s",
        evidence_refs=("stage-02/refinement_log.json",),
    )
    rc_executor._write_stage_meta(
        stage_dir, Stage.PROBLEM_DECOMPOSE, "run-paused", result
    )
    payload = cast(
        dict[str, Any],
        json.loads((stage_dir / "decision.json").read_text(encoding="utf-8")),
    )
    assert payload["status"] == "paused"
    assert payload["decision"] == "resume"
    assert payload["next_stage"] == int(Stage.PROBLEM_DECOMPOSE)


def test_execute_stage_creates_stage_dir_writes_artifacts_and_meta(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    fake_llm = FakeLLMClientWithConfig("# Goal\n\nMocked goal body")
    monkeypatch.setattr(
        "researchclaw.pipeline.executor.LLMClient.from_rc_config",
        lambda _config: fake_llm,
    )

    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-1",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.DONE
    assert "goal.md" in result.artifacts
    assert "hardware_profile.json" in result.artifacts
    assert (run_dir / "stage-01").is_dir()
    assert (
        (run_dir / "stage-01" / "goal.md")
        .read_text(encoding="utf-8")
        .startswith("# Goal")
    )
    assert (run_dir / "stage-01" / "hardware_profile.json").exists()
    assert len(fake_llm.calls) == 1

    decision = cast(
        dict[str, Any],
        json.loads(
            (run_dir / "stage-01" / "decision.json").read_text(encoding="utf-8")
        ),
    )
    assert decision["run_id"] == "run-1"
    assert decision["status"] == "done"
    assert decision["output_artifacts"] == ["goal.md", "hardware_profile.json"]


def test_execute_stage_contract_validation_missing_output_file_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def bad_executor(
        _stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
    ):
        _ = llm
        return rc_executor.StageResult(
            stage=Stage.TOPIC_INIT, status=StageStatus.DONE, artifacts=("goal.md",)
        )

    monkeypatch.setitem(rc_executor._STAGE_EXECUTORS, Stage.TOPIC_INIT, bad_executor)
    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-2",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert "Missing or empty output: goal.md" in (result.error or "")


def test_execute_stage_contract_validation_missing_output_directory_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    _write_prior_artifact(run_dir, 5, "shortlist.jsonl", '{"title": "x"}')

    def bad_executor(
        _stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
    ):
        _ = llm
        return rc_executor.StageResult(
            stage=Stage.KNOWLEDGE_EXTRACT,
            status=StageStatus.DONE,
            artifacts=("cards/",),
        )

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.KNOWLEDGE_EXTRACT, bad_executor
    )
    result = rc_executor.execute_stage(
        Stage.KNOWLEDGE_EXTRACT,
        run_dir=run_dir,
        run_id="run-3",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert "Missing output directory: cards/" in (result.error or "")


def test_execute_stage_missing_required_input_returns_failed(
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    result = rc_executor.execute_stage(
        Stage.PROBLEM_DECOMPOSE,
        run_dir=run_dir,
        run_id="run-4",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert "Missing input: goal.md" in (result.error or "")


def test_execute_stage_gate_behavior_auto_approve_true_keeps_done(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    _write_prior_artifact(run_dir, 4, "candidates.jsonl", '{"title": "paper"}')

    def good_executor(
        stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
        **_kwargs: object,
    ):
        _ = llm
        (stage_dir / "shortlist.jsonl").write_text(
            '{"title": "paper"}\n', encoding="utf-8"
        )
        return rc_executor.StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.DONE,
            artifacts=("shortlist.jsonl",),
        )

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.LITERATURE_SCREEN, good_executor
    )
    result = rc_executor.execute_stage(
        Stage.LITERATURE_SCREEN,
        run_dir=run_dir,
        run_id="run-5",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.DONE
    memory_entries = getattr(adapters.memory, "entries", [])
    assert any(
        ns == "gates" and "auto-approved" in content for ns, content in memory_entries
    )


def test_execute_stage_gate_behavior_auto_approve_false_blocks(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    _write_prior_artifact(run_dir, 4, "candidates.jsonl", '{"title": "paper"}')

    def good_executor(
        stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
        **_kwargs: object,
    ):
        _ = llm
        (stage_dir / "shortlist.jsonl").write_text(
            '{"title": "paper"}\n', encoding="utf-8"
        )
        return rc_executor.StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.DONE,
            artifacts=("shortlist.jsonl",),
        )

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.LITERATURE_SCREEN, good_executor
    )
    result = rc_executor.execute_stage(
        Stage.LITERATURE_SCREEN,
        run_dir=run_dir,
        run_id="run-6",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=False,
    )
    assert result.status == StageStatus.BLOCKED_APPROVAL
    assert result.decision == "block"
    message_calls = getattr(adapters.message, "calls", [])
    assert message_calls
    assert "Approval required" in message_calls[-1][2]


def test_execute_stage_llm_client_creation_error_falls_back_without_crash(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def boom(_config: RCConfig):
        raise RuntimeError("llm init failed")

    monkeypatch.setattr("researchclaw.pipeline.executor.LLMClient.from_rc_config", boom)
    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-7",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.DONE
    assert (run_dir / "stage-01" / "goal.md").exists()


def test_execute_stage_executor_exception_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def raising_executor(
        _stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
        **_kwargs: object,
    ):
        _ = llm
        raise RuntimeError("stage exploded")

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.TOPIC_INIT, raising_executor
    )
    result = rc_executor.execute_stage(
        Stage.TOPIC_INIT,
        run_dir=run_dir,
        run_id="run-8",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert result.status == StageStatus.FAILED
    assert result.decision == "retry"
    assert "stage exploded" in (result.error or "")


@pytest.mark.parametrize(
    "stage",
    [
        Stage.TOPIC_INIT,
        Stage.PROBLEM_DECOMPOSE,
        Stage.SEARCH_STRATEGY,
        Stage.LITERATURE_COLLECT,
        Stage.LITERATURE_SCREEN,
        Stage.KNOWLEDGE_EXTRACT,
        Stage.SYNTHESIS,
        Stage.HYPOTHESIS_GEN,
        Stage.EXPERIMENT_TASK_SPEC,
        Stage.CODE_AGENT_IMPLEMENT,
    ],
)
def test_stage_executor_mapping_values_are_callable(stage: Stage) -> None:
    assert callable(rc_executor._STAGE_EXECUTORS[stage])


class TestStageHealth:
    def test_stage_health_json_written(self, tmp_path: Path) -> None:
        from researchclaw.pipeline.executor import execute_stage
        from researchclaw.pipeline.stages import Stage

        config = RCConfig.load(
            Path(__file__).parent.parent / "config.researchclaw.example.yaml",
            check_paths=False,
        )
        result = execute_stage(
            Stage.TOPIC_INIT,
            run_dir=tmp_path,
            run_id="test-health",
            config=config,
            adapters=AdapterBundle(),
            auto_approve_gates=True,
        )
        health_path = tmp_path / "stage-01" / "stage_health.json"
        assert result is not None
        assert health_path.exists()

    def test_stage_health_has_required_fields(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from researchclaw.pipeline.executor import execute_stage
        from researchclaw.pipeline.stages import Stage

        config = RCConfig.load(
            Path(__file__).parent.parent / "config.researchclaw.example.yaml",
            check_paths=False,
        )

        with patch("researchclaw.pipeline.executor.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.chat.return_value = MagicMock(
                content='{"topic": "test", "research_questions": ["q1"]}'
            )
            mock_llm_cls.from_rc_config.return_value = mock_client

            execute_stage(
                Stage.TOPIC_INIT,
                run_dir=tmp_path,
                run_id="test-health-fields",
                config=config,
                adapters=AdapterBundle(),
                auto_approve_gates=True,
            )

        health_path = tmp_path / "stage-01" / "stage_health.json"
        if health_path.exists():
            data = json.loads(health_path.read_text(encoding="utf-8"))
            assert "stage_id" in data
            assert "run_id" in data
            assert "duration_sec" in data
            assert "status" in data
            assert "timestamp" in data
            assert data["duration_sec"] >= 0


    def test_stage_health_duration_positive(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from researchclaw.pipeline.executor import execute_stage
        from researchclaw.pipeline.stages import Stage

        config = RCConfig.load(
            Path(__file__).parent.parent / "config.researchclaw.example.yaml",
            check_paths=False,
        )

        with patch("researchclaw.pipeline.executor.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.chat.return_value = MagicMock(
                content='{"topic": "test", "sub_problems": []}'
            )
            mock_llm_cls.from_rc_config.return_value = mock_client

            execute_stage(
                Stage.TOPIC_INIT,
                run_dir=tmp_path,
                run_id="test-duration",
                config=config,
                adapters=AdapterBundle(),
                auto_approve_gates=True,
            )

        health_path = tmp_path / "stage-01" / "stage_health.json"
        if health_path.exists():
            data = json.loads(health_path.read_text(encoding="utf-8"))
            assert data["duration_sec"] >= 0

# ── P1-1: Topic keyword extraction tests ──


class TestExtractTopicKeywords:
    def test_basic_extraction(self) -> None:
        keywords = rc_executor._extract_topic_keywords(
            "Agent-based Reinforcement Learning for Automated Scientific Discovery"
        )
        assert "agent-based" in keywords
        assert "reinforcement" in keywords
        assert "learning" in keywords
        assert "automated" in keywords
        assert "scientific" in keywords
        assert "discovery" in keywords
        # Stop words excluded
        # Stop words excluded
        assert "for" not in keywords

    def test_includes_domain_keywords(self) -> None:
        keywords = rc_executor._extract_topic_keywords(
            "Neural network pruning", domains=("ml", "optimization")
        )
        assert "neural" in keywords
        assert "network" in keywords
        assert "pruning" in keywords
        assert "ml" in keywords
        assert "optimization" in keywords

    def test_deduplication(self) -> None:
        keywords = rc_executor._extract_topic_keywords(
            "Learning to learn meta-learning", domains=("learning",)
        )
        assert keywords.count("learning") == 1

    def test_empty_topic(self) -> None:
        keywords = rc_executor._extract_topic_keywords("")
        assert keywords == []


# ── P1-2: Topic constraint block test ──


class TestTopicConstraintBlock:
    def test_contains_topic(self) -> None:
        block = rc_executor._topic_constraint_block("Transformer attention for time series")
        assert "Transformer attention for time series" in block

    def test_contains_prohibition(self) -> None:
        block = rc_executor._topic_constraint_block("anything")
        assert "PROHIBITED" in block
        assert "environment" in block.lower()
        assert "infrastructure" in block.lower()

    def test_hard_constraint_markers(self) -> None:
        block = rc_executor._topic_constraint_block("test")
        assert "HARD TOPIC CONSTRAINT" in block
        assert "END CONSTRAINT" in block


# ── Multi-perspective debate tests ──


class TestParseDecision:
    def test_proceed_default(self) -> None:
        assert rc_executor._parse_decision("Some random text") == "proceed"

    def test_proceed_explicit(self) -> None:
        text = "## Decision\nPROCEED\n## Justification\nGood results."
        assert rc_executor._parse_decision(text) == "proceed"

    def test_pivot_detected(self) -> None:
        text = "## Decision\nPIVOT\n## Justification\nHypotheses flawed."
        assert rc_executor._parse_decision(text) == "pivot"

    def test_refine_detected(self) -> None:
        text = "## Decision\nREFINE\n## Justification\nNeed more tuning."
        assert rc_executor._parse_decision(text) == "refine"

    def test_pivot_case_insensitive(self) -> None:
        text = "## Decision\npivot\n## Justification\nBad approach."
        assert rc_executor._parse_decision(text) == "pivot"

    def test_pivot_takes_priority_over_proceed(self) -> None:
        text = "## Decision\nPIVOT\nWe should not PROCEED."
        assert rc_executor._parse_decision(text) == "pivot"

    def test_decision_in_body_not_heading(self) -> None:
        text = "The results suggest we should PIVOT to a new approach."
        assert rc_executor._parse_decision(text) == "pivot"


class TestResearchDecisionStructured:
    def test_decision_produces_structured_json(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-15"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 14, "analysis.md", "# Analysis\nResults ok.")
        fake_llm = FakeLLMClient("## Decision\nPROCEED\n## Justification\nGood.")
        result = rc_executor._execute_research_decision(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.decision == "proceed"
        assert "decision_structured.json" in result.artifacts
        import json
        data = json.loads((stage_dir / "decision_structured.json").read_text())
        assert data["decision"] == "proceed"

    def test_pivot_decision_from_llm(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-15"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 14, "analysis.md", "# Analysis\nBad results.")
        fake_llm = FakeLLMClient("## Decision\nPIVOT\n## Justification\nFlawed.")
        result = rc_executor._execute_research_decision(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.decision == "pivot"

    def test_no_llm_defaults_to_proceed(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-15"
        stage_dir.mkdir(parents=True)
        result = rc_executor._execute_research_decision(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )
        assert result.decision == "proceed"


class TestMultiPerspectiveGenerate:
    def test_generates_all_perspectives(self, tmp_path: Path) -> None:
        roles = {
            "role_a": {"system": "You are A.", "user": "Do A for {topic}."},
            "role_b": {"system": "You are B.", "user": "Do B for {topic}."},
        }
        fake_llm = FakeLLMClient("perspective output")
        perspectives_dir = tmp_path / "perspectives"
        result = rc_executor._multi_perspective_generate(
            fake_llm, roles, {"topic": "test"}, perspectives_dir
        )
        assert set(result.keys()) == {"role_a", "role_b"}
        assert (perspectives_dir / "role_a.md").exists()
        assert (perspectives_dir / "role_b.md").exists()
        assert len(fake_llm.calls) == 2

    def test_saves_perspective_content(self, tmp_path: Path) -> None:
        roles = {"critic": {"system": "Be critical.", "user": "Criticize {topic}."}}
        fake_llm = FakeLLMClient("critical analysis here")
        perspectives_dir = tmp_path / "perspectives"
        rc_executor._multi_perspective_generate(
            fake_llm, roles, {"topic": "ml"}, perspectives_dir
        )
        content = (perspectives_dir / "critic.md").read_text()
        assert content == "critical analysis here"

    def test_renders_variables_in_prompts(self, tmp_path: Path) -> None:
        roles = {"r1": {"system": "Sys for {topic}.", "user": "User for {topic}."}}
        fake_llm = FakeLLMClient("ok")
        rc_executor._multi_perspective_generate(
            fake_llm, roles, {"topic": "RL"}, tmp_path / "p"
        )
        call = fake_llm.calls[0]
        assert "RL" in call[0]["content"]


class TestSynthesizePerspectives:
    def test_combines_perspectives(self) -> None:
        fake_llm = FakeLLMClient("synthesized result")
        pm = rc_executor.PromptManager()
        perspectives = {"innovator": "idea A", "contrarian": "idea B"}
        result = rc_executor._synthesize_perspectives(
            fake_llm, perspectives, "hypothesis_synthesize", pm
        )
        assert result == "synthesized result"
        # Check the user prompt contained both perspectives
        call_content = fake_llm.calls[0][0]["content"]
        assert "innovator" in call_content
        assert "contrarian" in call_content


class TestHypothesisGenDebate:
    def _acp_config(self, rc_config: RCConfig, tmp_path: Path) -> RCConfig:
        data = rc_config.to_dict()
        data["llm"]["provider"] = "acp"
        data["llm"]["acp"] = {
            "agent": "codex",
            "cwd": str(tmp_path),
            "acpx_command": "acpx",
            "session_name": "main",
            "debate_max_rounds": 2,
            "debate_confidence_min": 0.6,
        }
        data["security"]["hitl_required_stages"] = list(
            data["security"]["hitl_required_stages"]
        )
        return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    def test_hypothesis_gen_with_llm_creates_perspectives(
        self,
        tmp_path: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_novelty_check(monkeypatch)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-08"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 7, "synthesis.md", "# Synthesis\nGap found.")
        fake_llm = FakeLLMClient("## H1\nTest hypothesis")
        result = rc_executor._execute_hypothesis_gen(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.status == StageStatus.DONE
        assert "hypotheses.md" in result.artifacts
        perspectives_dir = stage_dir / "perspectives"
        assert perspectives_dir.exists()
        # Should have 3 perspective files (innovator, pragmatist, contrarian)
        perspective_files = list(perspectives_dir.glob("*.md"))
        assert len(perspective_files) == 3

    def test_hypothesis_gen_without_llm_no_perspectives(
        self,
        tmp_path: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_novelty_check(monkeypatch)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-08"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 7, "synthesis.md", "# Synthesis\nGap found.")
        result = rc_executor._execute_hypothesis_gen(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )
        assert result.status == StageStatus.DONE
        assert "hypotheses.md" in result.artifacts
        # No perspectives directory when no LLM
        assert not (stage_dir / "perspectives").exists()

    def test_hypothesis_gen_acp_debate_success_preserves_clean_contract(
        self,
        tmp_path: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_novelty_check(monkeypatch)
        from researchclaw.pipeline.stage_impls import _hypothesis_debate

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-08"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 7, "synthesis.md", "# Synthesis\nGap found.")
        config = self._acp_config(rc_config, tmp_path)

        def fake_debate(*args: object, **kwargs: object) -> str:
            _ = args, kwargs
            return (
                "## H1: Routing improves robustness\n"
                "Hypothesis Statement: Routing improves robustness."
            )

        monkeypatch.setattr(_hypothesis_debate, "run_acp_debate", fake_debate)

        result = rc_executor._execute_hypothesis_gen(
            stage_dir, run_dir, config, adapters, llm=FakeLLMClient("fallback")
        )

        output = (stage_dir / "hypotheses.md").read_text(encoding="utf-8")
        assert "hypotheses.md" in result.artifacts
        assert "Routing improves robustness" in output
        assert not (stage_dir / "perspectives").exists()
        assert "judge" not in output.lower()
        assert "debate" not in output.lower()

    def test_hypothesis_gen_acp_debate_failure_falls_back_to_perspectives(
        self,
        tmp_path: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_novelty_check(monkeypatch)
        from researchclaw.pipeline.stage_impls import _hypothesis_debate

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-08"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 7, "synthesis.md", "# Synthesis\nGap found.")
        config = self._acp_config(rc_config, tmp_path)

        def fake_debate(*args: object, **kwargs: object) -> str:
            _ = args, kwargs
            raise RuntimeError("export failed")

        monkeypatch.setattr(_hypothesis_debate, "run_acp_debate", fake_debate)
        fake_llm = FakeLLMClient("## H1\nFallback hypothesis")

        result = rc_executor._execute_hypothesis_gen(
            stage_dir, run_dir, config, adapters, llm=fake_llm
        )

        assert result.status == StageStatus.DONE
        assert (stage_dir / "perspectives").exists()
        assert len(list((stage_dir / "perspectives").glob("*.md"))) == 3

    def test_hypothesis_gen_acp_and_perspective_failures_use_defaults(
        self,
        tmp_path: Path,
        rc_config: RCConfig,
        adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _stub_novelty_check(monkeypatch)
        from researchclaw.pipeline.stage_impls import _hypothesis_debate, _synthesis

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-08"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 7, "synthesis.md", "# Synthesis\nGap found.")
        config = self._acp_config(rc_config, tmp_path)

        def fake_debate(*args: object, **kwargs: object) -> str:
            _ = args, kwargs
            raise RuntimeError("export failed")

        monkeypatch.setattr(_hypothesis_debate, "run_acp_debate", fake_debate)
        monkeypatch.setattr(
            _synthesis,
            "_multi_perspective_generate",
            lambda *args, **kwargs: {},
        )

        result = rc_executor._execute_hypothesis_gen(
            stage_dir, run_dir, config, adapters, llm=FakeLLMClient("unused")
        )

        output = (stage_dir / "hypotheses.md").read_text(encoding="utf-8")
        assert result.status == StageStatus.DONE
        assert "test-driven science" in output


class TestResultAnalysisDebate:
    def test_result_analysis_writes_workspace_native_artifacts(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-14"
        stage_dir.mkdir(parents=True)
        _write_prior_artifact(run_dir, 1, "goal.md", "# Goal\nTest")
        _write_prior_artifact(run_dir, 8, "hypotheses.md", "# H1\nTest")
        fake_llm = FakeLLMClient("## Analysis\nResults look good.")
        result = rc_executor._execute_result_analysis(
            stage_dir, run_dir, rc_config, adapters, llm=fake_llm
        )
        assert result.status == StageStatus.DONE
        assert "analysis.md" in result.artifacts
        assert "experiment_summary.json" in result.artifacts
        assert "provenance.json" in result.artifacts
        assert (stage_dir / "analysis.md").exists()
        assert (stage_dir / "experiment_summary.json").exists()
        assert (stage_dir / "provenance.json").exists()

    def test_result_analysis_without_llm_no_perspectives(
        self, tmp_path: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        stage_dir = run_dir / "stage-14"
        stage_dir.mkdir(parents=True)
        result = rc_executor._execute_result_analysis(
            stage_dir, run_dir, rc_config, adapters, llm=None
        )
        assert result.status == StageStatus.DONE
        assert "analysis.md" in result.artifacts
        assert not (stage_dir / "perspectives").exists()


class TestParseMetricsFromStdout:
    """Tests for _parse_metrics_from_stdout() helper."""

    def test_parses_simple_name_value(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "loss: 0.0042\naccuracy: 0.95"
        metrics = _parse_metrics_from_stdout(stdout)
        assert metrics["loss"] == pytest.approx(0.0042)
        assert metrics["accuracy"] == pytest.approx(0.95)

    def test_parses_compound_names(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "UCB (Stochastic) cumulative_regret: 361.9233\nEXP3 (Adversarial) total_rewards: 13368.4811"
        metrics = _parse_metrics_from_stdout(stdout)
        assert "UCB (Stochastic) cumulative_regret" in metrics
        assert metrics["UCB (Stochastic) cumulative_regret"] == pytest.approx(361.9233)

    def test_ignores_non_numeric_lines(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "Running experiment...\nloss: 0.5\nDone."
        metrics = _parse_metrics_from_stdout(stdout)
        assert len(metrics) == 1
        assert metrics["loss"] == pytest.approx(0.5)

    def test_empty_stdout_returns_empty_dict(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        assert _parse_metrics_from_stdout("") == {}

    def test_handles_negative_values(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "UCB (Adversarial) cumulative_regret: -3877.5323"
        metrics = _parse_metrics_from_stdout(stdout)
        assert metrics["UCB (Adversarial) cumulative_regret"] == pytest.approx(-3877.5323)

    def test_filters_log_lines(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = (
            "Running experiments for support set size: 1\n"
            "Loading model weights: 42\n"
            "Training epoch: 5\n"
            "loss: 0.123\n"
            "accuracy: 0.95\n"
        )
        metrics = _parse_metrics_from_stdout(stdout)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert len(metrics) == 2  # log lines should be excluded

    def test_filters_long_name_lines(self) -> None:
        from researchclaw.pipeline.executor import _parse_metrics_from_stdout

        stdout = "this is a very long status message that should not be a metric: 42\n"
        metrics = _parse_metrics_from_stdout(stdout)
        assert len(metrics) == 0


class TestDetectRuntimeIssues:
    """Tests for _detect_runtime_issues() helper."""

    def _make_sandbox_result(
        self,
        metrics: dict | None = None,
        stdout: str = "",
        stderr: str = "",
    ):
        from types import SimpleNamespace

        return SimpleNamespace(
            metrics=metrics or {},
            stdout=stdout,
            stderr=stderr,
            returncode=0,
            elapsed_sec=1.0,
            timed_out=False,
        )

    def test_no_issues_returns_empty_string(self) -> None:
        r = self._make_sandbox_result(metrics={"loss": 0.5}, stdout="loss: 0.5")
        assert rc_executor._detect_runtime_issues(r) == ""

    def test_detects_nan_in_metrics(self) -> None:
        r = self._make_sandbox_result(metrics={"loss": float("nan")})
        result = rc_executor._detect_runtime_issues(r)
        assert "NaN" in result
        assert "loss" in result

    def test_detects_inf_in_metrics(self) -> None:
        r = self._make_sandbox_result(metrics={"loss": float("inf")})
        result = rc_executor._detect_runtime_issues(r)
        assert "Inf" in result

    def test_detects_nan_in_stdout(self) -> None:
        r = self._make_sandbox_result(stdout="accuracy: nan\nloss: 0.5")
        result = rc_executor._detect_runtime_issues(r)
        assert "NaN" in result or "nan" in result

    def test_detects_runtime_warning_in_stderr(self) -> None:
        stderr = (
            "optimizers.py:76: RuntimeWarning: invalid value encountered in divide\n"
            "  directions = np.vstack((directions[1:], new_direction / norm))\n"
        )
        r = self._make_sandbox_result(stderr=stderr)
        result = rc_executor._detect_runtime_issues(r)
        assert "RuntimeWarning" in result
        assert "invalid value" in result

    def test_detects_division_error_in_stderr(self) -> None:
        stderr = "ZeroDivisionError: division by zero\n"
        r = self._make_sandbox_result(stderr=stderr)
        result = rc_executor._detect_runtime_issues(r)
        assert "Error" in result

    def test_ignores_benign_stderr(self) -> None:
        # Non-warning stderr should be ignored
        r = self._make_sandbox_result(stderr="Loading module...\nDone.\n")
        assert rc_executor._detect_runtime_issues(r) == ""

    def test_combined_nan_and_stderr(self) -> None:
        r = self._make_sandbox_result(
            metrics={"accuracy": float("nan")},
            stderr="RuntimeWarning: invalid value\n",
        )
        result = rc_executor._detect_runtime_issues(r)
        assert "NaN" in result
        assert "RuntimeWarning" in result

    def test_detects_dummy_metric_identical_values(self) -> None:
        stdout = (
            "UCB (Stochastic) convergence_rate: 1.0000\n"
            "UCB (Adversarial) convergence_rate: 1.0000\n"
            "Thompson (Stochastic) convergence_rate: 1.0000\n"
            "Thompson (Adversarial) convergence_rate: 1.0000\n"
        )
        r = self._make_sandbox_result(stdout=stdout)
        result = rc_executor._detect_runtime_issues(r)
        assert "DUMMY" in result
        assert "convergence_rate" in result

    def test_no_dummy_metric_when_values_differ(self) -> None:
        stdout = (
            "UCB (Stochastic) regret: 78.5\n"
            "Thompson (Stochastic) regret: 121.0\n"
            "EpsilonGreedy (Stochastic) regret: 42.1\n"
        )
        r = self._make_sandbox_result(stdout=stdout)
        result = rc_executor._detect_runtime_issues(r)
        assert "DUMMY" not in result


class TestRemoveBibtexEntries:
    """Tests for _remove_bibtex_entries() helper."""

    def test_removes_specified_keys(self) -> None:
        bib = (
            '@article{smith2024,\n  title={Good Paper},\n  author={Smith},\n}\n\n'
            '@article{venus2024,\n  title={Venus Exploration},\n  author={NASA},\n}\n'
        )
        result = rc_executor._remove_bibtex_entries(bib, {"venus2024"})
        assert "smith2024" in result
        assert "venus2024" not in result

    def test_keeps_all_when_no_match(self) -> None:
        bib = '@article{smith2024,\n  title={Paper},\n}\n'
        result = rc_executor._remove_bibtex_entries(bib, {"other_key"})
        assert "smith2024" in result

    def test_empty_bib(self) -> None:
        assert rc_executor._remove_bibtex_entries("", {"key"}) == ""


class TestRemoveCitationsFromText:
    """Tests for _remove_citations_from_text() helper."""

    def test_removes_latex_cite(self) -> None:
        text = r"As shown in \cite{venus2024}, the results are..."
        result = rc_executor._remove_citations_from_text(text, {"venus2024"})
        assert "venus2024" not in result
        assert "results are" in result

    def test_removes_markdown_cite(self) -> None:
        text = "Prior work [venus2024] explored this topic."
        result = rc_executor._remove_citations_from_text(text, {"venus2024"})
        assert "venus2024" not in result

    def test_cleans_multi_cite_comma(self) -> None:
        text = r"\cite{good2024,venus2024}"
        result = rc_executor._remove_citations_from_text(text, {"venus2024"})
        assert r"\cite{good2024}" in result


class TestCollectRawExperimentMetrics:
    """Tests for _collect_raw_experiment_metrics() helper."""

    def test_returns_empty_when_no_runs(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        block, has_parsed = rc_executor._collect_raw_experiment_metrics(run_dir)
        assert block == ""
        assert not has_parsed

    def test_extracts_metrics_from_stdout(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True)
        payload = {
            "metrics": {},
            "stdout": "UCB regret: 361.92\nThompson regret: 576.24\n",
        }
        (runs_dir / "run-1.json").write_text(json.dumps(payload))
        result, has_parsed = rc_executor._collect_raw_experiment_metrics(run_dir)
        assert "361.92" in result
        assert "576.24" in result
        assert "1 run(s)" in result
        assert not has_parsed

    def test_extracts_from_metrics_dict(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True)
        payload = {"metrics": {"loss": 0.042, "accuracy": 0.95}, "stdout": ""}
        (runs_dir / "run-1.json").write_text(json.dumps(payload))
        result, has_parsed = rc_executor._collect_raw_experiment_metrics(run_dir)
        assert "loss" in result
        assert "0.042" in result
        assert has_parsed

    def test_deduplicates_metrics(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True)
        payload = {
            "metrics": {"loss": 0.5},
            "stdout": "loss: 0.5\nloss: 0.5\n",
        }
        (runs_dir / "run-1.json").write_text(json.dumps(payload))
        result, _ = rc_executor._collect_raw_experiment_metrics(run_dir)
        # "loss: 0.5" should appear only once (deduplicated)
        assert result.count("loss: 0.5") == 1


class TestCollectExperimentEvidence:
    """Tests for _collect_experiment_evidence() helper."""

    def test_returns_empty_when_no_artifacts(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        assert rc_executor._collect_experiment_evidence(run_dir) == ""

    def test_includes_main_py_code(self, run_dir: Path) -> None:
        exp_dir = run_dir / "stage-10" / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "main.py").write_text("print('hello')", encoding="utf-8")
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "main.py" in result
        assert "hello" in result

    def test_includes_run_metrics(self, run_dir: Path) -> None:
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"metrics": {"loss": 0.5}, "elapsed_sec": 3.2}),
            encoding="utf-8",
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "loss" in result
        assert "0.5" in result

    def test_includes_stderr_excerpt(self, run_dir: Path) -> None:
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "metrics": {"loss": 0.5},
                "stderr": "RuntimeWarning: divide by zero",
            }),
            encoding="utf-8",
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "divide by zero" in result

    def test_includes_refinement_summary(self, run_dir: Path) -> None:
        refine_dir = run_dir / "stage-13"
        refine_dir.mkdir(parents=True, exist_ok=True)
        (refine_dir / "refinement_log.json").write_text(
            json.dumps({
                "iterations": [{"iteration": 1}, {"iteration": 2}],
                "converged": True,
                "stop_reason": "no_improvement_for_2_iterations",
                "best_metric": 0.3,
            }),
            encoding="utf-8",
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "iterations_executed" in result
        assert "2" in result

    def test_includes_actual_trial_count(self, run_dir: Path) -> None:
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"metrics": {"loss": 0.5}}), encoding="utf-8"
        )
        result = rc_executor._collect_experiment_evidence(run_dir)
        assert "1 time(s)" in result
        assert "CRITICAL" in result


class TestWritePaperSections:
    """Tests for _write_paper_sections() multi-call writing."""

    def test_produces_three_part_draft(self) -> None:
        call_count = {"n": 0}
        parts = [
            "# Test Title\n\n## Abstract\nTest abstract.\n\n## Introduction\nTest intro.\n\n## Related Work\nTest related.",
            "## Method\nTest method.\n\n## Experiments\nTest experiments.",
            "## Results\nTest results.\n\n## Discussion\nTest discussion.\n\n## Limitations\nTest limits.\n\n## Conclusion\nTest conclusion.",
        ]

        class MultiCallLLM:
            def __init__(self):
                self.calls: list = []

            def chat(self, messages, **kwargs):
                self.calls.append(messages)
                from researchclaw.llm.client import LLMResponse
                idx = len(self.calls) - 1
                return LLMResponse(content=parts[min(idx, 2)], model="fake")

        llm = MultiCallLLM()
        from researchclaw.prompts import PromptManager
        pm = PromptManager()

        draft = rc_executor._write_paper_sections(
            llm=llm,
            pm=pm,
            preamble="Test preamble",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="Test outline",
        )

        assert llm.calls is not None
        assert len(llm.calls) == 3
        assert "## Abstract" in draft
        assert "## Method" in draft
        assert "## Results" in draft
        assert "## Conclusion" in draft

    def test_each_call_receives_prior_context(self) -> None:
        class ContextTrackingLLM:
            def __init__(self):
                self.user_prompts: list[str] = []

            def chat(self, messages, **kwargs):
                for m in messages:
                    if m.get("role") == "user":
                        self.user_prompts.append(m["content"])
                from researchclaw.llm.client import LLMResponse
                return LLMResponse(content="## Section\nContent here.", model="fake")

        llm = ContextTrackingLLM()
        from researchclaw.prompts import PromptManager
        pm = PromptManager()

        rc_executor._write_paper_sections(
            llm=llm,
            pm=pm,
            preamble="Preamble",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="Outline",
        )

        assert len(llm.user_prompts) == 3
        # Call 2 and 3 should contain "sections written so far"
        assert "sections written so far" in llm.user_prompts[1]
        assert "completing a paper" in llm.user_prompts[2]


class TestLoadHardwareProfile:
    """Tests for _load_hardware_profile()."""

    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "run"
        d.mkdir()
        return d

    def test_loads_valid_profile(self, run_dir: Path) -> None:
        stage = run_dir / "stage-01"
        stage.mkdir()
        profile = {"has_gpu": True, "gpu_type": "mps", "tier": "limited"}
        (stage / "hardware_profile.json").write_text(
            json.dumps(profile), encoding="utf-8"
        )
        result = rc_executor._load_hardware_profile(run_dir)
        assert result is not None
        assert result["gpu_type"] == "mps"

    def test_returns_none_when_missing(self, run_dir: Path) -> None:
        assert rc_executor._load_hardware_profile(run_dir) is None

    def test_returns_none_on_invalid_json(self, run_dir: Path) -> None:
        stage = run_dir / "stage-01"
        stage.mkdir()
        (stage / "hardware_profile.json").write_text("not json", encoding="utf-8")
        assert rc_executor._load_hardware_profile(run_dir) is None


class TestExpandSearchQueries:
    """Tests for _expand_search_queries()."""

    def test_adds_broader_queries(self) -> None:
        queries = ["gradient descent optimization algorithms"]
        topic = "Comparing gradient descent optimization algorithms on benchmark functions"
        result = rc_executor._expand_search_queries(queries, topic)
        assert len(result) > len(queries)

    def test_deduplicates(self) -> None:
        queries = ["gradient descent survey"]
        topic = "gradient descent optimization"
        result = rc_executor._expand_search_queries(queries, topic)
        lowered = [q.lower().strip() for q in result]
        assert len(lowered) == len(set(lowered))

    def test_preserves_original_queries(self) -> None:
        queries = ["query A", "query B"]
        topic = "some research topic about machine learning methods"
        result = rc_executor._expand_search_queries(queries, topic)
        assert result[0] == "query A"
        assert result[1] == "query B"

    def test_adds_survey_benchmark_variants(self) -> None:
        queries = ["deep learning"]
        topic = "deep learning for image classification with limited data"
        result = rc_executor._expand_search_queries(queries, topic)
        has_survey = any("survey" in q.lower() for q in result)
        has_benchmark = any("benchmark" in q.lower() for q in result)
        assert has_survey
        assert has_benchmark


# ── R4-1: Experiment Budget Guard Tests ──────────────────────────────


class TestComputeBudgetBlock:
    """Test compute_budget prompt block injection (R4-1a)."""

    def test_compute_budget_block_exists_in_prompt_manager(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("compute_budget")
        assert "time_budget_sec" in block or "Compute Budget" in block


# ── R4-2: Data Integrity Enforcement Tests ───────────────────────────


class TestDataIntegrityBlock:
    """Test paper draft blocked when no metrics exist (R4-2a)."""

    def test_paper_draft_blocked_with_no_metrics(
        self, tmp_path: Path, run_dir: Path, rc_config: RCConfig, adapters: AdapterBundle,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Write prior artifacts with NO metrics
        _write_prior_artifact(run_dir, 16, "outline.md", "# Outline\n## Abstract\n")
        # No experiment_summary.json, no run files with metrics
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"run_id": "run-1", "status": "failed", "metrics": {}, "timed_out": True}),
            encoding="utf-8",
        )

        stage_dir = run_dir / "stage-17"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Ensure domain detection returns an empirical domain so the block triggers
        from researchclaw.pipeline.stage_impls import _paper_writing
        monkeypatch.setattr(
            _paper_writing, "_detect_domain",
            lambda topic, domains=(): ("ml", "machine learning", "NeurIPS, ICML, ICLR"),
        )

        llm = FakeLLMClient("should not be called")
        result = rc_executor._execute_paper_draft(
            stage_dir, run_dir, rc_config, adapters, llm=llm
        )

        assert result.status == StageStatus.FAILED
        draft = (stage_dir / "paper_draft.md").read_text(encoding="utf-8")
        assert "Blocked" in draft or "BLOCKED" in draft or "no metrics" in draft.lower()
        # LLM should NOT have been called
        assert len(llm.calls) == 0

    def test_paper_draft_proceeds_with_metrics(
        self, tmp_path: Path, run_dir: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        _write_prior_artifact(run_dir, 16, "outline.md", "# Outline\n## Abstract\n")
        # Write experiment data with real metrics
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({
                "run_id": "run-1",
                "status": "completed",
                "metrics": {"best_loss": 0.123},
                "stdout": "best_loss: 0.123\n",
            }),
            encoding="utf-8",
        )

        stage_dir = run_dir / "stage-17"
        stage_dir.mkdir(parents=True, exist_ok=True)

        llm = FakeLLMClient("# Paper Title\n## Abstract\nSome abstract text.")
        result = rc_executor._execute_paper_draft(
            stage_dir, run_dir, rc_config, adapters, llm=llm
        )

        # Should proceed (LLM was called)
        assert len(llm.calls) >= 1
        # The prompt should contain anti-fabrication instructions
        all_prompts = " ".join(
            msg["content"] for call in llm.calls for msg in call
        )
        assert "Data Integrity" in all_prompts or "ONLY report numbers" in all_prompts


# ── R4-3: Conference-Grade Title Guidelines Tests ────────────────────


class TestTitleGuidelines:
    """Test title_guidelines and abstract_structure blocks (R4-3)."""

    def test_title_guidelines_block_exists(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("title_guidelines")
        assert "novelty" in block.lower() or "TITLE RULES" in block
        assert "14 words" in block or "15 words" in block or "concrete" in block.lower()

    def test_abstract_structure_block_exists(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("abstract_structure")
        assert "5-sentence" in block or "problem" in block.lower()

    def test_title_guidelines_injected_into_paper_draft(
        self, tmp_path: Path, run_dir: Path, rc_config: RCConfig, adapters: AdapterBundle
    ) -> None:
        _write_prior_artifact(run_dir, 16, "outline.md", "# Outline\n")
        runs_dir = run_dir / "stage-12" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "run-1.json").write_text(
            json.dumps({"run_id": "run-1", "status": "completed",
                        "metrics": {"best_loss": 0.1}, "stdout": "best_loss: 0.1\n"}),
            encoding="utf-8",
        )

        stage_dir = run_dir / "stage-17"
        stage_dir.mkdir(parents=True, exist_ok=True)

        llm = FakeLLMClient("# Paper Title\n## Abstract\nText.")
        rc_executor._execute_paper_draft(
            stage_dir, run_dir, rc_config, adapters, llm=llm
        )

        all_prompts = " ".join(
            msg["content"] for call in llm.calls for msg in call
        )
        assert "Title" in all_prompts or "TITLE" in all_prompts


# ── R4-4: Conference-Grade Writing Quality Tests ─────────────────────


class TestConferenceWritingQuality:
    """Test enhanced writing prompts and writing_guide.py (R4-4)."""

    def test_writing_guide_format_all(self) -> None:
        from researchclaw.writing_guide import format_writing_tips

        result = format_writing_tips()
        assert "Conference Writing Best Practices" in result
        assert "Title" in result
        assert "Common Rejections" in result

    def test_writing_guide_format_subset(self) -> None:
        from researchclaw.writing_guide import format_writing_tips

        result = format_writing_tips(["title", "abstract"])
        assert "Title" in result
        assert "Abstract" in result
        assert "Common Rejections" not in result

    def test_paper_draft_system_includes_principles(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        sp = pm.for_stage(
            "paper_draft",
            preamble="test",
            topic_constraint="test",
            exp_metrics_instruction="test",
            citation_instruction="test",
            outline="test",
        )
        # System prompt should mention key principles
        assert "NOVELTY" in sp.system or "novelty" in sp.system.lower()
        assert "fabricate" in sp.system.lower() or "real experimental" in sp.system.lower()


# ── R5-3: NaN/Divergence Fast-Fail Tests ────────────────────────────


class TestNaNDivergenceDetection:
    """Test NaN/Inf filtering and divergence detection (R5-3)."""

    def test_runtime_issues_detects_diverging_loss(self) -> None:
        from types import SimpleNamespace

        fake_result = SimpleNamespace(
            metrics={"best_loss": 500.0},
            stdout="best_loss: 500.0\n",
            stderr="",
        )
        issues = rc_executor._detect_runtime_issues(fake_result)
        assert "DIVERGING" in issues or "diverging" in issues.lower()

    def test_compute_budget_includes_nan_guard(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("compute_budget")
        assert "NaN" in block or "nan" in block.lower() or "divergence" in block.lower()


# ── R5-4: Experiment Harness Template Tests ──────────────────────────


class TestExperimentHarness:
    """Test the immutable experiment harness (R5-4)."""

    def test_harness_should_stop(self) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=1)
        assert not h.should_stop()  # Just created, not at 80% yet
        import time
        time.sleep(0.9)
        assert h.should_stop()  # Should be past 80% of 1s

    def test_harness_report_metric(self, capsys: pytest.CaptureFixture[str]) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=60)
        h.report_metric("best_loss", 0.123)
        captured = capsys.readouterr()
        assert "best_loss: 0.123" in captured.out
        assert h._metrics["best_loss"] == 0.123

    def test_harness_rejects_nan(self, capsys: pytest.CaptureFixture[str]) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=60)
        h.report_metric("bad", float("nan"))
        captured = capsys.readouterr()
        assert "bad" not in h._metrics
        assert "non-finite" in captured.err.lower() or "WARNING" in captured.err

    def test_harness_rejects_inf(self, capsys: pytest.CaptureFixture[str]) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=60)
        h.report_metric("bad", float("inf"))
        assert "bad" not in h._metrics

    def test_harness_finalize(self, tmp_path: Path) -> None:
        import os
        from researchclaw.experiment.harness_template import ExperimentHarness

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            h = ExperimentHarness(time_budget=60)
            h.report_metric("accuracy", 0.95)
            h.report_metric("loss", 0.05)
            h.log_result({"condition": "A", "value": 1.0})
            h.finalize()

            results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
            assert results["metrics"]["accuracy"] == 0.95
            assert results["metrics"]["loss"] == 0.05
            assert len(results["results"]) == 1
        finally:
            os.chdir(old_cwd)

    def test_harness_progress(self) -> None:
        from researchclaw.experiment.harness_template import ExperimentHarness

        h = ExperimentHarness(time_budget=1000)
        assert h.progress < 0.01  # Just started
        assert 0.0 <= h.progress <= 1.0

    def test_prompt_mentions_harness(self) -> None:
        from researchclaw.prompts import PromptManager

        pm = PromptManager()
        block = pm.block("compute_budget")
        assert "experiment_harness" in block or "ExperimentHarness" in block


# ===================================================================
# R6 Tests — Post-E2E Failure Analysis Fixes
# ===================================================================


class TestConsecutiveEmptyMetrics:
    """R6-4: Pipeline should detect consecutive empty-metrics REFINE cycles."""

    def test_detects_consecutive_empty(self, tmp_path: Path) -> None:
        """Two cycles with empty metrics should return True."""
        from researchclaw.pipeline.runner import _consecutive_empty_metrics

        run_dir = tmp_path / "run"
        # Current cycle (stage-14)
        s14 = run_dir / "stage-14"
        s14.mkdir(parents=True)
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))
        # Previous cycle (stage-14_v1)
        s14v1 = run_dir / "stage-14_v1"
        s14v1.mkdir(parents=True)
        (s14v1 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))

        assert _consecutive_empty_metrics(run_dir, pivot_count=1) is True

    def test_not_empty_when_metrics_exist(self, tmp_path: Path) -> None:
        """If any cycle has real metrics, return False."""
        from researchclaw.pipeline.runner import _consecutive_empty_metrics

        run_dir = tmp_path / "run"
        s14 = run_dir / "stage-14"
        s14.mkdir(parents=True)
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {"loss": 0.5}},
        }))
        s14v1 = run_dir / "stage-14_v1"
        s14v1.mkdir(parents=True)
        (s14v1 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))

        assert _consecutive_empty_metrics(run_dir, pivot_count=1) is False

    def test_false_when_no_previous_cycle(self, tmp_path: Path) -> None:
        """First cycle (no v1) should return False."""
        from researchclaw.pipeline.runner import _consecutive_empty_metrics

        run_dir = tmp_path / "run"
        s14 = run_dir / "stage-14"
        s14.mkdir(parents=True)
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }))

        # No stage-14_v1 exists
        assert _consecutive_empty_metrics(run_dir, pivot_count=1) is False


# ===================================================================
# R7 Tests — Experiment-Paper Quality Alignment
# ===================================================================


class TestMultiConditionEnforcement:
    """R7-1: Code generation prompt must enforce multi-condition experiments."""

    def test_code_generation_prompt_has_multi_condition_block(self) -> None:
        """The code_generation prompt should contain multi-condition instructions."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="test topic",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions:\n  - echo_chamber\n  - bridge_building\n  - random",
        )
        assert "MULTI-CONDITION REQUIREMENT" in sp.user
        assert "condition=" in sp.user
        assert "SUMMARY" in sp.user

    def test_multi_condition_labels_required(self) -> None:
        """Prompt must mention per-condition labeled output format."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="test",
            metric="loss",
            pkg_hint="",
            exp_plan="treatments: [A, B, C]",
        )
        assert "condition=<name>" in sp.user


class TestEvidenceBoundedWriting:
    """R7-2: Paper draft prompt must enforce evidence-bounded claims."""

    def test_paper_draft_has_evidence_bounding_rules(self) -> None:
        """System prompt should contain evidence-bounding rules."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "paper_draft",
            preamble="test preamble",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="# Outline",
        )
        assert "EVIDENCE-BOUNDING RULES" in sp.system
        assert "title" in sp.system.lower()
        assert "causal claim" in sp.system.lower() or "causal claims" in sp.system.lower()

    def test_hedging_language_guidance(self) -> None:
        """Should suggest hedged alternatives like 'Toward...' for partial data."""
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "paper_draft",
            preamble="",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="",
        )
        assert "Toward" in sp.system or "Investigating" in sp.system


# ===================================================================
# R8 Tests — AutoBench Round 1 Fixes
# ===================================================================


class TestBreadthFirstPrompt:
    """R8-1: Code generation prompt should require breadth-first condition ordering."""

    def test_breadth_first_in_code_generation(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="test",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions: [A, B, C]",
        )
        assert "BREADTH-FIRST" in sp.user
        assert "ONE representative" in sp.user


# ===================================================================
# R9 Tests — AutoBench Round 2 Fixes
# ===================================================================


class TestCodeGenTopicNeutral:
    """R9-1: Code generation prompt should be topic-neutral, not optimization-biased."""

    def test_no_gradient_descent_bias(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="multi-agent simulation",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions: [L1, L2, L3, L4]",
        )
        # Should NOT contain optimization-specific examples as recommended approaches
        assert "Adam" not in sp.user
        assert "SGD" not in sp.user
        assert "Rosenbrock" not in sp.user
        # "gradient descent" may appear as anti-pattern warning but not as example
        assert "e.g., gradient descent" not in sp.user

    def test_topic_relevant_guidance(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.for_stage(
            "code_generation",
            topic="multi-agent simulation",
            metric="primary_metric",
            pkg_hint="",
            exp_plan="conditions: [L1, L2, L3, L4]",
        )
        # Should contain generic guidance that works for any topic
        assert "simulation" in sp.user.lower() or "appropriate" in sp.user.lower()
        assert "ACTUAL experiment" in sp.user or "relevant to the TOPIC" in sp.user


class TestRefineTopicAlignment:
    """R9-2: Refine prompt should include topic-code alignment check."""

    def test_topic_alignment_in_refine_prompt(self) -> None:
        from researchclaw.prompts import PromptManager
        pm = PromptManager()
        sp = pm.sub_prompt(
            "iterative_improve",
            metric_key="primary_metric",
            metric_direction="maximize",
            files_context="# main.py\nprint('hello')",
            run_summaries="{}",
            condition_coverage_hint="",
            topic="multi-agent diversity scaling",
            exp_plan_anchor="",
        )
        assert "EXPERIMENT PLAN ANCHOR" in sp.user
        assert "multi-agent diversity scaling" in sp.user
        assert "NEVER rename" in sp.user


# =====================================================================
# _validate_draft_quality tests
# =====================================================================


def _make_prose(word_count: int) -> str:  # noqa: E302
    """Generate flowing prose text of approximately *word_count* words."""
    sentence = (
        "This is a flowing academic prose sentence "
        "that demonstrates our research findings. "
    )
    words_per = len(sentence.split())
    return sentence * (word_count // words_per + 1)


def _make_bullets(word_count: int) -> str:
    """Generate bullet-point text of approximately *word_count* words."""
    line = "- This is a bullet point about a research finding\n"
    words_per = len(line.split())
    return line * (word_count // words_per + 1)


def _make_comparative_prose(word_count: int) -> str:
    """Generate related-work style prose with comparative language."""
    sentence = (
        "Unlike prior work that focuses on simple baselines, "
        "our approach differs by incorporating novel techniques. "
        "In contrast to existing methods, we address key limitations. "
        "However, while previous approaches rely on heuristics, "
        "our method provides theoretical guarantees. "
    )
    words_per = len(sentence.split())
    return sentence * (word_count // words_per + 1)


def _make_results_prose(word_count: int) -> str:
    """Generate results prose with statistical measures."""
    sentence = (
        "Our method achieves 85.3 ± 1.2 accuracy averaged over 5 seeds. "
        "The baseline comparison yields a p-value of 0.003, confirming "
        "statistical significance with 95% confidence interval. "
    )
    words_per = len(sentence.split())
    return sentence * (word_count // words_per + 1)


def _build_draft(**section_overrides: str) -> str:
    """Build a paper draft with default prose sections."""
    defaults = {
        "Abstract": _make_prose(200),
        "Introduction": _make_prose(900),
        "Related Work": _make_comparative_prose(700),
        "Method": _make_prose(1200),
        "Experiments": _make_prose(1000),
        "Results": _make_results_prose(700),
        "Discussion": _make_prose(500),
        "Limitations": _make_prose(250),
        "Conclusion": _make_prose(250),
    }
    defaults.update(section_overrides)
    parts = ["# My Research Title\n"]
    for heading, body in defaults.items():
        parts.append(f"# {heading}\n{body}\n")
    return "\n".join(parts)


class TestValidateDraftQuality:
    """Tests for _validate_draft_quality()."""

    def test_short_section_triggers_warning(self) -> None:
        """Short Method section triggers expand warning."""
        draft = _build_draft(Method=_make_prose(200))
        result = rc_executor._validate_draft_quality(draft)
        assert any("Method" in w for w in result["overall_warnings"])
        assert any("EXPAND" in d or "Expand" in d
                    for d in result["revision_directives"])

    def test_bullet_density_triggers_warning(self) -> None:
        """Bullet-heavy Method section triggers rewrite warning."""
        draft = _build_draft(Method=_make_bullets(1200))
        result = rc_executor._validate_draft_quality(draft)
        assert any(
            "bullet" in w.lower() or "density" in w.lower()
            for w in result["overall_warnings"]
        )
        assert any("REWRITE" in d for d in result["revision_directives"])

    def test_clean_draft_no_warnings(self) -> None:
        """Balanced prose draft produces zero warnings."""
        draft = _build_draft()
        result = rc_executor._validate_draft_quality(draft)
        assert len(result["overall_warnings"]) == 0
        assert len(result["revision_directives"]) == 0

    def test_balance_warning(self) -> None:
        """Large imbalance between sections triggers balance warning."""
        draft = _build_draft(
            Introduction=_make_prose(1500),
            Results=_make_prose(100),
        )
        result = rc_executor._validate_draft_quality(draft)
        bal = [w for w in result["overall_warnings"]
               if "imbalance" in w.lower()]
        assert len(bal) >= 1, (
            f"Expected balance warning, got: {result['overall_warnings']}"
        )

    def test_writes_json_to_stage_dir(self, tmp_path: Path) -> None:
        """Quality report is written as draft_quality.json."""
        draft = _build_draft(Method=_make_prose(200))
        rc_executor._validate_draft_quality(draft, stage_dir=tmp_path)
        assert (tmp_path / "draft_quality.json").exists()
        data = json.loads(
            (tmp_path / "draft_quality.json").read_text(encoding="utf-8")
        )
        assert "section_analysis" in data
        assert "overall_warnings" in data
        assert "revision_directives" in data


class TestExperimentValidatorPrecision:
    def test_deep_validation_detects_undefined_helper_calls(self) -> None:
        from researchclaw.experiment.validator import deep_validate_files

        issues = deep_validate_files(
            {
                "main.py": (
                    "def main():\n"
                    "    create_empty_csv('tmp.csv', ['a'])\n\n"
                    "if __name__ == '__main__':\n"
                    "    main()\n"
                )
            }
        )

        assert any(
            "Call to undefined function 'create_empty_csv()'" in issue
            for issue in issues
        )

    def test_deep_validation_allows_inherited_single_core_method_subclass(
        self,
    ) -> None:
        from researchclaw.experiment.validator import deep_validate_files

        issues = deep_validate_files(
            {
                "main.py": (
                    "class BaseVerifier:\n"
                    "    def __init__(self, scale=1.0):\n"
                    "        self.scale = float(scale)\n\n"
                    "class ChildVerifier(BaseVerifier):\n"
                    "    def predict(self, value):\n"
                    "        total = value * self.scale\n"
                    "        shifted = total + 1.0\n"
                    "        centered = shifted - 0.5\n"
                    "        bounded = max(centered, 0.0)\n"
                    "        return {'score': bounded}\n"
                )
            }
        )

        assert not any(
            "Class 'ChildVerifier' has only 1 non-dunder method" in issue
            for issue in issues
        )

    def test_deep_validation_detects_duplicate_algorithm_classes_across_files(
        self,
    ) -> None:
        from researchclaw.experiment.validator import deep_validate_files

        issues = deep_validate_files(
            {
                "main.py": (
                    "class DuplicateVerifier:\n"
                    "    def __init__(self, bias=0.0):\n"
                    "        self.bias = float(bias)\n\n"
                    "    def predict(self, value):\n"
                    "        shifted = value + self.bias\n"
                    "        bounded = max(shifted, 0.0)\n"
                    "        return {'score': bounded}\n"
                ),
                "models.py": (
                    "class DuplicateVerifier:\n"
                    "    def __init__(self, bias=0.0):\n"
                    "        self.bias = float(bias)\n\n"
                    "    def predict(self, value):\n"
                    "        shifted = value + self.bias\n"
                    "        bounded = max(shifted, 0.0)\n"
                    "        return {'score': bounded}\n"
                ),
            }
        )

        assert any(
            "Class 'DuplicateVerifier' is defined in multiple files" in issue
            for issue in issues
        )
        assert not any(
            "Classes 'DuplicateVerifier' and 'DuplicateVerifier' have identical"
            in issue
            for issue in issues
        )
