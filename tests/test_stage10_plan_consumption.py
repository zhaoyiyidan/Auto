from __future__ import annotations

import json
import subprocess
from pathlib import Path

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.experiment.workspace import WorkspaceAgentResult
from researchclaw.pipeline.stages import StageStatus


PLAN_MD = """
# Experiment Plan

## Hypotheses
H1 tests whether corrected accumulation semantics match true large-batch training.

## Baselines
Compare against true large-batch and fixed accumulation variants.

## Ablations
Disable corrected clipping and scheduler stepping separately.

## Metrics
Report time-to-target loss and update cosine similarity.

## Decision Criteria
Support H1 when corrected accumulation matches true large-batch within seed noise.

## Expected Outputs
Write outputs/results.json and outputs/summary.md.
"""


def _init_git_workspace(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# Workspace\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, stdout=subprocess.PIPE)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _config(tmp_path: Path, workspace: Path) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "stage10-plan-test", "mode": "docs-first"},
            "research": {"topic": "gradient accumulation semantics"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": False, "use_message": False},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "test",
                "primary_model": "fake-model",
            },
            "experiment": {
                "mode": "workspace",
                "workspace_agent": {
                    "enabled": True,
                    "transport": "acp",
                    "workspace_path": str(workspace),
                    "session_name": "stage10-plan",
                    "agent": "codex",
                    "manifest_filename": "run_manifest.json",
                    "timeout_sec": 300,
                },
                "submitter": {"type": "manual"},
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _write_stage9(run_dir: Path, outputs: list[str]) -> None:
    stage9 = run_dir / "stage-09"
    stage9.mkdir(parents=True)
    (stage9 / "plan.md").write_text(PLAN_MD, encoding="utf-8")
    (stage9 / "expected_outputs.json").write_text(
        json.dumps(
            {
                "schema_version": "researchclaw.expected_outputs.v1",
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )


def _manifest(commit: str, result_paths: list[str]) -> dict[str, object]:
    return {
        "schema_version": "researchclaw.run_manifest.v1",
        "code_commit": commit,
        "launch": {
            "command": "python scripts/run_experiment.py",
            "cwd": ".",
            "env": {},
            "resources": {"gpus": 0, "time": "00:05:00", "partition": "", "mem_gb": 4},
        },
        "result_paths": result_paths,
    }


def test_stage10_reads_plan_md_not_task_spec(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from researchclaw.pipeline.stage_impls import _code_generation

    workspace = tmp_path / "workspace"
    commit = _init_git_workspace(workspace)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_stage9(run_dir, ["outputs/results.json"])
    stage_dir = run_dir / "stage-10"
    stage_dir.mkdir()
    (workspace / "run_manifest.json").write_text(
        json.dumps(_manifest(commit, ["outputs/results.json"])),
        encoding="utf-8",
    )

    captured_prompt: dict[str, str] = {}

    def fake_create_workspace_agent(*args, **kwargs):
        return object()

    def fake_run_workspace_agent_implement(**kwargs):
        captured_prompt["prompt"] = kwargs["prompt"]
        return WorkspaceAgentResult(
            base_sha="base",
            agent_commit_sha=commit,
            manifest_path="run_manifest.json",
            diff_stat="",
            raw_log="",
            provider_name="fake",
            elapsed_sec=0.1,
        )

    monkeypatch.setattr(
        "researchclaw.experiment.workspace_agent.create_workspace_agent",
        fake_create_workspace_agent,
    )
    monkeypatch.setattr(
        "researchclaw.pipeline.workspace_orchestrator.run_workspace_agent_implement",
        fake_run_workspace_agent_implement,
    )

    result = _code_generation._execute_code_agent_implement_or_repair(
        stage_dir,
        run_dir,
        _config(tmp_path, workspace),
        AdapterBundle(),
    )

    assert result.status is StageStatus.DONE
    assert "Experiment Plan" in captured_prompt["prompt"]
    assert "task_spec.yaml" not in captured_prompt["prompt"]


def test_stage11_run_manifest_must_cover_expected_outputs(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._execution import (
        _execute_manifest_validate_and_prepare,
    )

    workspace = tmp_path / "workspace"
    commit = _init_git_workspace(workspace)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_stage9(run_dir, ["outputs/results.json", "outputs/summary.md"])
    stage10 = run_dir / "stage-10"
    stage10.mkdir()
    (stage10 / "run_manifest.json").write_text(
        json.dumps(_manifest(commit, ["outputs/results.json"])),
        encoding="utf-8",
    )
    stage11 = run_dir / "stage-11"
    stage11.mkdir()

    result = _execute_manifest_validate_and_prepare(
        stage11,
        run_dir,
        _config(tmp_path, workspace),
        AdapterBundle(),
    )

    assert result.status is StageStatus.FAILED
    validation = json.loads((stage11 / "manifest_validation.json").read_text(encoding="utf-8"))
    assert validation["ok"] is False
    assert any("outputs/summary.md" in item for item in validation["errors"])


def test_run_manifest_no_primary_metric_field() -> None:
    from researchclaw.experiment.workspace import RunManifest

    manifest = RunManifest.from_json(json.dumps(_manifest("abc123", ["outputs/results.json"])))

    assert not hasattr(manifest, "metrics")
    assert "primary_metric" not in manifest.to_json()
    assert "metric_direction" not in manifest.to_json()
