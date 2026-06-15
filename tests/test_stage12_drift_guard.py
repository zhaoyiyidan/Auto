from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.stages import StageStatus


def _config(tmp_path: Path, workspace: Path) -> RCConfig:
    workspace.mkdir(parents=True, exist_ok=True)
    return RCConfig.from_dict(
        {
            "project": {"name": "stage12-drift-test", "mode": "docs-first"},
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
                    "session_name": "stage12-drift",
                    "agent": "codex",
                    "timeout_sec": 300,
                },
                "submitter": {"type": "manual", "wait_for_completion": True},
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _write_stage9(run_dir: Path, outputs: list[str]) -> None:
    stage9 = run_dir / "stage-09"
    stage9.mkdir(parents=True)
    (stage9 / "plan.md").write_text(
        "# Experiment Plan\n\n## Expected Outputs\n" + "\n".join(outputs),
        encoding="utf-8",
    )
    (stage9 / "expected_outputs.json").write_text(
        json.dumps(
            {
                "schema_version": "researchclaw.expected_outputs.v1",
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )


def _write_stage11(run_dir: Path, result_paths: list[str]) -> None:
    stage11 = run_dir / "stage-11"
    stage11.mkdir(parents=True)
    manifest = {
        "schema_version": "researchclaw.run_manifest.v1",
        "code_commit": "abc123",
        "launch": {
            "command": "python run.py",
            "cwd": ".",
            "env": {},
            "resources": {"gpus": 0, "time": "00:05:00", "partition": "", "mem_gb": 4},
        },
        "result_paths": result_paths,
    }
    (stage11 / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (stage11 / "manifest_validation.json").write_text(
        json.dumps({"ok": True, "result_paths": result_paths}),
        encoding="utf-8",
    )


def test_execution_record_flags_missing_expected_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from researchclaw.pipeline.stage_impls._execution import (
        _execute_harness_submit_and_collect,
    )

    workspace = tmp_path / "workspace"
    (workspace / "outputs").mkdir(parents=True)
    (workspace / "outputs" / "results.json").write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_stage9(run_dir, ["outputs/results.json", "outputs/summary.md"])
    _write_stage11(run_dir, ["outputs/results.json", "outputs/summary.md"])
    stage12 = run_dir / "stage-12"
    stage12.mkdir()

    def fake_create_submitter(config: Any) -> object:
        return object()

    def fake_submit_and_collect(**kwargs: Any) -> None:
        run_dir_arg = Path(kwargs["run_dir"])
        (run_dir_arg / "execution_record.json").write_text(
            json.dumps(
                {
                    "stage": 12,
                    "code_commit": "abc123",
                    "submitter": "fake",
                    "job_id": "job-1",
                    "submit_status": "submitted",
                    "final_status": "completed",
                    "log_path": "",
                    "result_paths": ["outputs/results.json", "outputs/summary.md"],
                    "result_hashes": {},
                    "elapsed_sec": 1.0,
                    "waited": True,
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        (run_dir_arg / "submit_result.json").write_text("{}", encoding="utf-8")
        (run_dir_arg / "result_artifacts.json").write_text(
            json.dumps(
                {
                    "artifacts": [
                        {"path": "outputs/results.json", "exists": True},
                        {"path": "outputs/summary.md", "exists": False},
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "researchclaw.experiment.submitter.create_submitter",
        fake_create_submitter,
    )
    monkeypatch.setattr(
        "researchclaw.pipeline.workspace_orchestrator.submit_and_collect",
        fake_submit_and_collect,
    )

    result = _execute_harness_submit_and_collect(
        stage12,
        run_dir,
        _config(tmp_path, workspace),
        AdapterBundle(),
    )

    assert result.status is StageStatus.DONE
    record = json.loads((stage12 / "execution_record.json").read_text(encoding="utf-8"))
    assert record["missing_expected_outputs"] == ["outputs/summary.md"]


def test_stage12_all_missing_outputs_still_reaches_route_decision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from researchclaw.pipeline.stage_impls._execution import (
        _execute_harness_submit_and_collect,
    )

    workspace = tmp_path / "workspace"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_stage9(run_dir, ["outputs/results.json", "outputs/summary.md"])
    _write_stage11(run_dir, ["outputs/results.json", "outputs/summary.md"])
    stage12 = run_dir / "stage-12"
    stage12.mkdir()

    def fake_create_submitter(config: Any) -> object:
        return object()

    def fake_submit_and_collect(**kwargs: Any) -> None:
        run_dir_arg = Path(kwargs["run_dir"])
        (run_dir_arg / "execution_record.json").write_text(
            json.dumps(
                {
                    "stage": 12,
                    "code_commit": "abc123",
                    "submitter": "fake",
                    "job_id": "job-1",
                    "submit_status": "submitted",
                    "final_status": "timeout",
                    "log_path": "",
                    "result_paths": ["outputs/results.json", "outputs/summary.md"],
                    "result_hashes": {},
                    "elapsed_sec": 300.0,
                    "waited": True,
                    "recorded_at": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        (run_dir_arg / "submit_result.json").write_text("{}", encoding="utf-8")
        (run_dir_arg / "result_artifacts.json").write_text(
            json.dumps(
                {
                    "artifacts": [
                        {"path": "outputs/results.json", "exists": False},
                        {"path": "outputs/summary.md", "exists": False},
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "researchclaw.experiment.submitter.create_submitter",
        fake_create_submitter,
    )
    monkeypatch.setattr(
        "researchclaw.pipeline.workspace_orchestrator.submit_and_collect",
        fake_submit_and_collect,
    )

    result = _execute_harness_submit_and_collect(
        stage12,
        run_dir,
        _config(tmp_path, workspace),
        AdapterBundle(),
    )

    assert result.status is StageStatus.DONE
    record = json.loads((stage12 / "execution_record.json").read_text(encoding="utf-8"))
    assert record["missing_expected_outputs"] == [
        "outputs/results.json",
        "outputs/summary.md",
    ]


def test_experiment_decision_output_drift_flag(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._execution import (
        _execute_experiment_route_decision,
    )

    workspace = tmp_path / "workspace"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_stage9(run_dir, ["outputs/results.json", "outputs/summary.md"])
    _write_stage11(run_dir, ["outputs/results.json", "outputs/summary.md"])
    stage12 = run_dir / "stage-12"
    stage12.mkdir()
    (stage12 / "execution_record.json").write_text(
        json.dumps(
            {
                "stage": 12,
                "code_commit": "abc123",
                "submitter": "fake",
                "job_id": "job-1",
                "submit_status": "submitted",
                "final_status": "completed",
                "log_path": "",
                "result_paths": ["outputs/results.json", "outputs/summary.md"],
                "result_hashes": {},
                "missing_expected_outputs": ["outputs/summary.md"],
                "elapsed_sec": 1.0,
                "waited": True,
                "recorded_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (stage12 / "result_artifacts.json").write_text(
        json.dumps({"artifacts": [{"path": "outputs/results.json", "exists": True}]}),
        encoding="utf-8",
    )
    stage13 = run_dir / "stage-13"
    stage13.mkdir()

    result = _execute_experiment_route_decision(
        stage13,
        run_dir,
        _config(tmp_path, workspace),
        AdapterBundle(),
    )

    decision = json.loads((stage13 / "experiment_decision.json").read_text(encoding="utf-8"))
    assert result.status is StageStatus.DONE
    assert decision["output_drift"] is True
    assert decision["route"] == "fix_code"


def test_execution_record_no_primary_metric_field() -> None:
    from researchclaw.experiment.workspace import ExecutionRecord

    names = {field.name for field in fields(ExecutionRecord)}

    assert "metrics" not in names
    assert "missing_expected_outputs" in names
