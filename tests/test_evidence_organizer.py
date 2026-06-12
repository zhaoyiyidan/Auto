from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from researchclaw.config import RCConfig
from researchclaw.pipeline import evidence_organizer
from researchclaw.pipeline.evidence_organizer import (
    build_evidence_bundle,
    build_organizer_prompt,
    postcheck_analysis,
)


def _config_for_workspace(workspace: Path) -> RCConfig:
    base = RCConfig()
    return replace(
        base,
        experiment=replace(
            base.experiment,
            workspace_agent=replace(
                base.experiment.workspace_agent,
                workspace_path=str(workspace),
            ),
        ),
    )


def _write_stage_artifact(
    run_dir: Path,
    stage_num: int,
    filename: str,
    payload: str | dict[str, Any],
) -> Path:
    stage_dir = run_dir / f"stage-{stage_num:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / filename
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _seed_bundle_inputs(tmp_path: Path) -> tuple[Path, Path, RCConfig]:
    run_dir = tmp_path / "run"
    workspace = tmp_path / "workspace"
    run_dir.mkdir()
    (workspace / "outputs" / "old_smoke").mkdir(parents=True)
    (workspace / "outputs").mkdir(exist_ok=True)
    (workspace / "outputs" / "metrics.json").write_text(
        json.dumps({"accuracy": 0.82}), encoding="utf-8"
    )
    (workspace / "outputs" / "run_summary.json").write_text(
        json.dumps({"training_run_count": 24}), encoding="utf-8"
    )
    (workspace / "outputs" / "old_smoke" / "metrics.json").write_text(
        json.dumps({"accuracy": 0.1}), encoding="utf-8"
    )

    _write_stage_artifact(run_dir, 9, "plan.md", "# Experiment Plan\n")
    _write_stage_artifact(
        run_dir,
        9,
        "expected_outputs.json",
        {
            "schema_version": "researchclaw.expected_outputs.v1",
            "outputs": ["outputs/metrics.json"],
        },
    )
    _write_stage_artifact(
        run_dir,
        10,
        "run_manifest.json",
        {
            "schema_version": "researchclaw.run_manifest.v1",
            "code_commit": "abc123",
            "launch": {
                "command": "python train.py",
                "cwd": ".",
                "env": {"SEED": "1"},
                "resources": {
                    "gpus": 1,
                    "time": "02:00:00",
                    "partition": "gpu",
                    "mem_gb": 32,
                },
            },
            "result_paths": [
                "outputs/metrics.json",
                "outputs/run_summary.json",
            ],
            "metrics": {"primary": "accuracy", "direction": "maximize"},
        },
    )
    _write_stage_artifact(
        run_dir,
        12,
        "execution_record.json",
        {
            "job_id": "job-123",
            "elapsed_sec": 42.5,
            "submitter": {"type": "slurm"},
            "metrics": {"accuracy": 0.82},
        },
    )
    _write_stage_artifact(
        run_dir,
        12,
        "result_artifacts.json",
        {"artifacts": ["outputs/metrics.json"]},
    )
    _write_stage_artifact(
        run_dir,
        12,
        "contract_evidence.json",
        {"ok": True, "violations": []},
    )
    _write_stage_artifact(
        run_dir,
        13,
        "experiment_decision.json",
        {"route": "continue"},
    )
    _write_stage_artifact(run_dir, 15, "decision.md", "PIVOT\n")
    return run_dir, workspace, _config_for_workspace(workspace)


def _all_bundle_paths(bundle: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("default_inputs", "optional_inputs", "result_files"):
        for entry in bundle.get(key, []):
            if isinstance(entry, dict) and entry.get("path"):
                paths.append(str(entry["path"]))
    return paths


def test_build_evidence_bundle_lists_current_declared_result_paths_only(
    tmp_path: Path,
) -> None:
    run_dir, workspace, config = _seed_bundle_inputs(tmp_path)

    bundle = build_evidence_bundle(run_dir, config)

    assert bundle["run_dir"] == str(run_dir.resolve())
    assert bundle["stage_dir"] == str((run_dir / "stage-14").resolve())
    assert bundle["workspace_path"] == str(workspace.resolve())

    defaults = {entry["label"]: entry for entry in bundle["default_inputs"]}
    assert defaults["experiment_plan"]["exists"] is True
    assert defaults["expected_outputs"]["exists"] is True
    assert defaults["run_manifest"]["exists"] is True
    assert defaults["execution_record"]["exists"] is True
    assert defaults["result_artifacts"]["exists"] is True
    assert defaults["experiment_decision"]["exists"] is True

    result_paths = [entry["path"] for entry in bundle["result_files"]]
    assert str((workspace / "outputs" / "metrics.json").resolve()) in result_paths
    assert str((workspace / "outputs" / "run_summary.json").resolve()) in result_paths
    assert not any("old_smoke" in path for path in result_paths)

    optionals = {entry["label"]: entry for entry in bundle["optional_inputs"]}
    assert optionals["stage_12_local_log"]["exists"] is False
    assert optionals["workspace_agent_result"]["exists"] is False
    assert optionals["manifest_validation"]["exists"] is False

    assert bundle["reproducibility"]["launch_command"] == "python train.py"
    assert bundle["reproducibility"]["launch_cwd"] == "."
    assert bundle["reproducibility"]["code_commit"] == "abc123"
    assert bundle["reproducibility"]["job_id"] == "job-123"
    assert bundle["reproducibility"]["elapsed_sec"] == 42.5
    assert bundle["reproducibility"]["submitter"] == {"type": "slurm"}
    assert not any("stage-15" in path for path in _all_bundle_paths(bundle))


def test_bundle_includes_expected_outputs_when_present(tmp_path: Path) -> None:
    run_dir, _workspace, config = _seed_bundle_inputs(tmp_path)

    bundle = build_evidence_bundle(run_dir, config)
    defaults = {entry["label"]: entry for entry in bundle["default_inputs"]}

    assert defaults["expected_outputs"]["exists"] is True
    assert defaults["expected_outputs"]["filename"] == "expected_outputs.json"
    assert defaults["expected_outputs"]["path"].endswith(
        "stage-09/expected_outputs.json"
    )


def test_build_organizer_prompt_contains_fixed_sections_and_boundaries() -> None:
    bundle = {
        "run_dir": "/tmp/run",
        "stage_dir": "/tmp/run/stage-14",
        "workspace_path": "/tmp/workspace",
        "default_inputs": [
            {"label": "experiment_plan", "path": "/tmp/run/stage-09/plan.md", "exists": True}
        ],
        "optional_inputs": [],
        "result_files": [
            {"path": "/tmp/workspace/outputs/metrics.json", "exists": True}
        ],
        "reproducibility": {"launch_command": "python train.py"},
    }

    prompt = build_organizer_prompt(bundle)

    for heading in (
        "## Experiment Objective",
        "## Experiment Plan",
        "## Executed Experiments",
        "## Results Summary",
        "## Artifact Locations",
        "## Reproducibility",
    ):
        assert heading in prompt
    for phrase in (
        "DO NOT",
        "research judgment",
        "recommendation",
        "next actions",
        "quality assessment",
        "missing evidence",
        "PROCEED",
        "PIVOT",
        "EXTEND",
    ):
        assert phrase in prompt
    assert "stage-15" not in prompt.lower()


def test_create_evidence_organizer_inherits_llm_acp_agent_not_workspace_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class DummySession:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def _resolve_acpx(self) -> str:
            return "/usr/bin/acpx"

    monkeypatch.setattr(evidence_organizer, "AcpWorkspaceSession", DummySession)
    base = RCConfig()
    config = replace(
        base,
        llm=replace(
            base.llm,
            acp=replace(base.llm.acp, agent="codex"),
        ),
        experiment=replace(
            base.experiment,
            workspace_agent=replace(
                base.experiment.workspace_agent,
                enabled=True,
                agent="claude",
            ),
            result_analysis_agent=replace(
                base.experiment.result_analysis_agent,
                agent="",
            ),
        ),
    )

    session = evidence_organizer.create_evidence_organizer_agent(config, tmp_path)

    assert session is not None
    assert captured["agent"] == "codex"


def test_create_evidence_organizer_uses_explicit_analysis_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class DummySession:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def _resolve_acpx(self) -> str:
            return "/usr/bin/acpx"

    monkeypatch.setattr(evidence_organizer, "AcpWorkspaceSession", DummySession)
    base = RCConfig()
    config = replace(
        base,
        llm=replace(
            base.llm,
            acp=replace(base.llm.acp, agent="codex"),
        ),
        experiment=replace(
            base.experiment,
            result_analysis_agent=replace(
                base.experiment.result_analysis_agent,
                agent="claude",
            ),
        ),
    )

    session = evidence_organizer.create_evidence_organizer_agent(config, tmp_path)

    assert session is not None
    assert captured["agent"] == "claude"


def test_postcheck_analysis_accepts_clean_six_section_document(tmp_path: Path) -> None:
    text = """# Experiment Analysis

## Experiment Objective
Test the hypothesis.

## Experiment Plan
Compare baseline and treatment.

## Executed Experiments
Two conditions with three seeds each.

## Results Summary
Treatment reached 0.82 accuracy.

## Artifact Locations
metrics.json contains the aggregate metrics.

## Reproducibility
Run python train.py from the workspace.
"""

    result = postcheck_analysis(text, tmp_path)

    assert result.ok is True
    assert result.violations == ()


@pytest.mark.parametrize(
    ("text", "expected_violation"),
    [
        ("", "empty"),
        ("# Experiment Analysis\n\n## Decision\nPIVOT\n", "forbidden_heading:decision"),
        ("# Experiment Analysis\n\nPROCEED\n", "standalone_decision_token:proceed"),
        (
            "# Experiment Analysis\n\n## Quality Assessment\nGood enough.\n",
            "forbidden_heading:quality assessment",
        ),
        (
            "# Experiment Analysis\n\n## Missing Or Ambiguous Evidence\nNone.\n",
            "forbidden_heading:missing or ambiguous evidence",
        ),
        (
            "# Experiment Analysis\n\nSee stage-15/decision.md for details.\n",
            "forbidden_reference:stage-15",
        ),
    ],
)
def test_postcheck_analysis_rejects_boundary_violations(
    tmp_path: Path,
    text: str,
    expected_violation: str,
) -> None:
    result = postcheck_analysis(text, tmp_path)

    assert result.ok is False
    assert expected_violation in result.violations


def test_postcheck_still_blocks_decision_language(tmp_path: Path) -> None:
    text = """# Experiment Analysis

## Experiment Objective
Measure the run.

## Results Summary
The result is sufficient.

## Decision
PROCEED
"""

    result = postcheck_analysis(text, tmp_path)

    assert result.ok is False
    assert "forbidden_heading:decision" in result.violations
    assert "standalone_decision_token:proceed" in result.violations
