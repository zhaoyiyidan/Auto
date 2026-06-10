from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest


def _hypothesis_node_cls() -> Any:
    try:
        from researchclaw.pipeline.hypothesis_store import HypothesisNode
    except ImportError:
        pytest.fail("HypothesisNode is not implemented")
    return HypothesisNode


def test_seed_branch_dir_links_shared_context_and_writes_single_hypothesis(
    tmp_path: Path,
) -> None:
    try:
        from researchclaw.experiment.protocol import parse_hypotheses_md
        from researchclaw.pipeline._helpers import _read_prior_artifact
        from researchclaw.pipeline.hypothesis_branch import seed_branch_dir
    except ImportError:
        pytest.fail("seed_branch_dir dependencies are not implemented")

    run_dir = tmp_path / "run"
    branch_run_dir = run_dir / "hypothesis_branches" / "h-001" / "attempt-001"
    for stage_number in range(1, 8):
        stage_dir = run_dir / f"stage-{stage_number:02d}"
        stage_dir.mkdir(parents=True)
        (stage_dir / f"context_{stage_number}.txt").write_text(
            f"context {stage_number}",
            encoding="utf-8",
        )
    (run_dir / "stage-03" / "hardware_profile.json").write_text(
        '{"gpu": "A100"}',
        encoding="utf-8",
    )
    node = _hypothesis_node_cls()(
        id="h-001",
        statement="Treatment improves accuracy.",
        prediction="Accuracy increases by at least 5 points.",
        falsification="Accuracy does not improve.",
        rationale="Prior runs show useful signal.",
        baselines=("baseline-a",),
        source="stage8_batch",
        parent_id=None,
        created_at="2026-01-01T00:00:00+00:00",
    )

    seed_branch_dir(branch_run_dir, run_dir, node)

    for stage_number in range(1, 8):
        link = branch_run_dir / f"stage-{stage_number:02d}"
        assert link.is_symlink()
        assert os.readlink(link) == f"../../../stage-{stage_number:02d}"
    assert (
        _read_prior_artifact(branch_run_dir, "hardware_profile.json")
        == '{"gpu": "A100"}'
    )

    branch_stage8 = branch_run_dir / "stage-08"
    assert branch_stage8.is_dir()
    assert not branch_stage8.is_symlink()
    hypotheses_md = (branch_stage8 / "hypotheses.md").read_text(encoding="utf-8")
    parsed = parse_hypotheses_md(hypotheses_md)
    assert len(parsed) == 1
    assert parsed[0].statement == "Treatment improves accuracy."
    assert parsed[0].prediction == "Accuracy increases by at least 5 points."
    assert parsed[0].falsification == "Accuracy does not improve."


def test_validate_branch_runs_stage9_to_stage15_and_writes_attempt_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from researchclaw.pipeline import hypothesis_branch
        from researchclaw.pipeline.executor import StageResult
        from researchclaw.pipeline.hypothesis_branch import validate_branch
        from researchclaw.pipeline.hypothesis_store import ValidationAttempt
        from researchclaw.pipeline.stages import Stage, StageStatus
    except ImportError:
        pytest.fail("validate_branch dependencies are not implemented")

    branch_run_dir = (
        tmp_path / "run" / "hypothesis_branches" / "h-001" / "attempt-001"
    )
    branch_run_dir.mkdir(parents=True)
    node = _hypothesis_node_cls()(
        id="h-001",
        statement="Treatment improves accuracy.",
    )
    attempt = ValidationAttempt(
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        branch_run_dir=str(branch_run_dir),
    )
    recorded: dict[str, Any] = {}

    def fake_execute_pipeline(**kwargs: Any) -> list[Any]:
        recorded.update(kwargs)
        return [
            StageResult(
                stage=Stage.EXPERIMENT_TASK_SPEC,
                status=StageStatus.DONE,
                artifacts=("experiment_protocol.json",),
            ),
            StageResult(
                stage=Stage.RESEARCH_DECISION,
                status=StageStatus.DONE,
                artifacts=("decision.md",),
                decision="pivot",
            ),
        ]

    monkeypatch.setattr(
        hypothesis_branch,
        "execute_pipeline",
        fake_execute_pipeline,
        raising=False,
    )

    result = validate_branch(
        branch_run_dir=branch_run_dir,
        node=node,
        attempt=attempt,
        config=object(),
        adapters=object(),
    )

    assert recorded["run_dir"] == branch_run_dir
    assert recorded["from_stage"] is Stage.EXPERIMENT_TASK_SPEC
    assert recorded["to_stage"] is Stage.RESEARCH_DECISION
    assert recorded["config"] is not None
    assert recorded["adapters"] is not None
    assert result["decision"] == "pivot"
    assert result["artifacts"] == ["decision.md"]

    payload = json.loads(
        (branch_run_dir / "attempt_result.json").read_text(encoding="utf-8")
    )
    assert payload == {
        "attempt_id": "h-001/attempt-001",
        "node_id": "h-001",
        "status": "succeeded",
        "decision": "pivot",
        "artifacts": ["decision.md"],
        "error": None,
    }
