from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _next_hypothesis() -> dict[str, Any]:
    return {
        "statement": "Follow-up treatment improves accuracy.",
        "prediction": "Accuracy increases by at least 5 points.",
        "falsification": "Accuracy does not improve.",
        "rationale": "The first branch found a useful mechanism.",
        "baselines": ["baseline-a"],
    }


def test_verdict_written_on_proceed(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage15_verdict import write_stage15_verdict

    verdict = write_stage15_verdict(
        tmp_path,
        decision="proceed",
        confidence=0.82,
        evidence_summary="Evidence supports the hypothesis.",
        key_metrics={"score": 0.91},
        generated_at="2026-01-01T00:00:00+00:00",
    )

    assert verdict["decision"] == "proceed"
    assert verdict["confidence"] == 0.82
    assert verdict["next_hypothesis"] is None
    assert json.loads((tmp_path / "verdict.json").read_text()) == verdict


def test_verdict_written_on_extend_requires_next_hypothesis(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.stage15_verdict import write_stage15_verdict

    verdict = write_stage15_verdict(
        tmp_path,
        decision="extend",
        next_hypothesis=_next_hypothesis(),
        strict=True,
    )

    assert verdict["decision"] == "extend"
    assert verdict["next_hypothesis"]["statement"]
    assert verdict["next_hypothesis"]["prediction"]
    assert verdict["next_hypothesis"]["falsification"]


def test_verdict_written_on_pivot_requires_next_hypothesis(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage15_verdict import write_stage15_verdict

    verdict = write_stage15_verdict(
        tmp_path,
        decision="pivot",
        next_hypothesis=_next_hypothesis(),
        strict=True,
    )

    assert verdict["decision"] == "pivot"
    assert verdict["next_hypothesis"]["statement"] == (
        "Follow-up treatment improves accuracy."
    )


def test_verdict_schema_invalid_raises() -> None:
    from researchclaw.pipeline.stage15_verdict import validate_stage15_verdict

    with pytest.raises(ValueError, match="statement"):
        validate_stage15_verdict(
            {
                "decision": "extend",
                "confidence": 0.8,
                "next_hypothesis": {
                    "prediction": "Prediction.",
                    "falsification": "Falsification.",
                },
            }
        )


def test_validate_branch_reads_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline import hypothesis_branch
    from researchclaw.pipeline.executor import StageResult
    from researchclaw.pipeline.hypothesis_branch import validate_branch
    from researchclaw.pipeline.hypothesis_store import HypothesisNode, ValidationAttempt
    from researchclaw.pipeline.stage15_verdict import write_stage15_verdict
    from researchclaw.pipeline.stages import Stage, StageStatus

    branch_run_dir = tmp_path / "run" / "hypothesis_branches" / "h-001" / "attempt-001"
    stage15 = branch_run_dir / "stage-15"
    stage15.mkdir(parents=True)

    def fake_execute_pipeline(**kwargs: Any) -> list[Any]:
        _ = kwargs
        write_stage15_verdict(
            stage15,
            decision="extend",
            next_hypothesis=_next_hypothesis(),
            key_metrics={"score": 0.91},
            strict=True,
        )
        return [
            StageResult(
                stage=Stage.RESEARCH_DECISION,
                status=StageStatus.DONE,
                artifacts=("decision.md",),
                decision="proceed",
            )
        ]

    monkeypatch.setattr(hypothesis_branch, "execute_pipeline", fake_execute_pipeline)

    result = validate_branch(
        branch_run_dir=branch_run_dir,
        node=HypothesisNode(id="h-001", statement="Treatment improves accuracy."),
        attempt=ValidationAttempt(
            attempt_id="h-001/attempt-001",
            node_id="h-001",
            branch_run_dir=str(branch_run_dir),
        ),
        config=object(),
        adapters=object(),
    )

    assert result["decision"] == "extend"
    assert result["next_hypothesis"]["statement"] == (
        "Follow-up treatment improves accuracy."
    )
    assert result["metrics"] == {"score": 0.91}
