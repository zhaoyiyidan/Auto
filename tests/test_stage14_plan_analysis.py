from __future__ import annotations

import inspect
import json
from pathlib import Path


PLAN_MD = """
# Experiment Plan

## Hypotheses
H1 tests corrected accumulation semantics.

## Baselines
True large-batch and fixed accumulation.

## Ablations
Disable corrected clipping.

## Metrics
Update cosine similarity and time-to-target loss.

## Decision Criteria
Support H1 when corrected accumulation matches large-batch behavior.

## Expected Outputs
outputs/results.json
"""


def _write_stage9(run_dir: Path) -> None:
    stage9 = run_dir / "stage-09"
    stage9.mkdir(parents=True)
    (stage9 / "plan.md").write_text(PLAN_MD, encoding="utf-8")
    (stage9 / "expected_outputs.json").write_text(
        json.dumps(
            {
                "schema_version": "researchclaw.expected_outputs.v1",
                "outputs": ["outputs/results.json", "outputs/summary.md"],
            }
        ),
        encoding="utf-8",
    )


def test_stage14_loads_plan_md_and_expected_outputs(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._analysis import (
        _load_plan_and_expected_outputs,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_stage9(run_dir)

    plan_md, expected_outputs = _load_plan_and_expected_outputs(run_dir)

    assert "Decision Criteria" in plan_md
    assert expected_outputs == ["outputs/results.json", "outputs/summary.md"]


def test_stage14_no_best_execution_by_primary_metric() -> None:
    from researchclaw.pipeline.stage_impls import _analysis

    assert not hasattr(_analysis, "_best_execution")
    assert not hasattr(_analysis, "_metrics_summary")


def test_stage15_uses_plan_md_decision_criteria() -> None:
    from researchclaw.pipeline.stage_impls import _analysis

    signature = inspect.signature(_analysis._hypothesis_protocol_decision)

    assert "plan_md" in signature.parameters
    assert "protocol" not in signature.parameters
