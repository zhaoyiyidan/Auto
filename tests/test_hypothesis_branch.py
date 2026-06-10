from __future__ import annotations

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
