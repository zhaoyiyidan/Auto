from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _coordinator_cls() -> Any:
    try:
        from researchclaw.pipeline.hypothesis_coordinator import (
            HypothesisValidationCoordinator,
        )
    except ImportError:
        pytest.fail("HypothesisValidationCoordinator is not implemented")
    return HypothesisValidationCoordinator


def _node_payload(run_dir: Path, node_id: str) -> dict[str, Any]:
    return json.loads(
        (run_dir / "hypothesis_tree" / "nodes" / node_id / "node.json").read_text(
            encoding="utf-8"
        )
    )


def test_coordinator_splits_stage8_hypotheses_into_sibling_nodes(
    tmp_path: Path,
) -> None:
    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    hypotheses_md = """
## H1: Accuracy hypothesis
Statement: Treatment improves accuracy.
Prediction: Accuracy increases by at least 5 points.
Falsification: Accuracy does not improve.
Rationale: Prior runs show useful signal.
Baselines: baseline-a

## H2: Robustness hypothesis
Statement: Treatment improves robustness.
Prediction: Error rate falls under perturbation.
Falsification: Robustness does not improve.
Rationale: Different mechanism.
Baselines: baseline-b

## H3: Duplicate accuracy hypothesis
Statement: Treatment improves accuracy.
Prediction: Accuracy increases by at least 5 points.
Falsification: Accuracy does not improve.
Rationale: Duplicate wording should dedupe by science hash.
Baselines: baseline-c
"""

    nodes = coordinator.split_stage8_hypotheses(
        hypotheses_md,
        created_at="2026-01-01T00:00:00+00:00",
    )

    assert [node.id for node in nodes] == ["h-001", "h-002"]
    assert [node.parent_id for node in nodes] == [None, None]
    assert _node_payload(tmp_path, "h-001")["statement"] == "Treatment improves accuracy."
    assert _node_payload(tmp_path, "h-002")["statement"] == "Treatment improves robustness."
    events = [
        json.loads(line)
        for line in (tmp_path / "hypothesis_tree" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event_type"] for event in events] == [
        "node_proposed",
        "node_proposed",
    ]
