from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _aggregator_cls() -> Any:
    try:
        from researchclaw.pipeline.evidence_aggregator import EvidenceAggregator
    except ImportError:
        pytest.fail("EvidenceAggregator is not implemented")
    return EvidenceAggregator


def test_evidence_aggregator_selects_best_attempt_per_node(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.hypothesis_store import HypothesisStore

    EvidenceAggregator = _aggregator_cls()
    store = HypothesisStore(tmp_path)
    node = store.create_node(
        statement="Treatment improves accuracy.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    first = store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(tmp_path / "branches" / node.id / "attempt-001"),
        created_at="2026-01-01T00:01:00+00:00",
    )
    second = store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(tmp_path / "branches" / node.id / "attempt-002"),
        created_at="2026-01-01T00:02:00+00:00",
    )
    store.update_attempt(
        first.attempt_id,
        status="succeeded",
        metrics={"score": 0.72},
        decision="proceed",
        finished_at="2026-01-01T00:03:00+00:00",
    )
    store.update_attempt(
        second.attempt_id,
        status="succeeded",
        metrics={"score": 0.91},
        decision="proceed",
        finished_at="2026-01-01T00:04:00+00:00",
    )

    winners = EvidenceAggregator(tmp_path).select_best_attempts(
        metric_name="score",
        direction="maximize",
    )

    assert list(winners) == [node.id]
    assert winners[node.id].attempt_id == second.attempt_id
    assert winners[node.id].metrics["score"] == 0.91


def test_evidence_aggregator_skips_degenerate_minimize_attempt(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.hypothesis_store import HypothesisStore

    EvidenceAggregator = _aggregator_cls()
    store = HypothesisStore(tmp_path)
    node = store.create_node(
        statement="Treatment lowers loss.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    degenerate = store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(tmp_path / "branches" / node.id / "attempt-001"),
        created_at="2026-01-01T00:01:00+00:00",
    )
    valid = store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(tmp_path / "branches" / node.id / "attempt-002"),
        created_at="2026-01-01T00:02:00+00:00",
    )
    store.update_attempt(
        degenerate.attempt_id,
        status="succeeded",
        metrics={"loss": 1e-9},
        decision="proceed",
        finished_at="2026-01-01T00:03:00+00:00",
    )
    store.update_attempt(
        valid.attempt_id,
        status="succeeded",
        metrics={"loss": 0.2},
        decision="proceed",
        finished_at="2026-01-01T00:04:00+00:00",
    )

    winners = EvidenceAggregator(tmp_path).select_best_attempts(
        metric_name="loss",
        direction="minimize",
    )

    assert winners[node.id].attempt_id == valid.attempt_id
    assert winners[node.id].metrics["loss"] == 0.2
