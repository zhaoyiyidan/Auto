from __future__ import annotations

import json
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


def test_evidence_aggregator_writes_validation_summary_schema(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.hypothesis_store import HypothesisStore

    EvidenceAggregator = _aggregator_cls()
    store = HypothesisStore(tmp_path)
    supported = store.create_node(
        statement="Supported hypothesis.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    refuted = store.create_node(
        statement="Refuted hypothesis.",
        created_at="2026-01-01T00:00:01+00:00",
    )
    for node, status, score in [
        (supported, "supported", 0.91),
        (refuted, "refuted", 0.22),
    ]:
        attempt = store.add_attempt(
            node_id=node.id,
            branch_run_dir=str(tmp_path / "branches" / node.id / "attempt-001"),
            created_at="2026-01-01T00:01:00+00:00",
        )
        store.update_attempt(
            attempt.attempt_id,
            status="succeeded",
            metrics={"score": score},
            decision="proceed" if status == "supported" else "pivot",
            finished_at="2026-01-01T00:02:00+00:00",
        )
        store.set_node_status(
            node.id,
            "validating",
            created_at="2026-01-01T00:03:00+00:00",
        )
        store.set_node_status(
            node.id,
            status,
            created_at="2026-01-01T00:04:00+00:00",
        )

    summary = EvidenceAggregator(tmp_path).write_validation_summary(
        metric_name="score",
        direction="maximize",
        generated_at="2026-01-01T00:05:00+00:00",
    )

    assert summary == {
        "generated": "2026-01-01T00:05:00+00:00",
        "counts": {
            "total": 2,
            "supported": 1,
            "refuted": 1,
            "inconclusive": 0,
            "superseded": 0,
        },
        "nodes": [
            {
                "node_id": "h-001",
                "status": "supported",
                "statement": "Supported hypothesis.",
                "best_attempt_id": "h-001/attempt-001",
                "decision": "proceed",
                "metrics": {"score": 0.91},
            },
            {
                "node_id": "h-002",
                "status": "refuted",
                "statement": "Refuted hypothesis.",
                "best_attempt_id": "h-002/attempt-001",
                "decision": "pivot",
                "metrics": {"score": 0.22},
            },
        ],
    }
    assert json.loads(
        (
            tmp_path
            / "hypothesis_aggregate"
            / "validation_summary.json"
        ).read_text(encoding="utf-8")
    ) == summary


def test_evidence_aggregator_writes_registry_for_all_outcomes(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.hypothesis_store import HypothesisStore

    EvidenceAggregator = _aggregator_cls()
    store = HypothesisStore(tmp_path)
    expected_rows: list[dict[str, Any]] = []
    for index, (status, decision, score) in enumerate(
        [
            ("supported", "proceed", 0.91),
            ("refuted", "pivot", 0.22),
            ("inconclusive", "inconclusive", 0.50),
        ],
        start=1,
    ):
        node = store.create_node(
            statement=f"{status.title()} hypothesis.",
            created_at=f"2026-01-01T00:00:0{index}+00:00",
        )
        branch_run_dir = tmp_path / "branches" / node.id / "attempt-001"
        attempt = store.add_attempt(
            node_id=node.id,
            branch_run_dir=str(branch_run_dir),
            created_at=f"2026-01-01T00:01:0{index}+00:00",
        )
        store.update_attempt(
            attempt.attempt_id,
            status="succeeded",
            metrics={"score": score},
            artifacts=["stage-15/decision.md"],
            decision=decision,
            finished_at=f"2026-01-01T00:02:0{index}+00:00",
        )
        store.set_node_status(
            node.id,
            "validating",
            created_at=f"2026-01-01T00:03:0{index}+00:00",
        )
        store.set_node_status(
            node.id,
            status,
            created_at=f"2026-01-01T00:04:0{index}+00:00",
        )
        expected_rows.append(
            {
                "node_id": node.id,
                "attempt_id": attempt.attempt_id,
                "outcome": status,
                "decision": decision,
                "metrics": {"score": score},
                "artifacts": ["stage-15/decision.md"],
                "branch_run_dir": str(branch_run_dir),
            }
        )

    rows = EvidenceAggregator(tmp_path).write_evidence_registry(
        metric_name="score",
        direction="maximize",
    )

    assert rows == expected_rows
    registry_path = tmp_path / "hypothesis_aggregate" / "evidence_registry.jsonl"
    assert [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
    ] == expected_rows


def test_evidence_aggregator_writes_paper_context_with_zero_supported_warning(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.hypothesis_store import HypothesisStore

    EvidenceAggregator = _aggregator_cls()
    store = HypothesisStore(tmp_path)
    for index, status in enumerate(["refuted", "inconclusive"], start=1):
        node = store.create_node(
            statement=f"{status.title()} hypothesis.",
            created_at=f"2026-01-01T00:00:0{index}+00:00",
        )
        attempt = store.add_attempt(
            node_id=node.id,
            branch_run_dir=str(tmp_path / "branches" / node.id / "attempt-001"),
            created_at=f"2026-01-01T00:01:0{index}+00:00",
        )
        store.update_attempt(
            attempt.attempt_id,
            status="succeeded",
            metrics={"score": 0.1 * index},
            artifacts=["stage-15/decision.md"],
            decision=status,
            finished_at=f"2026-01-01T00:02:0{index}+00:00",
        )
        store.set_node_status(
            node.id,
            "validating",
            created_at=f"2026-01-01T00:03:0{index}+00:00",
        )
        store.set_node_status(
            node.id,
            status,
            created_at=f"2026-01-01T00:04:0{index}+00:00",
        )

    context = EvidenceAggregator(tmp_path).write_paper_context(
        metric_name="score",
        direction="maximize",
        generated_at="2026-01-01T00:05:00+00:00",
    )

    assert "Quality warning: no supported hypotheses" in context
    assert "refuted: 1" in context
    assert "inconclusive: 1" in context
    assert "Refuted hypothesis." in context
    assert "Inconclusive hypothesis." in context
    assert (
        tmp_path / "hypothesis_aggregate" / "paper_context.md"
    ).read_text(encoding="utf-8") == context


def test_evidence_aggregator_write_all_is_idempotent_and_overwrites_atomically(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.hypothesis_store import HypothesisStore

    EvidenceAggregator = _aggregator_cls()
    store = HypothesisStore(tmp_path)
    node = store.create_node(
        statement="Supported hypothesis.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(tmp_path / "branches" / node.id / "attempt-001"),
        created_at="2026-01-01T00:01:00+00:00",
    )
    store.update_attempt(
        attempt.attempt_id,
        status="succeeded",
        metrics={"score": 0.91},
        artifacts=["stage-15/decision.md"],
        decision="proceed",
        finished_at="2026-01-01T00:02:00+00:00",
    )
    store.set_node_status(
        node.id,
        "validating",
        created_at="2026-01-01T00:03:00+00:00",
    )
    store.set_node_status(
        node.id,
        "supported",
        created_at="2026-01-01T00:04:00+00:00",
    )

    first = EvidenceAggregator(tmp_path).write_all(
        metric_name="score",
        direction="maximize",
        generated_at="2026-01-01T00:05:00+00:00",
    )
    aggregate_dir = tmp_path / "hypothesis_aggregate"
    (aggregate_dir / "validation_summary.json").write_text(
        "{not-json",
        encoding="utf-8",
    )

    second = EvidenceAggregator(tmp_path).write_all(
        metric_name="score",
        direction="maximize",
        generated_at="2026-01-01T00:05:00+00:00",
    )

    assert second == first
    assert json.loads(
        (aggregate_dir / "validation_summary.json").read_text(encoding="utf-8")
    ) == first["validation_summary"]
    assert (aggregate_dir / "evidence_registry.jsonl").is_file()
    assert (aggregate_dir / "paper_context.md").is_file()
    assert not list(aggregate_dir.glob("*.tmp"))


def test_aggregate_writes_root_schema_and_recommended_supported(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.hypothesis_store import HypothesisStore

    EvidenceAggregator = _aggregator_cls()
    store = HypothesisStore(tmp_path)
    supported = store.create_node(
        statement="Supported hypothesis.",
        prediction="Supported prediction.",
        falsification="Supported falsification.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    inconclusive = store.create_node(
        statement="Inconclusive hypothesis.",
        created_at="2026-01-01T00:00:01+00:00",
    )
    for node, status, decision, score in [
        (supported, "supported", "proceed", 0.91),
        (inconclusive, "inconclusive", "inconclusive", 0.2),
    ]:
        attempt = store.add_attempt(
            node_id=node.id,
            branch_run_dir=str(tmp_path / "branches" / node.id / "attempt-001"),
            created_at="2026-01-01T00:01:00+00:00",
        )
        store.update_attempt(
            attempt.attempt_id,
            status="succeeded",
            metrics={"score": score},
            artifacts=["stage-15/verdict.json"],
            decision=decision,
            finished_at="2026-01-01T00:02:00+00:00",
        )
        store.set_node_status(
            node.id,
            "validating",
            created_at="2026-01-01T00:03:00+00:00",
        )
        store.set_node_status(
            node.id,
            status,
            created_at="2026-01-01T00:04:00+00:00",
        )

    aggregate = EvidenceAggregator(tmp_path).aggregate(
        generated_at="2026-01-01T00:05:00+00:00",
    )

    assert aggregate["run_id"] == tmp_path.name
    assert aggregate["hypothesis_tree"] == {
        "total_nodes": 2,
        "terminal_nodes": 2,
        "supported": 1,
        "inconclusive": 1,
        "superseded": 0,
    }
    assert len(aggregate["validation_summary"]) == 2
    assert aggregate["validation_summary"][0]["decision"] == "supported"
    assert aggregate["validation_summary"][0]["stage15_decision"] == "proceed"
    assert aggregate["validation_summary"][1]["decision"] == "inconclusive"
    assert aggregate["validation_summary"][1]["stage15_decision"] == "inconclusive"
    assert aggregate["recommended_hypotheses"] == [
        {
            "node_id": supported.id,
            "statement": "Supported hypothesis.",
            "prediction": "Supported prediction.",
            "falsification": "Supported falsification.",
        }
    ]
    assert json.loads((tmp_path / "hypothesis_aggregate.json").read_text()) == aggregate


def test_aggregate_includes_terminal_failed_attempt_for_diagnostics(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.hypothesis_store import HypothesisStore

    EvidenceAggregator = _aggregator_cls()
    store = HypothesisStore(tmp_path)
    node = store.create_node(
        statement="Failed validation hypothesis.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(tmp_path / "branches" / node.id / "attempt-001"),
        created_at="2026-01-01T00:01:00+00:00",
    )
    store.update_attempt(
        attempt.attempt_id,
        status="failed",
        artifacts=["stage-10/stage-10-workspace-agent-result.json"],
        error="E10_CODE_AGENT_FAIL: timeout",
        finished_at="2026-01-01T00:02:00+00:00",
    )
    store.set_node_status(
        node.id,
        "validating",
        created_at="2026-01-01T00:03:00+00:00",
    )
    store.set_node_status(
        node.id,
        "inconclusive",
        created_at="2026-01-01T00:04:00+00:00",
    )

    aggregate = EvidenceAggregator(tmp_path).aggregate(
        generated_at="2026-01-01T00:05:00+00:00",
    )

    row = aggregate["validation_summary"][0]
    assert row["decision"] == "inconclusive"
    assert row["branch_run_dir"].endswith("branches/h-001/attempt-001")
    assert row["artifacts"] == ["stage-10/stage-10-workspace-agent-result.json"]
    assert aggregate["evidence_registry"][node.id]["attempt_id"] == attempt.attempt_id


def test_write_root_handoff_marks_zero_supported_as_inconclusive(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.hypothesis_store import HypothesisStore

    EvidenceAggregator = _aggregator_cls()
    store = HypothesisStore(tmp_path)
    node = store.create_node(
        statement="Unvalidated hypothesis.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(tmp_path / "branches" / node.id / "attempt-001"),
        created_at="2026-01-01T00:01:00+00:00",
    )
    store.update_attempt(
        attempt.attempt_id,
        status="failed",
        artifacts=["stage-10/stage-10-workspace-agent-result.json"],
        error="E10_CODE_AGENT_FAIL: timeout",
        finished_at="2026-01-01T00:02:00+00:00",
    )
    store.set_node_status(
        node.id,
        "validating",
        created_at="2026-01-01T00:03:00+00:00",
    )
    store.set_node_status(
        node.id,
        "inconclusive",
        created_at="2026-01-01T00:04:00+00:00",
    )
    aggregator = EvidenceAggregator(tmp_path)
    aggregate = aggregator.aggregate(
        generated_at="2026-01-01T00:05:00+00:00",
    )

    handoff = aggregator.write_root_handoff(
        aggregate,
        generated_at="2026-01-01T00:05:00+00:00",
    )

    analysis = (tmp_path / "stage-14" / "analysis.md").read_text(encoding="utf-8")
    decision_md = (tmp_path / "stage-15" / "decision.md").read_text(
        encoding="utf-8"
    )
    verdict = json.loads((tmp_path / "stage-15" / "verdict.json").read_text())
    provenance = json.loads((tmp_path / "stage-14" / "provenance.json").read_text())

    assert handoff["decision"] == "inconclusive"
    assert "Result quality: 2/10" in analysis
    assert "No branch produced a supported verdict" in decision_md
    assert verdict["decision"] == "inconclusive"
    assert verdict["key_metrics"]["supported"] == 0
    assert provenance["source"] == "hypothesis_aggregate.json"
