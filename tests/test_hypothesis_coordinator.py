from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
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


def _events(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "hypothesis_tree" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


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


def test_coordinator_validates_split_nodes_sequentially_and_continues_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    hypotheses_md = """
## H1: Accuracy hypothesis
Statement: Treatment improves accuracy.
Prediction: Accuracy increases by at least 5 points.
Falsification: Accuracy does not improve.
Rationale: Prior runs show useful signal.

## H2: Robustness hypothesis
Statement: Treatment improves robustness.
Prediction: Error rate falls under perturbation.
Falsification: Robustness does not improve.
Rationale: Different mechanism.

## H3: Calibration hypothesis
Statement: Treatment improves calibration.
Prediction: Calibration error falls.
Falsification: Calibration does not improve.
Rationale: Independent signal.
"""
    calls: list[tuple[str, str, str]] = []

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        calls.append(
            (
                node.id,
                attempt.attempt_id,
                Path(attempt.branch_run_dir).relative_to(tmp_path).as_posix(),
            )
        )
        if node.id == "h-002":
            raise RuntimeError("branch crashed")
        decision = "proceed" if node.id == "h-001" else "inconclusive"
        return SimpleNamespace(
            decision=decision,
            artifacts=("stage-15/decision.md",),
            metrics={"score": 0.9},
        )

    monkeypatch.setattr(
        coordinator,
        "validate_branch",
        fake_validate_branch,
        raising=False,
    )

    attempts = coordinator.split_and_validate_sequential(
        hypotheses_md,
        config=object(),
        adapters=object(),
        created_at="2026-01-01T00:00:00+00:00",
    )

    assert calls == [
        ("h-001", "h-001/attempt-001", "hypothesis_branches/h-001/attempt-001"),
        ("h-002", "h-002/attempt-001", "hypothesis_branches/h-002/attempt-001"),
        ("h-003", "h-003/attempt-001", "hypothesis_branches/h-003/attempt-001"),
    ]
    assert [
        (attempt.node_id, attempt.status, attempt.decision, attempt.error)
        for attempt in attempts
    ] == [
        ("h-001", "succeeded", "proceed", None),
        ("h-002", "failed", None, "branch crashed"),
        ("h-003", "succeeded", "inconclusive", None),
    ]
    verdicts = [
        (event["node_id"], event["data"]["attempt_id"], event["data"]["decision"])
        for event in _events(tmp_path)
        if event["event_type"] == "node_verdict"
    ]
    assert verdicts == [
        ("h-001", "h-001/attempt-001", "proceed"),
        ("h-003", "h-003/attempt-001", "inconclusive"),
    ]
