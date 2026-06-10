from __future__ import annotations

from typing import Any

import pytest


def _hypothesis_node_cls() -> Any:
    try:
        from researchclaw.pipeline.hypothesis_store import HypothesisNode
    except ModuleNotFoundError:
        pytest.fail("HypothesisNode is not implemented")
    return HypothesisNode


def _validation_attempt_cls() -> Any:
    try:
        from researchclaw.pipeline.hypothesis_store import ValidationAttempt
    except ImportError:
        pytest.fail("ValidationAttempt is not implemented")
    return ValidationAttempt


def test_hypothesis_node_hash_is_stable_and_content_based() -> None:
    HypothesisNode = _hypothesis_node_cls()
    node = HypothesisNode(
        id="h-001",
        statement="Treatment improves accuracy.",
        prediction="Accuracy increases by at least 5 points.",
        falsification="Accuracy does not improve.",
        rationale="Prior runs show useful signal.",
        baselines=("baseline-a", "baseline-b"),
        source="stage8_batch",
        parent_id=None,
        created_at="2026-01-01T00:00:00+00:00",
    )
    same_science = HypothesisNode(
        id="h-099",
        statement="Treatment improves accuracy.",
        prediction="Accuracy increases by at least 5 points.",
        falsification="Accuracy does not improve.",
        rationale="Different rationale is not part of the identity hash.",
        baselines=("other",),
        source="pivot",
        parent_id="h-010",
        created_at="2026-02-01T00:00:00+00:00",
    )
    different_science = HypothesisNode(
        id="h-100",
        statement="Treatment improves robustness.",
        prediction="Accuracy increases by at least 5 points.",
        falsification="Accuracy does not improve.",
        rationale="Prior runs show useful signal.",
        source="stage8_batch",
        parent_id=None,
        created_at="2026-01-01T00:00:00+00:00",
    )

    assert node.hypothesis_hash == same_science.hypothesis_hash
    assert node.hypothesis_hash != different_science.hypothesis_hash


def test_hypothesis_node_round_trips_through_dict() -> None:
    HypothesisNode = _hypothesis_node_cls()
    node = HypothesisNode(
        id="h-001",
        statement="Treatment improves accuracy.",
        prediction="Accuracy increases by at least 5 points.",
        falsification="Accuracy does not improve.",
        rationale="Prior runs show useful signal.",
        baselines=("baseline-a", "baseline-b"),
        source="stage8_batch",
        parent_id="root",
        created_at="2026-01-01T00:00:00+00:00",
    )

    payload = node.to_dict()
    loaded = HypothesisNode.from_dict(payload)

    assert loaded == node
    assert payload == {
        "id": "h-001",
        "statement": "Treatment improves accuracy.",
        "prediction": "Accuracy increases by at least 5 points.",
        "falsification": "Accuracy does not improve.",
        "rationale": "Prior runs show useful signal.",
        "baselines": ["baseline-a", "baseline-b"],
        "source": "stage8_batch",
        "parent_id": "root",
        "hypothesis_hash": node.hypothesis_hash,
        "created_at": "2026-01-01T00:00:00+00:00",
        "status": "proposed",
    }


def test_validation_attempt_defaults_and_round_trip() -> None:
    ValidationAttempt = _validation_attempt_cls()
    attempt = ValidationAttempt(
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        branch_run_dir="/tmp/run/hypothesis_branches/h-001/attempt-001",
        workspace_path="/tmp/workspaces/h-001-attempt-001",
        agent_session_name="researchclaw-h-001-attempt-001",
        stage_status={9: "done", 10: "running"},
        metrics={"score": 0.91},
        artifacts=["stage-15/decision.md"],
        decision="proceed",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:10:00+00:00",
    )

    payload = attempt.to_dict()
    loaded = ValidationAttempt.from_dict(payload)

    assert loaded == attempt
    assert payload == {
        "attempt_id": "h-001/attempt-001",
        "node_id": "h-001",
        "status": "queued",
        "branch_run_dir": "/tmp/run/hypothesis_branches/h-001/attempt-001",
        "workspace_path": "/tmp/workspaces/h-001-attempt-001",
        "agent_session_name": "researchclaw-h-001-attempt-001",
        "stage_status": {"9": "done", "10": "running"},
        "metrics": {"score": 0.91},
        "artifacts": ["stage-15/decision.md"],
        "decision": "proceed",
        "error": None,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:10:00+00:00",
    }


def test_validation_attempt_rejects_invalid_status() -> None:
    ValidationAttempt = _validation_attempt_cls()

    with pytest.raises(ValueError, match="Invalid validation attempt status"):
        ValidationAttempt(
            attempt_id="h-001/attempt-001",
            node_id="h-001",
            status="complete",
            branch_run_dir="/tmp/run/hypothesis_branches/h-001/attempt-001",
        )
