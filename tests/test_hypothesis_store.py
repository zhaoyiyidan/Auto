from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
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


def _hypothesis_store_cls() -> Any:
    try:
        from researchclaw.pipeline.hypothesis_store import HypothesisStore
    except ImportError:
        pytest.fail("HypothesisStore is not implemented")
    return HypothesisStore


def _events(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "hypothesis_tree" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


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


def test_hypothesis_store_create_node_writes_node_artifacts_and_event(
    tmp_path: Path,
) -> None:
    HypothesisStore = _hypothesis_store_cls()
    store = HypothesisStore(tmp_path)

    node = store.create_node(
        statement="Treatment improves accuracy.",
        prediction="Accuracy increases by at least 5 points.",
        falsification="Accuracy does not improve.",
        rationale="Prior runs show useful signal.",
        baselines=("baseline-a",),
        source="stage8_batch",
        parent_id=None,
        created_at="2026-01-01T00:00:00+00:00",
    )

    node_dir = tmp_path / "hypothesis_tree" / "nodes" / "h-001"
    node_payload = json.loads((node_dir / "node.json").read_text(encoding="utf-8"))

    assert node.id == "h-001"
    assert node_payload == node.to_dict()
    assert (node_dir / "hypothesis.md").read_text(encoding="utf-8") == (
        "# Hypothesis h-001\n\n"
        "## Statement\nTreatment improves accuracy.\n\n"
        "## Prediction\nAccuracy increases by at least 5 points.\n\n"
        "## Falsification\nAccuracy does not improve.\n\n"
        "## Rationale\nPrior runs show useful signal.\n\n"
        "## Baselines\n- baseline-a\n"
    )
    assert _events(tmp_path) == [
        {
            "event_type": "node_proposed",
            "node_id": "h-001",
            "data": {
                "parent_id": None,
                "source": "stage8_batch",
                "hypothesis_hash": node.hypothesis_hash,
            },
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    ]


def test_hypothesis_store_add_and_update_attempt_persist_under_node(
    tmp_path: Path,
) -> None:
    HypothesisStore = _hypothesis_store_cls()
    store = HypothesisStore(tmp_path)
    node = store.create_node(
        statement="Treatment improves accuracy.",
        created_at="2026-01-01T00:00:00+00:00",
    )

    attempt = store.add_attempt(
        node_id=node.id,
        branch_run_dir="/tmp/run/hypothesis_branches/h-001/attempt-001",
        workspace_path="/tmp/workspaces/h-001-attempt-001",
        agent_session_name="researchclaw-h-001-attempt-001",
        created_at="2026-01-01T00:01:00+00:00",
    )

    attempt_path = (
        tmp_path
        / "hypothesis_tree"
        / "nodes"
        / "h-001"
        / "attempts"
        / "h-001"
        / "attempt-001.json"
    )
    assert attempt.attempt_id == "h-001/attempt-001"
    assert json.loads(attempt_path.read_text(encoding="utf-8")) == attempt.to_dict()

    updated = store.update_attempt(
        attempt.attempt_id,
        status="succeeded",
        stage_status={9: "done", 15: "done"},
        metrics={"score": 0.91},
        artifacts=["stage-15/decision.md"],
        decision="proceed",
        finished_at="2026-01-01T00:10:00+00:00",
    )

    persisted = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert persisted == updated.to_dict()
    assert persisted["status"] == "succeeded"
    assert persisted["stage_status"] == {"9": "done", "15": "done"}
    assert [event["event_type"] for event in _events(tmp_path)] == [
        "node_proposed",
        "attempt_queued",
        "attempt_finished",
    ]


def test_hypothesis_store_set_node_status_enforces_state_machine(
    tmp_path: Path,
) -> None:
    HypothesisStore = _hypothesis_store_cls()
    store = HypothesisStore(tmp_path)
    node = store.create_node(
        statement="Treatment improves accuracy.",
        created_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="Illegal hypothesis node transition"):
        store.set_node_status(
            node.id,
            "supported",
            created_at="2026-01-01T00:01:00+00:00",
        )

    validating = store.set_node_status(
        node.id,
        "validating",
        created_at="2026-01-01T00:02:00+00:00",
    )
    supported = store.set_node_status(
        node.id,
        "supported",
        created_at="2026-01-01T00:03:00+00:00",
    )

    node_payload = json.loads(
        (
            tmp_path
            / "hypothesis_tree"
            / "nodes"
            / node.id
            / "node.json"
        ).read_text(encoding="utf-8")
    )
    assert validating.status == "validating"
    assert supported.status == "supported"
    assert node_payload["status"] == "supported"
    assert [event["event_type"] for event in _events(tmp_path)] == [
        "node_proposed",
        "node_status_changed",
        "node_verdict",
    ]

    with pytest.raises(ValueError, match="Illegal hypothesis node transition"):
        store.set_node_status(
            node.id,
            "validating",
            created_at="2026-01-01T00:04:00+00:00",
        )


def test_hypothesis_store_filelock_serializes_concurrent_event_appends(
    tmp_path: Path,
) -> None:
    HypothesisStore = _hypothesis_store_cls()
    store = HypothesisStore(tmp_path)
    count = 32
    barrier = threading.Barrier(count)

    def append_one(index: int) -> None:
        barrier.wait(timeout=5)
        store.append_event(
            event_type="concurrent_test",
            node_id=None,
            data={"index": index},
            timestamp=f"2026-01-01T00:00:{index:02d}+00:00",
        )

    with ThreadPoolExecutor(max_workers=count) as pool:
        list(pool.map(append_one, range(count)))

    events = _events(tmp_path)
    assert len(events) == count
    assert sorted(event["data"]["index"] for event in events) == list(range(count))
    assert (tmp_path / "hypothesis_tree" / ".lock").is_file()
