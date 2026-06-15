from __future__ import annotations

import json
from pathlib import Path
import threading
import time
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
        ("h-002", "h-002/attempt-001", "inconclusive"),
        ("h-003", "h-003/attempt-001", "inconclusive"),
    ]


def test_coordinator_maps_decisions_to_node_statuses_and_followup_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    hypotheses_md = """
## H1: Proceed hypothesis
Statement: Proceeding hypothesis.
Prediction: Proceed prediction.
Falsification: Proceed falsification.
Rationale: Proceed rationale.

## H2: Inconclusive hypothesis
Statement: Inconclusive hypothesis.
Prediction: Inconclusive prediction.
Falsification: Inconclusive falsification.
Rationale: Inconclusive rationale.

## H3: Extend hypothesis
Statement: Extend source hypothesis.
Prediction: Extend source prediction.
Falsification: Extend source falsification.
Rationale: Extend source rationale.

## H4: Pivot hypothesis
Statement: Pivot source hypothesis.
Prediction: Pivot source prediction.
Falsification: Pivot source falsification.
Rationale: Pivot source rationale.
"""
    followups = {
        "h-003": {
            "statement": "Extended child hypothesis.",
            "prediction": "Extended child prediction.",
            "falsification": "Extended child falsification.",
            "rationale": "Extended child rationale.",
            "baselines": ("extended-baseline",),
        },
        "h-004": {
            "statement": "Pivot sibling hypothesis.",
            "prediction": "Pivot sibling prediction.",
            "falsification": "Pivot sibling falsification.",
            "rationale": "Pivot sibling rationale.",
            "baselines": ("pivot-baseline",),
        },
    }

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        decisions = {
            "h-001": "proceed",
            "h-002": "inconclusive",
            "h-003": "extend",
            "h-004": "pivot",
        }
        return SimpleNamespace(
            decision=decisions[node.id],
            artifacts=("stage-15/decision.md",),
            metrics={"score": 0.9},
            next_hypothesis=followups.get(node.id),
        )

    monkeypatch.setattr(
        coordinator,
        "validate_branch",
        fake_validate_branch,
        raising=False,
    )

    coordinator.split_and_validate_sequential(
        hypotheses_md,
        config=object(),
        adapters=object(),
        created_at="2026-01-01T00:00:00+00:00",
    )

    assert _node_payload(tmp_path, "h-001")["status"] == "supported"
    assert _node_payload(tmp_path, "h-002")["status"] == "inconclusive"
    assert _node_payload(tmp_path, "h-003")["status"] == "superseded"
    assert _node_payload(tmp_path, "h-004")["status"] == "superseded"

    extend_child = _node_payload(tmp_path, "h-005")
    assert extend_child["statement"] == "Extended child hypothesis."
    assert extend_child["parent_id"] == "h-003"
    assert extend_child["source"] == "extend"
    assert extend_child["status"] == "proposed"

    pivot_sibling = _node_payload(tmp_path, "h-006")
    assert pivot_sibling["statement"] == "Pivot sibling hypothesis."
    assert pivot_sibling["parent_id"] is None
    assert pivot_sibling["source"] == "pivot"
    assert pivot_sibling["status"] == "proposed"

    verdicts = [
        (
            event["node_id"],
            event["data"].get("attempt_id"),
            event["data"].get("decision"),
            event["data"].get("to"),
        )
        for event in _events(tmp_path)
        if event["event_type"] == "node_verdict"
    ]
    assert verdicts == [
        ("h-001", "h-001/attempt-001", "proceed", "supported"),
        ("h-002", "h-002/attempt-001", "inconclusive", "inconclusive"),
        ("h-003", "h-003/attempt-001", "extend", "superseded"),
        ("h-004", "h-004/attempt-001", "pivot", "superseded"),
    ]


def test_coordinator_resumes_queued_attempts_without_result_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue, WorkItem

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    node = coordinator.store.create_node(
        statement="Treatment improves accuracy.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = coordinator.store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(
            tmp_path / "hypothesis_branches" / node.id / "attempt-001"
        ),
        created_at="2026-01-01T00:01:00+00:00",
    )
    DurableWorkQueue(tmp_path).append(
        WorkItem(
            node_id=node.id,
            attempt_id=attempt.attempt_id,
            branch_run_dir=attempt.branch_run_dir,
        ),
        created_at="2026-01-01T00:02:00+00:00",
    )
    calls: list[tuple[str, str]] = []

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        calls.append((node.id, attempt.attempt_id))
        return SimpleNamespace(
            decision="proceed",
            artifacts=("stage-15/decision.md",),
            metrics={"score": 0.9},
        )

    monkeypatch.setattr(
        coordinator,
        "validate_branch",
        fake_validate_branch,
        raising=False,
    )

    resumed = coordinator.resume_pending_work(
        config=object(),
        adapters=object(),
        created_at="2026-01-01T00:03:00+00:00",
    )

    assert calls == [(node.id, attempt.attempt_id)]
    assert [(item.node_id, item.status, item.decision) for item in resumed] == [
        (node.id, "succeeded", "proceed")
    ]
    assert _node_payload(tmp_path, node.id)["status"] == "supported"


def test_coordinator_enqueues_followup_attempts_for_extend_and_pivot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    hypotheses_md = """
## H1: Extend hypothesis
Statement: Extend source hypothesis.
Prediction: Extend source prediction.
Falsification: Extend source falsification.
Rationale: Extend source rationale.

## H2: Pivot hypothesis
Statement: Pivot source hypothesis.
Prediction: Pivot source prediction.
Falsification: Pivot source falsification.
Rationale: Pivot source rationale.
"""
    followups = {
        "h-001": {
            "statement": "Extended child hypothesis.",
            "prediction": "Extended child prediction.",
            "falsification": "Extended child falsification.",
            "rationale": "Extended child rationale.",
        },
        "h-002": {
            "statement": "Pivot sibling hypothesis.",
            "prediction": "Pivot sibling prediction.",
            "falsification": "Pivot sibling falsification.",
            "rationale": "Pivot sibling rationale.",
        },
    }

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        return SimpleNamespace(
            decision="extend" if node.id == "h-001" else "pivot",
            artifacts=("stage-15/decision.md",),
            metrics={},
            next_hypothesis=followups[node.id],
        )

    monkeypatch.setattr(
        coordinator,
        "validate_branch",
        fake_validate_branch,
        raising=False,
    )

    coordinator.split_and_validate_sequential(
        hypotheses_md,
        config=object(),
        adapters=object(),
        created_at="2026-01-01T00:00:00+00:00",
    )

    items = DurableWorkQueue(tmp_path).read_items()
    assert [(item.node_id, item.attempt_id) for item in items] == [
        ("h-003", "h-003/attempt-001"),
        ("h-004", "h-004/attempt-001"),
    ]
    assert [
        Path(item.branch_run_dir).relative_to(tmp_path).as_posix()
        for item in items
    ] == [
        "hypothesis_branches/h-003/attempt-001",
        "hypothesis_branches/h-004/attempt-001",
    ]
    assert _node_payload(tmp_path, "h-003")["parent_id"] == "h-001"
    assert _node_payload(tmp_path, "h-004")["parent_id"] is None


def test_coordinator_reduces_attempt_result_idempotently(
    tmp_path: Path,
) -> None:
    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    node = coordinator.store.create_node(
        statement="Treatment improves accuracy.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = coordinator.store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(
            tmp_path / "hypothesis_branches" / node.id / "attempt-001"
        ),
        created_at="2026-01-01T00:01:00+00:00",
    )
    coordinator.store.set_node_status(
        node.id,
        "validating",
        created_at="2026-01-01T00:02:00+00:00",
    )
    result_path = Path(attempt.branch_run_dir) / "attempt_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "attempt_id": attempt.attempt_id,
                "node_id": node.id,
                "status": "succeeded",
                "decision": "proceed",
                "artifacts": ["stage-15/decision.md"],
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    first = coordinator.reduce_attempt_result(
        result_path,
        created_at="2026-01-01T00:03:00+00:00",
    )
    second = coordinator.reduce_attempt_result(
        result_path,
        created_at="2026-01-01T00:04:00+00:00",
    )

    assert first == second
    events = _events(tmp_path)
    assert [
        event["event_type"]
        for event in events
        if event["event_type"] == "attempt_finished"
    ] == ["attempt_finished"]
    assert [
        event["data"]["attempt_id"]
        for event in events
        if event["event_type"] == "node_verdict"
    ] == [attempt.attempt_id]


def test_coordinator_honors_concurrent_branch_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue, WorkItem

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    queue = DurableWorkQueue(tmp_path)
    for index in range(4):
        node = coordinator.store.create_node(
            statement=f"Hypothesis {index}",
            created_at=f"2026-01-01T00:00:0{index}+00:00",
        )
        attempt = coordinator.store.add_attempt(
            node_id=node.id,
            branch_run_dir=str(
                tmp_path / "hypothesis_branches" / node.id / "attempt-001"
            ),
            created_at=f"2026-01-01T00:01:0{index}+00:00",
        )
        queue.append(
            WorkItem(
                node_id=node.id,
                attempt_id=attempt.attempt_id,
                branch_run_dir=attempt.branch_run_dir,
            ),
            created_at=f"2026-01-01T00:02:0{index}+00:00",
        )

    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return SimpleNamespace(
            decision="proceed",
            artifacts=("stage-15/decision.md",),
            metrics={},
        )

    monkeypatch.setattr(
        coordinator,
        "validate_branch",
        fake_validate_branch,
        raising=False,
    )

    completed = coordinator.run_pending_work_concurrent(
        config=object(),
        adapters=object(),
        max_concurrent=2,
        created_at="2026-01-01T00:03:00+00:00",
    )

    assert len(completed) == 4
    assert max_active == 2
    assert [attempt.status for attempt in completed] == ["succeeded"] * 4


def test_coordinator_reduces_attempt_results_concurrently_without_lost_verdicts(
    tmp_path: Path,
) -> None:
    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    result_paths: list[Path] = []
    for index in range(2):
        node = coordinator.store.create_node(
            statement=f"Hypothesis {index}",
            created_at=f"2026-01-01T00:00:0{index}+00:00",
        )
        attempt = coordinator.store.add_attempt(
            node_id=node.id,
            branch_run_dir=str(
                tmp_path / "hypothesis_branches" / node.id / "attempt-001"
            ),
            created_at=f"2026-01-01T00:01:0{index}+00:00",
        )
        coordinator.store.set_node_status(
            node.id,
            "validating",
            created_at=f"2026-01-01T00:02:0{index}+00:00",
        )
        result_path = Path(attempt.branch_run_dir) / "attempt_result.json"
        result_path.parent.mkdir(parents=True)
        result_path.write_text(
            json.dumps(
                {
                    "attempt_id": attempt.attempt_id,
                    "node_id": node.id,
                    "status": "succeeded",
                    "decision": "proceed",
                    "artifacts": ["stage-15/decision.md"],
                    "error": None,
                }
            ),
            encoding="utf-8",
        )
        result_paths.append(result_path)

    reduced = coordinator.reduce_attempt_results_concurrent(
        result_paths,
        max_concurrent=2,
        created_at="2026-01-01T00:03:00+00:00",
    )

    assert [attempt.status for attempt in reduced] == ["succeeded", "succeeded"]
    verdict_attempts = [
        event["data"]["attempt_id"]
        for event in _events(tmp_path)
        if event["event_type"] == "node_verdict"
    ]
    assert sorted(verdict_attempts) == [
        "h-001/attempt-001",
        "h-002/attempt-001",
    ]


def test_coordinator_requeues_abandoned_attempts_up_to_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline.hypothesis_coordinator import WorkerAbandoned
    from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue, WorkItem

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    node = coordinator.store.create_node(
        statement="Treatment improves accuracy.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = coordinator.store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(
            tmp_path / "hypothesis_branches" / node.id / "attempt-001"
        ),
        created_at="2026-01-01T00:01:00+00:00",
    )
    queue = DurableWorkQueue(tmp_path)
    queue.append(
        WorkItem(
            node_id=node.id,
            attempt_id=attempt.attempt_id,
            branch_run_dir=attempt.branch_run_dir,
        ),
        created_at="2026-01-01T00:02:00+00:00",
    )

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        raise WorkerAbandoned("worker died")

    monkeypatch.setattr(
        coordinator,
        "validate_branch",
        fake_validate_branch,
        raising=False,
    )

    completed = coordinator.run_pending_work_concurrent(
        config=object(),
        adapters=object(),
        max_concurrent=1,
        max_attempts_per_node=2,
        created_at="2026-01-01T00:03:00+00:00",
    )

    assert [(item.node_id, item.status, item.error) for item in completed] == [
        (node.id, "abandoned", "worker died")
    ]
    assert [item.attempt_id for item in DurableWorkQueue(tmp_path).read_items()] == [
        "h-001/attempt-002",
    ]
    assert [
        item.attempt_id
        for item in DurableWorkQueue(tmp_path).read_items(include_done=True)
    ] == [
        "h-001/attempt-001",
        "h-001/attempt-002",
    ]
    retry_dir = Path(DurableWorkQueue(tmp_path).read_items()[0].branch_run_dir)
    assert retry_dir.relative_to(tmp_path).as_posix() == (
        "hypothesis_branches/h-001/attempt-002"
    )
    retry_hypotheses = retry_dir / "stage-08" / "hypotheses.md"
    assert retry_hypotheses.is_file()
    assert "Statement: Treatment improves accuracy." in retry_hypotheses.read_text(
        encoding="utf-8"
    )


def test_failed_branch_result_marks_attempt_failed_and_node_inconclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue, WorkItem

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    node = coordinator.store.create_node(
        statement="Treatment improves accuracy.",
        prediction="Accuracy improves.",
        falsification="Accuracy does not improve.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = coordinator.store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(
            tmp_path / "hypothesis_branches" / node.id / "attempt-001"
        ),
        created_at="2026-01-01T00:01:00+00:00",
    )
    DurableWorkQueue(tmp_path).append(
        WorkItem(
            node_id=node.id,
            attempt_id=attempt.attempt_id,
            branch_run_dir=attempt.branch_run_dir,
        ),
        created_at="2026-01-01T00:02:00+00:00",
    )

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        return {
            "attempt_id": attempt.attempt_id,
            "node_id": node.id,
            "status": "failed",
            "decision": None,
            "artifacts": ["stage-10/stage-10-workspace-agent-result.json"],
            "error": "E10_CODE_AGENT_FAIL: timeout",
        }

    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    completed = coordinator.run_pending_work_concurrent(
        config=object(),
        adapters=object(),
        max_concurrent=1,
        created_at="2026-01-01T00:03:00+00:00",
    )

    assert [(item.node_id, item.status, item.error) for item in completed] == [
        (node.id, "failed", "E10_CODE_AGENT_FAIL: timeout")
    ]
    assert _node_payload(tmp_path, node.id)["status"] == "inconclusive"
    verdicts = [
        event
        for event in _events(tmp_path)
        if event["event_type"] == "node_verdict"
    ]
    assert verdicts[-1]["data"] == {
        "from": "validating",
        "to": "inconclusive",
        "attempt_id": attempt.attempt_id,
        "decision": "inconclusive",
        "error": "E10_CODE_AGENT_FAIL: timeout",
    }


def test_abandoned_attempt_marks_node_inconclusive_when_retry_cap_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline.hypothesis_coordinator import WorkerAbandoned
    from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue, WorkItem

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    node = coordinator.store.create_node(
        statement="Treatment improves accuracy.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = coordinator.store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(
            tmp_path / "hypothesis_branches" / node.id / "attempt-001"
        ),
        created_at="2026-01-01T00:01:00+00:00",
    )
    DurableWorkQueue(tmp_path).append(
        WorkItem(
            node_id=node.id,
            attempt_id=attempt.attempt_id,
            branch_run_dir=attempt.branch_run_dir,
        ),
        created_at="2026-01-01T00:02:00+00:00",
    )

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        raise WorkerAbandoned("worker died")

    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    completed = coordinator.run_pending_work_concurrent(
        config=object(),
        adapters=object(),
        max_concurrent=1,
        max_attempts_per_node=1,
        created_at="2026-01-01T00:03:00+00:00",
    )

    assert [(item.node_id, item.status, item.error) for item in completed] == [
        (node.id, "abandoned", "worker died")
    ]
    assert DurableWorkQueue(tmp_path).read_items() == []
    assert _node_payload(tmp_path, node.id)["status"] == "inconclusive"


def test_run_until_queue_empty_terminates_and_writes_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    coordinator.split_and_queue(
        """
## H1
Statement: First hypothesis.
Prediction: First prediction.
Falsification: First falsification.

## H2
Statement: Second hypothesis.
Prediction: Second prediction.
Falsification: Second falsification.
""",
        created_at="2026-01-01T00:00:00+00:00",
    )
    calls: list[str] = []

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        calls.append(node.id)
        return SimpleNamespace(
            decision="proceed",
            artifacts=("stage-15/verdict.json",),
            metrics={"score": 0.9},
        )

    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    nodes = coordinator.run_until_queue_empty(
        config=object(),
        adapters=object(),
        max_concurrent=1,
        created_at="2026-01-01T00:01:00+00:00",
    )

    assert sorted(calls) == ["h-001", "h-002"]
    assert [node.status for node in nodes] == ["supported", "supported"]
    aggregate = json.loads((tmp_path / "hypothesis_aggregate.json").read_text())
    assert aggregate["hypothesis_tree"]["total_nodes"] == 2
    assert len(aggregate["validation_summary"]) == 2


def test_run_until_queue_empty_runs_extend_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    coordinator.split_and_queue(
        """
## H1
Statement: Parent hypothesis.
Prediction: Parent prediction.
Falsification: Parent falsification.

## H2
Statement: Sibling hypothesis.
Prediction: Sibling prediction.
Falsification: Sibling falsification.
""",
        created_at="2026-01-01T00:00:00+00:00",
    )
    seen: list[str] = []

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        _ = attempt, config, adapters
        seen.append(node.id)
        if node.id == "h-001":
            return SimpleNamespace(
                decision="extend",
                artifacts=("stage-15/verdict.json",),
                metrics={"score": 0.7},
                next_hypothesis={
                    "statement": "Child hypothesis.",
                    "prediction": "Child prediction.",
                    "falsification": "Child falsification.",
                    "rationale": "Follow-up.",
                },
            )
        return SimpleNamespace(
            decision="proceed",
            artifacts=("stage-15/verdict.json",),
            metrics={"score": 0.9},
        )

    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    nodes = coordinator.run_until_queue_empty(
        config=object(),
        adapters=object(),
        max_concurrent=1,
        created_at="2026-01-01T00:01:00+00:00",
    )

    assert seen == ["h-001", "h-002", "h-003"]
    assert [(node.id, node.status, node.parent_id) for node in nodes] == [
        ("h-001", "superseded", None),
        ("h-002", "supported", None),
        ("h-003", "supported", "h-001"),
    ]
    child_hypotheses = (
        tmp_path
        / "hypothesis_branches"
        / "h-003"
        / "attempt-001"
        / "stage-08"
        / "hypotheses.md"
    )
    assert child_hypotheses.is_file()
    assert "Statement: Child hypothesis." in child_hypotheses.read_text(
        encoding="utf-8"
    )
    aggregate = json.loads((tmp_path / "hypothesis_aggregate.json").read_text())
    assert aggregate["hypothesis_tree"]["total_nodes"] == 3


def test_run_until_queue_empty_honors_max_tree_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    coordinator.split_and_queue(
        """
## H1
Statement: Depth-limited hypothesis.
Prediction: Depth-limited prediction.
Falsification: Depth-limited falsification.
""",
        created_at="2026-01-01T00:00:00+00:00",
    )

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        _ = node, attempt, config, adapters
        return SimpleNamespace(
            decision="extend",
            artifacts=("stage-15/verdict.json",),
            metrics={"score": 0.5},
            next_hypothesis={
                "statement": "Skipped child hypothesis.",
                "prediction": "Skipped child prediction.",
                "falsification": "Skipped child falsification.",
            },
        )

    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    nodes = coordinator.run_until_queue_empty(
        config=SimpleNamespace(
            hypothesis_validation=SimpleNamespace(
                max_tree_depth=0,
                max_total_attempts=10,
            )
        ),
        adapters=object(),
        max_concurrent=1,
        created_at="2026-01-01T00:01:00+00:00",
    )

    assert [(node.id, node.status) for node in nodes] == [
        ("h-001", "superseded")
    ]
    assert DurableWorkQueue(tmp_path).read_items() == []
    events = _events(tmp_path)
    assert any(
        event["event_type"] == "followup_skipped"
        and event["data"]["reason"] == "max_tree_depth"
        for event in events
    )


def test_run_until_queue_empty_dedupes_followup_hypotheses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    coordinator.split_and_queue(
        """
## H1
Statement: First parent hypothesis.
Prediction: First parent prediction.
Falsification: First parent falsification.

## H2
Statement: Second parent hypothesis.
Prediction: Second parent prediction.
Falsification: Second parent falsification.
""",
        created_at="2026-01-01T00:00:00+00:00",
    )

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        _ = attempt, config, adapters
        if node.id in {"h-001", "h-002"}:
            return SimpleNamespace(
                decision="extend",
                artifacts=("stage-15/verdict.json",),
                metrics={"score": 0.5},
                next_hypothesis={
                    "statement": "Shared follow-up hypothesis.",
                    "prediction": "Shared follow-up prediction.",
                    "falsification": "Shared follow-up falsification.",
                },
            )
        return SimpleNamespace(
            decision="proceed",
            artifacts=("stage-15/verdict.json",),
            metrics={"score": 0.9},
        )

    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    nodes = coordinator.run_until_queue_empty(
        config=SimpleNamespace(
            hypothesis_validation=SimpleNamespace(
                max_tree_depth=3,
                max_total_attempts=10,
            )
        ),
        adapters=object(),
        max_concurrent=2,
        created_at="2026-01-01T00:01:00+00:00",
    )

    assert [(node.id, node.status) for node in nodes] == [
        ("h-001", "superseded"),
        ("h-002", "superseded"),
        ("h-003", "supported"),
    ]
    dedup_events = [
        event for event in _events(tmp_path) if event["event_type"] == "followup_deduped"
    ]
    assert len(dedup_events) == 1
    assert dedup_events[0]["data"]["existing_node_id"] == "h-003"


def test_resume_skips_completed_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue, WorkItem

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    node = coordinator.store.create_node(
        statement="Completed hypothesis.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = coordinator.store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(tmp_path / "hypothesis_branches" / node.id / "attempt-001"),
        created_at="2026-01-01T00:01:00+00:00",
    )
    DurableWorkQueue(tmp_path).append(
        WorkItem(
            node_id=node.id,
            attempt_id=attempt.attempt_id,
            branch_run_dir=attempt.branch_run_dir,
        ),
        created_at="2026-01-01T00:02:00+00:00",
    )
    result_path = Path(attempt.branch_run_dir) / "attempt_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "attempt_id": attempt.attempt_id,
                "node_id": node.id,
                "status": "succeeded",
                "decision": "proceed",
                "artifacts": ["stage-15/verdict.json"],
                "metrics": {"score": 0.93},
            }
        ),
        encoding="utf-8",
    )

    def fail_validate_branch(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise AssertionError("completed attempt should not rerun")

    monkeypatch.setattr(coordinator, "validate_branch", fail_validate_branch)

    resumed = coordinator.resume_pending_work(
        config=object(),
        adapters=object(),
        created_at="2026-01-01T00:03:00+00:00",
    )

    assert [(attempt.status, attempt.decision) for attempt in resumed] == [
        ("succeeded", "proceed")
    ]
    assert DurableWorkQueue(tmp_path).read_items() == []


def test_run_until_queue_empty_reduces_existing_branch_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    node = coordinator.store.create_node(
        statement="Completed unqueued hypothesis.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = coordinator.store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(tmp_path / "hypothesis_branches" / node.id / "attempt-001"),
        created_at="2026-01-01T00:01:00+00:00",
    )
    coordinator.store.set_node_status(
        node.id,
        "validating",
        created_at="2026-01-01T00:02:00+00:00",
    )
    result_path = Path(attempt.branch_run_dir) / "attempt_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "attempt_id": attempt.attempt_id,
                "node_id": node.id,
                "status": "succeeded",
                "decision": "proceed",
                "artifacts": ["stage-15/verdict.json"],
                "metrics": {"score": 0.88},
            }
        ),
        encoding="utf-8",
    )

    def fail_validate_branch(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise AssertionError("existing result should be reduced without rerun")

    monkeypatch.setattr(coordinator, "validate_branch", fail_validate_branch)

    nodes = coordinator.run_until_queue_empty(
        config=object(),
        adapters=object(),
        max_concurrent=1,
        created_at="2026-01-01T00:03:00+00:00",
    )

    assert [(node.id, node.status) for node in nodes] == [
        ("h-001", "supported")
    ]
    assert (tmp_path / "hypothesis_aggregate.json").is_file()


def test_concurrent_branches_get_unique_workspaces_and_session_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from researchclaw.pipeline import hypothesis_branch

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    coordinator.split_and_queue(
        """
## H1
Statement: First workspace hypothesis.
Prediction: First prediction.
Falsification: First falsification.

## H2
Statement: Second workspace hypothesis.
Prediction: Second prediction.
Falsification: Second falsification.
""",
        created_at="2026-01-01T00:00:00+00:00",
    )
    provisioned: list[str] = []
    released: list[str] = []
    seen: list[tuple[str | None, str | None]] = []

    def fake_provision_workspace(
        attempt: Any,
        *,
        source_workspace: Path,
        workspace_root: Path | None = None,
    ) -> Any:
        _ = source_workspace
        workspace = Path(workspace_root or tmp_path / ".worktrees") / (
            attempt.attempt_id.replace("/", "-")
        )
        workspace.mkdir(parents=True, exist_ok=True)
        provisioned.append(str(workspace))
        return replace(attempt, workspace_path=str(workspace))

    def fake_release_workspace(attempt: Any, *, source_workspace: Path | None = None) -> None:
        _ = source_workspace
        released.append(str(attempt.workspace_path))

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        _ = node, config, adapters
        seen.append((attempt.workspace_path, attempt.agent_session_name))
        return SimpleNamespace(
            decision="proceed",
            artifacts=("stage-15/verdict.json",),
            metrics={"score": 0.9},
        )

    monkeypatch.setattr(
        hypothesis_branch,
        "provision_workspace",
        fake_provision_workspace,
    )
    monkeypatch.setattr(
        hypothesis_branch,
        "release_workspace",
        fake_release_workspace,
    )
    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    config = SimpleNamespace(
        hypothesis_validation=SimpleNamespace(
            workspace_isolation="shared",
            max_tree_depth=3,
            max_total_attempts=10,
        ),
        experiment=SimpleNamespace(
            workspace_agent=SimpleNamespace(workspace_path=str(tmp_path / "source"))
        ),
    )
    completed = coordinator.run_pending_work_concurrent(
        config=config,
        adapters=object(),
        max_concurrent=2,
        created_at="2026-01-01T00:01:00+00:00",
    )

    assert [attempt.status for attempt in completed] == ["succeeded", "succeeded"]
    assert len({workspace for workspace, _session in seen}) == 2
    assert len({session for _workspace, session in seen}) == 2
    assert all(session and session.startswith(f"{tmp_path.name}-h-") for _, session in seen)
    assert sorted(provisioned) == sorted(released)


def test_workspace_isolation_provision_failure_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline import hypothesis_branch
    from researchclaw.pipeline.hypothesis_coordinator import WorkspaceIsolationError

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    node = coordinator.store.create_node(
        statement="Treatment improves accuracy.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = coordinator.store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(tmp_path / "hypothesis_branches" / node.id / "attempt-001"),
        created_at="2026-01-01T00:01:00+00:00",
    )

    def fail_provision_workspace(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise RuntimeError("git worktree add failed")

    monkeypatch.setattr(
        hypothesis_branch,
        "provision_workspace",
        fail_provision_workspace,
    )
    config = SimpleNamespace(
        hypothesis_validation=SimpleNamespace(workspace_isolation="worktree"),
        experiment=SimpleNamespace(
            workspace_agent=SimpleNamespace(workspace_path=str(tmp_path / "source"))
        ),
    )

    with pytest.raises(WorkspaceIsolationError, match="git worktree add failed"):
        coordinator._prepare_attempt_for_run(
            node,
            attempt,
            config,
            created_at="2026-01-01T00:02:00+00:00",
            max_concurrent=1,
        )


def test_workspace_isolation_off_keeps_shared_workspace_on_provision_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline import hypothesis_branch

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    node = coordinator.store.create_node(
        statement="Treatment improves accuracy.",
        created_at="2026-01-01T00:00:00+00:00",
    )
    attempt = coordinator.store.add_attempt(
        node_id=node.id,
        branch_run_dir=str(tmp_path / "hypothesis_branches" / node.id / "attempt-001"),
        created_at="2026-01-01T00:01:00+00:00",
    )

    def fail_provision_workspace(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise AssertionError("provision_workspace should not be called")

    monkeypatch.setattr(
        hypothesis_branch,
        "provision_workspace",
        fail_provision_workspace,
    )
    config = SimpleNamespace(
        hypothesis_validation=SimpleNamespace(workspace_isolation="shared"),
        experiment=SimpleNamespace(
            workspace_agent=SimpleNamespace(workspace_path=str(tmp_path / "source"))
        ),
    )

    prepared = coordinator._prepare_attempt_for_run(
        node,
        attempt,
        config,
        created_at="2026-01-01T00:02:00+00:00",
        max_concurrent=1,
    )

    assert prepared.status == "running"
    assert prepared.workspace_path is None
    assert prepared.agent_session_name == f"{tmp_path.name}-{node.id}-attempt-001"


def test_workspace_provisioning_is_limited_to_active_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from researchclaw.pipeline import hypothesis_branch

    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    coordinator.split_and_queue(
        """
## H1
Statement: First hypothesis.

## H2
Statement: Second hypothesis.

## H3
Statement: Third hypothesis.

## H4
Statement: Fourth hypothesis.
""",
        created_at="2026-01-01T00:00:00+00:00",
    )
    lock = threading.Lock()
    active_workspaces = 0
    max_active_workspaces = 0

    def fake_provision_workspace(
        attempt: Any,
        *,
        source_workspace: Path,
        workspace_root: Path | None = None,
    ) -> Any:
        nonlocal active_workspaces, max_active_workspaces
        _ = source_workspace
        workspace = Path(workspace_root or tmp_path / ".worktrees") / (
            attempt.attempt_id.replace("/", "-")
        )
        workspace.mkdir(parents=True, exist_ok=True)
        with lock:
            active_workspaces += 1
            max_active_workspaces = max(max_active_workspaces, active_workspaces)
        return replace(attempt, workspace_path=str(workspace))

    def fake_release_workspace(
        attempt: Any,
        *,
        source_workspace: Path | None = None,
    ) -> None:
        nonlocal active_workspaces
        _ = attempt, source_workspace
        with lock:
            active_workspaces -= 1

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        _ = node, attempt, config, adapters
        time.sleep(0.03)
        return SimpleNamespace(
            decision="proceed",
            artifacts=("stage-15/verdict.json",),
            metrics={"score": 0.9},
        )

    monkeypatch.setattr(
        hypothesis_branch,
        "provision_workspace",
        fake_provision_workspace,
    )
    monkeypatch.setattr(
        hypothesis_branch,
        "release_workspace",
        fake_release_workspace,
    )
    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)
    config = SimpleNamespace(
        hypothesis_validation=SimpleNamespace(
            workspace_isolation="shared",
            max_tree_depth=3,
            max_total_attempts=10,
        ),
        experiment=SimpleNamespace(
            workspace_agent=SimpleNamespace(workspace_path=str(tmp_path / "source"))
        ),
    )

    completed = coordinator.run_pending_work_concurrent(
        config=config,
        adapters=object(),
        max_concurrent=2,
        created_at="2026-01-01T00:01:00+00:00",
    )

    assert [attempt.status for attempt in completed] == ["succeeded"] * 4
    assert max_active_workspaces == 2
    assert active_workspaces == 0


def test_coordinator_gate_written_when_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    HypothesisValidationCoordinator = _coordinator_cls()
    coordinator = HypothesisValidationCoordinator(tmp_path)
    coordinator.split_and_queue(
        """
## H1
Statement: Gate hypothesis.
Prediction: Gate prediction.
Falsification: Gate falsification.
""",
        created_at="2026-01-01T00:00:00+00:00",
    )

    def fake_validate_branch(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        return SimpleNamespace(
            decision="proceed",
            artifacts=("stage-15/verdict.json",),
            metrics={"score": 0.9},
        )

    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    coordinator.run_until_queue_empty(
        config=object(),
        adapters=object(),
        max_concurrent=1,
        created_at="2026-01-01T00:01:00+00:00",
        require_coordinator_gate=True,
    )

    gate = json.loads((tmp_path / "coordinator_gate.json").read_text())
    assert gate["status"] == "blocked_approval"
    assert gate["reason"] == "coordinator_gate_required"
