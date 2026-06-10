from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _queue_classes() -> tuple[Any, Any]:
    try:
        from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue, WorkItem
    except ImportError:
        pytest.fail("Durable hypothesis work queue is not implemented")
    return DurableWorkQueue, WorkItem


def test_durable_work_queue_appends_reads_and_dedupes_by_attempt_id(
    tmp_path: Path,
) -> None:
    DurableWorkQueue, WorkItem = _queue_classes()
    queue = DurableWorkQueue(tmp_path)
    first = WorkItem(
        node_id="h-001",
        attempt_id="h-001/attempt-001",
        branch_run_dir=str(tmp_path / "branches" / "h-001" / "attempt-001"),
    )
    duplicate = WorkItem(
        node_id="h-001",
        attempt_id="h-001/attempt-001",
        branch_run_dir=str(tmp_path / "branches" / "h-001" / "attempt-001"),
    )
    second = WorkItem(
        node_id="h-002",
        attempt_id="h-002/attempt-001",
        branch_run_dir=str(tmp_path / "branches" / "h-002" / "attempt-001"),
    )

    queue.append(first, created_at="2026-01-01T00:00:00+00:00")
    queue.append(duplicate, created_at="2026-01-01T00:00:01+00:00")
    queue.append(second, created_at="2026-01-01T00:00:02+00:00")

    queue_path = tmp_path / "hypothesis_tree" / "work_queue.jsonl"
    raw_lines = [line for line in queue_path.read_text(encoding="utf-8").splitlines()]
    assert len(raw_lines) == 3
    assert json.loads(raw_lines[0])["item"]["attempt_id"] == "h-001/attempt-001"

    items = queue.read_items()
    assert [item.attempt_id for item in items] == [
        "h-001/attempt-001",
        "h-002/attempt-001",
    ]
    assert [item.node_id for item in items] == ["h-001", "h-002"]
    assert items[0].branch_run_dir == str(
        tmp_path / "branches" / "h-001" / "attempt-001"
    )
