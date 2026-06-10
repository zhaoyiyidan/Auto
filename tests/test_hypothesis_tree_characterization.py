# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from researchclaw.pipeline import hypothesis_tree as ht


FIXED_NOW = "2026-01-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def _fixed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ht, "_utcnow_iso", lambda: FIXED_NOW)


def _events(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "hypothesis_tree" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def _tree(run_dir: Path) -> dict[str, Any]:
    return json.loads(
        (run_dir / "hypothesis_tree" / "tree.json").read_text(encoding="utf-8")
    )


def _node(run_dir: Path, node_id: str) -> dict[str, Any]:
    return json.loads(
        (run_dir / "hypothesis_tree" / "nodes" / node_id / "node.json").read_text(
            encoding="utf-8"
        )
    )


def _hypothesis(number: int) -> str:
    return f"# Hypotheses\nH{number}: hypothesis {number} explores mechanism {number}."


def test_proceed_marks_current_node_completed_and_records_decision(
    tmp_path: Path,
) -> None:
    # characterization
    assert ht.finalize_after_stage8(tmp_path, _hypothesis(1)) == "h-1"

    ht.record_stage15_decision(
        tmp_path,
        "proceed",
        "## Decision\nPROCEED\nEvidence is sufficient.",
        human_edited=False,
    )

    assert _node(tmp_path, "h-1")["status"] == "completed"
    assert ht.get_current_node_id(tmp_path) == "h-1"
    assert ht.read_pending_transition(tmp_path) is None
    assert _tree(tmp_path)["nodes"]["h-1"]["status"] == "completed"
    assert _events(tmp_path) == [
        {
            "event_type": "tree_initialized",
            "node_id": "root",
            "data": {},
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "node_created",
            "node_id": "h-1",
            "data": {
                "edge_type": "initialize",
                "parent_id": "root",
                "pivoted_from": None,
            },
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "node_status_changed",
            "node_id": "h-1",
            "data": {"from": "active", "to": "completed"},
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "decision_recorded",
            "node_id": "h-1",
            "data": {
                "decision": "proceed",
                "decision_text_excerpt": "## Decision\nPROCEED\nEvidence is sufficient.",
                "forced": False,
                "human_edited": False,
            },
            "timestamp": FIXED_NOW,
        },
    ]


def test_extend_creates_child_and_inactivates_source(tmp_path: Path) -> None:
    # characterization
    ht.finalize_after_stage8(tmp_path, _hypothesis(1))
    ht.record_stage15_decision(
        tmp_path,
        "extend",
        "## Decision\nEXTEND\nRefine the active hypothesis.",
        human_edited=True,
    )

    assert ht.finalize_after_stage8(tmp_path, _hypothesis(2)) == "h-2"

    assert _node(tmp_path, "h-1")["status"] == "inactive"
    assert _node(tmp_path, "h-2")["status"] == "active"
    assert _node(tmp_path, "h-2")["parent_id"] == "h-1"
    assert _node(tmp_path, "h-2")["pivoted_from"] is None
    assert ht.get_current_node_id(tmp_path) == "h-2"
    assert ht.read_pending_transition(tmp_path) is None
    assert _tree(tmp_path)["edges"][-1]["edge_type"] == "extend"
    assert _events(tmp_path) == [
        {
            "event_type": "tree_initialized",
            "node_id": "root",
            "data": {},
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "node_created",
            "node_id": "h-1",
            "data": {
                "edge_type": "initialize",
                "parent_id": "root",
                "pivoted_from": None,
            },
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "transition_pending",
            "node_id": "h-1",
            "data": {
                "decision_text_excerpt": "## Decision\nEXTEND\nRefine the active hypothesis.",
                "human_edited": True,
                "transition_type": "extend",
            },
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "node_status_changed",
            "node_id": "h-1",
            "data": {"from": "active", "to": "inactive"},
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "node_created",
            "node_id": "h-2",
            "data": {
                "edge_type": "extend",
                "parent_id": "h-1",
                "pivoted_from": None,
            },
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "transition_finalized",
            "node_id": "h-2",
            "data": {
                "duplicate": False,
                "human_edited": True,
                "source_node_id": "h-1",
                "transition_type": "extend",
            },
            "timestamp": FIXED_NOW,
        },
    ]


def test_pivot_creates_sibling_and_blocks_source(tmp_path: Path) -> None:
    # characterization
    ht.finalize_after_stage8(tmp_path, _hypothesis(1))
    ht.record_stage15_decision(
        tmp_path,
        "pivot",
        "## Decision\nPIVOT\nReplace the active hypothesis.",
        human_edited=False,
    )

    assert ht.finalize_after_stage8(tmp_path, _hypothesis(2)) == "h-2"

    assert _node(tmp_path, "h-1")["status"] == "blocked"
    assert _node(tmp_path, "h-2")["status"] == "active"
    assert _node(tmp_path, "h-2")["parent_id"] == "root"
    assert _node(tmp_path, "h-2")["pivoted_from"] == "h-1"
    assert ht.get_current_node_id(tmp_path) == "h-2"
    assert ht.read_pending_transition(tmp_path) is None
    assert _tree(tmp_path)["nodes"]["root"]["children"] == ["h-1", "h-2"]
    assert _tree(tmp_path)["edges"][-1]["edge_type"] == "pivot"
    assert _events(tmp_path) == [
        {
            "event_type": "tree_initialized",
            "node_id": "root",
            "data": {},
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "node_created",
            "node_id": "h-1",
            "data": {
                "edge_type": "initialize",
                "parent_id": "root",
                "pivoted_from": None,
            },
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "transition_pending",
            "node_id": "h-1",
            "data": {
                "decision_text_excerpt": "## Decision\nPIVOT\nReplace the active hypothesis.",
                "human_edited": False,
                "transition_type": "pivot",
            },
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "node_status_changed",
            "node_id": "h-1",
            "data": {"from": "active", "to": "blocked"},
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "node_created",
            "node_id": "h-2",
            "data": {
                "edge_type": "pivot",
                "parent_id": "root",
                "pivoted_from": "h-1",
            },
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "transition_finalized",
            "node_id": "h-2",
            "data": {
                "duplicate": False,
                "human_edited": False,
                "source_node_id": "h-1",
                "transition_type": "pivot",
            },
            "timestamp": FIXED_NOW,
        },
    ]


def test_forced_proceed_clears_pending_transition_and_records_forced_decision(
    tmp_path: Path,
) -> None:
    # characterization
    ht.finalize_after_stage8(tmp_path, _hypothesis(1))
    ht.record_stage15_decision(
        tmp_path,
        "extend",
        "## Decision\nEXTEND\nWould normally extend.",
        human_edited=False,
    )

    ht.record_forced_proceed(tmp_path, reason="max_pivots")

    assert _node(tmp_path, "h-1")["status"] == "completed"
    assert ht.get_current_node_id(tmp_path) == "h-1"
    assert ht.read_pending_transition(tmp_path) is None
    assert _events(tmp_path) == [
        {
            "event_type": "tree_initialized",
            "node_id": "root",
            "data": {},
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "node_created",
            "node_id": "h-1",
            "data": {
                "edge_type": "initialize",
                "parent_id": "root",
                "pivoted_from": None,
            },
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "transition_pending",
            "node_id": "h-1",
            "data": {
                "decision_text_excerpt": "## Decision\nEXTEND\nWould normally extend.",
                "human_edited": False,
                "transition_type": "extend",
            },
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "node_status_changed",
            "node_id": "h-1",
            "data": {"from": "active", "to": "completed"},
            "timestamp": FIXED_NOW,
        },
        {
            "event_type": "decision_recorded",
            "node_id": "h-1",
            "data": {
                "decision": "proceed",
                "forced": True,
                "reason": "max_pivots",
            },
            "timestamp": FIXED_NOW,
        },
    ]
