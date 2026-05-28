"""Append-only hypothesis lineage tree sidecar artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


TREE_VERSION = 1
TREE_DIRNAME = "hypothesis_tree"
ROOT_NODE_ID = "root"

_VALID_STATUSES = {"active", "inactive", "blocked", "completed"}
_VALID_TRANSITIONS = {"extend", "pivot", "refine"}
_DECISIONS = {"proceed", "extend", "pivot", "refine"}
_NODE_ID_RE = re.compile(r"^h-(\d+)$")


@dataclass(frozen=True)
class TreeNodeData:
    id: str
    label: str
    parent_id: str | None
    status: str
    created_at: str
    activated_at: str | None
    pivoted_from: str | None
    hypothesis_snippet: str
    attempt_number: int


@dataclass(frozen=True)
class TreeEvent:
    event_type: str
    node_id: str | None
    data: dict[str, Any]
    timestamp: str


@dataclass(frozen=True)
class PendingTransition:
    transition_type: str
    source_node_id: str
    created_at: str
    decision_text_excerpt: str
    human_edited: bool


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tree_dir(run_dir: Path) -> Path:
    return Path(run_dir) / TREE_DIRNAME


def _nodes_dir(run_dir: Path) -> Path:
    return _tree_dir(run_dir) / "nodes"


def _tree_path(run_dir: Path) -> Path:
    return _tree_dir(run_dir) / "tree.json"


def _events_path(run_dir: Path) -> Path:
    return _tree_dir(run_dir) / "events.jsonl"


def _current_path(run_dir: Path) -> Path:
    return _tree_dir(run_dir) / "current_node.txt"


def _pending_path(run_dir: Path) -> Path:
    return _tree_dir(run_dir) / "pending_transition.json"


def _node_dir(run_dir: Path, node_id: str) -> Path:
    return _nodes_dir(run_dir) / node_id


def _node_json_path(run_dir: Path, node_id: str) -> Path:
    return _node_dir(run_dir, node_id) / "node.json"


def _node_hypothesis_path(run_dir: Path, node_id: str) -> Path:
    return _node_dir(run_dir, node_id) / "hypothesis.md"


def _ensure_dirs(run_dir: Path) -> None:
    _nodes_dir(run_dir).mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        Path(tmp_path).replace(path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
    )


def _snippet(text: str, limit: int = 200) -> str:
    return text.strip().replace("\x00", "")[:limit]


def _decision_excerpt(text: str, limit: int = 500) -> str:
    return text.strip().replace("\x00", "")[:limit]


def _node_to_dict(node: TreeNodeData) -> dict[str, Any]:
    return {
        "id": node.id,
        "label": node.label,
        "parent_id": node.parent_id,
        "status": node.status,
        "created_at": node.created_at,
        "activated_at": node.activated_at,
        "pivoted_from": node.pivoted_from,
        "hypothesis_snippet": node.hypothesis_snippet,
        "attempt_number": node.attempt_number,
    }


def _node_from_dict(data: dict[str, Any]) -> TreeNodeData:
    return TreeNodeData(
        id=str(data["id"]),
        label=str(data.get("label") or data["id"]),
        parent_id=data.get("parent_id"),
        status=str(data.get("status") or "active"),
        created_at=str(data.get("created_at") or _utcnow_iso()),
        activated_at=data.get("activated_at"),
        pivoted_from=data.get("pivoted_from"),
        hypothesis_snippet=str(data.get("hypothesis_snippet") or ""),
        attempt_number=int(data.get("attempt_number") or 0),
    )


def _tree_node_record(node: TreeNodeData, children: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": node.id,
        "label": node.label,
        "parent_id": node.parent_id,
        "status": node.status,
        "children": list(children or []),
    }


def _pending_to_dict(pending: PendingTransition) -> dict[str, Any]:
    return {
        "transition_type": pending.transition_type,
        "source_node_id": pending.source_node_id,
        "created_at": pending.created_at,
        "decision_text_excerpt": pending.decision_text_excerpt,
        "human_edited": pending.human_edited,
    }


def _pending_from_dict(data: dict[str, Any]) -> PendingTransition:
    return PendingTransition(
        transition_type=str(data["transition_type"]).lower(),
        source_node_id=str(data["source_node_id"]),
        created_at=str(data.get("created_at") or _utcnow_iso()),
        decision_text_excerpt=str(data.get("decision_text_excerpt") or ""),
        human_edited=bool(data.get("human_edited")),
    )


def _event_to_dict(event: TreeEvent) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "node_id": event.node_id,
        "data": event.data,
        "timestamp": event.timestamp or _utcnow_iso(),
    }


def _empty_tree() -> dict[str, Any]:
    return {
        "version": TREE_VERSION,
        "generated": _utcnow_iso(),
        "nodes": {},
        "edges": [],
    }


def read_tree(run_dir: Path) -> dict[str, Any]:
    """Read tree.json, returning an empty skeleton when it is absent."""
    path = _tree_path(run_dir)
    if not path.exists():
        return _empty_tree()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return _empty_tree()
    data.setdefault("version", TREE_VERSION)
    data.setdefault("generated", _utcnow_iso())
    data.setdefault("nodes", {})
    data.setdefault("edges", [])
    return data


def write_tree(run_dir: Path, tree_data: dict[str, Any]) -> None:
    """Atomically write tree.json."""
    tree_data = dict(tree_data)
    tree_data["version"] = TREE_VERSION
    tree_data["generated"] = _utcnow_iso()
    tree_data.setdefault("nodes", {})
    tree_data.setdefault("edges", [])
    _atomic_write_json(_tree_path(run_dir), tree_data)


def append_event(run_dir: Path, event: TreeEvent) -> None:
    """Append one JSON object to events.jsonl without truncating history."""
    _tree_dir(run_dir).mkdir(parents=True, exist_ok=True)
    with _events_path(run_dir).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_event_to_dict(event), ensure_ascii=False, sort_keys=True))
        fh.write("\n")


def _write_node(run_dir: Path, node: TreeNodeData) -> None:
    _atomic_write_json(_node_json_path(run_dir, node.id), _node_to_dict(node))


def get_node(run_dir: Path, node_id: str) -> TreeNodeData | None:
    path = _node_json_path(run_dir, node_id)
    if not path.exists():
        return None
    return _node_from_dict(json.loads(path.read_text(encoding="utf-8")))


def get_current_node_id(run_dir: Path) -> str | None:
    path = _current_path(run_dir)
    if not path.exists():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def set_current_node(run_dir: Path, node_id: str) -> None:
    _atomic_write_text(_current_path(run_dir), f"{node_id}\n")


def init_tree_if_needed(run_dir: Path) -> TreeNodeData:
    """Create the virtual root and sidecar files if they do not exist."""
    _ensure_dirs(run_dir)
    existing_root = get_node(run_dir, ROOT_NODE_ID)
    if existing_root is not None:
        tree = read_tree(run_dir)
        if ROOT_NODE_ID not in tree["nodes"]:
            tree["nodes"][ROOT_NODE_ID] = _tree_node_record(existing_root)
            write_tree(run_dir, tree)
        if get_current_node_id(run_dir) is None:
            set_current_node(run_dir, ROOT_NODE_ID)
        return existing_root

    now = _utcnow_iso()
    root = TreeNodeData(
        id=ROOT_NODE_ID,
        label=ROOT_NODE_ID,
        parent_id=None,
        status="active",
        created_at=now,
        activated_at=now,
        pivoted_from=None,
        hypothesis_snippet="Virtual hypothesis root",
        attempt_number=0,
    )
    _write_node(run_dir, root)
    tree = _empty_tree()
    tree["nodes"][ROOT_NODE_ID] = _tree_node_record(root)
    write_tree(run_dir, tree)
    set_current_node(run_dir, ROOT_NODE_ID)
    append_event(
        run_dir,
        TreeEvent(
            event_type="tree_initialized",
            node_id=ROOT_NODE_ID,
            data={},
            timestamp=now,
        ),
    )
    return root


def _node_sort_key(path: Path) -> tuple[int, str]:
    match = _NODE_ID_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def _next_node_number(run_dir: Path) -> int:
    max_number = 0
    for child in _nodes_dir(run_dir).iterdir() if _nodes_dir(run_dir).exists() else []:
        match = _NODE_ID_RE.match(child.name)
        if match:
            max_number = max(max_number, int(match.group(1)))
    return max_number + 1


def _validate_status(status: str) -> None:
    if status not in _VALID_STATUSES:
        raise ValueError(f"Invalid hypothesis node status: {status}")


def create_node(
    run_dir: Path,
    parent_id: str,
    hypothesis_md: str,
    status: str,
    pivoted_from: str | None,
    edge_type: str,
) -> TreeNodeData:
    """Create a hypothesis node and update all sidecar artifacts."""
    _validate_status(status)
    init_tree_if_needed(run_dir)
    parent = get_node(run_dir, parent_id)
    if parent is None:
        raise ValueError(f"Unknown parent node: {parent_id}")
    if parent.status == "blocked":
        raise ValueError(f"Cannot create child under blocked parent: {parent_id}")

    node_number = _next_node_number(run_dir)
    node_id = f"h-{node_number}"
    now = _utcnow_iso()
    node = TreeNodeData(
        id=node_id,
        label=node_id,
        parent_id=parent_id,
        status=status,
        created_at=now,
        activated_at=now if status == "active" else None,
        pivoted_from=pivoted_from,
        hypothesis_snippet=_snippet(hypothesis_md),
        attempt_number=node_number,
    )
    _node_dir(run_dir, node_id).mkdir(parents=True, exist_ok=False)
    _atomic_write_text(_node_hypothesis_path(run_dir, node_id), hypothesis_md)
    _write_node(run_dir, node)

    tree = read_tree(run_dir)
    tree["nodes"].setdefault(parent_id, _tree_node_record(parent))
    parent_children = tree["nodes"][parent_id].setdefault("children", [])
    if node_id not in parent_children:
        parent_children.append(node_id)
    tree["nodes"][node_id] = _tree_node_record(node)
    tree["edges"].append(
        {
            "source_id": parent_id,
            "target_id": node_id,
            "edge_type": edge_type,
            "created_at": now,
        }
    )
    write_tree(run_dir, tree)
    append_event(
        run_dir,
        TreeEvent(
            event_type="node_created",
            node_id=node_id,
            data={
                "parent_id": parent_id,
                "edge_type": edge_type,
                "pivoted_from": pivoted_from,
            },
            timestamp=now,
        ),
    )
    return node


def update_node_status(run_dir: Path, node_id: str, new_status: str) -> None:
    """Update a node status in node.json and tree.json, preserving history."""
    _validate_status(new_status)
    node = get_node(run_dir, node_id)
    if node is None:
        raise ValueError(f"Unknown hypothesis node: {node_id}")
    old_status = node.status
    if old_status == new_status:
        tree = read_tree(run_dir)
        if node_id in tree["nodes"]:
            tree["nodes"][node_id]["status"] = new_status
            write_tree(run_dir, tree)
        return

    updated = TreeNodeData(
        id=node.id,
        label=node.label,
        parent_id=node.parent_id,
        status=new_status,
        created_at=node.created_at,
        activated_at=node.activated_at or (_utcnow_iso() if new_status == "active" else None),
        pivoted_from=node.pivoted_from,
        hypothesis_snippet=node.hypothesis_snippet,
        attempt_number=node.attempt_number,
    )
    _write_node(run_dir, updated)
    tree = read_tree(run_dir)
    if node_id in tree["nodes"]:
        tree["nodes"][node_id]["status"] = new_status
        write_tree(run_dir, tree)
    append_event(
        run_dir,
        TreeEvent(
            event_type="node_status_changed",
            node_id=node_id,
            data={"from": old_status, "to": new_status},
            timestamp=_utcnow_iso(),
        ),
    )


def write_pending_transition(run_dir: Path, pending: PendingTransition) -> None:
    if pending.transition_type not in _VALID_TRANSITIONS:
        raise ValueError(f"Invalid pending transition: {pending.transition_type}")
    init_tree_if_needed(run_dir)
    _atomic_write_json(_pending_path(run_dir), _pending_to_dict(pending))


def read_pending_transition(run_dir: Path) -> PendingTransition | None:
    path = _pending_path(run_dir)
    if not path.exists():
        return None
    return _pending_from_dict(json.loads(path.read_text(encoding="utf-8")))


def clear_pending_transition(run_dir: Path) -> None:
    try:
        _pending_path(run_dir).unlink()
    except FileNotFoundError:
        return


def _normalize_decision(decision: str) -> str:
    normalized = str(decision or "").strip().lower()
    if normalized not in _DECISIONS:
        return "proceed"
    return normalized


def _read_stage8_hypotheses(run_dir: Path) -> str:
    current = Path(run_dir) / "stage-08" / "hypotheses.md"
    if current.exists():
        try:
            return current.read_text(encoding="utf-8")
        except OSError:
            return ""
    candidates = sorted(Path(run_dir).glob("stage-08_v*/hypotheses.md"))
    for candidate in reversed(candidates):
        try:
            return candidate.read_text(encoding="utf-8")
        except OSError:
            continue
    return ""


def _find_node_by_hypothesis(run_dir: Path, hypothesis_md: str) -> str | None:
    target = hypothesis_md.strip()
    if not target:
        return None
    nodes_dir = _nodes_dir(run_dir)
    if not nodes_dir.exists():
        return None
    for node_path in sorted(nodes_dir.iterdir(), key=_node_sort_key):
        if node_path.name == ROOT_NODE_ID:
            continue
        hyp_path = node_path / "hypothesis.md"
        if not hyp_path.exists():
            continue
        try:
            if hyp_path.read_text(encoding="utf-8").strip() == target:
                return node_path.name
        except OSError:
            continue
    return None


def _ensure_current_hypothesis_node(run_dir: Path) -> str:
    init_tree_if_needed(run_dir)
    current = get_current_node_id(run_dir)
    if current and current != ROOT_NODE_ID and get_node(run_dir, current) is not None:
        return current

    hypotheses_md = _read_stage8_hypotheses(run_dir)
    if hypotheses_md.strip():
        existing = _find_node_by_hypothesis(run_dir, hypotheses_md)
        if existing:
            set_current_node(run_dir, existing)
            return existing
        node = create_node(
            run_dir,
            ROOT_NODE_ID,
            hypotheses_md,
            status="active",
            pivoted_from=None,
            edge_type="initialize",
        )
        set_current_node(run_dir, node.id)
        return node.id

    set_current_node(run_dir, ROOT_NODE_ID)
    return ROOT_NODE_ID


def _append_transition_pending(
    run_dir: Path,
    pending: PendingTransition,
) -> None:
    append_event(
        run_dir,
        TreeEvent(
            event_type="transition_pending",
            node_id=pending.source_node_id,
            data={
                "transition_type": pending.transition_type,
                "decision_text_excerpt": pending.decision_text_excerpt,
                "human_edited": pending.human_edited,
            },
            timestamp=pending.created_at,
        ),
    )


def _append_transition_finalized(
    run_dir: Path,
    pending: PendingTransition,
    *,
    node_id: str | None,
    duplicate: bool = False,
) -> None:
    append_event(
        run_dir,
        TreeEvent(
            event_type="transition_finalized",
            node_id=node_id,
            data={
                "transition_type": pending.transition_type,
                "source_node_id": pending.source_node_id,
                "human_edited": pending.human_edited,
                "duplicate": duplicate,
            },
            timestamp=_utcnow_iso(),
        ),
    )


def record_stage15_decision(
    run_dir: Path,
    decision: str,
    decision_md: str,
    *,
    human_edited: bool,
) -> None:
    """Record the authoritative Stage 15 decision for tree routing."""
    decision = _normalize_decision(decision)
    source_node_id = _ensure_current_hypothesis_node(run_dir)
    source = get_node(run_dir, source_node_id)
    if source is None:
        raise ValueError(f"Unknown current hypothesis node: {source_node_id}")
    if source.status == "blocked" and decision in _VALID_TRANSITIONS:
        raise ValueError(f"Cannot record transition from blocked node: {source_node_id}")

    if decision == "proceed":
        clear_pending_transition(run_dir)
        update_node_status(run_dir, source_node_id, "completed")
        append_event(
            run_dir,
            TreeEvent(
                event_type="decision_recorded",
                node_id=source_node_id,
                data={
                    "decision": decision,
                    "forced": False,
                    "human_edited": human_edited,
                    "decision_text_excerpt": _decision_excerpt(decision_md),
                },
                timestamp=_utcnow_iso(),
            ),
        )
        return

    pending = PendingTransition(
        transition_type=decision,
        source_node_id=source_node_id,
        created_at=_utcnow_iso(),
        decision_text_excerpt=_decision_excerpt(decision_md),
        human_edited=human_edited,
    )
    write_pending_transition(run_dir, pending)
    _append_transition_pending(run_dir, pending)


def record_forced_proceed(run_dir: Path, *, reason: str) -> None:
    """Clear any pending transition and mark the current node completed."""
    source_node_id = _ensure_current_hypothesis_node(run_dir)
    clear_pending_transition(run_dir)
    update_node_status(run_dir, source_node_id, "completed")
    append_event(
        run_dir,
        TreeEvent(
            event_type="decision_recorded",
            node_id=source_node_id,
            data={"decision": "proceed", "forced": True, "reason": reason},
            timestamp=_utcnow_iso(),
        ),
    )


def finalize_after_stage8(run_dir: Path, hypotheses_md: str) -> str | None:
    """Finalize a pending transition after Stage 8 writes hypotheses.md."""
    init_tree_if_needed(run_dir)
    hypotheses_md = hypotheses_md.strip()
    pending = read_pending_transition(run_dir)

    if not hypotheses_md:
        if pending is not None:
            clear_pending_transition(run_dir)
            append_event(
                run_dir,
                TreeEvent(
                    event_type="transition_abandoned",
                    node_id=pending.source_node_id,
                    data={
                        "transition_type": pending.transition_type,
                        "reason": "empty_hypotheses",
                        "human_edited": pending.human_edited,
                    },
                    timestamp=_utcnow_iso(),
                ),
            )
        return None

    duplicate_id = _find_node_by_hypothesis(run_dir, hypotheses_md)
    if pending is None:
        current = get_current_node_id(run_dir)
        if duplicate_id:
            if current in (None, ROOT_NODE_ID):
                set_current_node(run_dir, duplicate_id)
            return None
        hypothesis_nodes = [
            child
            for child in _nodes_dir(run_dir).iterdir()
            if _NODE_ID_RE.match(child.name)
        ]
        if current in (None, ROOT_NODE_ID) or not hypothesis_nodes:
            node = create_node(
                run_dir,
                ROOT_NODE_ID,
                hypotheses_md,
                status="active",
                pivoted_from=None,
                edge_type="initialize",
            )
            set_current_node(run_dir, node.id)
            return node.id
        return None

    if duplicate_id:
        if pending.transition_type == "refine":
            clear_pending_transition(run_dir)
            _append_transition_finalized(run_dir, pending, node_id=pending.source_node_id)
            return None
        if duplicate_id != pending.source_node_id:
            set_current_node(run_dir, duplicate_id)
            clear_pending_transition(run_dir)
            _append_transition_finalized(
                run_dir, pending, node_id=duplicate_id, duplicate=True
            )
            return None
        clear_pending_transition(run_dir)
        append_event(
            run_dir,
            TreeEvent(
                event_type="transition_abandoned",
                node_id=pending.source_node_id,
                data={
                    "transition_type": pending.transition_type,
                    "reason": "duplicate_source_hypothesis",
                    "human_edited": pending.human_edited,
                },
                timestamp=_utcnow_iso(),
            ),
        )
        return None

    source = get_node(run_dir, pending.source_node_id)
    if source is None:
        raise ValueError(f"Unknown pending source node: {pending.source_node_id}")
    if source.status == "blocked":
        raise ValueError(f"Cannot finalize transition from blocked node: {source.id}")

    if pending.transition_type == "refine":
        clear_pending_transition(run_dir)
        _append_transition_finalized(run_dir, pending, node_id=source.id)
        return None

    if pending.transition_type == "extend":
        if source.id != ROOT_NODE_ID:
            update_node_status(run_dir, source.id, "inactive")
        parent_id = source.id if source.id != ROOT_NODE_ID else ROOT_NODE_ID
        node = create_node(
            run_dir,
            parent_id,
            hypotheses_md,
            status="active",
            pivoted_from=None,
            edge_type="extend",
        )
    elif pending.transition_type == "pivot":
        parent_id = source.parent_id or ROOT_NODE_ID
        parent = get_node(run_dir, parent_id)
        if parent is None:
            raise ValueError(f"Unknown pivot parent node: {parent_id}")
        if parent.status == "blocked":
            raise ValueError(f"Cannot finalize pivot with blocked parent: {parent_id}")
        if source.id != ROOT_NODE_ID:
            update_node_status(run_dir, source.id, "blocked")
        node = create_node(
            run_dir,
            parent_id,
            hypotheses_md,
            status="active",
            pivoted_from=source.id,
            edge_type="pivot",
        )
    else:
        raise ValueError(f"Invalid pending transition: {pending.transition_type}")

    set_current_node(run_dir, node.id)
    clear_pending_transition(run_dir)
    _append_transition_finalized(run_dir, pending, node_id=node.id)
    return node.id
