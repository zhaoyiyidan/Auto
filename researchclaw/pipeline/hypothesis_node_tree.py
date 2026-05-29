"""Materialized, human-readable hypothesis lineage view."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researchclaw.pipeline.hypothesis_tree import (
    ROOT_NODE_ID,
    TREE_DIRNAME,
    _atomic_write_json,
    _atomic_write_text,
    _node_hypothesis_path,
    _node_json_path,
    _tree_path,
    _utcnow_iso,
    get_current_node_id,
    get_node,
    read_tree,
)


NODE_TREE_DIRNAME = "node_tree"
NODE_TREE_VERSION = 1
ARTIFACTS_DIRNAME = "_artifacts"
NODE_META_FILENAME = "_node.json"
NODE_HYP_FILENAME = "_hypothesis.md"
INDEX_FILENAME = "index.json"


def node_tree_dir(run_dir: Path) -> Path:
    return Path(run_dir) / TREE_DIRNAME / NODE_TREE_DIRNAME


def node_path_ids(run_dir: Path, node_id: str) -> list[str]:
    tree = read_tree(run_dir)
    nodes = tree.get("nodes", {})
    if node_id not in nodes:
        return []

    path: list[str] = []
    seen: set[str] = set()
    current: str | None = node_id
    while current is not None:
        if current in seen or current not in nodes:
            return []
        seen.add(current)
        path.append(current)
        current = nodes[current].get("parent_id")

    path.reverse()
    if path and path[0] == ROOT_NODE_ID:
        return path
    return []


def node_tree_path_for(run_dir: Path, node_id: str) -> Path:
    path_ids = node_path_ids(run_dir, node_id)
    if not path_ids:
        raise ValueError(f"Unknown hypothesis node: {node_id}")
    return node_tree_dir(run_dir) / Path(*path_ids)


def read_index(run_dir: Path) -> dict[str, Any]:
    path = node_tree_dir(run_dir) / INDEX_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def rebuild_node_tree(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    if not _tree_path(run_dir).exists():
        return

    tree = read_tree(run_dir)
    nodes = tree.get("nodes", {})
    edge_types = _edge_types_by_target(tree)
    index_nodes: dict[str, dict[str, Any]] = {}

    for node_id in sorted(nodes, key=_node_sort_key):
        path_ids = node_path_ids(run_dir, node_id)
        if not path_ids:
            continue
        materialized_dir = node_tree_dir(run_dir) / Path(*path_ids)
        materialized_dir.mkdir(parents=True, exist_ok=True)
        _refresh_node_files(run_dir, node_id, materialized_dir)

        node = get_node(run_dir, node_id)
        artifacts_dir = materialized_dir / ARTIFACTS_DIRNAME
        index_nodes[node_id] = {
            "id": node_id,
            "parent_id": nodes[node_id].get("parent_id"),
            "status": nodes[node_id].get("status"),
            "edge_from_parent": edge_types.get(node_id),
            "pivoted_from": node.pivoted_from if node is not None else None,
            "path": path_ids,
            "tree_path": "/".join(path_ids),
            "artifacts": _artifact_ids(artifacts_dir),
        }

    _atomic_write_json(
        node_tree_dir(run_dir) / INDEX_FILENAME,
        {
            "version": NODE_TREE_VERSION,
            "generated": _utcnow_iso(),
            "current_node_id": get_current_node_id(run_dir),
            "nodes": index_nodes,
        },
    )


def _refresh_node_files(run_dir: Path, node_id: str, materialized_dir: Path) -> None:
    node_json = _node_json_path(run_dir, node_id)
    if node_json.exists():
        _atomic_write_text(
            materialized_dir / NODE_META_FILENAME,
            node_json.read_text(encoding="utf-8"),
        )

    hypothesis_md = _node_hypothesis_path(run_dir, node_id)
    if hypothesis_md.exists():
        _atomic_write_text(
            materialized_dir / NODE_HYP_FILENAME,
            hypothesis_md.read_text(encoding="utf-8"),
        )


def _edge_types_by_target(tree: dict[str, Any]) -> dict[str, str]:
    edge_types: dict[str, str] = {}
    for edge in tree.get("edges", []):
        target_id = edge.get("target_id")
        edge_type = edge.get("edge_type")
        if target_id is not None and edge_type is not None:
            edge_types[str(target_id)] = str(edge_type)
    return edge_types


def _artifact_ids(artifacts_dir: Path) -> list[str]:
    if not artifacts_dir.exists():
        return []
    return sorted(
        child.name
        for child in artifacts_dir.iterdir()
        if child.is_dir() and child.name.startswith("cycle-")
    )


def _node_sort_key(node_id: str) -> tuple[int, str]:
    if node_id == ROOT_NODE_ID:
        return (0, node_id)
    prefix, _, suffix = node_id.partition("-")
    if prefix == "h" and suffix.isdigit():
        return (int(suffix), node_id)
    return (10**9, node_id)
