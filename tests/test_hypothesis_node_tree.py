from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researchclaw.pipeline import hypothesis_node_tree as nt
from researchclaw.pipeline import hypothesis_tree as ht


def _tree(run_dir: Path) -> dict[str, Any]:
    return json.loads(
        (run_dir / "hypothesis_tree" / "tree.json").read_text(encoding="utf-8")
    )


def _node_json(run_dir: Path, node_id: str) -> dict[str, Any]:
    return json.loads(
        (run_dir / "hypothesis_tree" / "nodes" / node_id / "node.json").read_text(
            encoding="utf-8"
        )
    )


def _node_tree_json(run_dir: Path, *ids: str) -> dict[str, Any]:
    return json.loads(
        (
            run_dir
            / "hypothesis_tree"
            / "node_tree"
            / Path(*ids)
            / "_node.json"
        ).read_text(encoding="utf-8")
    )


def _hypothesis_text(number: int = 1) -> str:
    return f"# Hypotheses\nH{number}: hypothesis {number} explores mechanism {number}."


def _create_extend_chain(run_dir: Path) -> None:
    ht.finalize_after_stage8(run_dir, _hypothesis_text(1))
    ht.record_stage15_decision(
        run_dir, "extend", "## Decision\nEXTEND", human_edited=False
    )
    ht.finalize_after_stage8(run_dir, _hypothesis_text(2))


def _create_pivot_chain(run_dir: Path) -> None:
    ht.finalize_after_stage8(run_dir, _hypothesis_text(1))
    ht.record_stage15_decision(
        run_dir, "pivot", "## Decision\nPIVOT", human_edited=False
    )
    ht.finalize_after_stage8(run_dir, _hypothesis_text(2))


def test_node_path_ids_linear_chain(tmp_path: Path) -> None:
    _create_extend_chain(tmp_path)

    assert nt.node_path_ids(tmp_path, "h-2") == ["root", "h-1", "h-2"]


def test_node_path_ids_unknown_returns_empty(tmp_path: Path) -> None:
    _create_extend_chain(tmp_path)

    assert nt.node_path_ids(tmp_path, "h-99") == []


def test_node_tree_path_for_is_nested(tmp_path: Path) -> None:
    _create_extend_chain(tmp_path)

    assert str(nt.node_tree_path_for(tmp_path, "h-2")).endswith(
        "node_tree/root/h-1/h-2"
    )


def test_rebuild_creates_dirs_and_meta(tmp_path: Path) -> None:
    _create_extend_chain(tmp_path)

    nt.rebuild_node_tree(tmp_path)

    assert (
        tmp_path
        / "hypothesis_tree"
        / "node_tree"
        / "root"
        / "h-1"
        / "_node.json"
    ).is_file()
    assert (
        tmp_path
        / "hypothesis_tree"
        / "node_tree"
        / "root"
        / "h-1"
        / "_hypothesis.md"
    ).is_file()
    assert (
        tmp_path
        / "hypothesis_tree"
        / "node_tree"
        / "root"
        / "h-1"
        / "h-2"
        / "_node.json"
    ).is_file()
    assert (
        tmp_path
        / "hypothesis_tree"
        / "node_tree"
        / "root"
        / "h-1"
        / "h-2"
        / "_hypothesis.md"
    ).is_file()
    assert _node_tree_json(tmp_path, "root", "h-1") == _node_json(tmp_path, "h-1")


def test_rebuild_noop_when_no_tree(tmp_path: Path) -> None:
    nt.rebuild_node_tree(tmp_path)

    assert not (tmp_path / "hypothesis_tree" / "node_tree").exists()


def test_rebuild_preserves_artifacts_dir(tmp_path: Path) -> None:
    _create_extend_chain(tmp_path)
    artifact = (
        tmp_path
        / "hypothesis_tree"
        / "node_tree"
        / "root"
        / "h-1"
        / "_artifacts"
        / "cycle-001"
        / "x.txt"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("keep", encoding="utf-8")

    nt.rebuild_node_tree(tmp_path)

    assert artifact.read_text(encoding="utf-8") == "keep"


def test_index_json_derived_fields(tmp_path: Path) -> None:
    _create_extend_chain(tmp_path)

    nt.rebuild_node_tree(tmp_path)

    index = nt.read_index(tmp_path)
    node = index["nodes"]["h-2"]
    assert index["current_node_id"] == "h-2"
    assert node["parent_id"] == "h-1"
    assert node["edge_from_parent"] == "extend"
    assert node["tree_path"] == "root/h-1/h-2"
    assert node["path"] == ["root", "h-1", "h-2"]
    assert node["artifacts"] == []


def test_index_pivoted_from(tmp_path: Path) -> None:
    _create_pivot_chain(tmp_path)

    nt.rebuild_node_tree(tmp_path)

    assert nt.read_index(tmp_path)["nodes"]["h-2"]["pivoted_from"] == "h-1"


def test_rebuild_idempotent(tmp_path: Path) -> None:
    _create_extend_chain(tmp_path)
    nt.rebuild_node_tree(tmp_path)
    node_path = (
        tmp_path
        / "hypothesis_tree"
        / "node_tree"
        / "root"
        / "h-1"
        / "_node.json"
    )
    before = node_path.read_bytes()

    nt.rebuild_node_tree(tmp_path)

    assert node_path.read_bytes() == before
    assert sorted(
        path.name
        for path in (tmp_path / "hypothesis_tree" / "node_tree" / "root").iterdir()
    ) == ["h-1"]


def test_old_flat_tree_still_readable(tmp_path: Path) -> None:
    _create_extend_chain(tmp_path)
    tree_before = (tmp_path / "hypothesis_tree" / "tree.json").read_bytes()
    node_before = (
        tmp_path / "hypothesis_tree" / "nodes" / "h-1" / "node.json"
    ).read_bytes()

    nt.rebuild_node_tree(tmp_path)

    assert (tmp_path / "hypothesis_tree" / "tree.json").read_bytes() == tree_before
    assert (
        tmp_path / "hypothesis_tree" / "nodes" / "h-1" / "node.json"
    ).read_bytes() == node_before
    assert _tree(tmp_path)["nodes"]["h-1"]["children"] == ["h-2"]
