"""Legacy hypothesis compatibility tests.

These tests keep old hypothesis cycle archives readable without keeping the
runner wired to create new archives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researchclaw.pipeline import hypothesis_cycle_archive as ca
from researchclaw.pipeline import hypothesis_node_tree as nt
from researchclaw.pipeline import hypothesis_tree as ht


def _hypothesis_text(number: int = 1) -> str:
    return f"# Hypotheses\nH{number}: hypothesis {number} explores mechanism {number}."


def _manifest(cycle_dir: Path) -> dict[str, Any]:
    return json.loads((cycle_dir / "manifest.json").read_text(encoding="utf-8"))


def _cycle_root(run_dir: Path, *ids: str) -> Path:
    return run_dir / "hypothesis_tree" / "node_tree" / Path(*ids) / "_artifacts"


def _write_stage_artifacts(run_dir: Path) -> None:
    stage8 = run_dir / "stage-08"
    stage8.mkdir(parents=True, exist_ok=True)
    (stage8 / "hypotheses.md").write_text(_hypothesis_text(1), encoding="utf-8")

    stage14 = run_dir / "stage-14"
    stage14.mkdir(parents=True, exist_ok=True)
    (stage14 / "analysis.md").write_text("# Analysis\nUseful evidence.", encoding="utf-8")
    (stage14 / "experiment_summary.json").write_text(
        json.dumps({"metrics_summary": {"accuracy": {"mean": 0.8}}}),
        encoding="utf-8",
    )

    stage15 = run_dir / "stage-15"
    stage15.mkdir(parents=True, exist_ok=True)
    (stage15 / "decision.md").write_text("## Decision\nEXTEND", encoding="utf-8")
    (stage15 / "decision_structured.json").write_text(
        json.dumps({"decision": "extend"}),
        encoding="utf-8",
    )


def _create_initial_current(run_dir: Path) -> None:
    ht.finalize_after_stage8(run_dir, _hypothesis_text(1))


def _create_pivot_current(run_dir: Path) -> None:
    _create_initial_current(run_dir)
    ht.record_stage15_decision(
        run_dir, "pivot", "## Decision\nPIVOT", human_edited=False
    )
    ht.finalize_after_stage8(run_dir, _hypothesis_text(2))


def test_archive_returns_none_when_current_root(tmp_path: Path) -> None:
    ht.init_tree_if_needed(tmp_path)

    assert ca.archive_current_hypothesis_cycle(tmp_path, decision="extend") is None
    assert not (tmp_path / "hypothesis_tree" / "node_tree" / "root" / "_artifacts").exists()


def test_archive_returns_none_when_no_current(tmp_path: Path) -> None:
    assert ca.archive_current_hypothesis_cycle(tmp_path, decision="extend") is None


def test_archive_extend_under_parent_node(tmp_path: Path) -> None:
    _create_initial_current(tmp_path)
    _write_stage_artifacts(tmp_path)

    cycle = ca.archive_current_hypothesis_cycle(tmp_path, decision="extend")

    assert cycle == _cycle_root(tmp_path, "root", "h-1") / "cycle-001"
    assert cycle.is_dir()
    manifest = _manifest(cycle)
    assert manifest["node_id"] == "h-1"
    assert manifest["decision"] == "extend"


def test_archive_copies_present_stage_dirs(tmp_path: Path) -> None:
    _create_initial_current(tmp_path)
    _write_stage_artifacts(tmp_path)

    cycle = ca.archive_current_hypothesis_cycle(tmp_path, decision="extend")

    manifest = _manifest(cycle)
    assert manifest["included_stages"] == ["stage-08", "stage-14", "stage-15"]
    assert manifest["missing_stages"] == [
        "stage-09",
        "stage-10",
        "stage-11",
        "stage-12",
        "stage-13",
    ]
    assert (cycle / "stage-15" / "decision.md").is_file()


def test_archive_increments_cycle_id(tmp_path: Path) -> None:
    _create_initial_current(tmp_path)
    _write_stage_artifacts(tmp_path)
    first = ca.archive_current_hypothesis_cycle(tmp_path, decision="extend")
    (tmp_path / "decision_history.json").write_text(
        json.dumps([{"decision": "extend"}]), encoding="utf-8"
    )

    second = ca.archive_current_hypothesis_cycle(tmp_path, decision="extend")

    assert first.name == "cycle-001"
    assert second.name == "cycle-002"
    assert (_cycle_root(tmp_path, "root", "h-1") / "cycle-001").is_dir()
    assert (_cycle_root(tmp_path, "root", "h-1") / "cycle-002").is_dir()


def test_archive_idempotent_same_signature(tmp_path: Path) -> None:
    _create_initial_current(tmp_path)
    _write_stage_artifacts(tmp_path)

    first = ca.archive_current_hypothesis_cycle(tmp_path, decision="extend")
    second = ca.archive_current_hypothesis_cycle(tmp_path, decision="extend")

    assert second == first
    assert [path.name for path in ca._existing_cycles(first.parent)] == ["cycle-001"]


def test_archive_does_not_overwrite_existing_cycle(tmp_path: Path) -> None:
    _create_initial_current(tmp_path)
    _write_stage_artifacts(tmp_path)
    cycle_root = _cycle_root(tmp_path, "root", "h-1")
    existing = cycle_root / "cycle-001"
    existing.mkdir(parents=True)
    (existing / "sentinel.txt").write_text("keep", encoding="utf-8")

    cycle = ca.archive_current_hypothesis_cycle(tmp_path, decision="extend")

    assert cycle.name == "cycle-002"
    assert (existing / "sentinel.txt").read_text(encoding="utf-8") == "keep"


def test_archive_pivot_records_pivoted_from_in_index(tmp_path: Path) -> None:
    _create_pivot_current(tmp_path)
    _write_stage_artifacts(tmp_path)

    ca.archive_current_hypothesis_cycle(tmp_path, decision="proceed")

    assert nt.read_index(tmp_path)["nodes"]["h-2"]["pivoted_from"] == "h-1"


def test_manifest_signature_present(tmp_path: Path) -> None:
    _create_initial_current(tmp_path)
    _write_stage_artifacts(tmp_path)

    cycle = ca.archive_current_hypothesis_cycle(tmp_path, decision="extend")

    signature = _manifest(cycle)["signature"]
    assert signature["pivot_count"] == 0
    assert signature["decision"] == "extend"
    assert len(signature["decision_hash"]) == 64
    assert len(signature["hypotheses_hash"]) == 64
    assert len(signature["summary_hash"]) == 64


def test_archive_ignores_large_binaries(tmp_path: Path) -> None:
    _create_initial_current(tmp_path)
    _write_stage_artifacts(tmp_path)
    stage12 = tmp_path / "stage-12"
    stage12.mkdir(parents=True)
    (stage12 / "model.ckpt").write_bytes(b"binary")
    (stage12 / "metrics.json").write_text("{}", encoding="utf-8")

    cycle = ca.archive_current_hypothesis_cycle(tmp_path, decision="extend")

    assert not (cycle / "stage-12" / "model.ckpt").exists()
    assert (cycle / "stage-12" / "metrics.json").is_file()
