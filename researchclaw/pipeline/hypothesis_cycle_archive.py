"""Archive Stage 8-15 experiment cycles under hypothesis tree nodes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from researchclaw.pipeline.hypothesis_node_tree import (
    ARTIFACTS_DIRNAME,
    node_tree_dir,
    node_tree_path_for,
    rebuild_node_tree,
)
from researchclaw.pipeline.hypothesis_tree import (
    ROOT_NODE_ID,
    _atomic_write_json,
    _utcnow_iso,
    get_current_node_id,
)


ARCHIVE_STAGE_RANGE = range(8, 16)
CYCLE_PREFIX = "cycle-"
MANIFEST_FILENAME = "manifest.json"
IGNORE_PATTERNS = (
    "*.ckpt",
    "*.pt",
    "*.pth",
    "*.bin",
    "__pycache__",
    ".venv",
    "node_modules",
    "*.tmp",
)


def archive_current_hypothesis_cycle(
    run_dir: Path,
    *,
    decision: str | None = None,
) -> Path | None:
    run_dir = Path(run_dir)
    node_id = get_current_node_id(run_dir)
    if node_id is None or node_id == ROOT_NODE_ID:
        return None

    rebuild_node_tree(run_dir)
    try:
        node_tree_path = node_tree_path_for(run_dir, node_id)
    except ValueError:
        return None

    cycle_root = node_tree_path / ARTIFACTS_DIRNAME
    cycle_root.mkdir(parents=True, exist_ok=True)
    pivot_count = _read_pivot_count(run_dir)
    signature = _cycle_signature(run_dir, decision, pivot_count)

    newest = _newest_matching_cycle(cycle_root, node_id, pivot_count, signature)
    if newest is not None:
        return newest

    cycle_id = _next_cycle_id(cycle_root)
    partial = cycle_root / f"{cycle_id}.partial"
    final = cycle_root / cycle_id
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    included_stages: list[str] = []
    missing_stages: list[str] = []
    source_stage_dirs: dict[str, str] = {}
    for stage_num in ARCHIVE_STAGE_RANGE:
        stage_name = f"stage-{stage_num:02d}"
        source = run_dir / stage_name
        if not source.is_dir():
            missing_stages.append(stage_name)
            continue
        try:
            shutil.copytree(
                source,
                partial / stage_name,
                ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
                symlinks=False,
            )
        except Exception:
            shutil.rmtree(partial / stage_name, ignore_errors=True)
            missing_stages.append(stage_name)
            continue
        included_stages.append(stage_name)
        source_stage_dirs[stage_name] = str(source.resolve())

    manifest = {
        "version": 1,
        "node_id": node_id,
        "cycle_id": cycle_id,
        "archived_at": _utcnow_iso(),
        "decision": decision,
        "pivot_count": pivot_count,
        "tree_path": "/".join(node_tree_path.relative_to(node_tree_dir(run_dir)).parts),
        "run_dir": str(run_dir.resolve()),
        "included_stages": included_stages,
        "missing_stages": missing_stages,
        "source_stage_dirs": source_stage_dirs,
        "signature": signature,
    }
    _atomic_write_json(partial / MANIFEST_FILENAME, manifest)
    os.replace(partial, final)
    rebuild_node_tree(run_dir)
    return final


def _existing_cycles(cycle_root: Path) -> list[Path]:
    if not cycle_root.exists():
        return []
    return sorted(
        child
        for child in cycle_root.iterdir()
        if child.is_dir()
        and child.name.startswith(CYCLE_PREFIX)
        and not child.name.endswith(".partial")
    )


def _next_cycle_id(cycle_root: Path) -> str:
    max_number = 0
    for cycle in _existing_cycles(cycle_root):
        suffix = cycle.name.removeprefix(CYCLE_PREFIX)
        if suffix.isdigit():
            max_number = max(max_number, int(suffix))
    return f"{CYCLE_PREFIX}{max_number + 1:03d}"


def _read_pivot_count(run_dir: Path) -> int:
    history_path = Path(run_dir) / "decision_history.json"
    if not history_path.exists():
        return 0
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    if isinstance(data, list):
        return len(data)
    return 0


def _cycle_signature(
    run_dir: Path,
    decision: str | None,
    pivot_count: int,
) -> dict[str, Any]:
    return {
        "pivot_count": pivot_count,
        "decision": decision,
        "decision_hash": _hash_first_existing(
            Path(run_dir) / "stage-15" / "decision_structured.json",
            Path(run_dir) / "stage-15" / "decision.md",
        ),
        "hypotheses_hash": _hash_path(Path(run_dir) / "stage-08" / "hypotheses.md"),
        "summary_hash": _hash_path(
            Path(run_dir) / "stage-14" / "experiment_summary.json"
        ),
    }


def _newest_matching_cycle(
    cycle_root: Path,
    node_id: str,
    pivot_count: int,
    signature: dict[str, Any],
) -> Path | None:
    cycles = _existing_cycles(cycle_root)
    if not cycles:
        return None
    newest = cycles[-1]
    manifest_path = newest / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if (
        manifest.get("node_id") == node_id
        and manifest.get("pivot_count") == pivot_count
        and manifest.get("signature") == signature
    ):
        return newest
    return None


def _hash_first_existing(*paths: Path) -> str | None:
    for path in paths:
        digest = _hash_path(path)
        if digest is not None:
            return digest
    return None


def _hash_path(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()
