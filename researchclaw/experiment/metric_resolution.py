"""Deprecated metric-resolution shim for legacy callers."""

from __future__ import annotations

from pathlib import Path


SAFE_DEFAULT_METRIC = ("primary_metric", "maximize")


def resolve_experiment_metric(run_dir: Path) -> tuple[str, str]:
    """Return a compatibility default for non-refactored legacy callers."""
    _ = Path(run_dir)
    return SAFE_DEFAULT_METRIC
