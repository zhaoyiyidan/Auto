"""Deprecated metric-resolution shim for legacy callers."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


SAFE_DEFAULT_METRIC = ("primary_metric", "maximize")


def _coerce_direction(value: Any) -> str | None:
    direction = str(value or "").strip().lower()
    if direction in {"maximize", "max", "higher", "higher_is_better"}:
        return "maximize"
    if direction in {"minimize", "min", "lower", "lower_is_better"}:
        return "minimize"
    return None


def _read_json_metric(path: Path) -> tuple[str, str] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    candidates = [
        payload,
        payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
        payload.get("evaluation") if isinstance(payload.get("evaluation"), dict) else {},
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        name = (
            candidate.get("primary_metric")
            or candidate.get("metric")
            or candidate.get("name")
            or SAFE_DEFAULT_METRIC[0]
        )
        direction = _coerce_direction(
            candidate.get("metric_direction")
            or candidate.get("direction")
        )
        if direction:
            return str(name or SAFE_DEFAULT_METRIC[0]), direction
        primary = candidate.get("primary")
        if isinstance(primary, dict):
            direction = _coerce_direction(primary.get("direction"))
            if direction:
                return str(primary.get("name") or SAFE_DEFAULT_METRIC[0]), direction
    return None


def resolve_experiment_metric(run_dir: Path) -> tuple[str, str]:
    """Resolve the primary metric name and direction from Stage 9 artifacts."""
    run_dir = Path(run_dir)
    for candidate in (
        run_dir / "stage-09" / "experiment_spec.json",
        run_dir / "stage-09" / "expected_outputs.json",
        run_dir / "run_manifest.json",
    ):
        resolved = _read_json_metric(candidate)
        if resolved is not None:
            return resolved
    plan_path = run_dir / "stage-09" / "plan.md"
    if plan_path.exists():
        try:
            text = plan_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        direction_match = re.search(
            r"\bdirection\s*[:=]\s*(maximize|minimize|max|min|higher|lower)\b",
            text,
            re.IGNORECASE,
        )
        if direction_match:
            direction = _coerce_direction(direction_match.group(1))
            if direction:
                metric_match = re.search(
                    r"\bprimary\s+metric\s*[:=]\s*([A-Za-z0-9_.-]+)",
                    text,
                    re.IGNORECASE,
                )
                metric = metric_match.group(1) if metric_match else SAFE_DEFAULT_METRIC[0]
                return metric, direction
    return SAFE_DEFAULT_METRIC
