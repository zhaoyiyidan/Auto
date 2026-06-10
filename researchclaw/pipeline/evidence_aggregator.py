"""Aggregate per-hypothesis branch evidence into terminal artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from researchclaw.pipeline.hypothesis_store import (
    TREE_DIRNAME,
    ValidationAttempt,
)


def _metric_value(metrics: dict[str, Any], metric_name: str) -> float | None:
    value = metrics.get(metric_name)
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


class EvidenceAggregator:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.tree_dir = self.run_dir / TREE_DIRNAME

    def _read_attempts(self) -> list[ValidationAttempt]:
        attempts: list[ValidationAttempt] = []
        nodes_dir = self.tree_dir / "nodes"
        if not nodes_dir.exists():
            return attempts
        for path in sorted(nodes_dir.glob("*/attempts/**/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            attempts.append(ValidationAttempt.from_dict(payload))
        return attempts

    def select_best_attempts(
        self,
        *,
        metric_name: str,
        direction: str,
    ) -> dict[str, ValidationAttempt]:
        grouped: dict[str, list[tuple[float, ValidationAttempt]]] = {}
        for attempt in self._read_attempts():
            if attempt.status != "succeeded":
                continue
            value = _metric_value(attempt.metrics, metric_name)
            if value is None:
                continue
            grouped.setdefault(attempt.node_id, []).append((value, attempt))

        winners: dict[str, ValidationAttempt] = {}
        minimize = str(direction or "").lower() == "minimize"
        for node_id in sorted(grouped):
            candidates = sorted(
                grouped[node_id],
                key=lambda item: item[0],
                reverse=not minimize,
            )
            if minimize and len(candidates) > 1:
                best_value = candidates[0][0]
                second_value = candidates[1][0]
                if 0 < best_value < second_value * 1e-3:
                    candidates = candidates[1:]
            if candidates:
                winners[node_id] = candidates[0][1]
        return winners
