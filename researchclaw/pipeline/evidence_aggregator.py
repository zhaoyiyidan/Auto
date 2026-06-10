"""Aggregate per-hypothesis branch evidence into terminal artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from researchclaw.pipeline.hypothesis_store import (
    HypothesisNode,
    TREE_DIRNAME,
    ValidationAttempt,
    _atomic_write_json,
    _atomic_write_text,
    _utcnow_iso,
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

    def _read_nodes(self) -> list[HypothesisNode]:
        nodes: list[HypothesisNode] = []
        nodes_dir = self.tree_dir / "nodes"
        if not nodes_dir.exists():
            return nodes
        for path in sorted(nodes_dir.glob("*/node.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            nodes.append(HypothesisNode.from_dict(payload))
        return nodes

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

    def write_validation_summary(
        self,
        *,
        metric_name: str,
        direction: str,
        generated_at: str | None = None,
    ) -> dict[str, Any]:
        generated_at = generated_at or _utcnow_iso()
        winners = self.select_best_attempts(
            metric_name=metric_name,
            direction=direction,
        )
        nodes = self._read_nodes()
        counts = {
            "total": len(nodes),
            "supported": 0,
            "refuted": 0,
            "inconclusive": 0,
            "superseded": 0,
        }
        rows: list[dict[str, Any]] = []
        for node in nodes:
            if node.status in counts:
                counts[node.status] += 1
            best = winners.get(node.id)
            rows.append(
                {
                    "node_id": node.id,
                    "status": node.status,
                    "statement": node.statement,
                    "best_attempt_id": best.attempt_id if best else None,
                    "decision": best.decision if best else None,
                    "metrics": best.metrics if best else {},
                }
            )
        summary = {
            "generated": generated_at,
            "counts": counts,
            "nodes": rows,
        }
        output_dir = self.run_dir / "hypothesis_aggregate"
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output_dir / "validation_summary.json", summary)
        return summary

    def write_evidence_registry(
        self,
        *,
        metric_name: str,
        direction: str,
    ) -> list[dict[str, Any]]:
        winners = self.select_best_attempts(
            metric_name=metric_name,
            direction=direction,
        )
        rows: list[dict[str, Any]] = []
        for node in self._read_nodes():
            attempt = winners.get(node.id)
            if attempt is None:
                continue
            rows.append(
                {
                    "node_id": node.id,
                    "attempt_id": attempt.attempt_id,
                    "outcome": node.status,
                    "decision": attempt.decision,
                    "metrics": attempt.metrics,
                    "artifacts": attempt.artifacts,
                    "branch_run_dir": attempt.branch_run_dir,
                }
            )
        output_dir = self.run_dir / "hypothesis_aggregate"
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            output_dir / "evidence_registry.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )
        return rows

    def write_paper_context(
        self,
        *,
        metric_name: str,
        direction: str,
        generated_at: str | None = None,
    ) -> str:
        summary = self.write_validation_summary(
            metric_name=metric_name,
            direction=direction,
            generated_at=generated_at,
        )
        counts = summary["counts"]
        lines = [
            "# Hypothesis Validation Context",
            "",
            f"Generated: {summary['generated']}",
            "",
            "## Verdict Counts",
            "",
            f"- total: {counts['total']}",
            f"- supported: {counts['supported']}",
            f"- refuted: {counts['refuted']}",
            f"- inconclusive: {counts['inconclusive']}",
            f"- superseded: {counts['superseded']}",
            "",
        ]
        if counts["supported"] == 0:
            lines.extend(
                [
                    "Quality warning: no supported hypotheses.",
                    "",
                ]
            )
        lines.extend(["## Hypotheses", ""])
        for row in summary["nodes"]:
            lines.append(
                "- {node_id} [{status}] {statement} "
                "(best_attempt={best_attempt_id}, decision={decision})".format(
                    **row
                )
            )
        context = "\n".join(lines) + "\n"
        output_dir = self.run_dir / "hypothesis_aggregate"
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(output_dir / "paper_context.md", context)
        return context
