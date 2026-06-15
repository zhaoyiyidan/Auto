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
from researchclaw.pipeline.stage15_verdict import write_stage15_verdict


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

    def write_all(
        self,
        *,
        metric_name: str,
        direction: str,
        generated_at: str | None = None,
    ) -> dict[str, Any]:
        summary = self.write_validation_summary(
            metric_name=metric_name,
            direction=direction,
            generated_at=generated_at,
        )
        registry = self.write_evidence_registry(
            metric_name=metric_name,
            direction=direction,
        )
        context = self.write_paper_context(
            metric_name=metric_name,
            direction=direction,
            generated_at=generated_at,
        )
        return {
            "validation_summary": summary,
            "evidence_registry": registry,
            "paper_context": context,
        }

    def aggregate(
        self,
        *,
        generated_at: str | None = None,
        metric_name: str = "score",
        direction: str = "maximize",
    ) -> dict[str, Any]:
        generated_at = generated_at or _utcnow_iso()
        nodes = self._read_nodes()
        attempts = self._read_attempts()
        best_attempts = self.select_best_attempts(
            metric_name=metric_name,
            direction=direction,
        )
        for attempt in reversed(attempts):
            if attempt.node_id in best_attempts:
                continue
            if attempt.status in {"succeeded", "failed", "abandoned"}:
                best_attempts[attempt.node_id] = attempt

        tree_counts = {
            "total_nodes": len(nodes),
            "terminal_nodes": sum(
                1
                for node in nodes
                if node.status in {"supported", "refuted", "inconclusive", "superseded"}
            ),
            "supported": sum(1 for node in nodes if node.status == "supported"),
            "inconclusive": sum(1 for node in nodes if node.status == "inconclusive"),
            "superseded": sum(1 for node in nodes if node.status == "superseded"),
        }
        validation_summary: list[dict[str, Any]] = []
        evidence_registry: dict[str, Any] = {}
        recommended_hypotheses: list[dict[str, Any]] = []
        node_by_id = {node.id: node for node in nodes}
        for node in nodes:
            attempt = best_attempts.get(node.id)
            artifacts = list(attempt.artifacts if attempt else [])
            branch_run_dir = attempt.branch_run_dir if attempt else ""
            row = {
                "node_id": node.id,
                "decision": node.status,
                "stage15_decision": attempt.decision if attempt else None,
                "confidence": _metric_value(attempt.metrics, "confidence") if attempt else None,
                "key_metrics": attempt.metrics if attempt else {},
                "branch_run_dir": branch_run_dir,
                "artifacts": artifacts,
            }
            validation_summary.append(row)
            evidence_registry[node.id] = {
                "attempt_id": attempt.attempt_id if attempt else None,
                "status": node.status,
                "branch_run_dir": branch_run_dir,
                "artifacts": artifacts,
            }
            if node.status == "supported":
                recommended_hypotheses.append(
                    {
                        "node_id": node.id,
                        "statement": node.statement,
                        "prediction": node.prediction,
                        "falsification": node.falsification,
                    }
                )

        context_lines = [
            "# Per-Hypothesis Validation Aggregate",
            "",
            f"Generated: {generated_at}",
            f"Total nodes: {tree_counts['total_nodes']}",
            f"Supported: {tree_counts['supported']}",
            f"Inconclusive: {tree_counts['inconclusive']}",
            f"Superseded: {tree_counts['superseded']}",
            "",
        ]
        for row in validation_summary:
            node = node_by_id.get(str(row["node_id"]))
            statement = node.statement if node else ""
            context_lines.append(
                f"- {row['node_id']}: {row['decision'] or 'unknown'} - {statement}"
            )
        aggregate = {
            "run_id": self.run_dir.name,
            "generated_at": generated_at,
            "hypothesis_tree": tree_counts,
            "validation_summary": validation_summary,
            "evidence_registry": evidence_registry,
            "recommended_hypotheses": recommended_hypotheses,
            "context_for_writing": "\n".join(context_lines).strip() + "\n",
        }
        _atomic_write_json(self.run_dir / "hypothesis_aggregate.json", aggregate)
        return aggregate

    def write_root_handoff(
        self,
        aggregate: dict[str, Any] | None = None,
        *,
        generated_at: str | None = None,
    ) -> dict[str, Any]:
        """Write root Stage 14/15 artifacts from per-hypothesis aggregate data."""
        generated_at = generated_at or _utcnow_iso()
        if aggregate is None:
            aggregate_path = self.run_dir / "hypothesis_aggregate.json"
            if aggregate_path.is_file():
                aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            else:
                aggregate = self.aggregate(generated_at=generated_at)

        decision = _root_decision_from_aggregate(aggregate)
        confidence = _root_confidence_from_aggregate(aggregate)
        key_metrics = _root_key_metrics(aggregate)
        analysis_md = _format_root_analysis(aggregate, generated_at=generated_at)
        decision_md = _format_root_decision(
            aggregate,
            decision=decision,
            confidence=confidence,
            generated_at=generated_at,
        )

        stage14_dir = self.run_dir / "stage-14"
        stage15_dir = self.run_dir / "stage-15"
        stage14_dir.mkdir(parents=True, exist_ok=True)
        stage15_dir.mkdir(parents=True, exist_ok=True)

        _atomic_write_text(stage14_dir / "analysis.md", analysis_md)
        _atomic_write_json(
            stage14_dir / "experiment_summary.json",
            {
                "source": "hypothesis_aggregate.json",
                "generated_at": generated_at,
                "hypothesis_tree": aggregate.get("hypothesis_tree", {}),
                "validation_summary": aggregate.get("validation_summary", []),
                "recommended_hypotheses": aggregate.get(
                    "recommended_hypotheses", []
                ),
            },
        )
        _atomic_write_json(
            stage14_dir / "provenance.json",
            {
                "source": "hypothesis_aggregate.json",
                "aggregate_path": str(
                    (self.run_dir / "hypothesis_aggregate.json").resolve()
                ),
                "generated_at": generated_at,
            },
        )
        _atomic_write_text(stage15_dir / "decision.md", decision_md)
        _atomic_write_json(
            stage15_dir / "decision_structured.json",
            {
                "decision": decision,
                "confidence": confidence,
                "source": "per_hypothesis_aggregate",
                "generated": generated_at,
                "key_metrics": key_metrics,
            },
        )
        verdict = write_stage15_verdict(
            stage15_dir,
            decision=decision,
            decision_md=decision_md,
            confidence=confidence,
            evidence_summary=decision_md[:1000],
            key_metrics=key_metrics,
            generated_at=generated_at,
            strict=False,
        )
        return {
            "decision": decision,
            "confidence": confidence,
            "analysis_path": str(stage14_dir / "analysis.md"),
            "decision_path": str(stage15_dir / "decision.md"),
            "verdict": verdict,
        }


def _root_key_metrics(aggregate: dict[str, Any]) -> dict[str, Any]:
    tree = aggregate.get("hypothesis_tree", {})
    if not isinstance(tree, dict):
        tree = {}
    rows = aggregate.get("validation_summary", [])
    if not isinstance(rows, list):
        rows = []
    return {
        "total_nodes": int(tree.get("total_nodes") or 0),
        "terminal_nodes": int(tree.get("terminal_nodes") or 0),
        "supported": int(tree.get("supported") or 0),
        "inconclusive": int(tree.get("inconclusive") or 0),
        "superseded": int(tree.get("superseded") or 0),
        "validation_summary_count": len(rows),
    }


def _root_decision_from_aggregate(aggregate: dict[str, Any]) -> str:
    metrics = _root_key_metrics(aggregate)
    return "proceed" if metrics["supported"] > 0 else "inconclusive"


def _root_confidence_from_aggregate(aggregate: dict[str, Any]) -> float:
    metrics = _root_key_metrics(aggregate)
    total = max(1, metrics["total_nodes"])
    if metrics["supported"] <= 0:
        return 0.0
    return min(1.0, metrics["supported"] / total)


def _quality_rating_from_aggregate(aggregate: dict[str, Any]) -> int:
    metrics = _root_key_metrics(aggregate)
    if metrics["supported"] <= 0:
        return 2
    if metrics["supported"] == metrics["terminal_nodes"]:
        return 7
    return 5


def _format_root_analysis(
    aggregate: dict[str, Any],
    *,
    generated_at: str,
) -> str:
    metrics = _root_key_metrics(aggregate)
    rows = aggregate.get("validation_summary", [])
    if not isinstance(rows, list):
        rows = []
    context = str(aggregate.get("context_for_writing") or "").strip()
    quality = _quality_rating_from_aggregate(aggregate)
    lines = [
        "# Per-Hypothesis Validation Analysis",
        "",
        "## Experiment Objective",
        (
            "Summarize branch-level Stage 9-15 validation results generated by "
            "the per-hypothesis coordinator."
        ),
        "",
        "## Experiment Plan",
        (
            "Each Stage 8 hypothesis was validated in an isolated branch attempt. "
            "This root analysis is a handoff artifact derived from "
            "`hypothesis_aggregate.json`; it is not an additional experiment."
        ),
        "",
        "## Executed Experiments",
        f"Total hypothesis nodes: {metrics['total_nodes']}",
        f"Terminal nodes: {metrics['terminal_nodes']}",
        f"Supported nodes: {metrics['supported']}",
        f"Inconclusive nodes: {metrics['inconclusive']}",
        f"Superseded nodes: {metrics['superseded']}",
        "",
        "## Results Summary",
        f"Result quality: {quality}/10",
    ]
    if metrics["supported"] <= 0:
        lines.append(
            "No branch produced a supported verdict; downstream writing must not "
            "claim validated experimental findings."
        )
    else:
        lines.append(
            "At least one branch produced a supported verdict; downstream writing "
            "should cite branch artifacts and preserve branch-level caveats."
        )
    if context:
        lines.extend(["", context])
    lines.extend(["", "## Artifact Locations"])
    lines.append("- Root aggregate: hypothesis_aggregate.json")
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- {node_id}: decision={decision}, stage15_decision={stage15}, "
            "branch_run_dir={branch_run_dir}".format(
                node_id=row.get("node_id", ""),
                decision=row.get("decision", ""),
                stage15=row.get("stage15_decision", ""),
                branch_run_dir=row.get("branch_run_dir", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            f"Generated: {generated_at}",
            "Source: hypothesis_aggregate.json",
            "",
        ]
    )
    return "\n".join(lines)


def _format_root_decision(
    aggregate: dict[str, Any],
    *,
    decision: str,
    confidence: float,
    generated_at: str,
) -> str:
    metrics = _root_key_metrics(aggregate)
    lines = [
        "# Per-Hypothesis Research Decision",
        "",
        f"Decision: {decision.upper()}",
        f"Confidence: {confidence:.2f}",
        f"Generated: {generated_at}",
        "",
    ]
    if decision == "proceed":
        lines.append(
            "At least one branch produced a supported verdict. Use the "
            "per-hypothesis aggregate and branch artifacts as the evidence base."
        )
    else:
        lines.append(
            "No branch produced a supported verdict. Treat the run as "
            "inconclusive and do not present failed branch attempts as validated "
            "results."
        )
    lines.extend(
        [
            "",
            "## Aggregate Counts",
            f"- total_nodes: {metrics['total_nodes']}",
            f"- terminal_nodes: {metrics['terminal_nodes']}",
            f"- supported: {metrics['supported']}",
            f"- inconclusive: {metrics['inconclusive']}",
            f"- superseded: {metrics['superseded']}",
            "",
        ]
    )
    return "\n".join(lines)
