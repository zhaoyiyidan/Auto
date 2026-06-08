"""Deterministic hypothesis judgment from experiment protocols."""

from __future__ import annotations

import json
import math
from typing import Any

from researchclaw.experiment.protocol import DecisionRule, ExperimentProtocol
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import _safe_json_loads


VERDICTS = ("supported", "refuted", "inconclusive")


def evaluate_decision_rule(
    rule: DecisionRule,
    summary: dict[str, Any],
    *,
    llm: LLMClient | None = None,
) -> str:
    """Evaluate a single decision rule against an experiment summary."""

    value = _resolve_metric_value(rule.metric, summary)
    baseline = (
        _resolve_metric_value(rule.baseline_metric, summary)
        if rule.baseline_metric
        else None
    )
    passed = _apply_comparator(rule, value, baseline)
    if passed is None:
        if llm is not None:
            return _llm_fallback(rule, summary, llm)
        return "inconclusive"
    if rule.supported_if == "fail":
        passed = not passed
    return "supported" if passed else "refuted"


def judge_hypotheses(
    protocol: ExperimentProtocol,
    summary: dict[str, Any],
    *,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Judge every hypothesis with deterministic rules and optional LLM fallback."""

    hypothesis_ids = [hypothesis.id for hypothesis in protocol.hypotheses]
    for rule in protocol.decision_rules:
        if rule.hypothesis_id not in hypothesis_ids:
            hypothesis_ids.append(rule.hypothesis_id)

    by_hypothesis: dict[str, list[DecisionRule]] = {item: [] for item in hypothesis_ids}
    for rule in protocol.decision_rules:
        by_hypothesis.setdefault(rule.hypothesis_id, []).append(rule)

    per_hypothesis: dict[str, dict[str, Any]] = {}
    for hypothesis_id in hypothesis_ids:
        rule_verdicts = [
            evaluate_decision_rule(rule, summary, llm=llm)
            for rule in by_hypothesis.get(hypothesis_id, ())
        ]
        if not rule_verdicts:
            verdict = "inconclusive"
        elif "supported" in rule_verdicts:
            verdict = "supported"
        elif "refuted" in rule_verdicts:
            verdict = "refuted"
        else:
            verdict = "inconclusive"
        statement = next(
            (
                hypothesis.statement
                for hypothesis in protocol.hypotheses
                if hypothesis.id == hypothesis_id
            ),
            "",
        )
        per_hypothesis[hypothesis_id] = {
            "hypothesis_id": hypothesis_id,
            "statement": statement,
            "verdict": verdict,
            "rule_verdicts": rule_verdicts,
        }

    counts = {verdict: 0 for verdict in VERDICTS}
    for item in per_hypothesis.values():
        counts[str(item.get("verdict", "inconclusive"))] += 1
    return {
        "per_hypothesis": per_hypothesis,
        "counts": counts,
        "decision": decision_from_verdicts(per_hypothesis),
    }


def decision_from_verdicts(per_hypothesis: dict[str, Any]) -> str:
    verdicts = []
    for item in per_hypothesis.values():
        if isinstance(item, dict):
            verdicts.append(str(item.get("verdict", "inconclusive")))
        else:
            verdicts.append(str(item))
    if any(verdict == "supported" for verdict in verdicts):
        return "proceed"
    if any(verdict == "refuted" for verdict in verdicts):
        return "pivot"
    return "extend"


def _resolve_metric_value(name: str, summary: dict[str, Any]) -> float | None:
    if not name:
        return None
    primary = str(summary.get("primary_metric") or "")
    if name == primary or name == "primary_metric":
        value = _numeric(summary.get("best_metric"))
        if value is not None:
            return value

    metrics_summary = summary.get("metrics_summary")
    if isinstance(metrics_summary, dict):
        if name in metrics_summary:
            value = metrics_summary[name]
            if isinstance(value, dict):
                numeric = _numeric(value.get("mean"))
                if numeric is not None:
                    return numeric
            numeric = _numeric(value)
            if numeric is not None:
                return numeric

    best_run = summary.get("best_run")
    metrics = best_run.get("metrics") if isinstance(best_run, dict) else None
    if isinstance(metrics, dict):
        numeric = _numeric(metrics.get(name))
        if numeric is not None:
            return numeric

    return None


def _numeric(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _apply_comparator(
    rule: DecisionRule,
    value: float | None,
    baseline: float | None,
) -> bool | None:
    if value is None:
        return None
    comparator = rule.comparator
    threshold = rule.threshold
    if comparator == "gt":
        return value > threshold
    if comparator == "gte":
        return value >= threshold
    if comparator == "lt":
        return value < threshold
    if comparator == "lte":
        return value <= threshold
    if baseline is None:
        return None
    delta = value - baseline
    if comparator == "delta_gt":
        return delta > threshold
    if comparator == "delta_lt":
        return delta < threshold
    if comparator == "abs_delta_lt":
        return abs(delta) < threshold
    if comparator == "within_pct":
        if baseline == 0:
            return None
        return abs(delta / baseline) * 100.0 <= threshold
    return None


def _llm_fallback(
    rule: DecisionRule,
    summary: dict[str, Any],
    llm: LLMClient,
) -> str:
    prompt = (
        "A deterministic experiment-protocol decision rule was inconclusive "
        "because a metric was missing or non-numeric. Read the rule and summary, "
        "then decide the hypothesis verdict. Return only JSON with keys "
        '`verdict` ("supported", "refuted", or "inconclusive") and `rationale`.\n\n'
        f"RULE:\n{json.dumps(rule.to_dict(), indent=2, sort_keys=True)}\n\n"
        f"SUMMARY:\n{json.dumps(summary, indent=2, default=str, sort_keys=True)[:12000]}"
    )
    try:
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are a strict scientific hypothesis judge. You only use "
                "the provided summary and never invent missing measurements."
            ),
            json_mode=True,
            max_tokens=1000,
        )
        text = response.content if hasattr(response, "content") else str(response)
        payload = _safe_json_loads(text, {})
        if isinstance(payload, dict):
            verdict = str(payload.get("verdict") or "").strip().lower()
            if verdict in VERDICTS:
                return verdict
    except Exception:  # noqa: BLE001
        return "inconclusive"
    return "inconclusive"
