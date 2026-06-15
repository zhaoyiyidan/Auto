"""Canonical machine-readable Stage 15 verdicts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from researchclaw.pipeline.hypothesis_store import _atomic_write_json, _utcnow_iso


STAGE15_DECISIONS = {"proceed", "extend", "pivot", "inconclusive"}
FOLLOWUP_REQUIRED_FIELDS = ("statement", "prediction", "falsification")


def normalize_next_hypothesis(payload: Any) -> dict[str, Any]:
    """Return a canonical follow-up hypothesis or raise ``ValueError``."""
    if not isinstance(payload, dict):
        raise ValueError("next_hypothesis must be a mapping")
    missing = [
        field
        for field in FOLLOWUP_REQUIRED_FIELDS
        if not str(payload.get(field) or "").strip()
    ]
    if missing:
        raise ValueError(
            "next_hypothesis missing required field(s): " + ", ".join(missing)
        )
    baselines = payload.get("baselines") or []
    if not isinstance(baselines, (list, tuple)):
        baselines = [str(baselines)]
    return {
        "statement": str(payload.get("statement") or "").strip(),
        "prediction": str(payload.get("prediction") or "").strip(),
        "falsification": str(payload.get("falsification") or "").strip(),
        "rationale": str(payload.get("rationale") or "").strip(),
        "baselines": [str(item) for item in baselines],
    }


def validate_stage15_verdict(payload: Any) -> dict[str, Any]:
    """Validate and canonicalize a Stage 15 verdict payload."""
    if not isinstance(payload, dict):
        raise ValueError("Stage 15 verdict must be a JSON object")
    decision = str(payload.get("decision") or "inconclusive").strip().lower()
    if decision not in STAGE15_DECISIONS:
        raise ValueError(f"Invalid Stage 15 decision: {decision}")
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    raw_next = payload.get("next_hypothesis")
    next_hypothesis = None
    if decision in {"extend", "pivot"}:
        next_hypothesis = normalize_next_hypothesis(raw_next)
    elif raw_next is not None:
        next_hypothesis = normalize_next_hypothesis(raw_next)

    key_metrics = payload.get("key_metrics") or {}
    if not isinstance(key_metrics, dict):
        key_metrics = {}

    return {
        "decision": decision,
        "confidence": confidence,
        "next_hypothesis": next_hypothesis,
        "evidence_summary": str(payload.get("evidence_summary") or "").strip(),
        "key_metrics": key_metrics,
    }


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_next_hypothesis_from_text(text: str) -> dict[str, Any] | None:
    """Extract a follow-up hypothesis from JSON or simple labeled Markdown."""
    parsed = _extract_json_payload(text)
    if parsed:
        raw = parsed.get("next_hypothesis") or parsed.get("hypothesis")
        if raw is not None:
            return normalize_next_hypothesis(raw)

    labels = {
        "statement": r"(?:statement|hypothesis)\s*:",
        "prediction": r"prediction\s*:",
        "falsification": r"falsification\s*:",
        "rationale": r"rationale\s*:",
    }
    found: dict[str, str] = {}
    for field, label_pattern in labels.items():
        match = re.search(
            rf"(?im)^\s*(?:[-*]\s*)?{label_pattern}\s*(.+?)\s*$",
            text,
        )
        if match:
            found[field] = match.group(1).strip()
    if any(field in found for field in FOLLOWUP_REQUIRED_FIELDS):
        return normalize_next_hypothesis(found)
    return None


def build_stage15_verdict(
    *,
    decision: str,
    decision_md: str = "",
    confidence: float | None = None,
    next_hypothesis: dict[str, Any] | None = None,
    evidence_summary: str = "",
    key_metrics: dict[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Build a canonical verdict.

    In non-strict mode, EXTEND/PIVOT without a valid follow-up falls back to an
    inconclusive verdict so branch validation can finish without fabricating a
    child hypothesis.
    """
    normalized_decision = str(decision or "inconclusive").strip().lower()
    if normalized_decision not in STAGE15_DECISIONS:
        normalized_decision = "inconclusive"
    followup = next_hypothesis
    if followup is None and normalized_decision in {"extend", "pivot"}:
        try:
            followup = extract_next_hypothesis_from_text(decision_md)
        except ValueError:
            if strict:
                raise
            followup = None
    payload = {
        "decision": normalized_decision,
        "confidence": 0.0 if confidence is None else confidence,
        "next_hypothesis": followup,
        "evidence_summary": evidence_summary or decision_md[:1000],
        "key_metrics": key_metrics or {},
    }
    if not strict and normalized_decision in {"extend", "pivot"} and followup is None:
        payload["decision"] = "inconclusive"
    return validate_stage15_verdict(payload)


def write_stage15_verdict(
    stage_dir: Path,
    *,
    decision: str,
    decision_md: str = "",
    confidence: float | None = None,
    next_hypothesis: dict[str, Any] | None = None,
    evidence_summary: str = "",
    key_metrics: dict[str, Any] | None = None,
    generated_at: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Write ``stage-15/verdict.json`` atomically and return the payload."""
    verdict = build_stage15_verdict(
        decision=decision,
        decision_md=decision_md,
        confidence=confidence,
        next_hypothesis=next_hypothesis,
        evidence_summary=evidence_summary,
        key_metrics=key_metrics,
        strict=strict,
    )
    payload = dict(verdict)
    payload["generated"] = generated_at or _utcnow_iso()
    _atomic_write_json(Path(stage_dir) / "verdict.json", payload)
    return payload


def read_stage15_verdict(path: Path) -> dict[str, Any]:
    """Read and validate a Stage 15 verdict file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    verdict = validate_stage15_verdict(payload)
    if isinstance(payload, dict) and payload.get("generated") is not None:
        verdict["generated"] = payload.get("generated")
    return verdict
