"""Stages 14-15: Result analysis and research decision."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._domain import _detect_domain, _is_ml_domain
from researchclaw.pipeline._helpers import (
    StageResult,
    _build_context_preamble,
    _chat_with_prompt,
    _collect_experiment_results,
    _collect_json_context,
    _get_evolution_overlay,
    _multi_perspective_generate,
    _read_prior_artifact,
    _safe_json_loads,
    _synthesize_perspectives,
    _utcnow_iso,
)
from researchclaw.pipeline.evidence_organizer import (
    build_evidence_bundle,
    build_organizer_prompt,
    create_evidence_organizer_agent,
    postcheck_analysis,
    run_evidence_organizer,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _load_execution_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("stage-*/execution_record.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _load_workspace_registry(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("stage-*/workspace_experiment_registry.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _load_plan_and_expected_outputs(run_dir: Path) -> tuple[str, list[str]]:
    plan_md = _read_prior_artifact(run_dir, "plan.md") or ""
    expected_text = _read_prior_artifact(run_dir, "expected_outputs.json")
    expected_outputs: list[str] = []
    if expected_text:
        try:
            payload = json.loads(expected_text)
        except json.JSONDecodeError:
            payload = {}
        outputs = payload.get("outputs") if isinstance(payload, dict) else None
        if isinstance(outputs, list):
            expected_outputs = [
                item.strip()
                for item in outputs
                if isinstance(item, str) and item.strip()
            ]
    return plan_md, expected_outputs


def _merge_result_hashes(
    execution_records: list[dict[str, Any]],
    registry_records: list[dict[str, Any]],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for record in [*registry_records, *execution_records]:
        payload = record.get("result_hashes", {})
        if isinstance(payload, dict):
            hashes.update({str(k): str(v) for k, v in payload.items()})
    return hashes


def _unique_commits(
    execution_records: list[dict[str, Any]],
    registry_records: list[dict[str, Any]],
) -> list[str]:
    commits: list[str] = []
    for record in [*registry_records, *execution_records]:
        for key in ("agent_commit_sha", "code_commit"):
            value = str(record.get(key, ""))
            if value and value not in commits:
                commits.append(value)
    return commits


def _execute_result_analysis(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    _ = adapters, llm, prompts
    summary, provenance = _build_experiment_summary(run_dir, config)
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (stage_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        bundle = build_evidence_bundle(run_dir, config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage 14 failed to build evidence bundle")
        return StageResult(
            stage=Stage.RESULT_ANALYSIS,
            status=StageStatus.FAILED,
            artifacts=("experiment_summary.json", "provenance.json"),
            error=f"E14_ANALYSIS_ERR: failed to build evidence bundle: {exc}",
        )

    try:
        session = create_evidence_organizer_agent(config, run_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage 14 failed to create evidence organizer agent")
        return StageResult(
            stage=Stage.RESULT_ANALYSIS,
            status=StageStatus.FAILED,
            artifacts=("experiment_summary.json", "provenance.json"),
            error=f"E14_ANALYSIS_ERR: evidence organizer agent unavailable: {exc}",
        )
    if session is None:
        return StageResult(
            stage=Stage.RESULT_ANALYSIS,
            status=StageStatus.FAILED,
            artifacts=("experiment_summary.json", "provenance.json"),
            error="E14_ANALYSIS_ERR: evidence organizer agent unavailable",
        )

    agent_cfg = config.experiment.result_analysis_agent
    max_postcheck_retries = max(
        0,
        int(getattr(agent_cfg, "max_postcheck_retries", 1)),
    )
    timeout_sec = int(getattr(agent_cfg, "timeout_sec", 1800))
    analysis_path = stage_dir / "analysis.md"
    attempts: list[dict[str, Any]] = []
    violations: tuple[str, ...] = ()
    for attempt_index in range(max_postcheck_retries + 1):
        prompt = build_organizer_prompt(bundle, violations if attempt_index else ())
        try:
            run_evidence_organizer(session, prompt, timeout_sec)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Stage 14 evidence organizer agent failed")
            _write_analysis_postcheck(
                stage_dir,
                {
                    "ok": False,
                    "attempts": attempts,
                    "error": f"agent failed: {exc}",
                },
            )
            return StageResult(
                stage=Stage.RESULT_ANALYSIS,
                status=StageStatus.FAILED,
                artifacts=("experiment_summary.json", "provenance.json"),
                error=f"E14_ANALYSIS_ERR: evidence organizer agent failed: {exc}",
            )
        try:
            analysis_text = analysis_path.read_text(encoding="utf-8")
        except OSError:
            analysis_text = ""
        postcheck = postcheck_analysis(analysis_text, run_dir)
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "ok": postcheck.ok,
                "violations": list(postcheck.violations),
                "strict_retry": attempt_index > 0,
            }
        )
        if postcheck.ok:
            _write_analysis_postcheck(
                stage_dir,
                {"ok": True, "attempts": attempts},
            )
            return StageResult(
                stage=Stage.RESULT_ANALYSIS,
                status=StageStatus.DONE,
                artifacts=("analysis.md", "experiment_summary.json", "provenance.json"),
                evidence_refs=("stage-14/analysis.md", "stage-14/provenance.json"),
            )
        violations = postcheck.violations

    _write_analysis_postcheck(
        stage_dir,
        {"ok": False, "attempts": attempts, "violations": list(violations)},
    )
    return StageResult(
        stage=Stage.RESULT_ANALYSIS,
        status=StageStatus.FAILED,
        artifacts=("analysis.md", "experiment_summary.json", "provenance.json"),
        error=(
            "E14_ANALYSIS_ERR: analysis.md crossed organizer boundaries: "
            + ", ".join(violations)
        ),
        decision="retry",
    )


def _build_experiment_summary(
    run_dir: Path,
    config: RCConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _ = config
    plan_md, expected_outputs = _load_plan_and_expected_outputs(run_dir)
    execution_records = _load_execution_records(run_dir)
    registry_records = _load_workspace_registry(run_dir)
    result_hashes = _merge_result_hashes(execution_records, registry_records)
    latest_execution = execution_records[-1] if execution_records else {}
    completed_runs = [
        record
        for record in execution_records
        if str(record.get("final_status", "")).strip().lower()
        in {"completed", "succeeded", "success"}
    ]
    missing_expected_outputs: list[str] = []
    for record in execution_records:
        for item in record.get("missing_expected_outputs", []) or []:
            if isinstance(item, str) and item not in missing_expected_outputs:
                missing_expected_outputs.append(item)
    observed_result_paths: list[str] = []
    for record in execution_records:
        for item in record.get("result_paths", []) or []:
            if isinstance(item, str) and item not in observed_result_paths:
                observed_result_paths.append(item)
    summary = {
        "plan": plan_md,
        "expected_outputs": expected_outputs,
        "observed_result_paths": observed_result_paths,
        "missing_expected_outputs": missing_expected_outputs,
        "output_drift": bool(missing_expected_outputs),
        "iterations": len([r for r in registry_records if int(r.get("stage", 0)) == 13]),
        "n_runs": len(execution_records),
        "n_completed_runs": len(completed_runs),
        "latest_run": latest_execution,
        "runs": execution_records,
    }
    provenance = {
        "base_sha": str(registry_records[0].get("base_sha", "")) if registry_records else "",
        "commits": _unique_commits(execution_records, registry_records),
        "session_name": str(registry_records[0].get("session_name", "")) if registry_records else "",
        "result_hashes": result_hashes,
        "registry_records": registry_records,
    }
    return summary, provenance


def _write_analysis_postcheck(stage_dir: Path, payload: dict[str, Any]) -> None:
    (stage_dir / "analysis_postcheck.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _parse_decision(text: str) -> str:
    """Extract PROCEED/PIVOT/EXTEND from Stage 15 decision text.

    Looks for the first standalone keyword on its own line after a
    ``## Decision`` heading.  Falls back to a keyword scan of the first
    few lines after the heading, but only matches the keyword itself
    (not mentions inside explanatory prose like "PIVOT is not warranted").
    """
    import re as _re

    route_keywords = ("PROCEED", "PIVOT", "EXTEND")
    text_upper = text.upper()
    # Look in the first occurrence after "## Decision" heading
    decision_section = ""
    for keyword in ("## DECISION", "## Decision", "## decision"):
        if keyword.upper() in text_upper:
            idx = text_upper.index(keyword.upper())
            decision_section = text[idx : idx + 200]
            break
    search_text = decision_section or text[:500]

    # First try: look for a line that is just the keyword (possibly with
    # whitespace / markdown bold / trailing punctuation).
    for line in search_text.splitlines():
        stripped = line.strip().strip("*").strip("#").strip().rstrip(".:;,-")
        upper = stripped.upper()
        if upper in route_keywords:
            return upper.lower()

    # Prefer explicit final recommendation lines over option lists that mention
    # every allowed keyword.
    for line in search_text.splitlines():
        line_upper = line.upper()
        if not any(
            marker in line_upper
            for marker in ("FINAL", "RECOMMENDATION", "RECOMMEND", "DECISION")
        ):
            continue
        matches = [
            kw for kw in route_keywords
            if _re.search(r"\b" + kw + r"\b", line_upper)
        ]
        if len(matches) == 1:
            return matches[0].lower()
        if "EXTEND" in matches:
            return "extend"

    # Fallback: regex for standalone word boundaries so that
    # "PIVOT is not warranted" does NOT match as a decision.
    for kw in ("EXTEND", "PIVOT", "PROCEED"):
        # Only match if the keyword appears as the FIRST keyword-class token
        # on its own (not embedded in a sentence saying "not PIVOT").
        pattern = _re.compile(
            r"(?:^|##\s*Decision\s*\n\s*)" + kw, _re.IGNORECASE | _re.MULTILINE
        )
        if pattern.search(search_text):
            return kw.lower()

    # Last resort: position-based for hypothesis-level route decisions only.
    # Experiment routing happens before Stage 15.
    search_upper = search_text.upper()
    positions = {
        kw: search_upper.rfind(kw)
        for kw in ("PIVOT", "EXTEND")
    }
    best_kw, best_pos = max(positions.items(), key=lambda item: item[1])
    if best_pos >= 0:
        return best_kw.lower()
    return "proceed"


def _decision_quality_warnings(decision_md: str) -> list[str]:
    """Return lightweight quality warnings for research decision text."""
    quality_warnings: list[str] = []
    dec_lower = decision_md.lower()
    if "baseline" not in dec_lower and "control" not in dec_lower:
        quality_warnings.append("Decision text does not mention baselines")
    if (
        "seed" not in dec_lower
        and "replicat" not in dec_lower
        and "run" not in dec_lower
    ):
        quality_warnings.append(
            "Decision text does not mention multi-seed/replicate runs"
        )
    if (
        "metric" not in dec_lower
        and "accuracy" not in dec_lower
        and "loss" not in dec_lower
    ):
        quality_warnings.append("Decision text does not mention evaluation metrics")
    return quality_warnings


def _write_decision_structured_json(
    stage_dir: Path,
    decision_md: str,
    decision: str,
) -> dict[str, Any]:
    """Write the structured Stage 15 decision payload and return it."""
    decision_payload = {
        "decision": decision,
        "raw_text_excerpt": decision_md[:500],
        "quality_warnings": _decision_quality_warnings(decision_md),
        "generated": _utcnow_iso(),
    }
    (stage_dir / "decision_structured.json").write_text(
        json.dumps(decision_payload, indent=2), encoding="utf-8"
    )
    return decision_payload


def _decision_review_unavailable(decision: str) -> str:
    """Return a transparent degraded-mode note without fabricating rationale."""
    decision_upper = str(decision or "proceed").upper()
    lines = [
        "# Decision Review",
        "",
        f"Decision Reviewed: {decision_upper}",
        "",
        "agent rationale unavailable: Stage 15 could not generate an "
        "agent-authored review note because the LLM was unavailable or review "
        "generation failed.",
        "",
        "This explanatory artifact is in degraded mode. The authoritative route "
        "remains `decision.md`; this file does not provide a Short Rationale or "
        "infer reasons beyond the finalized decision.",
        "",
        "Sections not generated:",
        f"- Decision Reviewed: {decision_upper}",
        "- Short Rationale: not generated",
        "- Evidence Considered: not generated",
        "- Criteria Assessment: not generated",
        "- Why This Decision: not generated",
        "- Why Not The Alternatives: not generated",
        "- Caveats For Reviewer: not generated",
        "- Recommended Human Review Focus: not generated",
        "",
        "Reviewer action: inspect `decision.md` and the prior analysis artifacts "
        "directly before approving this gate.",
    ]
    return "\n".join(lines) + "\n"


def _write_decision_review(
    *,
    stage_dir: Path,
    run_dir: Path,
    decision: str,
    decision_md: str,
    llm: LLMClient | None,
    prompts: PromptManager | None = None,
) -> None:
    """Generate and save the human-facing Stage 15 decision explanation."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / "decision_review.md"
    if llm is None:
        path.write_text(_decision_review_unavailable(decision), encoding="utf-8")
        return
    try:
        pm = prompts or PromptManager()
        analysis = _read_prior_artifact(run_dir, "analysis.md") or ""
        review_prompt = pm.sub_prompt(
            "decision_review",
            decision=str(decision).upper(),
            decision_md=decision_md,
            analysis=analysis,
        )
        resp = _chat_with_prompt(
            llm,
            review_prompt.system,
            review_prompt.user,
            strip_thinking=True,
        )
        text = (resp.content or "").strip()
        path.write_text(
            text or _decision_review_unavailable(decision),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        logger.warning("Stage 15: decision_review generation failed", exc_info=True)
        path.write_text(_decision_review_unavailable(decision), encoding="utf-8")


# ---------------------------------------------------------------------------
# Requirements gate. Reads the manifest's optional `requirements:` list, calls
# the LLM judge, persists the verdict, and decides PROCEED with a
# `requirements_unmet` flag when any must_pass item remains failing.
# ---------------------------------------------------------------------------

_REQUIREMENTS_RETRY_FILE = "requirements_retry_count.txt"


def _read_requirements_from_manifest(run_dir: Path) -> list[dict[str, object]]:
    """Pull the `requirements:` list out of the run's topic manifest.

    Lookup order:
      1. ``run_dir/stage-09/requirements.json``  (written by prepare_run.py
         for ARC-Bench topics that declared requirements)
      2. ``run_dir/stage-07/topic_manifest.json``  (raw manifest snapshot)
      3. ``run_dir/topic_manifest.json``  (legacy)

    Returns an empty list when no requirements are declared — callers treat
    that as "skip the agent-mode gate, fall through to standard decision."
    """
    candidates = (
        run_dir / "stage-09" / "requirements.json",
        run_dir / "stage-07" / "topic_manifest.json",
        run_dir / "topic_manifest.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # stage-09/requirements.json stores the list directly; manifest snapshots
        # nest it under "requirements".
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            req = data.get("requirements")
            if isinstance(req, list):
                return [r for r in req if isinstance(r, dict)]
    return []


def _read_experiment_summary(run_dir: Path) -> dict[str, object]:
    """Return the most recent experiment_summary.json contents (or {})."""
    p = run_dir / "experiment_summary_best.json"
    if not p.is_file():
        for sd in sorted(run_dir.glob("stage-14*/experiment_summary.json"), reverse=True):
            p = sd
            break
    if p and p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _read_agent_results_canonical(run_dir: Path) -> dict[str, object]:
    """Read canonical result payloads from stage-12 workspace artifacts.

    Prefers
    a true canonical file (one with ``metrics``/``primary_metric``/``hypotheses``
    keys), but skips metadata-only stubs and falls back to
    ``analysis/summary.json`` so runs without the canonical-output
    contract still surface scientific data to the requirements judge.
    """
    sandbox_meta_keys = {"source", "returncode", "elapsed_sec", "timed_out", "artifacts", "status"}
    workspace_roots = (
        run_dir / "stage-12" / "runs" / "workspace" / "sandbox",
        run_dir / "stage-12" / "runs",
    )
    for ws in workspace_roots:
        if not ws.is_dir():
            continue
        for path in (
            ws / "results.json",
            ws / "analysis" / "summary.json",
            ws / "analysis" / "flux_analysis_summary.json",
            ws / "output" / "data" / "results.json",
        ):
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            has_agent_keys = (
                "metrics" in data
                or "primary_metric" in data
                or "hypotheses" in data
                or "structured_results" in data
            )
            if has_agent_keys:
                return data
            non_meta = set(data.keys()) - sandbox_meta_keys
            if not non_meta:
                continue
            # Older convention: numeric / bool top-level keys → wrap.
            numeric = {
                k: v for k, v in data.items()
                if k in non_meta and isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            booleans = {
                k: v for k, v in data.items()
                if k in non_meta and isinstance(v, bool)
            }
            if numeric or booleans:
                wrapped: dict[str, object] = {"metrics": numeric}
                if booleans:
                    wrapped["hypotheses"] = {k: {"supported": v} for k, v in booleans.items()}
                return wrapped
    return {}


def _read_retry_count(run_dir: Path) -> int:
    p = run_dir / _REQUIREMENTS_RETRY_FILE
    if not p.is_file():
        return 0
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def _format_agent_decision_md(
    verdict: dict[str, object],
    decision: str,
    retry_count: int,
    rerun_triggered: bool,
) -> str:
    lines = [
        "# Research Decision (agent-mode requirements gate)",
        "",
        f"## Decision: {decision.upper()}",
        "",
        f"## Verdict: {verdict.get('verdict', '?')} "
        f"(retry_count={retry_count}, rerun_triggered={rerun_triggered})",
        "",
    ]
    if verdict.get("verdict") == "partial":
        lines += [
            "All must_pass requirements met; some optional requirements remain "
            "unmet.  Proceeding to paper-writing.",
            "",
        ]
    elif verdict.get("verdict") == "reject":
        lines += [
            "Requirements unmet — proceeding to "
            "paper-writing with `requirements_unmet=true` flag.  Downstream "
            "stages should surface this caveat in the writeup.",
            "",
        ]
    else:
        lines += [
            "All must_pass requirements met.  Proceeding to paper-writing.",
            "",
        ]
    lines += [
        "## Per-requirement",
        "",
        "| id | must_pass | met | evidence | missing |",
        "|---|---|---|---|---|",
    ]
    for r in verdict.get("per_requirement", []) or []:
        lines.append(
            f"| {r.get('id','?')} | {bool(r.get('must_pass'))} | "
            f"{bool(r.get('met'))} | {str(r.get('evidence',''))[:80]} | "
            f"{str(r.get('missing',''))[:80]} |"
        )
    delta = str(verdict.get("delta_feedback") or "").strip()
    if delta:
        lines += ["", "## Delta feedback for rerun", "", delta]
    return "\n".join(lines) + "\n"


def _format_hypothesis_decision_md(verdict: dict[str, Any]) -> str:
    decision = str(verdict.get("decision") or "extend").lower()
    lines = [
        "# Research Decision (hypothesis protocol judge)",
        "",
        f"## Decision: {decision.upper()}",
        "",
        "## Hypothesis Verdicts",
        "",
        "| hypothesis | verdict |",
        "|---|---|",
    ]
    per_hypothesis = verdict.get("per_hypothesis")
    if isinstance(per_hypothesis, dict):
        for hypothesis_id, item in per_hypothesis.items():
            row = item if isinstance(item, dict) else {}
            lines.append(
                f"| {hypothesis_id} | {str(row.get('verdict', 'inconclusive'))} |"
            )
    counts = verdict.get("counts")
    if isinstance(counts, dict):
        lines += [
            "",
            "## Counts",
            "",
            f"- supported: {int(counts.get('supported', 0) or 0)}",
            f"- refuted: {int(counts.get('refuted', 0) or 0)}",
            f"- inconclusive: {int(counts.get('inconclusive', 0) or 0)}",
        ]
    return "\n".join(lines) + "\n"


def _hypothesis_protocol_decision(
    *,
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    llm: LLMClient | None,
    plan_md: str = "",
    expected_outputs: list[str] | None = None,
) -> StageResult | None:
    _ = config
    if not plan_md:
        plan_md, expected_outputs = _load_plan_and_expected_outputs(run_dir)
    expected_outputs = expected_outputs or []
    if not plan_md.strip():
        return None
    if llm is None:
        return None

    summary = _read_experiment_summary(run_dir)
    analysis = _read_prior_artifact(run_dir, "analysis.md") or ""
    prompt = (
        "Use the experiment plan's decision criteria to decide whether the "
        "research pipeline should PROCEED, PIVOT, or EXTEND. The plan is the "
        "authoritative source for hypotheses, baselines, ablations, metrics, "
        "and decision criteria. Do not invent a different protocol.\n\n"
        "Return Markdown with a clear '## Decision' section containing exactly "
        "one of PROCEED, PIVOT, or EXTEND.\n\n"
        f"# Experiment Plan\n{plan_md}\n\n"
        f"# Expected Outputs\n{json.dumps(expected_outputs, indent=2)}\n\n"
        f"# Experiment Summary\n{json.dumps(summary, indent=2, sort_keys=True)}\n\n"
        f"# Analysis\n{analysis}\n"
    )
    response = _chat_with_prompt(
        llm,
        "You judge research decisions against the experiment plan only.",
        prompt,
        strip_thinking=True,
    )
    decision_md = str(getattr(response, "content", "") or "").strip()
    if not decision_md:
        return None
    decision = _parse_decision(decision_md)

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "decision.md").write_text(decision_md, encoding="utf-8")
    decision_payload = {
        "decision": decision,
        "source": "plan_decision_criteria",
        "expected_outputs": expected_outputs,
        "generated": _utcnow_iso(),
    }
    (stage_dir / "decision_structured.json").write_text(
        json.dumps(decision_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    from researchclaw.pipeline.hypothesis_mode import (
        per_hypothesis_validation_enabled,
    )

    if not per_hypothesis_validation_enabled(config):
        try:
            from researchclaw.pipeline.hypothesis_tree import record_stage15_decision

            record_stage15_decision(run_dir, decision, decision_md, human_edited=False)
        except Exception:
            logger.warning(
                "Failed to record hypothesis tree decision (plan judge)",
                exc_info=True,
            )
    _write_decision_review(
        stage_dir=stage_dir,
        run_dir=run_dir,
        decision=decision,
        decision_md=decision_md,
        llm=llm,
    )
    return StageResult(
        stage=Stage.RESEARCH_DECISION,
        status=StageStatus.DONE,
        artifacts=("decision.md", "decision_structured.json", "decision_review.md"),
        evidence_refs=("stage-15/decision.md",),
        decision=decision,
    )


def _agent_requirements_decision(
    *,
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    llm: LLMClient | None,
) -> StageResult | None:
    """Run the requirements judge and produce a stage-15 decision.

    Returns ``None`` when there are no manifest requirements, when the LLM
    is unavailable, or when the gate decides to fall through to the
    standard decision logic.  Otherwise returns a fully-formed
    :class:`StageResult` with ``decision`` set to ``proceed``.
    """
    requirements = _read_requirements_from_manifest(run_dir)
    if not requirements:
        logger.info("Stage 15: agent-mode but no manifest requirements declared — falling through")
        return None
    if llm is None:
        logger.warning("Stage 15: agent-mode requirements gate requires an LLM client; falling through")
        return None

    from researchclaw.pipeline.requirements_judge import judge_requirements

    summary = _read_experiment_summary(run_dir)
    agent_results = _read_agent_results_canonical(run_dir)
    verdict = judge_requirements(requirements, summary, agent_results, llm)

    retry_count = _read_retry_count(run_dir)
    rerun_triggered = False
    decision = "proceed"
    requirements_unmet = verdict.get("verdict") == "reject"

    decision_md = _format_agent_decision_md(verdict, decision, retry_count, rerun_triggered)
    (stage_dir / "decision.md").write_text(decision_md, encoding="utf-8")
    decision_payload = {
        "decision": decision,
        "verdict": verdict,
        "retry_count": retry_count,
        "rerun_triggered": rerun_triggered,
        "requirements_unmet": requirements_unmet,
        "generated": _utcnow_iso(),
        "source": "agent_requirements_gate",
    }
    (stage_dir / "decision_structured.json").write_text(
        json.dumps(decision_payload, indent=2), encoding="utf-8"
    )
    # Also persist the verdict for the runner / downstream stages to pick up
    (run_dir / "requirements_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    logger.info(
        "Agent requirements gate: verdict=%s, decision=%s, requirements_unmet=%s",
        verdict.get("verdict"), decision, requirements_unmet,
    )
    from researchclaw.pipeline.hypothesis_mode import (
        per_hypothesis_validation_enabled,
    )

    if not per_hypothesis_validation_enabled(config):
        try:
            from researchclaw.pipeline.hypothesis_tree import record_stage15_decision

            record_stage15_decision(
                run_dir, decision, decision_md, human_edited=False
            )
        except Exception:
            logger.warning(
                "Failed to record hypothesis tree decision (agent gate)",
                exc_info=True,
            )
    _write_decision_review(
        stage_dir=stage_dir,
        run_dir=run_dir,
        decision=decision,
        decision_md=decision_md,
        llm=llm,
    )
    return StageResult(
        stage=Stage.RESEARCH_DECISION,
        status=StageStatus.DONE,
        artifacts=("decision.md", "decision_structured.json", "decision_review.md"),
        evidence_refs=("stage-15/decision.md",),
        decision=decision,
    )


def _execute_research_decision(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    agent_decision = _agent_requirements_decision(
        stage_dir=stage_dir, run_dir=run_dir, config=config, llm=llm,
    )
    if agent_decision is not None:
        return agent_decision

    plan_md, expected_outputs = _load_plan_and_expected_outputs(run_dir)
    hypothesis_decision = _hypothesis_protocol_decision(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=config,
        llm=llm,
        plan_md=plan_md,
        expected_outputs=expected_outputs,
    )
    if hypothesis_decision is not None:
        return hypothesis_decision

    analysis = _read_prior_artifact(run_dir, "analysis.md") or ""

    _degenerate_hint = ""

    # Phase 2: Inject experiment diagnosis into decision prompt
    _diagnosis_hint = ""
    _diag_path = run_dir / "experiment_diagnosis.json"
    if _diag_path.exists():
        try:
            _diag_data = json.loads(_diag_path.read_text(encoding="utf-8"))
            _qa = _diag_data.get("quality_assessment", {})
            _mode = _qa.get("mode", "unknown")
            _sufficient = _qa.get("sufficient", False)
            _deficiency_types = _qa.get("deficiency_types", [])
            if not _sufficient:
                _diagnosis_hint = (
                    "\n\n## EXPERIMENT DIAGNOSIS (from automated analysis)\n"
                    f"Quality mode: {_mode}\n"
                    f"Sufficient for full paper: NO\n"
                    f"Issues found: {', '.join(_deficiency_types)}\n\n"
                    "IMPORTANT: The experiment has significant issues. "
                    "Stage 15 may choose PIVOT for paper-level hypothesis changes "
                    "or PROCEED with appropriate quality caveats. Experiment-level "
                    "repair routing is handled before this stage.\n"
                )
                logger.info(
                    "Stage 15: Injected experiment diagnosis — mode=%s, issues=%s",
                    _mode, _deficiency_types,
                )
        except (json.JSONDecodeError, OSError):
            pass

    # Improvement C: Check ablation quality and surface caveats to Stage 15.
    _ablation_quality_hint = ""
    # BUG-DA8-16: Prefer experiment_summary_best.json (promoted best) over
    # alphabetically-last stage-14* (which could be a stale versioned dir)
    _exp_sum_path = run_dir / "experiment_summary_best.json"
    if not _exp_sum_path.is_file():
        _exp_sum_path = None
        for _s14 in sorted(run_dir.glob("stage-14*/experiment_summary.json"), reverse=True):
            _exp_sum_path = _s14
            break
    if _exp_sum_path and _exp_sum_path.is_file():
        try:
            from researchclaw.pipeline.stage_impls._paper_writing import _check_ablation_effectiveness
            _abl_exp = json.loads(_exp_sum_path.read_text(encoding="utf-8"))
            _abl_warnings = _check_ablation_effectiveness(_abl_exp, threshold=0.02)
            if _abl_warnings:
                _trivial_count = sum(1 for w in _abl_warnings if "ineffective" in w.lower() or "trivial" in w.lower())
                _total_abl = max(1, len(_abl_warnings))
                if _trivial_count / _total_abl > 0.5:
                    _ablation_quality_hint = (
                        "\n\n## ABLATION QUALITY ASSESSMENT (CRITICAL)\n"
                        f"{_trivial_count}/{_total_abl} ablations show <2% difference from baseline "
                        f"(trivially similar). This means the ablation design is broken.\n"
                        "Surface this limitation in the paper decision: PIVOT if the "
                        "hypothesis is unsupported, otherwise PROCEED with caveats.\n"
                        "Warnings:\n" + "\n".join(f"- {w}" for w in _abl_warnings) + "\n"
                    )
                    logger.warning(
                        "C: %d/%d ablations trivial; surfacing Stage 15 quality caveat",
                        _trivial_count,
                        _total_abl,
                    )
        except Exception:  # noqa: BLE001
            pass

    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "research_decision")
        sp = _pm.for_stage("research_decision", evolution_overlay=_overlay, analysis=analysis)
        _plan_hint = ""
        if plan_md.strip():
            _plan_hint = (
                "\n\n## EXPERIMENT PLAN DECISION CRITERIA\n"
                "Use this plan as the source of truth for hypotheses, baselines, "
                "ablations, metrics, expected outputs, and decision criteria.\n\n"
                f"{plan_md}\n\n"
                "Expected output paths:\n"
                f"{json.dumps(expected_outputs, indent=2)}\n"
            )
        _user = sp.user + _plan_hint + _degenerate_hint + _diagnosis_hint + _ablation_quality_hint
        resp = _chat_with_prompt(llm, sp.system, _user)
        decision_md = resp.content
    else:
        decision_md = f"""# Research Decision

## Available Decisions
- PROCEED: results are sufficient; continue to paper writing.
- PIVOT: current hypotheses are fundamentally flawed; regenerate hypotheses.
- EXTEND: current hypotheses produced useful evidence; generate follow-up hypotheses.
Experiment-level repair and rerun decisions are handled before this stage.

## Decision
PROCEED

## Justification
Current evidence suggests measurable progress with actionable limitations.

## Next Actions
- Build detailed paper outline
- Expand ablation and uncertainty analysis in writing

Generated: {_utcnow_iso()}
"""
    (stage_dir / "decision.md").write_text(decision_md, encoding="utf-8")

    # --- Extract structured decision ---
    decision = _parse_decision(decision_md)

    # T3.1: Validate decision quality — check for minimum experiment rigor
    decision_payload = _write_decision_structured_json(
        stage_dir, decision_md, decision
    )
    _quality_warnings = decision_payload["quality_warnings"]
    if _quality_warnings:
        logger.warning("T3.1: Decision quality warnings: %s", _quality_warnings)

    logger.info("Research decision: %s", decision)
    from researchclaw.pipeline.hypothesis_mode import (
        per_hypothesis_validation_enabled,
    )

    if not per_hypothesis_validation_enabled(config):
        try:
            from researchclaw.pipeline.hypothesis_tree import record_stage15_decision

            record_stage15_decision(
                run_dir, decision, decision_md, human_edited=False
            )
        except Exception:
            logger.warning("Failed to record hypothesis tree decision", exc_info=True)

    _write_decision_review(
        stage_dir=stage_dir,
        run_dir=run_dir,
        decision=decision,
        decision_md=decision_md,
        llm=llm,
        prompts=prompts,
    )

    return StageResult(
        stage=Stage.RESEARCH_DECISION,
        status=StageStatus.DONE,
        artifacts=("decision.md", "decision_structured.json", "decision_review.md"),
        evidence_refs=("stage-15/decision.md",),
        decision=decision,
    )
