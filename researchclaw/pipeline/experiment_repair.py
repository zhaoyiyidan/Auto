"""Experiment diagnosis prompt and summary helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from researchclaw.pipeline.experiment_diagnosis import (
    DeficiencyType,
    ExperimentDiagnosis,
)
from researchclaw.pipeline._helpers import _parse_metrics_from_stdout

logger = logging.getLogger(__name__)

MAX_REPAIR_CYCLES = 3


# ---------------------------------------------------------------------------
# Repair prompt generation
# ---------------------------------------------------------------------------


def build_repair_prompt(
    diagnosis: ExperimentDiagnosis,
    original_code: dict[str, str],
    experiment_plan: dict | None = None,
    time_budget_sec: int = 2400,
) -> str:
    """Build a structured repair prompt for the workspace code agent.

    Parameters
    ----------
    diagnosis:
        The structured diagnosis from ``diagnose_experiment()``.
    original_code:
        Mapping of filename → source code for the current experiment.
    experiment_plan:
        The experiment design plan.
    time_budget_sec:
        Available time budget for the experiment.

    Returns
    -------
    str
        A formatted prompt suitable for the persistent workspace code agent.
    """
    sections: list[str] = []

    sections.append("# EXPERIMENT REPAIR TASK\n")
    sections.append(
        "The previous experiment run had failures. Your job is to fix "
        "the specific issues identified below. Do NOT rewrite from scratch — "
        "fix ONLY the identified problems.\n"
    )

    # Diagnosis section
    sections.append(diagnosis.to_repair_prompt())

    # Scope reduction guidance
    if any(d.type == DeficiencyType.TIME_GUARD_DOMINANT for d in diagnosis.deficiencies):
        n_planned = diagnosis.total_planned
        n_completed = len(diagnosis.conditions_completed)
        max_conditions = max(3, n_completed + 1)
        sections.append(
            f"\n## SCOPE REDUCTION REQUIRED\n"
            f"The experiment had {n_planned} conditions but only {n_completed} "
            f"completed within the time budget of {time_budget_sec}s.\n"
            f"**Reduce to at most {max_conditions} conditions:**\n"
            f"1. Keep the BASELINE condition (no modification)\n"
            f"2. Keep the PROPOSED method (paper's main contribution)\n"
            f"3. Keep 1 ablation (remove most impactful component)\n"
            f"4. Remove all other conditions\n"
            f"5. Reduce epochs by 30-50% if still tight on time\n"
            f"6. Reduce seeds from 3 to 2 if needed\n"
        )

    # Dependency fixes
    dep_issues = [d for d in diagnosis.deficiencies if d.type == DeficiencyType.MISSING_DEPENDENCY]
    if dep_issues:
        sections.append("\n## DEPENDENCY FIXES\n")
        sections.append("Add these to requirements.txt:\n")
        for d in dep_issues:
            # Extract package name from description
            sections.append(f"- {d.description}")

    # Original code
    sections.append("\n## CURRENT CODE (fix in-place)\n")
    for filename, content in sorted(original_code.items()):
        # Truncate very long files
        if len(content) > 5000:
            content = content[:5000] + "\n... (truncated)"
        sections.append(f"### {filename}\n```python\n{content}\n```\n")

    # Constraints
    sections.append(
        f"\n## CONSTRAINTS\n"
        f"- Time budget: {time_budget_sec} seconds total\n"
        f"- Pre-cached datasets: CIFAR-10, CIFAR-100, MNIST, FashionMNIST, STL-10 at /opt/datasets\n"
        f"- Every condition MUST output: condition=CONDNAME metric=VALUE\n"
        f"- The code must run without errors for at least 1 seed per condition\n"
    )

    # Workspace-agent instructions
    sections.append(
        "\n## WORKSPACE AGENT INSTRUCTIONS\n"
        "- Modify the configured workspace repository directly.\n"
        "- Update run_manifest.json so the harness can submit the experiment.\n"
        "- Make the final git commit after all task changes are ready, including run_manifest.json.\n"
        "- Set run_manifest.json code_commit to final HEAD; amend the same commit if needed.\n"
        "- Finish by verifying `git status --porcelain` is empty.\n"
        "- Do not submit the job yourself; the ResearchClaw harness owns submission.\n"
        "- Do not return pasted source files as markdown code blocks.\n"
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Best results selection
# ---------------------------------------------------------------------------


def select_best_results(
    run_dir: Path,
    cycle_history: list[Any],
) -> dict | None:
    """Select the best experiment_summary across all repair cycles.

    Looks for experiment_summary.json files in versioned stage directories
    and returns the one with the best primary metric / most conditions.

    Returns None if no valid summary found.
    """
    candidates: list[tuple[float, int, dict]] = []

    # Check main stage-14
    main_summary = _try_load_summary(run_dir / "stage-14" / "experiment_summary.json")
    if main_summary:
        score = _summary_quality_score(main_summary)
        candidates.append((score, 0, main_summary))

    # Check repair versions
    for i in range(1, MAX_REPAIR_CYCLES + 1):
        path = run_dir / f"stage-14_repair_v{i}" / "experiment_summary.json"
        summary = _try_load_summary(path)
        if summary:
            score = _summary_quality_score(summary)
            candidates.append((score, i, summary))

    if not candidates:
        return None

    # Sort by quality score (descending)
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_cycle, best_summary = candidates[0]
    logger.info(
        "Best experiment results from cycle %d (score=%.2f)", best_cycle, best_score
    )
    return best_summary


def _try_load_summary(path: Path) -> dict | None:
    """Try to load and parse an experiment_summary.json."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _summary_quality_score(summary: dict) -> float:
    """Compute a simple quality score for ranking summaries.

    Higher = better. Considers:
    - Number of completed conditions (×10)
    - Whether primary_metric is non-NaN (×5)
    - Number of metric keys (×1)
    """
    import math

    score = 0.0
    n_conditions = len(summary.get("condition_summaries", {}))
    score += n_conditions * 10.0

    pm = summary.get("best_run", {}).get("metrics", {}).get("primary_metric")
    if isinstance(pm, (int, float)) and math.isfinite(pm):
        score += 5.0

    n_keys = summary.get("total_metric_keys", 0)
    score += n_keys * 1.0

    return score


# ---------------------------------------------------------------------------
# Helper: load experiment artifacts
# ---------------------------------------------------------------------------


def _load_experiment_summary(run_dir: Path) -> dict | None:
    """Load the most recent experiment_summary.json."""
    for candidate in sorted(run_dir.glob("stage-14*/experiment_summary.json"), reverse=True):
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _load_experiment_code(run_dir: Path) -> dict[str, str]:
    """Load workspace manifest context from the most recent stage directory."""
    code: dict[str, str] = {}

    for manifest in sorted(run_dir.glob("stage-10*/run_manifest.json"), reverse=True):
        try:
            code[str(manifest.relative_to(run_dir))] = manifest.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass
        if code:
            return code

    for task_spec in sorted(run_dir.glob("stage-09*/task_spec.yaml"), reverse=True):
        try:
            code[str(task_spec.relative_to(run_dir))] = task_spec.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass
        if code:
            return code

    return code


def _load_experiment_plan(run_dir: Path) -> dict | None:
    """Load experiment task spec from stage-09."""
    for candidate in sorted(run_dir.glob("stage-09*/task_spec.yaml"), reverse=True):
        try:
            import yaml as _yaml_repair

            data = _yaml_repair.safe_load(candidate.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, UnicodeDecodeError):
            continue
    return None


def _collect_experiment_output(run_dir: Path) -> tuple[str, str]:
    """Collect stdout/stderr from execution records."""
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    for record_path in sorted(run_dir.glob("stage-12*/execution_record.json"), reverse=True):
        try:
            data = json.loads(record_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                stdout_parts.append(str(data.get("stdout", "")))
                stderr_parts.append(str(data.get("stderr", "")))
        except (json.JSONDecodeError, OSError):
            continue

    for stage_dir in sorted(run_dir.glob("stage-12*")):
        runs_dir = stage_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_file in sorted(runs_dir.glob("*.json"))[:5]:
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    stdout_parts.append(str(data.get("stdout", "")))
                    stderr_parts.append(str(data.get("stderr", "")))
            except (json.JSONDecodeError, OSError):
                continue

    return "\n".join(stdout_parts).strip(), "\n".join(stderr_parts).strip()


def _build_experiment_summary_from_run(
    run_result: dict,
    code: dict[str, str],
) -> dict:
    """Build an experiment_summary.json from a single execution result.

    Parses condition-level metrics from stdout and builds the standard
    summary format expected by ``assess_experiment_quality()``.
    """
    metrics = run_result.get("metrics", {})
    stdout = run_result.get("stdout", "")

    # Also parse metrics from stdout if the submitter did not capture them.
    if not metrics and stdout:
        metrics = _parse_metrics_from_stdout(stdout)

    # Group metrics by condition
    condition_summaries: dict[str, dict] = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        parts = key.split("/")
        if len(parts) >= 3:
            # Format: condition_name/seed/metric_name
            cond_name = parts[0]
            metric_name = parts[-1]
            if cond_name not in condition_summaries:
                condition_summaries[cond_name] = {"metrics": {}, "seeds": {}}
            condition_summaries[cond_name]["metrics"][metric_name] = value
            seed_key = "/".join(parts[1:-1])
            condition_summaries[cond_name]["seeds"].setdefault(seed_key, {})[metric_name] = value
        elif len(parts) == 2:
            # BUG-199: Stage 13 repair output can produce 2-part keys
            # (condition_name/metric_name) without a seed component.
            # Treat as a single-seed result.
            cond_name, metric_name = parts
            if cond_name not in condition_summaries:
                condition_summaries[cond_name] = {"metrics": {}, "seeds": {}}
            condition_summaries[cond_name]["metrics"][metric_name] = value
            condition_summaries[cond_name]["seeds"].setdefault("0", {})[metric_name] = value
        elif len(parts) == 1:
            # Top-level metric like "primary_metric"
            pass

    # Compute per-condition mean metrics
    for cond_name, cdata in condition_summaries.items():
        seeds = cdata.get("seeds", {})
        if seeds:
            cdata["n_seeds"] = len(seeds)
            # Average each metric across seeds
            all_metrics: dict[str, list[float]] = {}
            for seed_data in seeds.values():
                for mk, mv in seed_data.items():
                    if isinstance(mv, (int, float)):
                        all_metrics.setdefault(mk, []).append(float(mv))
            for mk, values in all_metrics.items():
                if values:
                    cdata["metrics"][mk] = sum(values) / len(values)
        # Remove seeds from final output (not standard format)
        cdata.pop("seeds", None)

    return {
        "condition_summaries": condition_summaries,
        "best_run": {
            "metrics": metrics,
            "status": "completed" if run_result.get("returncode") == 0 else "failed",
            "stdout": stdout[:5000],
            "stderr": run_result.get("stderr", "")[:2000],
        },
        "metrics_summary": {},
        "total_conditions": len(condition_summaries),
        "total_metric_keys": len(metrics),
    }
