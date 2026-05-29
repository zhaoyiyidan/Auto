"""Experiment Repair Loop — diagnose and delegate fixes to the workspace agent.

Orchestrates the cycle:
  1. Diagnose failures (``experiment_diagnosis.py``)
  2. Send a targeted repair prompt to the persistent workspace code agent
  3. Let the normal manifest validation and submitter stages execute results

Integrates between Stage 14 (result_analysis) and Stage 15 (research_decision).
"""

from __future__ import annotations

import json
import logging
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from researchclaw.pipeline.experiment_diagnosis import (
    DeficiencyType,
    ExperimentDiagnosis,
    ExperimentQualityAssessment,
    PaperMode,
    assess_experiment_quality,
    diagnose_experiment,
)
from researchclaw.pipeline._helpers import _parse_metrics_from_stdout

logger = logging.getLogger(__name__)

MAX_REPAIR_CYCLES = 3


@dataclass
class RepairCycleResult:
    """Result of one repair cycle."""

    cycle: int
    diagnosis: ExperimentDiagnosis
    repair_applied: bool = False
    repair_description: str = ""
    new_assessment: ExperimentQualityAssessment | None = None
    error: str = ""


@dataclass
class ExperimentRepairResult:
    """Final result of the entire repair loop."""

    success: bool  # True if experiment is now sufficient for full_paper
    total_cycles: int
    final_mode: PaperMode
    final_assessment: ExperimentQualityAssessment | None = None
    cycle_history: list[RepairCycleResult] = field(default_factory=list)
    best_experiment_summary: dict | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "total_cycles": self.total_cycles,
            "final_mode": self.final_mode.value,
            "cycle_history": [
                {
                    "cycle": cr.cycle,
                    "repair_applied": cr.repair_applied,
                    "repair_description": cr.repair_description,
                    "error": cr.error,
                    "diagnosis_summary": cr.diagnosis.summary if cr.diagnosis else "",
                }
                for cr in self.cycle_history
            ],
        }


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
        "- Commit the code changes with git.\n"
        "- Update run_manifest.json so the harness can submit the experiment.\n"
        "- Do not submit the job yourself; the ResearchClaw harness owns submission.\n"
        "- Do not return pasted source files as markdown code blocks.\n"
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Best results selection
# ---------------------------------------------------------------------------


def select_best_results(
    run_dir: Path,
    cycle_history: list[RepairCycleResult],
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
# Full repair loop
# ---------------------------------------------------------------------------


def run_repair_loop(
    run_dir: Path,
    config: Any,
    run_id: str = "",
) -> ExperimentRepairResult:
    """Execute a workspace-native repair request.

    After Stage 14 diagnosis finds quality issues, this function builds one
    targeted prompt for the persistent workspace code agent. It does not parse
    generated code, copy experiment directories, or execute jobs; the normal
    manifest validation and harness stages own those steps.

    Parameters
    ----------
    run_dir:
        Path to the pipeline run directory (contains stage-* subdirs).
    config:
        RCConfig instance with experiment and workspace-agent settings.
    run_id:
        Pipeline run ID for logging.

    Returns
    -------
    ExperimentRepairResult
    """
    repair_cfg = config.experiment.repair

    # Load initial experiment summary
    summary = _load_experiment_summary(run_dir)
    if not summary:
        logger.warning("[%s] Repair loop: no experiment_summary.json found", run_id)
        return ExperimentRepairResult(
            success=False, total_cycles=0, final_mode=PaperMode.TECHNICAL_REPORT,
        )

    # Initial quality assessment — pass user-configured thresholds
    ref_log = _load_refinement_log(run_dir)
    _min_cond = getattr(repair_cfg, "min_conditions", 3)
    qa = assess_experiment_quality(summary, ref_log, min_conditions=_min_cond)
    if qa.sufficient:
        logger.info("[%s] Repair loop: experiment already sufficient (%s)", run_id, qa.mode.value)
        return ExperimentRepairResult(
            success=True, total_cycles=0, final_mode=qa.mode,
            final_assessment=qa, best_experiment_summary=summary,
        )

    workspace_enabled = getattr(
        getattr(config.experiment, "workspace_agent", None),
        "enabled",
        False,
    )

    # Load legacy code snapshot only as extra context for the repair prompt.
    code = _load_experiment_code(run_dir)
    if not code and not workspace_enabled:
        logger.warning("[%s] Repair loop: no experiment code found", run_id)
        return ExperimentRepairResult(
            success=False, total_cycles=0, final_mode=qa.mode,
        )

    # Collect stdout/stderr for diagnosis
    stdout, stderr = _collect_experiment_output(run_dir)

    # Load experiment plan
    plan = _load_experiment_plan(run_dir)

    cycle_history: list[RepairCycleResult] = []
    best_summary = summary
    best_mode = qa.mode
    max_cycles = min(repair_cfg.max_cycles, MAX_REPAIR_CYCLES)
    loop_start = _time.monotonic()
    prior_diagnoses: list[dict] = []

    for cycle in range(1, max_cycles + 1):
        logger.info("[%s] Repair cycle %d/%d starting...", run_id, cycle, max_cycles)
        print(f"[{run_id}] Repair cycle {cycle}/{max_cycles}...")

        # 1. Diagnose current state
        diag = diagnose_experiment(
            experiment_summary=summary,
            experiment_plan=plan,
            refinement_log=ref_log,
            stdout=stdout,
            stderr=stderr,
            prior_diagnoses=prior_diagnoses or None,
        )
        prior_diagnoses.append(diag.to_dict() if hasattr(diag, "to_dict") else {})

        # 2. Build repair prompt
        repair_prompt = build_repair_prompt(
            diag, code, experiment_plan=plan,
            time_budget_sec=config.experiment.time_budget_sec,
        )

        # 3. Ask the workspace code agent to modify the repository.
        fixed_code = _get_repaired_code(
            repair_prompt, code, None, config, run_dir, cycle,
        )

        if not fixed_code:
            cycle_result = RepairCycleResult(
                cycle=cycle, diagnosis=diag,
                repair_applied=False,
                error="Failed to generate repaired code",
            )
            cycle_history.append(cycle_result)
            logger.warning("[%s] Repair cycle %d: code generation failed", run_id, cycle)
            break

        cycle_result = RepairCycleResult(
            cycle=cycle,
            diagnosis=diag,
            repair_applied=True,
            repair_description="Workspace agent repair requested; rerun stages 11-14 to evaluate results",
        )
        cycle_history.append(cycle_result)

        logger.info(
            "[%s] Repair cycle %d: workspace agent accepted repair request",
            run_id, cycle,
        )
        print(f"[{run_id}] Repair cycle {cycle}: workspace agent repair requested")
        break

    elapsed = _time.monotonic() - loop_start
    logger.info(
        "[%s] Repair loop completed: %d cycles in %.1fs, current mode=%s",
        run_id, len(cycle_history), elapsed, best_mode.value,
    )

    return ExperimentRepairResult(
        success=False,
        total_cycles=len(cycle_history),
        final_mode=best_mode,
        cycle_history=cycle_history,
        best_experiment_summary=best_summary,
    )


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


def _load_refinement_log(run_dir: Path) -> dict | None:
    """Load the most recent refinement_log.json."""
    for candidate in sorted(run_dir.glob("stage-13*/refinement_log.json"), reverse=True):
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _load_experiment_code(run_dir: Path) -> dict[str, str]:
    """Load experiment code from the most recent stage directory.

    Prefers: stage-13/experiment_final/ → stage-10/experiment/ → stage-10/*.py
    """
    code: dict[str, str] = {}

    # Try refined code first
    for refine_dir in sorted(run_dir.glob("stage-13*/experiment_final"), reverse=True):
        if refine_dir.is_dir():
            for py_file in sorted(refine_dir.glob("*.py")):
                try:
                    code[py_file.name] = py_file.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    pass
            # Also grab requirements.txt, setup.py
            for extra in ("requirements.txt", "setup.py"):
                extra_path = refine_dir / extra
                if extra_path.exists():
                    try:
                        code[extra] = extra_path.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        pass
            if code:
                return code

    # Fall back to stage-10 experiment directory
    for exp_dir in sorted(run_dir.glob("stage-10*/experiment"), reverse=True):
        if exp_dir.is_dir():
            for py_file in sorted(exp_dir.glob("*.py")):
                try:
                    code[py_file.name] = py_file.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    pass
            for extra in ("requirements.txt", "setup.py"):
                extra_path = exp_dir / extra
                if extra_path.exists():
                    try:
                        code[extra] = extra_path.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        pass
            if code:
                return code

    # Last resort: any .py files in stage-10*
    for stage_dir in sorted(run_dir.glob("stage-10*"), reverse=True):
        for py_file in sorted(stage_dir.glob("*.py")):
            try:
                code[py_file.name] = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                pass
        if code:
            return code

    return code


def _load_experiment_plan(run_dir: Path) -> dict | None:
    """Load experiment plan from stage-09."""
    for candidate in sorted(run_dir.glob("stage-09*/experiment_design.json"), reverse=True):
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _collect_experiment_output(run_dir: Path) -> tuple[str, str]:
    """Collect stdout/stderr from experiment runs."""
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    for stage_dir in sorted(run_dir.glob("stage-14*")):
        runs_dir = stage_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_file in sorted(runs_dir.glob("*.json"))[:5]:
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    stdout_parts.append(data.get("stdout", ""))
                    stderr_parts.append(data.get("stderr", ""))
            except (json.JSONDecodeError, OSError):
                continue

    return "\n".join(stdout_parts).strip(), "\n".join(stderr_parts).strip()


# ---------------------------------------------------------------------------
# Helper: request workspace-native repair
# ---------------------------------------------------------------------------


def _get_repaired_code(
    repair_prompt: str,
    current_code: dict[str, str],
    llm: Any,
    config: Any,
    run_dir: Path,
    cycle: int,
) -> dict[str, str] | None:
    """Get repaired code via the workspace-native code agent.

    Returns merged code dict (current + repaired files) or None on failure.
    """
    if getattr(getattr(config.experiment, "workspace_agent", None), "enabled", False):
        result = _repair_via_workspace_agent(
            repair_prompt, current_code, llm, config, run_dir, cycle,
        )
        if result:
            return result
        logger.info("Workspace-native repair unavailable")

    return None


def _repair_via_workspace_agent(
    repair_prompt: str,
    current_code: dict[str, str],
    llm: Any,
    config: Any,
    run_dir: Path,
    cycle: int,
) -> dict[str, str] | None:
    """Attempt repair through the workspace-native agent implement path."""
    try:
        from researchclaw.experiment.workspace_agent import create_workspace_agent
        from researchclaw.pipeline.workspace_orchestrator import run_workspace_agent_implement

        workspace_cfg = config.experiment.workspace_agent
        stage_dir = run_dir / f"stage-14-workspace-repair-v{cycle}"
        prompt = (
            "You are a workspace-native code agent repairing an existing git "
            "repository. Apply the requested repair in place, commit your code "
            "changes with git, and write the run manifest for ResearchClaw's "
            "submitter. Do not submit the job yourself.\n\n"
            f"MANIFEST FILENAME: {workspace_cfg.manifest_filename}\n\n"
            f"REPAIR TASK:\n{repair_prompt}\n\n"
            f"CURRENT RESEARCHCLAW CODE SNAPSHOT FILES:\n{sorted(current_code.keys())}\n"
        )
        agent = create_workspace_agent(config, llm=llm)
        result = run_workspace_agent_implement(
            workspace_path=Path(workspace_cfg.workspace_path),
            run_dir=stage_dir,
            stage=14,
            agent=agent,
            prompt=prompt,
            timeout_sec=workspace_cfg.timeout_sec,
            iteration=cycle,
            close_policy=getattr(workspace_cfg, "close_policy", "keep"),
        )
        if not result.ok:
            logger.warning("Workspace-native repair failed: %s", result.error)
            return None
        merged = dict(current_code)
        merged.update(_collect_workspace_text_files(Path(workspace_cfg.workspace_path)))
        return merged
    except Exception as exc:  # noqa: BLE001
        logger.warning("Workspace-native repair failed: %s", exc)
        return None


def _collect_workspace_text_files(workspace: Path) -> dict[str, str]:
    """Collect root-level text files for legacy repair-loop compatibility."""
    files: dict[str, str] = {}
    for path in sorted(workspace.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix not in (".py", ".txt", ".yaml", ".yml", ".json", ".cfg", ".ini", ".sh"):
            continue
        try:
            files[path.name] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files


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
            # BUG-199: Stage 13 refinement produces 2-part keys
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
