from __future__ import annotations

from collections.abc import Callable
import json
import importlib
import logging
import math
import os
import shutil
import tempfile
import threading
import time as _time
from pathlib import Path

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.evolution import EvolutionStore, extract_lessons
from researchclaw.experiment.metric_resolution import resolve_experiment_metric
from researchclaw.knowledge.base import write_stage_to_kb
from researchclaw.notify.pipeline import notify_terminal_failure
from researchclaw.pipeline.executor import StageResult, execute_stage
from researchclaw.pipeline.hypothesis_mode import per_hypothesis_validation_enabled
from researchclaw.pipeline.stages import (
    DECISION_ROLLBACK,
    EXPERIMENT_ROUTE_TARGETS,
    MAX_DECISION_PIVOTS,
    MAX_EXPERIMENT_ITERATIONS,
    NONCRITICAL_STAGES,
    STAGE_SEQUENCE,
    Stage,
    StageStatus,
)


def _utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _should_start(stage: Stage, from_stage: Stage, started: bool) -> bool:
    if started:
        return True
    return stage == from_stage


def _build_pipeline_summary(
    *,
    run_id: str,
    results: list[StageResult],
    from_stage: Stage,
    run_dir: Path | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "run_id": run_id,
        "stages_executed": len(results),
        "stages_done": sum(1 for item in results if item.status == StageStatus.DONE),
        "stages_paused": sum(
            1 for item in results if item.status == StageStatus.PAUSED
        ),
        "stages_blocked": sum(
            1 for item in results if item.status == StageStatus.BLOCKED_APPROVAL
        ),
        "stages_failed": sum(
            1 for item in results if item.status == StageStatus.FAILED
        ),
        "degraded": any(r.decision == "degraded" for r in results),
        "from_stage": int(from_stage),
        "final_stage": int(results[-1].stage) if results else int(from_stage),
        "final_status": results[-1].status.value if results else "no_stages",
        "generated": _utcnow_iso(),
        "content_metrics": _collect_content_metrics(run_dir),
    }
    return summary


def _write_pipeline_summary(run_dir: Path, summary: dict[str, object]) -> None:
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _write_checkpoint(
    run_dir: Path, stage: Stage, run_id: str,
    adapters: "AdapterBundle | None" = None,
) -> None:
    """Write checkpoint atomically via temp file + rename to prevent corruption."""
    checkpoint: dict[str, object] = {
        "last_completed_stage": int(stage),
        "last_completed_name": stage.name,
        "run_id": run_id,
        "timestamp": _utcnow_iso(),
    }

    # Embed HITL session data if available
    if adapters is not None:
        hitl_session = getattr(adapters, "hitl", None)
        if hitl_session is not None:
            try:
                checkpoint["hitl"] = hitl_session.hitl_checkpoint_data()
            except Exception:
                pass
    target = run_dir / "checkpoint.json"
    fd, tmp_path = tempfile.mkstemp(dir=run_dir, suffix=".tmp", prefix="checkpoint_")
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(checkpoint, indent=2))
        Path(tmp_path).replace(target)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _write_heartbeat(run_dir: Path, stage: Stage, run_id: str) -> None:
    """Write heartbeat file for sentinel watchdog monitoring."""
    import os

    heartbeat = {
        "pid": os.getpid(),
        "last_stage": int(stage),
        "last_stage_name": stage.name,
        "run_id": run_id,
        "timestamp": _utcnow_iso(),
    }
    (run_dir / "heartbeat.json").write_text(
        json.dumps(heartbeat, indent=2), encoding="utf-8"
    )


def read_checkpoint(run_dir: Path) -> Stage | None:
    """Read checkpoint and return the NEXT stage to execute, or None if no checkpoint."""
    cp_path = run_dir / "checkpoint.json"
    if not cp_path.exists():
        return None
    try:
        data = json.loads(cp_path.read_text(encoding="utf-8"))
        last_num = data.get("last_completed_stage")
        if last_num is None:
            return None
        for i, stage in enumerate(STAGE_SEQUENCE):
            if int(stage) == last_num:
                if i + 1 < len(STAGE_SEQUENCE):
                    return STAGE_SEQUENCE[i + 1]
                return None
        return None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def resume_from_checkpoint(
    run_dir: Path, default_stage: Stage = Stage.TOPIC_INIT
) -> Stage:
    """Resolve the stage to resume from using checkpoint metadata."""
    next_stage = read_checkpoint(run_dir)
    return next_stage if next_stage is not None else default_stage


def _collect_content_metrics(run_dir: Path | None) -> dict[str, object]:
    """Collect content authenticity metrics from stage outputs."""
    metrics: dict[str, object] = {
        "template_ratio": None,
        "citation_verify_score": None,
        "total_citations": None,
        "verified_citations": None,
        "degraded_sources": [],
    }
    if run_dir is None:
        return metrics

    draft_path = run_dir / "stage-17" / "paper_draft.md"
    if draft_path.exists():
        try:
            quality_module = importlib.import_module("researchclaw.quality")
            compute_template_ratio = quality_module.compute_template_ratio
            text = draft_path.read_text(encoding="utf-8")
            metrics["template_ratio"] = round(compute_template_ratio(text), 4)
        except (
            AttributeError,
            ModuleNotFoundError,
            UnicodeDecodeError,
            OSError,
            ValueError,
            TypeError,
        ):
            pass

    verify_path = run_dir / "stage-23" / "verification_report.json"
    if verify_path.exists():
        try:
            vdata = json.loads(verify_path.read_text(encoding="utf-8"))
            if isinstance(vdata, dict):
                summary = vdata.get("summary", vdata)
                total = summary.get("total", 0) if isinstance(summary, dict) else None
                verified = summary.get("verified", 0) if isinstance(summary, dict) else None
                if isinstance(total, int | float) and isinstance(verified, int | float):
                    total_num = int(total)
                    verified_num = int(verified)
                    metrics["total_citations"] = total_num
                    metrics["verified_citations"] = verified_num
                    if total_num > 0:
                        metrics["citation_verify_score"] = round(
                            verified_num / total_num, 4
                        )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    return metrics


logger = logging.getLogger(__name__)


def _notify_stage_complete(
    on_stage_complete: Callable[[Stage], None] | None,
    stage: Stage,
) -> None:
    if on_stage_complete is None:
        return
    try:
        on_stage_complete(stage)
    except Exception:
        logger.debug("Stage completion callback failed", exc_info=True)


def _run_experiment_diagnosis(run_dir: Path, config: RCConfig, run_id: str) -> None:
    """Run experiment diagnosis after Stage 14 and save reports.

    Produces:
    - ``run_dir/experiment_diagnosis.json`` — structured diagnosis + quality assessment
    """
    try:
        from researchclaw.pipeline.experiment_diagnosis import (
            diagnose_experiment,
            assess_experiment_quality,
        )

        # Find the most recent stage-14 experiment_summary.json
        summary_path = None
        for candidate in sorted(run_dir.glob("stage-14*/experiment_summary.json")):
            summary_path = candidate
        if not summary_path or not summary_path.exists():
            return

        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        # Collect stdout/stderr from execution records and any retained run payloads.
        stdout, stderr = "", ""
        for record_path in sorted(run_dir.glob("stage-12*/execution_record.json"), reverse=True)[:2]:
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(record, dict):
                stdout += str(record.get("stdout", "")) + "\n"
                stderr += str(record.get("stderr", "")) + "\n"
        runs_dir = None
        for _candidate_runs in sorted(run_dir.glob("stage-1[23]*/runs"), reverse=True):
            if _candidate_runs.is_dir():
                runs_dir = _candidate_runs
                break
        if runs_dir is None:
            runs_dir = summary_path.parent / "runs"
        if runs_dir.is_dir():
            for run_file in sorted(runs_dir.glob("*.json"))[:5]:
                try:
                    run_data = json.loads(run_file.read_text(encoding="utf-8"))
                    if isinstance(run_data, dict):
                        stdout += run_data.get("stdout", "") + "\n"
                        stderr += run_data.get("stderr", "") + "\n"
                except (json.JSONDecodeError, OSError):
                    continue

        # Load experiment plan from stage-09
        plan = None
        for candidate in sorted(run_dir.glob("stage-09*/plan.md")):
            try:
                expected_path = candidate.parent / "expected_outputs.json"
                expected_outputs = {}
                if expected_path.is_file():
                    try:
                        expected_outputs = json.loads(expected_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        expected_outputs = {}
                plan = {
                    "plan_md": candidate.read_text(encoding="utf-8"),
                    "expected_outputs": expected_outputs,
                }
            except Exception:
                pass

        # Run diagnosis
        diag = diagnose_experiment(
            experiment_summary=summary,
            experiment_plan=plan,
            stdout=stdout.strip(),
            stderr=stderr.strip(),
        )

        # Run quality assessment
        qa = assess_experiment_quality(summary, experiment_plan=plan)

        # Save diagnosis report
        diag_report = {
            "diagnosis": diag.to_dict(),
            "quality_assessment": {
                "mode": qa.mode.value,
                "sufficient": qa.sufficient,
                "repair_possible": qa.repair_possible,
                "deficiency_types": [d.type.value for d in qa.deficiencies],
            },
            "repair_needed": not qa.sufficient,
            "generated": _utcnow_iso(),
        }
        (run_dir / "experiment_diagnosis.json").write_text(
            json.dumps(diag_report, indent=2), encoding="utf-8"
        )

        if not qa.sufficient:
            logger.info(
                "[%s] Experiment diagnosis: mode=%s, deficiencies=%d — repair needed",
                run_id, qa.mode.value, len(diag.deficiencies),
            )
            print(
                f"[{run_id}] Experiment diagnosis: {qa.mode.value} "
                f"({len(diag.deficiencies)} issues found, repair needed)"
            )
        else:
            logger.info(
                "[%s] Experiment diagnosis: mode=%s, sufficient=True — quality OK",
                run_id, qa.mode.value,
            )
            print(f"[{run_id}] Experiment diagnosis: {qa.mode.value} — quality OK")

    except Exception as exc:
        logger.warning("Experiment diagnosis failed: %s", exc)


def _read_experiment_iterations(run_dir: Path) -> list[dict[str, object]]:
    history_path = run_dir / "experiment_loop_history.json"
    if not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(payload, dict) and isinstance(payload.get("iterations"), list):
        return [item for item in payload["iterations"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _max_experiment_iterations(config: RCConfig) -> int:
    repair_cfg = getattr(getattr(config, "experiment", None), "repair", None)
    configured = getattr(repair_cfg, "max_cycles", MAX_EXPERIMENT_ITERATIONS)
    try:
        return max(0, int(configured))
    except (TypeError, ValueError):
        return MAX_EXPERIMENT_ITERATIONS


def _record_experiment_iteration(
    run_dir: Path,
    *,
    route: str,
    jump: str,
    target: Stage,
) -> int:
    iterations = _read_experiment_iterations(run_dir)
    attempt = len(iterations) + 1
    iterations.append(
        {
            "iteration": attempt,
            "route": route,
            "jump": jump,
            "target_stage": target.name,
            "target_stage_num": int(target),
            "timestamp": _utcnow_iso(),
        }
    )
    payload = {
        "schema_version": "researchclaw.experiment_loop_history.v1",
        "iterations": iterations,
    }
    (run_dir / "experiment_loop_history.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return attempt


def _route_to_stage(route: str) -> Stage | None:
    return EXPERIMENT_ROUTE_TARGETS.get(route)


def _pause_for_experiment_hitl(
    run_dir: Path,
    run_id: str,
    stage: Stage,
    route: str,
) -> StageResult:
    iterations = _read_experiment_iterations(run_dir)
    payload = {
        "schema_version": "researchclaw.experiment_hitl_required.v1",
        "run_id": run_id,
        "stage": int(stage),
        "stage_name": stage.name,
        "route": route,
        "iterations": len(iterations),
        "timestamp": _utcnow_iso(),
    }
    (run_dir / "experiment_hitl_required.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return StageResult(
        stage=stage,
        status=StageStatus.PAUSED,
        artifacts=("experiment_hitl_required.json",),
        error=(
            f"Experiment repair budget exhausted while routing {route}; "
            "human review required"
        ),
        decision="hitl",
    )


_BACKWARD_ROUTES: frozenset[str] = frozenset({"fix_code", "rerun", "hitl", "abort"})


def is_forward_progress(result: StageResult) -> bool:
    """Return True when a result advances the pipeline high-water mark."""
    if result.status is not StageStatus.DONE:
        return False
    if result.stage in (
        Stage.MANIFEST_VALIDATE_AND_PREPARE,
        Stage.EXPERIMENT_ROUTE_DECISION,
    ):
        return result.decision not in _BACKWARD_ROUTES
    return True


def _recurse_pipeline(*, from_stage: Stage, **kwargs: object) -> list[StageResult]:
    """Re-enter the global stage loop from ``from_stage`` with one run context."""
    return execute_pipeline(from_stage=from_stage, **kwargs)


def _run_per_hypothesis_coordinator(
    *,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    split_from_stage8: bool,
) -> bool:
    if not per_hypothesis_validation_enabled(config):
        return False
    from researchclaw.pipeline.hypothesis_coordinator import (
        HypothesisValidationCoordinator,
    )

    coordinator = HypothesisValidationCoordinator(run_dir)
    if split_from_stage8:
        hypotheses_md = _read_stage_artifact(
            run_dir,
            Stage.HYPOTHESIS_GEN,
            "hypotheses.md",
        )
        if not hypotheses_md.strip():
            return False
        queue_path = run_dir / "hypothesis_tree" / "work_queue.jsonl"
        if not queue_path.exists():
            coordinator.split_and_queue(hypotheses_md)
    elif not (run_dir / "hypothesis_tree").exists():
        return False

    validation = getattr(config, "hypothesis_validation", None)
    coordinator.run_until_queue_empty(
        config=config,
        adapters=adapters,
        max_concurrent=int(getattr(validation, "max_concurrent_branches", 1) or 1),
        max_attempts_per_node=int(
            getattr(validation, "max_attempts_per_node", 1) or 1
        ),
        require_coordinator_gate=bool(
            getattr(validation, "require_coordinator_gate", False)
        ),
    )
    return True


def execute_pipeline(
    *,
    run_dir: Path,
    run_id: str,
    config: RCConfig,
    adapters: AdapterBundle,
    from_stage: Stage = Stage.TOPIC_INIT,
    to_stage: Stage | None = None,
    auto_approve_gates: bool = False,
    stop_on_gate: bool = False,
    skip_noncritical: bool = False,
    kb_root: Path | None = None,
    cancel_event: "threading.Event | None" = None,
    initialize_run_globals: bool = True,
    on_stage_complete: Callable[[Stage], None] | None = None,
) -> list[StageResult]:
    """Execute pipeline stages sequentially from *from_stage* to *to_stage* (inclusive)."""

    results: list[StageResult] = []
    started = False
    total_stages = len(STAGE_SEQUENCE)
    per_hypothesis_validation_ran = False
    skip_per_hypothesis_validation_stages = False

    # Force the domain detector to honor a deployed profile (if any) so
    # every stage picks the same adapter.  Safe no-op when empty.
    if initialize_run_globals:
        try:
            from researchclaw.domains.detector import set_forced_profile
            forced = getattr(config.project, "profile", "") or ""
            set_forced_profile(forced)
        except Exception:  # noqa: BLE001
            pass

    # ── Integration hooks: EventLog, ExperimentMemory ──
    event_log = None
    if initialize_run_globals:
        try:
            from researchclaw.pipeline.event_log import EventLog, EventType, create_event
            event_log = EventLog(log_dir=run_dir)
            event_log.append(create_event(
                EventType.PIPELINE_START, run_id=run_id,
                stages=total_stages, from_stage=int(from_stage),
            ))
        except Exception:
            logger.debug("Event log initialisation skipped")

    exp_memory = None
    if initialize_run_globals:
        try:
            from researchclaw.memory.experiment_memory import ExperimentMemory
            _mem_dir = run_dir / "experiment_memory"
            _mem_dir.mkdir(parents=True, exist_ok=True)
            exp_memory = ExperimentMemory(store_dir=str(_mem_dir))
        except Exception:
            logger.debug("Experiment memory initialisation skipped")

    if (
        int(from_stage) > int(Stage.RESEARCH_DECISION)
        and _run_per_hypothesis_coordinator(
            run_dir=run_dir,
            config=config,
            adapters=adapters,
            split_from_stage8=False,
        )
    ):
        per_hypothesis_validation_ran = True

    forward_kwargs = {
        "run_dir": run_dir,
        "run_id": run_id,
        "config": config,
        "adapters": adapters,
        "to_stage": to_stage,
        "auto_approve_gates": auto_approve_gates,
        "stop_on_gate": stop_on_gate,
        "skip_noncritical": skip_noncritical,
        "kb_root": kb_root,
        "cancel_event": cancel_event,
        "initialize_run_globals": initialize_run_globals,
        "on_stage_complete": on_stage_complete,
    }

    for stage in STAGE_SEQUENCE:
        started = _should_start(stage, from_stage, started)
        if not started:
            continue

        if (
            skip_per_hypothesis_validation_stages
            and Stage.EXPERIMENT_TASK_SPEC <= stage <= Stage.RESEARCH_DECISION
        ):
            continue

        if (
            not per_hypothesis_validation_ran
            and Stage.EXPERIMENT_TASK_SPEC <= stage <= Stage.RESEARCH_DECISION
            and (run_dir / "hypothesis_tree").exists()
            and _run_per_hypothesis_coordinator(
                run_dir=run_dir,
                config=config,
                adapters=adapters,
                split_from_stage8=False,
            )
        ):
            per_hypothesis_validation_ran = True
            skip_per_hypothesis_validation_stages = True
            if to_stage is not None and int(to_stage) <= int(Stage.RESEARCH_DECISION):
                break
            continue

        # ── Check for cancellation before each stage ──
        if cancel_event is not None and cancel_event.is_set():
            logger.info("[%s] Pipeline cancelled before stage %s", run_id, stage.name)
            print(f"[{run_id}] Pipeline cancelled by user.")
            break

        stage_num = int(stage)
        prefix = f"[{run_id}] Stage {stage_num:02d}/{total_stages}"
        print(f"{prefix} {stage.name} — running...")

        # ── Event log: stage start ──
        if event_log:
            try:
                event_log.append(create_event(
                    EventType.STAGE_START, run_id=run_id, stage=stage.name,
                ))
            except Exception:
                pass

        # Ensure the best stage-14 experiment data is promoted before paper
        # writing begins. Recursive PIVOT/EXTEND paths can otherwise leave the
        # latest iteration in place even when an earlier stage-14 is better.
        if stage == Stage.PAPER_OUTLINE:
            _promote_best_stage14(run_dir, config)

        t0 = _time.monotonic()

        result = execute_stage(
            stage,
            run_dir=run_dir,
            run_id=run_id,
            config=config,
            adapters=adapters,
            auto_approve_gates=auto_approve_gates,
        )
        elapsed = _time.monotonic() - t0

        # ── Event log: stage end ──
        if event_log:
            try:
                etype = EventType.STAGE_END if result.status == StageStatus.DONE else EventType.STAGE_FAIL
                event_log.append(create_event(
                    etype, run_id=run_id, stage=stage.name,
                    status=result.status.value, elapsed_sec=round(elapsed, 1),
                    error=result.error,
                ))
            except Exception:
                pass

        # ── ExperimentSpec: generate after design, validate after analysis ──
        if stage == Stage.EXPERIMENT_TASK_SPEC and result.status == StageStatus.DONE:
            try:
                from researchclaw.pipeline.experiment_spec import ExperimentSpec, MetricDef, generate_spec
                spec_text = generate_spec(config.research.topic, "")
                spec_path = run_dir / f"stage-{int(stage):02d}" / "experiment_spec.md"
                spec_path.write_text(spec_text, encoding="utf-8")
                logger.info("Experiment spec generated: %s", spec_path)
            except Exception:
                logger.debug("Experiment spec generation skipped")

        if stage == Stage.RESULT_ANALYSIS and result.status == StageStatus.DONE:
            try:
                from researchclaw.pipeline.experiment_spec import parse_spec, validate_results_against_spec
                spec_path = run_dir / "stage-09" / "experiment_spec.md"
                if spec_path.exists():
                    spec = parse_spec(spec_path.read_text(encoding="utf-8"))
                    results_path = run_dir / "results.json"
                    exp_results = {}
                    if results_path.exists():
                        exp_results = json.loads(results_path.read_text(encoding="utf-8"))
                    violations = validate_results_against_spec(spec, exp_results)
                    if violations:
                        logger.warning("Spec violations: %s", violations)
                        (run_dir / f"stage-{int(stage):02d}" / "spec_violations.json").write_text(
                            json.dumps(violations, indent=2), encoding="utf-8"
                        )
            except Exception:
                logger.debug("Experiment spec validation skipped")

        # ── Pitfall detection after code generation / experiment run ──
        if stage in (Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR, Stage.HARNESS_SUBMIT_AND_COLLECT) and result.status == StageStatus.DONE:
            try:
                from researchclaw.pipeline.pitfall_detector import PitfallDetector
                detector = PitfallDetector()
                code_path = run_dir / f"stage-{int(stage):02d}"
                code_files = list(code_path.rglob("*.py"))
                code_text = "\n".join(f.read_text(errors="ignore") for f in code_files[:5])
                pitfalls = detector.detect_all(code=code_text, results={}, experiment_config={})
                if pitfalls:
                    critical = [p for p in pitfalls if p.severity == "critical"]
                    if critical:
                        logger.warning("CRITICAL pitfalls detected: %s", [p.description for p in critical])
                    pitfall_report = [{"type": p.type.value, "severity": p.severity, "description": p.description} for p in pitfalls]
                    (run_dir / f"stage-{int(stage):02d}" / "pitfall_report.json").write_text(
                        json.dumps(pitfall_report, indent=2), encoding="utf-8"
                    )
            except Exception:
                logger.debug("Pitfall detection skipped")

        # ── Experiment memory: record outcome after experiment stages ──
        if stage in (Stage.HARNESS_SUBMIT_AND_COLLECT, Stage.EXPERIMENT_ROUTE_DECISION) and result.status == StageStatus.DONE and exp_memory:
            try:
                from researchclaw.memory.experiment_memory import ExperimentOutcome
                import time as _time_mod
                results_path = run_dir / "results.json"
                metric_val = 0.0
                metric_name, _metric_dir = resolve_experiment_metric(run_dir)
                if results_path.exists():
                    rdata = json.loads(results_path.read_text(encoding="utf-8"))
                    metric_val = rdata.get(metric_name, 0.0)
                exp_memory.record_outcome(ExperimentOutcome(
                    run_id=run_id, stage=stage.name,
                    hypothesis=config.research.topic, config={},
                    metric_name=metric_name,
                    metric_value=float(metric_val) if metric_val else 0.0,
                    baseline_value=0.0, improvement=0.0,
                    success=result.status == StageStatus.DONE,
                    failure_mode=result.error,
                    packages_used=[], hyperparameters={},
                    timestamp=_time_mod.time(), duration_sec=elapsed,
                ))
            except Exception:
                logger.debug("Experiment memory recording skipped")

        if result.status == StageStatus.DONE:
            arts = ", ".join(result.artifacts) if result.artifacts else "none"
            if result.decision == "degraded":
                print(
                    f"{prefix} {stage.name} — DEGRADED ({elapsed:.1f}s) "
                    f"— continuing with sanitization → {arts}"
                )
            else:
                print(f"{prefix} {stage.name} — done ({elapsed:.1f}s) → {arts}")
        elif result.status == StageStatus.FAILED:
            err = result.error or "unknown error"
            print(f"{prefix} {stage.name} — FAILED ({elapsed:.1f}s) — {err}")
        elif result.status == StageStatus.BLOCKED_APPROVAL:
            print(f"{prefix} {stage.name} — blocked (awaiting approval)")
        elif result.status == StageStatus.PAUSED:
            err = result.error or "paused"
            print(f"{prefix} {stage.name} -- PAUSED ({elapsed:.1f}s) -- {err}")
        results.append(result)

        experiment_route: str | None = None
        if (
            stage == Stage.MANIFEST_VALIDATE_AND_PREPARE
            and result.status == StageStatus.DONE
            and result.decision == "fix_code"
        ):
            experiment_route = "fix_code"
        elif (
            stage == Stage.EXPERIMENT_ROUTE_DECISION
            and result.status == StageStatus.DONE
        ):
            decision = result.decision or "continue"
            if decision in {"fix_code", "rerun"}:
                experiment_route = decision
            elif decision not in {"continue", "proceed", "abort", "hitl"}:
                break

        if experiment_route is not None:
            target = _route_to_stage(experiment_route)
            if target is None:
                break
            if len(_read_experiment_iterations(run_dir)) >= _max_experiment_iterations(config):
                results.append(
                    _pause_for_experiment_hitl(
                        run_dir,
                        run_id,
                        stage,
                        experiment_route,
                    )
                )
                break
            attempt = _record_experiment_iteration(
                run_dir,
                route=experiment_route,
                jump=f"{int(stage)}->{experiment_route}",
                target=target,
            )
            # Experiment route targets must stay inside the Stage 10/12 repair band.
            _version_rollback_stages(run_dir, target, attempt, incremental=True)
            routed_results = _recurse_pipeline(
                from_stage=target,
                **forward_kwargs,
            )
            results.extend(routed_results)
            break

        if kb_root is not None and result.status == StageStatus.DONE:
            try:
                stage_dir = run_dir / f"stage-{int(stage):02d}"
                write_stage_to_kb(
                    kb_root,
                    stage_id=int(stage),
                    stage_name=stage.name.lower(),
                    run_id=run_id,
                    artifacts=list(result.artifacts),
                    stage_dir=stage_dir,
                    backend=config.knowledge_base.backend,
                    topic=config.research.topic,
                )
            except Exception:  # noqa: BLE001
                pass

        if result.status == StageStatus.DONE:
            _write_checkpoint(run_dir, stage, run_id, adapters=adapters)
            _notify_stage_complete(on_stage_complete, stage)
            if (
                stage == Stage.HYPOTHESIS_GEN
                and not per_hypothesis_validation_enabled(config)
            ):
                try:
                    from researchclaw.pipeline.hypothesis_tree import (
                        finalize_after_stage8,
                    )

                    hypotheses_md = _read_stage_artifact(
                        run_dir, Stage.HYPOTHESIS_GEN, "hypotheses.md"
                    )
                    new_node_id = finalize_after_stage8(run_dir, hypotheses_md)
                    if new_node_id:
                        logger.info(
                            "Hypothesis tree: created node %s", new_node_id
                        )
                except Exception:
                    logger.warning(
                        "Hypothesis tree finalize failed (non-blocking)",
                        exc_info=True,
                    )
                try:
                    from researchclaw.pipeline.hypothesis_node_tree import (
                        rebuild_node_tree,
                    )

                    rebuild_node_tree(run_dir)
                except Exception:
                    logger.warning(
                        "Hypothesis node_tree rebuild failed (non-blocking)",
                        exc_info=True,
                    )

        # ── Stop after to_stage if specified ──
        if to_stage is not None and stage == to_stage:
            logger.info("[%s] Reached --to-stage %s, stopping.", run_id, stage.name)
            print(f"[{run_id}] Reached --to-stage {stage.name}, stopping pipeline.")
            break

        if (
            stage == Stage.HYPOTHESIS_GEN
            and result.status == StageStatus.DONE
            and _run_per_hypothesis_coordinator(
                run_dir=run_dir,
                config=config,
                adapters=adapters,
                split_from_stage8=True,
            )
        ):
            per_hypothesis_validation_ran = True
            skip_per_hypothesis_validation_stages = True
            if to_stage is not None and int(to_stage) <= int(Stage.RESEARCH_DECISION):
                break

        # --- Heartbeat for sentinel watchdog ---
        if result.status == StageStatus.DONE:
            _write_heartbeat(run_dir, stage, run_id)

        # --- PIVOT/EXTEND decision handling ---
        if (
            stage == Stage.RESEARCH_DECISION
            and result.status == StageStatus.DONE
            and result.decision in DECISION_ROLLBACK
            and not per_hypothesis_validation_enabled(config)
        ):
            pivot_count = _read_pivot_count(run_dir)
            if pivot_count < MAX_DECISION_PIVOTS:
                rollback_target = DECISION_ROLLBACK[result.decision]
                if result.decision == "extend":
                    _write_extension_context(run_dir)
                elif result.decision == "pivot":
                    _clear_extension_context(run_dir)
                _record_decision_history(
                    run_dir, result.decision, rollback_target, pivot_count + 1
                )
                logger.info(
                    "Decision %s: rolling back to %s (attempt %d/%d)",
                    result.decision.upper(),
                    rollback_target.name,
                    pivot_count + 1,
                    MAX_DECISION_PIVOTS,
                )
                print(
                    f"[{run_id}] Decision: {result.decision.upper()} → "
                    f"rollback to {rollback_target.name} "
                    f"(attempt {pivot_count + 1}/{MAX_DECISION_PIVOTS})"
                )
                # Version existing stage directories before overwriting.
                _version_rollback_stages(
                    run_dir, rollback_target, pivot_count + 1,
                )
                # Recurse from rollback target.
                pivot_results = _recurse_pipeline(
                    from_stage=rollback_target,
                    **forward_kwargs,
                )
                results.extend(pivot_results)
                _promote_best_stage14(run_dir, config)
                break  # Exit current loop; recursive call handles the rest
            else:
                # Quality gate: check if experiment results are actually usable
                _quality_ok, _quality_msg = _check_experiment_quality(
                    run_dir, pivot_count
                )
                if not _quality_ok:
                    logger.warning(
                        "Max pivot attempts (%d) reached — forcing PROCEED "
                        "with quality warning: %s",
                        MAX_DECISION_PIVOTS,
                        _quality_msg,
                    )
                    print(
                        f"[{run_id}] QUALITY WARNING: {_quality_msg}"
                    )
                    # Write quality warning to run directory
                    _qw_path = run_dir / "quality_warning.txt"
                    _qw_path.write_text(
                        f"Max pivots ({MAX_DECISION_PIVOTS}) reached.\n"
                        f"Quality gate failed: {_quality_msg}\n"
                        f"Paper will be written but may have significant issues.\n",
                        encoding="utf-8",
                    )
                else:
                    logger.warning(
                        "Max pivot attempts (%d) reached — forcing PROCEED",
                        MAX_DECISION_PIVOTS,
                    )
                print(
                    f"[{run_id}] Max pivot attempts reached — forcing PROCEED"
                )

                _clear_extension_context(run_dir)

                # After forced PROCEED, promote the best stage-14 experiment
                # summary across all decision iterations.
                _promote_best_stage14(run_dir, config)
                try:
                    from researchclaw.pipeline.hypothesis_tree import (
                        record_forced_proceed,
                    )

                    record_forced_proceed(run_dir, reason="max_pivots")
                except Exception:
                    logger.warning(
                        "Hypothesis tree forced-PROCEED update failed",
                        exc_info=True,
                    )

        # --- HITL: Handle abort decision ---
        if result.decision == "abort":
            logger.info("[%s] Pipeline aborted by user at stage %s", run_id, stage.name)
            print(f"[{run_id}] Pipeline aborted by user at {stage.name}")
            break

        if result.status == StageStatus.FAILED:
            if skip_noncritical and stage in NONCRITICAL_STAGES:
                logger.warning("Noncritical stage %s failed - skipping", stage.name)
            else:
                notify_terminal_failure(
                    config=config,
                    run_id=run_id,
                    stage_name=stage.name,
                    stage_num=int(stage),
                    error=result.error,
                    run_dir=run_dir,
                )
                break

        if result.status == StageStatus.PAUSED:
            logger.warning(
                "[%s] Pipeline paused at %s: %s",
                run_id,
                stage.name,
                result.error or result.decision,
            )
            break

        # --- HITL: Handle rejected stage (from HITL review) ---
        if result.status == StageStatus.REJECTED:
            logger.info(
                "[%s] Stage %s rejected by reviewer — pipeline stopped",
                run_id, stage.name,
            )
            print(f"[{run_id}] Stage {stage.name} rejected — pipeline stopped")
            break

        if result.status == StageStatus.BLOCKED_APPROVAL and stop_on_gate:
            break

    summary = _build_pipeline_summary(
        run_id=run_id,
        results=results,
        from_stage=from_stage,
        run_dir=run_dir,
    )
    _write_pipeline_summary(run_dir, summary)

    # ── Event log: pipeline end ──
    if event_log:
        try:
            done_count = sum(1 for r in results if r.status == StageStatus.DONE)
            failed_count = sum(1 for r in results if r.status == StageStatus.FAILED)
            event_log.append(create_event(
                EventType.PIPELINE_END, run_id=run_id,
                stages_done=done_count, stages_failed=failed_count,
            ))
        except Exception:
            pass

    # --- Evolution: extract and store lessons ---
    lessons: list[object] = []
    try:
        lessons = extract_lessons(results, run_id=run_id, run_dir=run_dir)
        if lessons:
            store = EvolutionStore(run_dir / "evolution")
            store.append_many(lessons)
            logger.info("Extracted %d lessons from pipeline run", len(lessons))
    except Exception:  # noqa: BLE001
        logger.warning("Evolution lesson extraction failed (non-blocking)")

    # --- MetaClaw bridge: convert high-severity lessons to skills ---
    try:
        _metaclaw_post_pipeline(config, results, lessons, run_id, run_dir)
    except Exception:  # noqa: BLE001
        logger.warning("MetaClaw post-pipeline hook failed (non-blocking)")

    # --- Package deliverables into a single folder ---
    try:
        deliverables_dir = _package_deliverables(run_dir, run_id, config)
        if deliverables_dir is not None:
            print(f"[{run_id}] Deliverables packaged → {deliverables_dir}")
    except Exception:  # noqa: BLE001
        logger.warning("Deliverables packaging failed (non-blocking)")

    # --- HITL: Finalize session state ---
    try:
        hitl_session = getattr(adapters, "hitl", None)
        if hitl_session is not None:
            has_abort = any(
                r.decision == "abort" for r in results
            )
            has_failure = any(
                r.status == StageStatus.FAILED for r in results
            )
            if has_abort:
                hitl_session.abort()
            elif has_failure:
                hitl_session.abort()
            else:
                hitl_session.complete()
    except Exception:  # noqa: BLE001
        logger.debug("HITL session finalization failed (non-blocking)")

    return results


def _package_deliverables(
    run_dir: Path,
    run_id: str,
    config: RCConfig,
) -> Path | None:
    """Collect all final user-facing deliverables into a single ``deliverables/`` folder.

    Returns the deliverables directory path, or None if nothing was packaged.

    Packaged artifacts (best-available version selected automatically):
    - paper_final.md          — Final paper (Markdown)
    - paper.tex               — Conference-ready LaTeX
    - references.bib          — BibTeX bibliography
    - code/                   — Experiment code package
    - verification_report.json — Citation verification report (if available)
    """
    dest = run_dir / "deliverables"
    dest.mkdir(parents=True, exist_ok=True)

    packaged: list[str] = []

    # --- 0. Resolve effective conference template ---
    # Mirrors the stage-22 domain-aware override: when the topic belongs to a
    # non-ML domain (hep_ph, etc.) and the user has left the default
    # neurips_2025, swap in the domain's preferred physics template so the
    # bundled .sty, regenerated .tex, and manifest are all consistent.
    effective_conf = config.export.target_conference
    try:
        from researchclaw.domains.detector import detect_domain as _dd_detect
        from researchclaw.domains.prompt_adapter import get_adapter as _dd_adapter

        _dd_dom = _dd_detect(topic=config.research.topic)
        _dd_blocks = _dd_adapter(_dd_dom).get_export_publish_blocks(
            {"topic": config.research.topic}
        )
        _pref_tpl = (_dd_blocks.preferred_template or "").strip()
        if _pref_tpl and effective_conf == "neurips_2025":
            effective_conf = _pref_tpl
            logger.info(
                "Deliverables: domain=%s — overriding target_conference "
                "'neurips_2025' → '%s'.",
                getattr(_dd_dom, "domain_id", "?"),
                effective_conf,
            )
    except Exception:  # noqa: BLE001
        logger.debug("Deliverables: domain-aware template override skipped")

    # --- 1. Final paper (Markdown) ---
    # Prefer verified version (stage 23) over base version (stage 22)
    paper_md = None
    for candidate in [
        run_dir / "stage-23" / "paper_final_verified.md",
        run_dir / "stage-22" / "paper_final.md",
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            paper_md = candidate
            break
    if paper_md is not None:
        shutil.copy2(paper_md, dest / "paper_final.md")
        packaged.append("paper_final.md")

    # --- 2. LaTeX paper ---
    # BUG-183: Stage 22's paper.tex has been sanitized (fabricated numbers
    # replaced with ---).  Regenerating from Markdown would undo this because
    # the Markdown was never sanitized.  Prefer Stage-22 paper.tex when a
    # sanitization report exists.  Only regenerate from verified Markdown if
    # no sanitization was performed (i.e., the run was clean).
    tex_regenerated = False
    _sanitization_report = run_dir / "stage-22" / "sanitization_report.json"
    _was_sanitized = _sanitization_report.exists()
    verified_md = run_dir / "stage-23" / "paper_final_verified.md"
    if (
        not _was_sanitized
        and paper_md is not None
        and paper_md == verified_md
        and verified_md.exists()
        and verified_md.stat().st_size > 0
    ):
        try:
            from researchclaw.templates import get_template, markdown_to_latex
            from researchclaw.pipeline.executor import _extract_paper_title

            tpl = get_template(effective_conf)
            v_text = verified_md.read_text(encoding="utf-8")
            tex_content = markdown_to_latex(
                v_text,
                tpl,
                title=_extract_paper_title(v_text),
                authors=config.export.authors,
                bib_file=config.export.bib_file,
            )
            # IMP-17: Quality check — ensure regenerated LaTeX has
            # proper structure (abstract, multiple sections)
            _has_abstract = (
                "\\begin{abstract}" in tex_content
                and tex_content.split("\\begin{abstract}")[1]
                .split("\\end{abstract}")[0]
                .strip()
            )
            _section_count = tex_content.count("\\section{")
            if _has_abstract and _section_count >= 3:
                (dest / "paper.tex").write_text(tex_content, encoding="utf-8")
                packaged.append("paper.tex")
                tex_regenerated = True
                logger.info(
                    "Deliverables: regenerated paper.tex from verified markdown"
                )
            else:
                logger.warning(
                    "Regenerated paper.tex has poor structure "
                    "(abstract=%s, sections=%d) — using Stage 22 version",
                    bool(_has_abstract),
                    _section_count,
                )
        except Exception:  # noqa: BLE001
            logger.debug("paper.tex regeneration from verified md failed")
    elif _was_sanitized:
        logger.info(
            "Deliverables: using Stage 22 paper.tex (sanitized) — "
            "skipping markdown regeneration to preserve sanitization"
        )

    if not tex_regenerated:
        tex_src = run_dir / "stage-22" / "paper.tex"
        if tex_src.exists() and tex_src.stat().st_size > 0:
            shutil.copy2(tex_src, dest / "paper.tex")
            packaged.append("paper.tex")

    # --- 3. References (BibTeX) ---
    # Prefer verified bib (stage 23) over base bib (stage 22)
    bib_src = None
    for candidate in [
        run_dir / "stage-23" / "references_verified.bib",
        run_dir / "stage-22" / "references.bib",
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            bib_src = candidate
            break
    if bib_src is not None:
        shutil.copy2(bib_src, dest / "references.bib")
        packaged.append("references.bib")

    # --- 4. Experiment code package ---
    code_src = run_dir / "stage-22" / "code"
    if code_src.is_dir():
        code_dest = dest / "code"
        if code_dest.exists():
            shutil.rmtree(code_dest)
        shutil.copytree(code_src, code_dest)
        packaged.append("code/")

    # --- 5. Verification report (optional) ---
    verify_src = run_dir / "stage-23" / "verification_report.json"
    if verify_src.exists() and verify_src.stat().st_size > 0:
        shutil.copy2(verify_src, dest / "verification_report.json")
        packaged.append("verification_report.json")

    # --- 5b. Sanitization report (degraded mode) ---
    san_src = run_dir / "stage-22" / "sanitization_report.json"
    if san_src.exists() and san_src.stat().st_size > 0:
        shutil.copy2(san_src, dest / "sanitization_report.json")
        packaged.append("sanitization_report.json")

    # --- 6. Charts (optional) ---
    charts_src = run_dir / "stage-22" / "charts"
    if charts_src.is_dir() and any(charts_src.iterdir()):
        charts_dest = dest / "charts"
        if charts_dest.exists():
            shutil.rmtree(charts_dest)
        shutil.copytree(charts_src, charts_dest)
        packaged.append("charts/")

    # --- 7. Conference style files (.sty, .bst) ---
    try:
        from researchclaw.templates import get_template

        tpl = get_template(effective_conf)
        style_files = tpl.get_style_files()
        for sf in style_files:
            shutil.copy2(sf, dest / sf.name)
            packaged.append(sf.name)
        if style_files:
            logger.info(
                "Deliverables: bundled %d style files for %s",
                len(style_files),
                tpl.display_name,
            )
    except Exception:  # noqa: BLE001
        logger.debug("Style file bundling skipped (template lookup failed)")

    # --- 8. Verify & repair cite key coverage (IMP-12 + IMP-14) ---
    tex_path = dest / "paper.tex"
    bib_path = dest / "references.bib"
    if tex_path.exists() and bib_path.exists():
        try:
            tex_text = tex_path.read_text(encoding="utf-8")
            bib_text = bib_path.read_text(encoding="utf-8")
            import re as _re

            # IMP-15: Deduplicate .bib entries
            _seen_bib_keys: set[str] = set()
            _deduped_entries: list[str] = []
            for _bm in _re.finditer(
                r"(@\w+\{([^,]+),.*?\n\})", bib_text, _re.DOTALL
            ):
                _bkey = _bm.group(2).strip()
                if _bkey not in _seen_bib_keys:
                    _seen_bib_keys.add(_bkey)
                    _deduped_entries.append(_bm.group(1))
            if len(_deduped_entries) < len(
                list(_re.finditer(r"@\w+\{", bib_text))
            ):
                bib_text = "\n\n".join(_deduped_entries) + "\n"
                bib_path.write_text(bib_text, encoding="utf-8")
                logger.info(
                    "Deliverables: deduplicated .bib → %d entries",
                    len(_deduped_entries),
                )

            # Collect all cite keys from \cite{key1, key2}
            all_cite_keys: set[str] = set()
            for cm in _re.finditer(r"\\cite\{([^}]+)\}", tex_text):
                all_cite_keys.update(k.strip() for k in cm.group(1).split(","))
            bib_keys = set(_re.findall(r"@\w+\{([^,]+),", bib_text))
            missing = all_cite_keys - bib_keys

            # IMP-14: Strip orphaned \cite{key} from paper.tex
            if missing:
                logger.warning(
                    "Deliverables: stripping %d orphaned cite keys from "
                    "paper.tex: %s",
                    len(missing),
                    sorted(missing)[:10],
                )

                def _filter_cite(m: _re.Match[str]) -> str:
                    keys = [k.strip() for k in m.group(1).split(",")]
                    kept = [k for k in keys if k not in missing]
                    if not kept:
                        return ""
                    return "\\cite{" + ", ".join(kept) + "}"

                tex_text = _re.sub(r"\\cite\{([^}]+)\}", _filter_cite, tex_text)
                # Clean up whitespace artifacts: double spaces, space before period
                tex_text = _re.sub(r"  +", " ", tex_text)
                tex_text = _re.sub(r" ([.,;:)])", r"\1", tex_text)
                tex_path.write_text(tex_text, encoding="utf-8")
                logger.info(
                    "Deliverables: paper.tex repaired — all remaining cite "
                    "keys verified"
                )
            else:
                logger.info(
                    "Deliverables: all %d cite keys verified in references.bib",
                    len(all_cite_keys),
                )
        except Exception:  # noqa: BLE001
            logger.debug("Cite key verification/repair skipped")

    # --- 9. IMP-18: Compile LaTeX to verify paper.tex ---
    if tex_path.exists() and bib_path.exists():
        try:
            from researchclaw.templates.compiler import compile_latex

            compile_result = compile_latex(tex_path, max_attempts=3, timeout=120)
            if compile_result.success:
                logger.info("IMP-18: paper.tex compiles successfully")
                # Keep the generated PDF
                pdf_path = dest / tex_path.stem
                pdf_file = dest / (tex_path.stem + ".pdf")
                if pdf_file.exists():
                    packaged.append(f"{tex_path.stem}.pdf")
            else:
                logger.warning(
                    "IMP-18: paper.tex compilation failed after %d attempts: %s",
                    compile_result.attempts,
                    compile_result.errors[:3],
                )
            if compile_result.fixes_applied:
                logger.info(
                    "IMP-18: Applied %d auto-fixes: %s",
                    len(compile_result.fixes_applied),
                    compile_result.fixes_applied,
                )
        except Exception:  # noqa: BLE001
            logger.debug("IMP-18: LaTeX compilation skipped (non-blocking)")

    if not packaged:
        # Nothing to package — remove empty dir
        dest.rmdir()
        return None

    # --- Write manifest ---
    manifest = {
        "run_id": run_id,
        "target_conference": effective_conf,
        "files": packaged,
        "generated": _utcnow_iso(),
        "notes": {
            "paper_final.md": "Final paper in Markdown format",
            "paper.tex": f"Conference-ready LaTeX ({effective_conf})",
            "references.bib": "BibTeX bibliography (verified citations only)",
            "code/": "Experiment source code with requirements.txt",
            "verification_report.json": "Citation integrity & relevance verification",
            "charts/": "Result visualizations",
        },
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info(
        "Deliverables packaged: %s (%d items)",
        dest,
        len(packaged),
    )
    return dest


def _version_rollback_stages(
    run_dir: Path,
    rollback_target: Stage,
    attempt: int,
    *,
    incremental: bool = False,
) -> None:
    """Snapshot stage directories that will be re-executed by a PIVOT/EXTEND
    or by an explicit incremental re-entry.

    Default behavior renames ``stage-NN/`` to ``stage-NN_v{attempt}/`` so the
    next run starts from a clean slate.

    When ``incremental=True``, directories whose number is >= HARNESS_SUBMIT_AND_COLLECT (12)
    are *copied* via ``shutil.copytree`` instead of renamed, so the live
    stage-12 workspace persists across re-entries. Stages before HARNESS_SUBMIT_AND_COLLECT
    in the rollback range are still renamed.
    """
    import shutil

    rollback_num = int(rollback_target)
    decision_num = int(Stage.RESEARCH_DECISION)
    exp_run_num = int(Stage.HARNESS_SUBMIT_AND_COLLECT)

    for stage_num in range(rollback_num, decision_num + 1):
        stage_dir = run_dir / f"stage-{stage_num:02d}"
        if not stage_dir.exists():
            continue
        version_dir = run_dir / f"stage-{stage_num:02d}_v{attempt}"
        if version_dir.exists():
            shutil.rmtree(version_dir)
        if incremental and stage_num >= exp_run_num:
            shutil.copytree(stage_dir, version_dir, symlinks=False)
            logger.debug(
                "Snapshotted (copytree) %s → %s (incremental)",
                stage_dir.name,
                version_dir.name,
            )
        else:
            stage_dir.rename(version_dir)
            logger.debug(
                "Versioned (rename) %s → %s", stage_dir.name, version_dir.name
            )


def _read_stage_artifact(run_dir: Path, stage: Stage, filename: str) -> str:
    """Read a current stage artifact, falling back to latest versioned copy."""

    current = run_dir / f"stage-{int(stage):02d}" / filename
    if current.exists():
        try:
            return current.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    candidates = sorted(run_dir.glob(f"stage-{int(stage):02d}_v*/{filename}"))
    for candidate in reversed(candidates):
        try:
            return candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return ""


def _write_extension_context(run_dir: Path) -> None:
    """Write context for an EXTEND rollback into Stage 8.

    EXTEND keeps the current hypothesis line alive, so the next hypothesis
    generation pass needs the previous hypothesis plus the experiment evidence
    that motivated follow-up hypotheses.
    """

    hypotheses = _read_stage_artifact(run_dir, Stage.HYPOTHESIS_GEN, "hypotheses.md")
    analysis = _read_stage_artifact(run_dir, Stage.RESULT_ANALYSIS, "analysis.md")
    summary = _read_stage_artifact(
        run_dir, Stage.RESULT_ANALYSIS, "experiment_summary.json"
    )
    decision = _read_stage_artifact(run_dir, Stage.RESEARCH_DECISION, "decision.md")

    if summary:
        try:
            summary = json.dumps(json.loads(summary), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass

    sections = [
        "# Hypothesis Extension Context",
        "",
        "Generated after an EXTEND research decision. Use this as prior "
        "evidence for follow-up hypotheses, not as a blank-slate pivot.",
        "",
        "## Previous Hypotheses",
        hypotheses or "_No previous hypotheses artifact found._",
        "",
        "## Latest Analysis",
        analysis or "_No latest analysis artifact found._",
        "",
        "## Latest Experiment Summary",
        "```json",
        summary or "{}",
        "```",
        "",
        "## Decision Rationale",
        decision or "_No decision artifact found._",
        "",
    ]
    (run_dir / "hypothesis_extension_context.md").write_text(
        "\n".join(sections), encoding="utf-8"
    )


def _clear_extension_context(run_dir: Path) -> None:
    context_path = run_dir / "hypothesis_extension_context.md"
    try:
        context_path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("Failed to remove stale extension context: %s", exc)


def _promote_best_stage14(run_dir: Path, config: RCConfig) -> None:
    """BUG-205: After forced PROCEED, promote the best stage-14 experiment.

    Scans all ``stage-14*`` directories, scores them by primary metric,
    and copies the best experiment_summary.json into ``stage-14/`` if the
    current ``stage-14/`` is not already the best.
    """
    import shutil

    _ = config
    metric_key, metric_dir = resolve_experiment_metric(run_dir)

    candidates: list[tuple[float, Path]] = []
    for d in sorted(run_dir.glob("stage-14*")):
        summary_path = d / "experiment_summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ms = data.get("metrics_summary", {})
        pm_val: float | None = None
        # BUG-DA8-03: Exact match first, then substring fallback
        # (avoids "accuracy" matching "balanced_accuracy")
        if metric_key in ms:
            _v = ms[metric_key]
            try:
                pm_val = float(_v["mean"] if isinstance(_v, dict) else _v)
            except (TypeError, ValueError, KeyError):
                pass
        if pm_val is None:
            for k, v in ms.items():
                if metric_key in k:
                    try:
                        pm_val = float(v["mean"] if isinstance(v, dict) else v)
                    except (TypeError, ValueError, KeyError):
                        pass
                    break
        if pm_val is not None:
            if math.isnan(pm_val):
                continue
            candidates.append((pm_val, d))

    if not candidates:
        return  # nothing to promote

    current_dir = run_dir / "stage-14"

    # Sort: best first
    candidates.sort(key=lambda x: x[0], reverse=(metric_dir == "maximize"))

    # BUG-226: Detect degenerate near-zero metrics (broken normalization or
    # collapsed training).  When minimising, a value >1000x smaller than the
    # second-best almost certainly comes from a degenerate iteration.
    if metric_dir == "minimize" and len(candidates) > 1:
        _bv, _bd = candidates[0]
        _sv = candidates[1][0]
        if 0 < _bv < _sv * 1e-3:
            logger.warning(
                "BUG-226: Degenerate best value %.6g is >1000× smaller than "
                "second-best %.6g — skipping degenerate iteration %s",
                _bv, _sv, _bd.name,
            )
            candidates.pop(0)

    best_val, best_dir = candidates[0]

    # BUG-223: Always write canonical best summary at run root BEFORE any
    # early return, so downstream consumers (Stage 17, Stage 20, Stage 22,
    # VerifiedRegistry) always find experiment_summary_best.json.
    _best_src = best_dir / "experiment_summary.json"
    if _best_src.exists():
        shutil.copy2(_best_src, run_dir / "experiment_summary_best.json")
        logger.info(
            "BUG-223: Wrote experiment_summary_best.json from %s (%.4f)",
            best_dir.name, best_val,
        )
        # BUG-225: Also copy analysis.md from the best iteration so Stage 17
        # doesn't read stale analysis from a degenerate non-versioned stage-14.
        _best_analysis = best_dir / "analysis.md"
        if _best_analysis.exists():
            shutil.copy2(_best_analysis, run_dir / "analysis_best.md")

    if best_dir == current_dir:
        logger.info("BUG-205: stage-14/ already has the best result (%.4f)", best_val)
        return

    # Promote: copy best summary into stage-14/
    current_summary = current_dir / "experiment_summary.json"
    best_summary = best_dir / "experiment_summary.json"
    # BUG-213: Also promote when stage-14/ is missing or empty
    if best_summary.exists():
        current_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "BUG-205: Promoting %s (%.4f) over stage-14/",
            best_dir.name, best_val,
        )
        shutil.copy2(best_summary, current_summary)
        # Also copy charts, analysis, and figure plans if they exist
        for fname in [
            "analysis.md",
            "results_table.tex",
            "figure_plan.json",           # BUG-213: must travel with metrics
            "figure_plan_final.json",     # BUG-213: ditto
        ]:
            src = best_dir / fname
            if src.exists():
                shutil.copy2(src, current_dir / fname)
        # Copy charts directory
        best_charts = best_dir / "charts"
        current_charts = current_dir / "charts"
        if best_charts.is_dir():
            if current_charts.is_dir():
                shutil.rmtree(current_charts)
            shutil.copytree(best_charts, current_charts)


def _check_experiment_quality(
    run_dir: Path, pivot_count: int
) -> tuple[bool, str]:
    """Quality gate before forced PROCEED.

    Returns (ok, message). ok=False means experiment results have critical
    quality issues and the forced-PROCEED paper will likely be poor.
    """
    # BUG-DA8-18: Check experiment_summary_best.json first (repair-promoted)
    summary_path = run_dir / "experiment_summary_best.json"
    if not summary_path.exists():
        summary_path = run_dir / "stage-14" / "experiment_summary.json"
    if not summary_path.exists():
        for v in range(pivot_count, 0, -1):
            alt = run_dir / f"stage-14_v{v}" / "experiment_summary.json"
            if alt.exists():
                summary_path = alt
                break

    if not summary_path.exists():
        return False, "No experiment_summary.json found — no metrics produced"

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, "experiment_summary.json is malformed"

    # Check 1: Are all metrics zero?
    ms = data.get("metrics_summary", {})
    if isinstance(ms, dict):
        values: list[float] = []
        for k, v in ms.items():
            if isinstance(v, (int, float)):
                values.append(float(v))
            # BUG-212: metrics_summary values are often dicts {min,max,mean,count}
            elif isinstance(v, dict) and "mean" in v:
                _mv = v["mean"]
                if isinstance(_mv, (int, float)):
                    values.append(float(_mv))
        if values and all(v == 0.0 for v in values):
            return False, "All experiment metrics are zero — experiments likely failed"

    # Check 2: Zero variance across conditions (R13-1)
    # Look for ablation_warnings or condition comparison data
    ablation_warnings = data.get("ablation_warnings", [])
    # BUG-212: Key is "condition_summaries", not "conditions"
    conditions = data.get(
        "condition_summaries", data.get("condition_metrics", {})
    )
    if isinstance(conditions, dict) and len(conditions) >= 2:
        primary_values: list[float] = []
        for cond_name, cond_data in conditions.items():
            if isinstance(cond_data, dict):
                # BUG-212: Primary metric lives inside cond_data["metrics"]
                _metrics = cond_data.get("metrics", cond_data)
                pm = _metrics.get(
                    "primary_metric",
                    _metrics.get("primary_metric_mean"),
                )
                if isinstance(pm, (int, float)):
                    primary_values.append(float(pm))
        if len(primary_values) >= 2 and len(set(primary_values)) == 1:
            return False, (
                f"All {len(primary_values)} conditions have identical primary_metric "
                f"({primary_values[0]}) — condition implementations are likely broken"
            )

    # Check 3: Too many ablation warnings
    if isinstance(ablation_warnings, list) and len(ablation_warnings) >= 3:
        return False, (
            f"{len(ablation_warnings)} ablation warnings — most conditions "
            f"produce identical results"
        )

    # Check 4: Analysis quality score (if available)
    quality = data.get("analysis_quality", data.get("quality_score"))
    if isinstance(quality, (int, float)) and quality < 3.0:
        return False, f"Analysis quality score {quality}/10 — below minimum threshold"

    return True, "Quality checks passed"


def _read_pivot_count(run_dir: Path) -> int:
    """Read how many PIVOT/EXTEND decisions have been made so far."""
    history_path = run_dir / "decision_history.json"
    if not history_path.exists():
        return 0
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
    except (json.JSONDecodeError, OSError):
        pass
    return 0


def _record_decision_history(
    run_dir: Path, decision: str, rollback_target: Stage, attempt: int
) -> None:
    """Append a decision event to the history log."""
    history_path = run_dir / "decision_history.json"
    history: list[dict[str, object]] = []
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                history = data
        except (json.JSONDecodeError, OSError):
            pass
    history.append({
        "decision": decision,
        "rollback_target": rollback_target.name,
        "rollback_stage_num": int(rollback_target),
        "attempt": attempt,
        "timestamp": _utcnow_iso(),
    })
    history_path.write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )



def _read_quality_score(run_dir: Path) -> float | None:
    """Extract quality score from the most recent quality_report.json."""
    report_path = run_dir / "stage-20" / "quality_report.json"
    if not report_path.exists():
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # Try common keys: score_1_to_10, score, quality_score
            for key in ("score_1_to_10", "score", "quality_score", "overall_score"):
                if key in data:
                    return float(data[key])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def _write_iteration_context(
    run_dir: Path, iteration: int, reviews: str, quality_score: float | None
) -> None:
    """Write iteration feedback file so next round can read it."""
    ctx = {
        "iteration": iteration,
        "quality_score": quality_score,
        "reviews_excerpt": reviews[:3000] if reviews else "",
        "generated": _utcnow_iso(),
    }
    (run_dir / "iteration_context.json").write_text(
        json.dumps(ctx, indent=2), encoding="utf-8"
    )


def execute_iterative_pipeline(
    *,
    run_dir: Path,
    run_id: str,
    config: RCConfig,
    adapters: AdapterBundle,
    auto_approve_gates: bool = False,
    kb_root: Path | None = None,
    max_iterations: int = 3,
    quality_threshold: float = 7.0,
    convergence_rounds: int = 2,
) -> dict[str, object]:
    """Run the full pipeline with iterative quality improvement.

    After the first full pass (stages 1-22), if the quality gate score is below
    *quality_threshold*, re-run stages 16-22 (paper writing + finalization) with
    review feedback injected.  Stop when:
      - Score >= quality_threshold, OR
      - Score hasn't improved for *convergence_rounds* consecutive iterations, OR
      - *max_iterations* reached.

    Returns a summary dict with iteration history.
    """
    iteration_scores: list[float | None] = []
    all_results: list[list[StageResult]] = []

    # --- First full pass ---
    logger.info("Iteration 1/%d: running full pipeline (stages 1-22)", max_iterations)
    results = execute_pipeline(
        run_dir=run_dir,
        run_id=f"{run_id}-iter1",
        config=config,
        adapters=adapters,
        auto_approve_gates=auto_approve_gates,
        kb_root=kb_root,
    )
    all_results.append(results)
    score = _read_quality_score(run_dir)
    iteration_scores.append(score)
    logger.info("Iteration 1 score: %s", score)

    # --- Iterative improvement ---
    for iteration in range(2, max_iterations + 1):
        # Check if we've met quality threshold
        if score is not None and score >= quality_threshold:
            logger.info(
                "Quality threshold %.1f met (score=%.1f). Stopping.",
                quality_threshold,
                score,
            )
            break

        # Check convergence (score hasn't improved)
        if len(iteration_scores) >= convergence_rounds:
            recent = iteration_scores[-convergence_rounds:]
            if all(s is not None for s in recent):
                recent_scores = [float(s) for s in recent if s is not None]
                if max(recent_scores) - min(recent_scores) < 0.5:
                    logger.info(
                        "Convergence detected: scores %s unchanged for %d rounds. Stopping.",
                        recent,
                        convergence_rounds,
                    )
                    break

        # Write iteration context with feedback from reviews
        reviews_text = ""
        reviews_path = run_dir / "stage-18" / "reviews.md"
        if reviews_path.exists():
            reviews_text = reviews_path.read_text(encoding="utf-8")
        _write_iteration_context(run_dir, iteration, reviews_text, score)

        # Re-run from PAPER_OUTLINE (stage 16) through EXPORT_PUBLISH (stage 22)
        logger.info(
            "Iteration %d/%d: re-running stages 16-22 with feedback",
            iteration,
            max_iterations,
        )
        results = execute_pipeline(
            run_dir=run_dir,
            run_id=f"{run_id}-iter{iteration}",
            config=config,
            adapters=adapters,
            from_stage=Stage.PAPER_OUTLINE,
            auto_approve_gates=auto_approve_gates,
            kb_root=kb_root,
        )
        all_results.append(results)
        score = _read_quality_score(run_dir)
        iteration_scores.append(score)
        logger.info("Iteration %d score: %s", iteration, score)

    # --- Build iterative summary ---
    converged = False
    if len(iteration_scores) >= convergence_rounds:
        recent_window = iteration_scores[-convergence_rounds:]
        if all(s is not None for s in recent_window):
            recent_scores = [float(s) for s in recent_window if s is not None]
            converged = max(recent_scores) - min(recent_scores) < 0.5

    summary: dict[str, object] = {
        "run_id": run_id,
        "total_iterations": len(iteration_scores),
        "iteration_scores": iteration_scores,
        "quality_threshold": quality_threshold,
        "converged": converged,
        "final_score": iteration_scores[-1] if iteration_scores else None,
        "met_threshold": score is not None and score >= quality_threshold,
        "stages_per_iteration": [len(r) for r in all_results],
        "generated": _utcnow_iso(),
    }
    (run_dir / "iteration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # --- Package deliverables into a single folder ---
    try:
        deliverables_dir = _package_deliverables(run_dir, run_id, config)
        if deliverables_dir is not None:
            print(f"[{run_id}] Deliverables packaged →{deliverables_dir}")
    except Exception:  # noqa: BLE001
        logger.warning("Deliverables packaging failed (non-blocking)")

    return summary


def _metaclaw_post_pipeline(
    config: RCConfig,
    results: list[StageResult],
    lessons: list[object],
    run_id: str,
    run_dir: Path,
) -> None:
    """MetaClaw bridge: post-pipeline hook.

    1. Convert high-severity lessons into MetaClaw skills.
    2. Record skill effectiveness feedback.
    3. Signal session end to MetaClaw proxy.
    """
    bridge = getattr(config, "metaclaw_bridge", None)
    if not bridge or not getattr(bridge, "enabled", False):
        return

    from researchclaw.llm.client import LLMClient

    # 1. Lesson-to-skill conversion
    l2s = getattr(bridge, "lesson_to_skill", None)
    if l2s and getattr(l2s, "enabled", False) and lessons:
        try:
            from researchclaw.metaclaw_bridge.lesson_to_skill import (
                convert_lessons_to_skills,
            )

            min_sev = getattr(l2s, "min_severity", "warning")
            llm = LLMClient.from_rc_config(config)
            new_skills = convert_lessons_to_skills(
                lessons,
                llm,
                getattr(bridge, "skills_dir", "~/.metaclaw/skills"),
                min_severity=min_sev,
                max_skills=getattr(l2s, "max_skills_per_run", 3),
            )
            if new_skills:
                logger.info(
                    "MetaClaw: generated %d new skills from lessons: %s",
                    len(new_skills),
                    new_skills,
                )
        except Exception:  # noqa: BLE001
            logger.warning("MetaClaw lesson-to-skill conversion failed", exc_info=True)

    # 2. Skill effectiveness feedback
    try:
        from researchclaw.metaclaw_bridge.skill_feedback import (
            SkillFeedbackStore,
            record_stage_skills,
        )
        from researchclaw.metaclaw_bridge.stage_skill_map import get_stage_config

        feedback_store = SkillFeedbackStore(run_dir / "evolution" / "skill_effectiveness.jsonl")
        for result in results:
            stage_num = int(getattr(result, "stage", 0))
            stage_name = {
                1: "topic_init", 2: "problem_decompose", 3: "search_strategy",
                4: "literature_collect", 5: "literature_screen", 6: "knowledge_extract",
                7: "synthesis", 8: "hypothesis_gen", 9: "experiment_plan",
                10: "code_agent_implement_or_repair",
                11: "manifest_validate_and_prepare",
                12: "harness_submit_and_collect",
                13: "experiment_route_decision",
                14: "result_analysis", 15: "research_decision",
                16: "paper_outline", 17: "paper_draft", 18: "peer_review",
                19: "paper_revision", 20: "quality_gate", 21: "knowledge_archive",
                22: "export_publish", 23: "citation_verify",
            }.get(stage_num, "")
            if not stage_name:
                continue

            stage_config = get_stage_config(stage_name)
            active_skills = stage_config.get("skills", [])
            status = str(getattr(result, "status", ""))
            success = "done" in status.lower()

            if active_skills:
                record_stage_skills(
                    feedback_store,
                    stage_name,
                    run_id,
                    success,
                    active_skills,
                )
    except Exception:  # noqa: BLE001
        logger.warning("MetaClaw skill feedback recording failed")

    # 3. Signal session end (fire-and-forget)
    try:
        from researchclaw.metaclaw_bridge.session import MetaClawSession
        import json as _json
        import urllib.request as _urllib_req

        session = MetaClawSession(run_id)
        end_headers = session.end()
        # Send a minimal request to signal session end
        proxy_url = getattr(bridge, "proxy_url", "http://localhost:30000")
        url = f"{proxy_url.rstrip('/')}/v1/chat/completions"
        body = _json.dumps({
            "model": "session-end",
            "messages": [{"role": "user", "content": "session complete"}],
            "max_tokens": 1,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        headers.update(end_headers)
        req = _urllib_req.Request(url, data=body, headers=headers)
        try:
            _urllib_req.urlopen(req, timeout=5)
        except Exception:  # noqa: BLE001
            pass  # Best-effort signal
    except Exception:  # noqa: BLE001
        pass
