"""Stages 11-13: manifest validation, harness execution, and route decision."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.experiment.manifest_validation import validate_manifest
from researchclaw.experiment.workspace import RunManifest
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import StageResult, _read_prior_artifact
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager


def _execute_resource_planning(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    return _execute_manifest_validate_and_prepare(
        stage_dir, run_dir, config, adapters, llm=llm, prompts=prompts
    )


def _execute_manifest_validate_and_prepare(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    _ = adapters, llm, prompts
    manifest_text = _read_prior_artifact(run_dir, "run_manifest.json")
    if not manifest_text:
        return _failed(Stage.MANIFEST_VALIDATE_AND_PREPARE, "E11_MANIFEST_INVALID: missing run_manifest.json")
    try:
        manifest = RunManifest.from_json(manifest_text)
    except Exception as exc:  # noqa: BLE001
        return _failed(Stage.MANIFEST_VALIDATE_AND_PREPARE, f"E11_MANIFEST_INVALID: invalid run_manifest.json: {exc}")

    workspace = Path(config.experiment.workspace_agent.workspace_path).resolve()
    validation = validate_manifest(manifest, workspace, allow_dirty=False)
    (stage_dir / "manifest_validation.json").write_text(
        json.dumps(validation.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not validation.ok:
        _write_repair_request(
            run_dir,
            origin_stage=11,
            reason="manifest_invalid",
            errors=validation.errors,
            iteration=_read_experiment_iteration_count(run_dir),
            generated=_utcnow_iso(),
            diagnosis_ref=(
                "experiment_diagnosis.json"
                if (run_dir / "experiment_diagnosis.json").is_file()
                else ""
            ),
        )
        return StageResult(
            stage=Stage.MANIFEST_VALIDATE_AND_PREPARE,
            status=StageStatus.FAILED,
            artifacts=("manifest_validation.json",),
            error="E11_MANIFEST_INVALID: " + "; ".join(validation.errors),
            decision="fix_code",
        )

    (stage_dir / "run_manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return StageResult(
        stage=Stage.MANIFEST_VALIDATE_AND_PREPARE,
        status=StageStatus.DONE,
        artifacts=("manifest_validation.json", "run_manifest.json"),
        evidence_refs=("stage-11/manifest_validation.json", "stage-11/run_manifest.json"),
    )


def _execute_experiment_run(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    return _execute_harness_submit_and_collect(
        stage_dir, run_dir, config, adapters, llm=llm, prompts=prompts
    )


def _execute_harness_submit_and_collect(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    _ = adapters, llm, prompts
    validation_text = _read_prior_artifact(run_dir, "manifest_validation.json")
    manifest_text = _read_prior_artifact(run_dir, "run_manifest.json")
    if not validation_text or not manifest_text:
        return _failed(Stage.HARNESS_SUBMIT_AND_COLLECT, "E12_HARNESS_FAIL: missing validated run manifest")
    try:
        validation_payload = json.loads(validation_text)
        manifest = RunManifest.from_json(manifest_text)
    except Exception as exc:  # noqa: BLE001
        return _failed(Stage.HARNESS_SUBMIT_AND_COLLECT, f"E12_HARNESS_FAIL: invalid manifest inputs: {exc}")
    if not bool(validation_payload.get("ok", False)):
        return _failed(Stage.HARNESS_SUBMIT_AND_COLLECT, "E12_HARNESS_FAIL: manifest validation is not ok")

    from researchclaw.experiment import submitter as submitter_factory
    from researchclaw.pipeline import workspace_orchestrator

    submitter = submitter_factory.create_submitter(config)
    submitter_cfg = getattr(config.experiment, "submitter", None)
    wait = bool(getattr(submitter_cfg, "wait_for_completion", True))
    timeout_sec = int(
        getattr(submitter_cfg, "wait_timeout_sec", 0)
        or getattr(config.experiment, "time_budget_sec", 300)
    )
    poll_interval_sec = int(getattr(submitter_cfg, "poll_interval_sec", 15))
    agent_result = _latest_agent_result(run_dir)
    try:
        workspace_orchestrator.submit_and_collect(
            manifest=manifest,
            submitter=submitter,
            workspace_path=Path(config.experiment.workspace_agent.workspace_path).resolve(),
            run_dir=stage_dir,
            stage=12,
            wait=wait,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            base_sha=str(agent_result.get("base_sha", "")),
            agent_commit_sha=manifest.code_commit,
            provider=str(agent_result.get("provider_name", "")),
            session_name=str(getattr(config.experiment.workspace_agent, "session_name", "")),
        )
    except Exception as exc:  # noqa: BLE001
        return _failed(Stage.HARNESS_SUBMIT_AND_COLLECT, f"E12_HARNESS_FAIL: {exc}")

    output_artifacts = (
        "execution_record.json",
        "submit_result.json",
        "result_artifacts.json",
    )
    artifacts = _read_result_artifacts(stage_dir)
    if artifacts and not any(bool(item.get("exists")) for item in artifacts):
        return StageResult(
            stage=Stage.HARNESS_SUBMIT_AND_COLLECT,
            status=StageStatus.FAILED,
            artifacts=output_artifacts,
            error="E12_HARNESS_FAIL: all declared result_paths are missing",
        )
    return StageResult(
        stage=Stage.HARNESS_SUBMIT_AND_COLLECT,
        status=StageStatus.DONE,
        artifacts=output_artifacts,
        evidence_refs=("stage-12/execution_record.json", "stage-12/result_artifacts.json"),
    )


def _execute_experiment_route_decision(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    _ = adapters, llm, prompts
    execution_text = _read_prior_artifact(run_dir, "execution_record.json")
    if not execution_text:
        return _failed(
            Stage.EXPERIMENT_ROUTE_DECISION,
            "E13_ROUTE_FAIL: missing execution_record.json",
        )
    try:
        execution = json.loads(execution_text)
    except json.JSONDecodeError as exc:
        return _failed(
            Stage.EXPERIMENT_ROUTE_DECISION,
            f"E13_ROUTE_FAIL: invalid execution_record.json: {exc}",
        )
    if not isinstance(execution, dict):
        return _failed(
            Stage.EXPERIMENT_ROUTE_DECISION,
            "E13_ROUTE_FAIL: execution_record.json must contain an object",
        )

    artifacts = _load_json_text(_read_prior_artifact(run_dir, "result_artifacts.json"))
    validation = _load_json_text(_read_prior_artifact(run_dir, "manifest_validation.json"))
    diagnosis_path = run_dir / "experiment_diagnosis.json"
    diagnosis = _load_json_file(diagnosis_path)
    route, reason, details = _route_from_experiment_evidence(
        execution=execution,
        artifacts=artifacts,
        validation=validation,
        diagnosis=diagnosis,
    )
    iteration = _read_experiment_iteration_count(run_dir)
    generated = _utcnow_iso()
    decision = {
        "schema_version": "researchclaw.experiment_decision.v1",
        "route": route,
        "reason": reason,
        "evidence": _experiment_decision_evidence(
            execution=execution,
            diagnosis=diagnosis,
            config=config,
        ),
        "iteration": iteration,
        "max_iterations": 3,
        "generated": generated,
    }
    (stage_dir / "experiment_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if route == "fix_code":
        _write_repair_request(
            run_dir,
            origin_stage=13,
            reason=reason,
            errors=details,
            iteration=iteration,
            generated=generated,
            diagnosis_ref="experiment_diagnosis.json" if diagnosis_path.is_file() else "",
        )
    elif route == "revise_task_spec":
        _write_refine_request(
            run_dir,
            reason=reason,
            suggested_changes=details,
            iteration=iteration,
            generated=generated,
            diagnosis_ref="experiment_diagnosis.json" if diagnosis_path.is_file() else "",
        )

    return StageResult(
        stage=Stage.EXPERIMENT_ROUTE_DECISION,
        status=StageStatus.DONE,
        artifacts=("experiment_decision.json",),
        evidence_refs=("stage-13/experiment_decision.json",),
        decision=route,
    )


def _load_json_text(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _route_from_experiment_evidence(
    *,
    execution: dict[str, Any],
    artifacts: dict[str, Any],
    validation: dict[str, Any],
    diagnosis: dict[str, Any],
) -> tuple[str, str, list[str]]:
    _ = validation
    if _diagnosis_requires_task_spec_revision(diagnosis):
        return (
            "revise_task_spec",
            "task_spec_objective_mismatch",
            _diagnosis_details(diagnosis)
            or ["Revise the experiment task spec to align objective, metrics, and expected outputs."],
        )

    final_status = str(execution.get("final_status", "")).strip().lower()
    if final_status in {"queued", "pending", "running", "submitted"}:
        return ("rerun", "execution_incomplete", [f"final_status={final_status}"])

    successful = {"completed", "complete", "succeeded", "success", "done", "passed"}
    if final_status not in successful:
        status = final_status or "missing"
        return ("fix_code", "execution_failed", [f"final_status={status}"])

    metrics = execution.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return ("fix_code", "metrics_missing", ["execution_record.metrics is empty"])

    artifact_rows = artifacts.get("artifacts")
    if isinstance(artifact_rows, list) and artifact_rows:
        if not any(bool(item.get("exists")) for item in artifact_rows if isinstance(item, dict)):
            return ("fix_code", "result_artifacts_missing", ["all declared result artifacts are missing"])

    if _diagnosis_sufficient(diagnosis) is False and (
        _has_real_deficiency(diagnosis) or not _metrics_have_numeric_value(metrics)
    ):
        return (
            "fix_code",
            "diagnosis_insufficient",
            _diagnosis_details(diagnosis)
            or ["experiment_diagnosis.json reports insufficient experiment quality"],
        )

    return ("continue", "metrics meet objective; artifacts available", [])


def _metrics_have_numeric_value(metrics: Any) -> bool:
    """True if *metrics* (the execution_record.metrics dict) holds any finite number.

    Guards the phantom-deficiency relaxation: a non-empty metrics dict that
    nonetheless carries no real numeric result (e.g. ``{"status": "ok"}``) is a
    genuinely empty run, so ``no_conditions`` alone should still route fix_code.
    """
    if not isinstance(metrics, dict):
        return False

    def _finite(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return value == value and value not in (float("inf"), float("-inf"))

    for value in metrics.values():
        if _finite(value):
            return True
        if isinstance(value, dict):
            for sub in value.values():
                if _finite(sub):
                    return True
    return False


# FIX#3: deficiency types that, ON THEIR OWN, must NOT trigger a fix_code
# repair when the run otherwise succeeded (successful final_status, non-empty
# metrics, artifacts present). The empty-condition phantom was produced by the
# old hardcoded condition_summaries={}; with that fixed it should rarely appear,
# but this guards against a spurious diagnosis re-triggering a needless repair.
_PHANTOM_ONLY_DEFICIENCIES = frozenset({"no_conditions"})


def _has_real_deficiency(diagnosis: dict[str, Any]) -> bool:
    """True if the diagnosis carries a deficiency beyond the empty-condition phantom."""
    types: set[str] = set()
    diag = diagnosis.get("diagnosis")
    if isinstance(diag, dict):
        for item in diag.get("deficiencies", []) or []:
            if isinstance(item, dict) and item.get("type"):
                types.add(str(item.get("type")))
    quality = diagnosis.get("quality_assessment")
    if isinstance(quality, dict):
        for t in quality.get("deficiency_types", []) or []:
            types.add(str(t))
    if not types:
        # No enumerable deficiency types but sufficient is False — be
        # conservative and treat it as real so genuine problems still route.
        return True
    return bool(types - _PHANTOM_ONLY_DEFICIENCIES)


def _diagnosis_sufficient(diagnosis: dict[str, Any]) -> bool | None:
    if not diagnosis:
        return None
    quality = diagnosis.get("quality_assessment")
    if isinstance(quality, dict) and "sufficient" in quality:
        return bool(quality.get("sufficient"))
    if "diagnosis_sufficient" in diagnosis:
        return bool(diagnosis.get("diagnosis_sufficient"))
    if "sufficient" in diagnosis:
        return bool(diagnosis.get("sufficient"))
    return None


def _diagnosis_requires_task_spec_revision(diagnosis: dict[str, Any]) -> bool:
    if _diagnosis_sufficient(diagnosis) is not False:
        return False
    haystack = " ".join(_diagnosis_details(diagnosis)).lower()
    quality = diagnosis.get("quality_assessment")
    if isinstance(quality, dict):
        haystack += " " + " ".join(str(item) for item in quality.get("deficiency_types", []))
        if quality.get("repair_possible") is False:
            haystack += " repair_not_possible"
    markers = (
        "task_spec",
        "task spec",
        "objective_mismatch",
        "objective mismatch",
        "plan_mismatch",
        "expected_outputs",
    )
    return any(marker in haystack for marker in markers)


def _diagnosis_details(diagnosis: dict[str, Any]) -> list[str]:
    details: list[str] = []
    diag = diagnosis.get("diagnosis")
    if isinstance(diag, dict):
        deficiencies = diag.get("deficiencies")
        if isinstance(deficiencies, list):
            for item in deficiencies:
                if not isinstance(item, dict):
                    continue
                text = str(
                    item.get("description")
                    or item.get("summary")
                    or item.get("type")
                    or ""
                ).strip()
                if text:
                    details.append(text)
        summary = str(diag.get("summary", "")).strip()
        if summary:
            details.append(summary)
    quality = diagnosis.get("quality_assessment")
    if isinstance(quality, dict):
        details.extend(str(item) for item in quality.get("deficiency_types", []) if item)
    return details


def _experiment_decision_evidence(
    *,
    execution: dict[str, Any],
    diagnosis: dict[str, Any],
    config: RCConfig,
) -> dict[str, Any]:
    metrics = execution.get("metrics") if isinstance(execution.get("metrics"), dict) else {}
    metric_key = str(getattr(config.experiment, "metric_key", "primary_metric"))
    best_metric = metrics.get(metric_key)
    if best_metric is None:
        for value in metrics.values():
            if isinstance(value, (int, float)):
                best_metric = value
                break
    quality = diagnosis.get("quality_assessment") if isinstance(diagnosis, dict) else {}
    if not isinstance(quality, dict):
        quality = {}
    return {
        "primary_metric": metric_key,
        "best_metric": best_metric,
        "metric_direction": str(getattr(config.experiment, "metric_direction", "maximize")),
        "n_runs": int(execution.get("n_runs", 1 if metrics else 0) or 0),
        "diagnosis_mode": quality.get("mode"),
        "diagnosis_sufficient": _diagnosis_sufficient(diagnosis),
    }


def _write_repair_request(
    run_dir: Path,
    *,
    origin_stage: int,
    reason: str,
    errors: list[str],
    iteration: int,
    generated: str,
    diagnosis_ref: str,
) -> None:
    payload = {
        "schema_version": "researchclaw.repair_request.v1",
        "origin_stage": origin_stage,
        "reason": reason,
        "errors": errors or [reason],
        "diagnosis_ref": diagnosis_ref,
        "iteration": iteration,
        "generated": generated,
    }
    (run_dir / "repair_request.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_refine_request(
    run_dir: Path,
    *,
    reason: str,
    suggested_changes: list[str],
    iteration: int,
    generated: str,
    diagnosis_ref: str,
) -> None:
    payload = {
        "schema_version": "researchclaw.refine_request.v1",
        "origin_stage": 13,
        "reason": reason,
        "suggested_changes": suggested_changes or [reason],
        "diagnosis_ref": diagnosis_ref,
        "iteration": iteration,
        "generated": generated,
    }
    (run_dir / "refine_request.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_experiment_iteration_count(run_dir: Path) -> int:
    history = _load_json_file(run_dir / "experiment_loop_history.json")
    if not history:
        return 0
    entries = history.get("iterations") or history.get("history")
    if isinstance(entries, list):
        return len(entries)
    for key in ("iteration", "iterations", "count", "total_iterations"):
        value = history.get(key)
        if isinstance(value, int):
            return value
    return 0


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _latest_agent_result(run_dir: Path) -> dict[str, Any]:
    text = _read_prior_artifact(run_dir, "stage-10-workspace-agent-result.json")
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_result_artifacts(stage_dir: Path) -> list[dict[str, Any]]:
    path = stage_dir / "result_artifacts.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    return artifacts if isinstance(artifacts, list) else []


def _result_summary(execution_text: str) -> str:
    try:
        payload = json.loads(execution_text)
    except json.JSONDecodeError:
        return execution_text[:1000]
    if not isinstance(payload, dict):
        return execution_text[:1000]
    return json.dumps(
        {
            "final_status": payload.get("final_status"),
            "metrics": payload.get("metrics", {}),
            "result_paths": payload.get("result_paths", []),
            "result_hashes": payload.get("result_hashes", {}),
        },
        sort_keys=True,
    )


def _estimate_stage12_footprint_bytes(run_dir: Path) -> int:
    total = 0
    for directory in run_dir.glob("stage-12*"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def _failed(stage: Stage, error: str) -> StageResult:
    return StageResult(stage=stage, status=StageStatus.FAILED, artifacts=(), error=error)
