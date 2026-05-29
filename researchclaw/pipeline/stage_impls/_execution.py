"""Stages 11-13: manifest validation, harness execution, and code-agent refine."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
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


def _workspace_refine_prompt(
    *,
    topic: str,
    metric_key: str,
    metric_direction: str,
    exp_plan: str,
    project_files: list[str],
    run_summaries: list[str],
    manifest_filename: str,
    execution_record: str = "",
    result_artifacts: str = "",
) -> str:
    summaries = "\n".join(run_summaries[:20]) if run_summaries else "(no prior runs)"
    results_section = ""
    if execution_record or result_artifacts:
        results_section = (
            f"\n\nEXECUTION RECORD:\n{execution_record or '{}'}\n\n"
            f"RESULT ARTIFACTS:\n{result_artifacts or '{}'}\n"
        )
    return (
        "You are a workspace-native code agent working inside an existing git "
        "repository. Improve the experiment in place. Do not emit code blocks "
        "for ResearchClaw to parse.\n\n"
        f"TOPIC:\n{topic}\n\n"
        f"TARGET: {metric_direction} {metric_key}\n\n"
        f"ORIGINAL EXPERIMENT PLAN:\n{exp_plan}\n\n"
        f"KNOWN PROJECT FILES FROM PRIOR STAGES:\n{project_files}\n\n"
        f"PRIOR RUN SUMMARIES:\n{summaries}\n\n"
        f"{results_section}"
        "Completion contract (MUST):\n"
        "1. MUST inspect the existing workspace before editing.\n"
        "2. MUST improve the existing repository in place, using its structure.\n"
        "3. MUST prepare a launch command or script for the improved run.\n"
        "4. MUST git add and git commit the code changes you made.\n"
        f"5. MUST write {manifest_filename} in the workspace root or .researchclaw/.\n"
        "6. MUST include code_commit, launch.command, launch.cwd, launch.env, "
        "launch.resources, and result_paths in the manifest.\n\n"
        "Boundaries (MUST NOT):\n"
        "1. MUST NOT submit the job yourself. Do not submit the job yourself; "
        "ResearchClaw's submitter will run the manifest command.\n"
        "2. MUST NOT fabricate a job_id or final result registry entry.\n"
        "3. MUST NOT assume a fixed entrypoint, file layout, or script name.\n"
        "4. MUST NOT emit code blocks for ResearchClaw to parse as the output.\n"
    )


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
    if not validation.ok:
        manifest = _ask_agent_to_fix_manifest(
            config=config,
            stage_dir=stage_dir,
            workspace=workspace,
            errors=validation.errors,
            manifest=manifest,
        ) or manifest
        validation = validate_manifest(manifest, workspace, allow_dirty=False)

    (stage_dir / "manifest_validation.json").write_text(
        json.dumps(validation.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not validation.ok:
        return StageResult(
            stage=Stage.MANIFEST_VALIDATE_AND_PREPARE,
            status=StageStatus.FAILED,
            artifacts=("manifest_validation.json",),
            error="E11_MANIFEST_INVALID: " + "; ".join(validation.errors),
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


def _execute_iterative_refine(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    return _execute_code_agent_refine(
        stage_dir, run_dir, config, adapters, llm=llm, prompts=prompts
    )


def _execute_code_agent_refine(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    _ = adapters
    execution_text = _read_prior_artifact(run_dir, "execution_record.json")
    artifacts_text = _read_prior_artifact(run_dir, "result_artifacts.json") or "{}"
    if not execution_text:
        return _failed(Stage.CODE_AGENT_REFINE, "E13_REFINE_FAIL: missing execution_record.json")

    from researchclaw.experiment import workspace_agent as workspace_agent_factory
    from researchclaw.pipeline import workspace_orchestrator

    workspace = Path(config.experiment.workspace_agent.workspace_path).resolve()
    manifest_filename = str(
        getattr(config.experiment.workspace_agent, "manifest_filename", "run_manifest.json")
        or "run_manifest.json"
    )
    prompt = _workspace_refine_prompt(
        topic=config.research.topic,
        metric_key=config.experiment.metric_key,
        metric_direction=config.experiment.metric_direction,
        exp_plan=_read_prior_artifact(run_dir, "task_spec.yaml") or "",
        project_files=[],
        run_summaries=[_result_summary(execution_text)],
        manifest_filename=manifest_filename,
        execution_record=execution_text,
        result_artifacts=artifacts_text,
    )
    agent = workspace_agent_factory.create_workspace_agent(config, llm=llm, prompts=prompts)
    result = workspace_orchestrator.run_workspace_agent_implement(
        workspace_path=workspace,
        run_dir=stage_dir,
        stage=13,
        agent=agent,
        prompt=prompt,
        timeout_sec=int(getattr(config.experiment.workspace_agent, "timeout_sec", 600)),
        close_policy="keep",
    )
    (stage_dir / "stage-13-workspace-agent-result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not result.ok or result.agent_commit_sha is None or result.agent_commit_sha == result.base_sha:
        return StageResult(
            stage=Stage.CODE_AGENT_REFINE,
            status=StageStatus.FAILED,
            artifacts=("stage-13-workspace-agent-result.json",),
            error=f"E13_REFINE_FAIL: {result.error or 'agent did not commit'}",
        )
    manifest_source = _manifest_source(workspace, result.manifest_path, manifest_filename)
    if manifest_source is None:
        return StageResult(
            stage=Stage.CODE_AGENT_REFINE,
            status=StageStatus.FAILED,
            artifacts=("stage-13-workspace-agent-result.json",),
            error="E13_REFINE_FAIL: missing run_manifest.json",
        )
    shutil.copy2(manifest_source, stage_dir / "run_manifest.json")
    refine_record = asdict(result)
    refine_record["source_execution_record"] = "execution_record.json"
    refine_record["source_result_artifacts"] = "result_artifacts.json"
    (stage_dir / "refine_record.json").write_text(
        json.dumps(refine_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.CODE_AGENT_REFINE,
        status=StageStatus.DONE,
        artifacts=("refine_record.json", "run_manifest.json"),
        evidence_refs=("stage-13/refine_record.json", "stage-13/run_manifest.json"),
    )


def _ask_agent_to_fix_manifest(
    *,
    config: RCConfig,
    stage_dir: Path,
    workspace: Path,
    errors: list[str],
    manifest: RunManifest,
) -> RunManifest | None:
    from researchclaw.experiment import workspace_agent as workspace_agent_factory
    from researchclaw.pipeline import workspace_orchestrator

    prompt = (
        "The current run_manifest.json failed ResearchClaw validation.\n"
        "Fix only the manifest and any minimal supporting files required for it. "
        "Do not submit the job.\n\n"
        "Validation errors:\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n\nCurrent manifest:\n"
        + manifest.to_json()
    )
    agent = workspace_agent_factory.create_workspace_agent(config)
    result = workspace_orchestrator.run_workspace_agent_implement(
        workspace_path=workspace,
        run_dir=stage_dir,
        stage=11,
        agent=agent,
        prompt=prompt,
        timeout_sec=int(getattr(config.experiment.workspace_agent, "timeout_sec", 600)),
        close_policy="keep",
    )
    manifest_path = _manifest_source(
        workspace,
        result.manifest_path,
        str(getattr(config.experiment.workspace_agent, "manifest_filename", "run_manifest.json")),
    )
    if manifest_path is None:
        return None
    shutil.copy2(manifest_path, stage_dir / "run_manifest.json")
    return RunManifest.from_path(manifest_path)


def _manifest_source(
    workspace: Path,
    manifest_path: str | None,
    manifest_filename: str,
) -> Path | None:
    candidates: list[Path] = []
    if manifest_path:
        path = Path(manifest_path)
        candidates.append(path if path.is_absolute() else workspace / path)
    candidates.extend([workspace / manifest_filename, workspace / ".researchclaw" / manifest_filename])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _latest_agent_result(run_dir: Path) -> dict[str, Any]:
    text = _read_prior_artifact(run_dir, "stage-10-workspace-agent-result.json")
    if not text:
        text = _read_prior_artifact(run_dir, "stage-13-workspace-agent-result.json")
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
