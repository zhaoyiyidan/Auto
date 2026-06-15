"""Stage 10: workspace-native code-agent implementation or repair."""

from __future__ import annotations

import json
import hashlib
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import StageResult, _read_prior_artifact
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _run_manifest_schema_example(expected_outputs: list[str]) -> str:
    result_paths = expected_outputs or ["outputs/results.json"]
    example = {
        "schema_version": "researchclaw.run_manifest.v1",
        "code_commit": "ACTUAL_GIT_COMMIT_SHA",
        "launch": {
            "command": "python scripts/run_experiment.py",
            "cwd": "/absolute/path/to/workspace",
            "env": {
                "PYTHONPATH": "/absolute/path/to/workspace:${PYTHONPATH:-}",
            },
            "resources": {
                "gpus": 0,
                "time": "01:00:00",
                "partition": "",
                "mem_gb": 16,
            },
        },
        "result_paths": result_paths,
    }
    return json.dumps(example, indent=2)


def _rendered_prompt_text(system: str, user: str) -> str:
    return f"{system}\n\n{user}" if system else user


def _check_rl_compatibility(code: str) -> list[str]:
    """Return obvious RL algorithm/environment compatibility warnings."""
    source = str(code or "").lower()
    if "dqn" not in source:
        return []
    continuous_envs = (
        "pendulum",
        "halfcheetah",
        "hopper",
        "walker2d",
        "ant-",
        "humanoid",
        "bipedalwalker",
        "mountaincarcontinuous",
        "lunarlandercontinuous",
    )
    matched = next((env for env in continuous_envs if env in source), "")
    if matched:
        return [
            "DQN is incompatible with continuous-action environments"
            f" such as {matched}."
        ]
    return []


def _workspace_codegen_prompt(
    *,
    topic: str,
    plan_md: str,
    expected_outputs: list[str],
    manifest_filename: str,
    prompt_manager: PromptManager | None = None,
) -> str:
    expected_json = json.dumps(
        {
            "schema_version": "researchclaw.expected_outputs.v1",
            "outputs": expected_outputs,
        },
        indent=2,
    )
    pm = prompt_manager or PromptManager()
    rendered = pm.sub_prompt(
        "workspace_codegen",
        topic=topic,
        plan_md=plan_md,
        expected_outputs_json=expected_json,
        manifest_filename=manifest_filename,
        manifest_schema_example=pm.block(
            "manifest_schema_example",
            manifest_example=_run_manifest_schema_example(expected_outputs),
        ),
        stage10_validation_boundary=pm.block("stage10_validation_boundary"),
    )
    return _rendered_prompt_text(rendered.system, rendered.user)


def _repair_or_refine_prompt(
    *,
    topic: str,
    plan_md: str,
    expected_outputs: list[str],
    project_files: list[str],
    run_summaries: list[str],
    manifest_filename: str,
    repair_request: dict[str, Any] | None = None,
    execution_record: str = "",
    result_artifacts: str = "",
    diagnosis: str = "",
    prompt_manager: PromptManager | None = None,
) -> str:
    summaries = "\n".join(run_summaries[:20]) if run_summaries else "(no prior runs)"
    request_section = ""
    if repair_request:
        request_section = (
            "REPAIR REQUEST:\n"
            f"{json.dumps(repair_request, indent=2, sort_keys=True)}\n\n"
        )
    if diagnosis:
        request_section += f"EXPERIMENT DIAGNOSIS:\n{diagnosis}\n\n"
    results_section = ""
    if execution_record or result_artifacts:
        results_section = (
            f"\n\nEXECUTION RECORD:\n{execution_record or '{}'}\n\n"
            f"RESULT ARTIFACTS:\n{result_artifacts or '{}'}\n"
        )
    expected_json = json.dumps(
        {
            "schema_version": "researchclaw.expected_outputs.v1",
            "outputs": expected_outputs,
        },
        indent=2,
    )
    pm = prompt_manager or PromptManager()
    rendered = pm.sub_prompt(
        "workspace_repair",
        request_section=request_section,
        topic=topic,
        plan_md=plan_md,
        expected_outputs_json=expected_json,
        run_summaries=summaries,
        project_files=json.dumps(project_files[:20], indent=2),
        results_section=results_section,
        manifest_filename=manifest_filename,
        manifest_schema_example=pm.block(
            "manifest_schema_example",
            manifest_example=_run_manifest_schema_example(expected_outputs),
        ),
        stage10_validation_boundary=pm.block("stage10_validation_boundary"),
    )
    return _rendered_prompt_text(rendered.system, rendered.user)


def _execute_code_agent_implement_or_repair(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    _ = adapters
    plan_md = _read_prior_artifact(run_dir, "plan.md")
    expected_outputs_text = _read_prior_artifact(run_dir, "expected_outputs.json")
    if not plan_md:
        return StageResult(
            stage=Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
            status=StageStatus.FAILED,
            artifacts=(),
            error="E10_CODE_AGENT_FAIL: missing plan.md",
        )
    if not expected_outputs_text:
        return StageResult(
            stage=Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
            status=StageStatus.FAILED,
            artifacts=(),
            error="E10_CODE_AGENT_FAIL: missing expected_outputs.json",
        )
    expected_outputs = _parse_expected_outputs(expected_outputs_text)
    if not expected_outputs:
        return StageResult(
            stage=Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
            status=StageStatus.FAILED,
            artifacts=(),
            error="E10_CODE_AGENT_FAIL: invalid expected_outputs.json",
        )

    from researchclaw.experiment import workspace_agent as workspace_agent_factory
    from researchclaw.pipeline import workspace_orchestrator

    workspace = Path(config.experiment.workspace_agent.workspace_path).resolve()
    manifest_filename = _manifest_filename(config)
    repair_request = _read_repair_request(run_dir)
    if repair_request:
        diagnosis = ""
        diagnosis_ref = str(repair_request.get("diagnosis_ref", "")).strip()
        if diagnosis_ref:
            diagnosis_path = run_dir / diagnosis_ref
            if diagnosis_path.is_file():
                diagnosis = diagnosis_path.read_text(encoding="utf-8")
        prompt = _repair_or_refine_prompt(
            topic=config.research.topic,
            plan_md=plan_md,
            expected_outputs=expected_outputs,
            project_files=[],
            run_summaries=[],
            manifest_filename=manifest_filename,
            repair_request=repair_request,
            execution_record=_read_prior_artifact(run_dir, "execution_record.json") or "",
            result_artifacts=_read_prior_artifact(run_dir, "result_artifacts.json") or "",
            diagnosis=diagnosis,
            prompt_manager=prompts,
        )
    else:
        prompt = _workspace_codegen_prompt(
            topic=config.research.topic,
            plan_md=plan_md,
            expected_outputs=expected_outputs,
            manifest_filename=manifest_filename,
            prompt_manager=prompts,
        )
    result_snapshot_before = _snapshot_result_paths(workspace, expected_outputs)
    agent = workspace_agent_factory.create_workspace_agent(
        config,
        llm=llm,
        prompts=prompts,
    )
    result = workspace_orchestrator.run_workspace_agent_implement(
        workspace_path=workspace,
        run_dir=stage_dir,
        stage=10,
        agent=agent,
        prompt=prompt,
        timeout_sec=int(getattr(config.experiment.workspace_agent, "timeout_sec", 600)),
        close_policy="keep",
    )
    _write_agent_result(stage_dir, result)
    touched_result_paths = _changed_result_paths(
        result_snapshot_before,
        _snapshot_result_paths(workspace, expected_outputs),
    )
    if touched_result_paths:
        return StageResult(
            stage=Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
            status=StageStatus.FAILED,
            artifacts=("stage-10-workspace-agent-result.json",),
            error=(
                "E10_CODE_AGENT_FAIL: Stage 10 created or modified final "
                "result artifacts: " + ", ".join(touched_result_paths)
            ),
        )
    if (
        not result.ok
        or result.agent_commit_sha is None
        or result.agent_commit_sha == result.base_sha
    ):
        return StageResult(
            stage=Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
            status=StageStatus.FAILED,
            artifacts=("stage-10-workspace-agent-result.json",),
            error=f"E10_CODE_AGENT_FAIL: {result.error or 'agent did not commit'}",
        )

    manifest_source = _manifest_source(workspace, result.manifest_path, manifest_filename)
    if manifest_source is None:
        return StageResult(
            stage=Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
            status=StageStatus.FAILED,
            artifacts=("stage-10-workspace-agent-result.json",),
            error="E10_CODE_AGENT_FAIL: missing run_manifest.json",
        )
    shutil.copy2(manifest_source, stage_dir / "run_manifest.json")
    if repair_request:
        _consume_repair_request(run_dir, repair_request)
    return StageResult(
        stage=Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
        status=StageStatus.DONE,
        artifacts=("stage-10-workspace-agent-result.json", "run_manifest.json"),
        evidence_refs=(
            "stage-10/stage-10-workspace-agent-result.json",
            "stage-10/run_manifest.json",
        ),
    )


def _execute_code_generation(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    return _execute_code_agent_implement_or_repair(
        stage_dir,
        run_dir,
        config,
        adapters,
        llm=llm,
        prompts=prompts,
    )


def _read_repair_request(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "repair_request.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _consume_repair_request(run_dir: Path, repair_request: dict[str, Any]) -> None:
    source = run_dir / "repair_request.json"
    if not source.is_file():
        return
    iteration = repair_request.get("iteration", 0)
    if not isinstance(iteration, int):
        iteration = 0
    target = run_dir / f"repair_request_consumed_v{iteration}.json"
    suffix = 1
    while target.exists():
        target = run_dir / f"repair_request_consumed_v{iteration}_{suffix}.json"
        suffix += 1
    source.replace(target)


def _parse_expected_outputs(text: str) -> list[str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    if not isinstance(outputs, list):
        return []
    return [
        str(item).strip()
        for item in outputs
        if isinstance(item, str) and item.strip()
    ]


def _manifest_filename(config: RCConfig) -> str:
    return str(
        getattr(
            getattr(config.experiment, "workspace_agent", None),
            "manifest_filename",
            "run_manifest.json",
        )
        or "run_manifest.json"
    )


def _snapshot_result_paths(
    workspace: Path,
    result_paths: list[str],
) -> dict[str, dict[str, object]]:
    snapshot: dict[str, dict[str, object]] = {}
    root = workspace.resolve()
    for rel in result_paths:
        rel_path = str(rel).strip()
        if not rel_path:
            continue
        path = (root / rel_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            snapshot[rel_path] = {"exists": False, "outside_workspace": True}
            continue
        snapshot[rel_path] = _snapshot_one_path(path)
    return snapshot


def _snapshot_one_path(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    if path.is_file():
        stat = path.stat()
        return {
            "exists": True,
            "type": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256_file(path),
        }
    if path.is_dir():
        files: dict[str, dict[str, object]] = {}
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            stat = child.stat()
            files[child.relative_to(path).as_posix()] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256_file(child),
            }
        return {"exists": True, "type": "dir", "files": files}
    return {"exists": True, "type": "other"}


def _changed_result_paths(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
) -> list[str]:
    changed: list[str] = []
    for rel in sorted(set(before) | set(after)):
        if before.get(rel) != after.get(rel):
            changed.append(rel)
    return changed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_source(
    workspace: Path,
    manifest_path: str | None,
    manifest_filename: str,
) -> Path | None:
    candidates: list[Path] = []
    if manifest_path:
        path = Path(manifest_path)
        candidates.append(path if path.is_absolute() else workspace / path)
    candidates.extend(
        [
            workspace / manifest_filename,
            workspace / ".researchclaw" / manifest_filename,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _write_agent_result(stage_dir: Path, result: object) -> None:
    (stage_dir / "stage-10-workspace-agent-result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
