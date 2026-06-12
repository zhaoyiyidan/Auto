"""Stage 10: workspace-native code-agent implementation or repair."""

from __future__ import annotations

import json
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
) -> str:
    expected_json = json.dumps(
        {
            "schema_version": "researchclaw.expected_outputs.v1",
            "outputs": expected_outputs,
        },
        indent=2,
    )
    return (
        "You are a workspace-native code agent working inside an existing git "
        "repository. Implement the experiment described by Stage 9. Do not "
        "redesign the experiment.\n\n"
        f"TOPIC:\n{topic}\n\n"
        f"STAGE 9 PLAN (plan.md):\n{plan_md}\n\n"
        f"EXPECTED OUTPUTS (expected_outputs.json):\n{expected_json}\n\n"
        "You MUST modify the workspace as needed to implement and run this "
        "experiment. You MUST write or update "
        f"{manifest_filename}. The manifest result_paths MUST include every "
        "path listed in expected_outputs.json. Extra result paths are allowed.\n\n"
        "Required run_manifest.json format example:\n"
        f"{_run_manifest_schema_example(expected_outputs)}\n\n"
        "Make a final git commit containing every task change, including the "
        "manifest. Set code_commit to the final HEAD and finish with clean git "
        "status."
    )


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
    return (
        "You are repairing or refining a workspace experiment implementation. "
        "Do not redesign the Stage 9 experiment plan; fix the implementation "
        "so it satisfies the plan and expected outputs.\n\n"
        f"{request_section}"
        f"TOPIC:\n{topic}\n\n"
        f"STAGE 9 PLAN:\n{plan_md}\n\n"
        f"EXPECTED OUTPUTS:\n{expected_json}\n\n"
        f"PRIOR RUN SUMMARIES:\n{summaries}\n\n"
        f"PROJECT FILES:\n{json.dumps(project_files[:20], indent=2)}"
        f"{results_section}\n\n"
        f"Write or update {manifest_filename}; result_paths must include every "
        "expected output path. Required run_manifest.json format example:\n"
        f"{_run_manifest_schema_example(expected_outputs)}\n\n"
        "Make a final git commit containing every task change and finish with "
        "clean git status."
    )


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
        )
    else:
        prompt = _workspace_codegen_prompt(
            topic=config.research.topic,
            plan_md=plan_md,
            expected_outputs=expected_outputs,
            manifest_filename=manifest_filename,
        )
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
