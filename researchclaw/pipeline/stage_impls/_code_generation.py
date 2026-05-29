"""Stage 10: workspace-native code-agent implementation."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.experiment.workspace import TaskSpec
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import StageResult, _read_prior_artifact
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


_CONTINUOUS_ENVS = {
    "pendulum",
    "halfcheetah",
    "hopper",
    "walker2d",
    "ant",
    "humanoid",
    "swimmer",
    "reacher",
    "invertedpendulum",
    "inverteddoublependulum",
    "mountaincarcontinuous",
    "lunarlander-continuous",
}


def _workspace_codegen_prompt(
    *,
    topic: str,
    exp_plan: str,
    metric: str,
    pkg_hint: str,
    compute_budget: str,
    extra_guidance: str,
    manifest_filename: str,
) -> str:
    return (
        "You are a workspace-native code agent working inside an existing git "
        "repository. Modify this repository to implement the experiment. Do not "
        "emit code blocks for ResearchClaw to parse.\n\n"
        f"TOPIC:\n{topic}\n\n"
        f"TASK SPEC:\n{exp_plan}\n\n"
        f"PRIMARY METRIC: {metric}\n\n"
        f"PACKAGE HINTS:\n{pkg_hint}\n\n"
        f"COMPUTE BUDGET:\n{compute_budget}\n\n"
        f"EXTRA GUIDANCE:\n{extra_guidance}\n\n"
        "Completion contract (MUST):\n"
        "1. MUST inspect the existing workspace before editing.\n"
        "2. MUST modify the existing repository in place, using its structure.\n"
        "3. MUST prepare a launch command or script for the experiment run.\n"
        "4. MUST git add and git commit the code changes you made.\n"
        f"5. MUST write {manifest_filename} in the workspace root or .researchclaw/.\n"
        "6. MUST include schema_version, code_commit, launch.command, launch.cwd, "
        "launch.env, launch.resources, result_paths, and metrics in the manifest.\n\n"
        "Boundaries (MUST NOT):\n"
        "1. MUST NOT submit the job yourself. Do not submit the job yourself; "
        "ResearchClaw's submitter will run the manifest command.\n"
        "2. MUST NOT fabricate a job_id or final result registry entry.\n"
        "3. MUST NOT assume a fixed entrypoint, file layout, or script name.\n"
        "4. MUST NOT emit code blocks for ResearchClaw to parse as the output.\n"
    )


def _execute_code_agent_implement(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    _ = adapters
    task_spec_text = _read_prior_artifact(run_dir, "task_spec.yaml")
    if not task_spec_text:
        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.FAILED,
            artifacts=(),
            error="E10_CODE_AGENT_FAIL: missing task_spec.yaml",
        )
    try:
        task_spec = TaskSpec.from_yaml(task_spec_text)
    except Exception as exc:  # noqa: BLE001
        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"E10_CODE_AGENT_FAIL: invalid task_spec.yaml: {exc}",
        )

    from researchclaw.experiment import workspace_agent as workspace_agent_factory
    from researchclaw.pipeline import workspace_orchestrator

    workspace = _resolve_workspace_path(config, task_spec.workspace)
    manifest_filename = _manifest_filename(config)
    prompt = _workspace_codegen_prompt(
        topic=config.research.topic,
        exp_plan=task_spec_text,
        metric=task_spec.primary_metric,
        pkg_hint="",
        compute_budget="\n".join(task_spec.constraints),
        extra_guidance="",
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
            stage=Stage.CODE_GENERATION,
            status=StageStatus.FAILED,
            artifacts=("stage-10-workspace-agent-result.json",),
            error=f"E10_CODE_AGENT_FAIL: {result.error or 'agent did not commit'}",
        )

    manifest_source = _manifest_source(workspace, result.manifest_path, manifest_filename)
    if manifest_source is None:
        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.FAILED,
            artifacts=("stage-10-workspace-agent-result.json",),
            error="E10_CODE_AGENT_FAIL: missing run_manifest.json",
        )
    shutil.copy2(manifest_source, stage_dir / "run_manifest.json")
    return StageResult(
        stage=Stage.CODE_GENERATION,
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
    return _execute_code_agent_implement(
        stage_dir,
        run_dir,
        config,
        adapters,
        llm=llm,
        prompts=prompts,
    )


def _resolve_workspace_path(config: RCConfig, workspace: str) -> Path:
    configured = Path(config.experiment.workspace_agent.workspace_path)
    if not workspace or workspace == ".":
        return configured
    path = Path(workspace).expanduser()
    if path.is_absolute():
        return path
    return configured.parent / path


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


def _check_rl_compatibility(code: str) -> list[str]:
    errors: list[str] = []
    code_lower = code.lower()
    if "dqn" not in code_lower:
        return errors
    for env_name in _CONTINUOUS_ENVS:
        if env_name in code_lower:
            errors.append(
                f"RL COMPATIBILITY ERROR: DQN is used with continuous-action "
                f"environment '{env_name}'. DQN only works with DISCRETE action "
                "spaces. Use SAC, TD3, or PPO instead."
            )
    return errors
