"""Stage 9: workspace-native experiment task specification."""

from __future__ import annotations

import logging
from pathlib import Path

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.experiment.workspace import TaskSpec
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import (
    StageResult,
    _chat_with_prompt,
    _extract_yaml_block,
    _read_prior_artifact,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _execute_experiment_task_spec(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    _ = adapters, prompts
    hypotheses = _read_prior_artifact(run_dir, "hypotheses.md") or ""
    spec = _task_spec_from_llm(config, hypotheses, llm)
    if spec is None:
        spec = _template_task_spec(config, hypotheses)

    (stage_dir / "task_spec.yaml").write_text(spec.to_yaml(), encoding="utf-8")
    return StageResult(
        stage=Stage.EXPERIMENT_DESIGN,
        status=StageStatus.DONE,
        artifacts=("task_spec.yaml",),
        evidence_refs=("stage-09/task_spec.yaml",),
    )


def _execute_experiment_design(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    return _execute_experiment_task_spec(
        stage_dir,
        run_dir,
        config,
        adapters,
        llm=llm,
        prompts=prompts,
    )


def _task_spec_from_llm(
    config: RCConfig,
    hypotheses: str,
    llm: LLMClient | None,
) -> TaskSpec | None:
    if llm is None:
        return None
    workspace = _workspace_path(config)
    prompt = (
        "Create a ResearchClaw code-agent task spec as YAML only.\n"
        "Required keys: workspace, objective, constraints, primary_metric, "
        "metric_direction, allowed_scope, forbidden_scope, expected_outputs.\n"
        f"Workspace: {workspace}\n"
        f"Topic: {config.research.topic}\n"
        f"Primary metric: {config.experiment.metric_key}\n"
        f"Metric direction: {config.experiment.metric_direction}\n"
        "The code agent will edit this existing workspace and must write "
        "run_manifest.json after committing code.\n\n"
        f"Hypotheses:\n{hypotheses}"
    )
    try:
        response = _chat_with_prompt(
            llm,
            "You output only valid YAML for a code-agent task specification.",
            prompt,
            max_tokens=2048,
        )
        return TaskSpec.from_yaml(_extract_yaml_block(response.content))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stage 09: failed to parse LLM task spec: %s", exc)
        return None


def _template_task_spec(config: RCConfig, hypotheses: str) -> TaskSpec:
    objective = (
        f"Implement and evaluate the experiment for topic: {config.research.topic}."
    )
    if hypotheses.strip():
        objective += "\n\nHypotheses:\n" + hypotheses.strip()
    return TaskSpec(
        workspace=_workspace_path(config),
        objective=objective,
        constraints=[
            "Use the existing workspace; do not create a fresh project elsewhere.",
            f"Respect the configured time budget: {config.experiment.time_budget_sec} seconds.",
            "Commit all code changes before writing run_manifest.json.",
        ],
        primary_metric=str(config.experiment.metric_key),
        metric_direction=str(config.experiment.metric_direction),
        allowed_scope=["."],
        forbidden_scope=[".git/", ".researchclaw/"],
        expected_outputs=["outputs/metrics.json"],
    )


def _workspace_path(config: RCConfig) -> str:
    workspace_agent = getattr(config.experiment, "workspace_agent", None)
    return str(getattr(workspace_agent, "workspace_path", "."))
