"""Stage 9: workspace-native experiment task specification."""

from __future__ import annotations

import logging
import json
from pathlib import Path

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.domains.detector import detect_domain
from researchclaw.experiment.execution_contract import default_contract
from researchclaw.experiment.protocol import (
    ComparisonSpec,
    DecisionRule,
    ExperimentProtocol,
    MetricSpec,
    parse_hypotheses_md,
)
from researchclaw.experiment.workspace import TaskSpec
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import (
    StageResult,
    _chat_with_prompt,
    _read_prior_artifact,
    _safe_json_loads,
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
    hypotheses_md = _read_prior_artifact(run_dir, "hypotheses.md") or ""
    hint = _domain_metric_hint(config)
    protocol = _protocol_from_llm(config, hypotheses_md, hint, llm)
    if protocol is None:
        protocol = _template_protocol(config, hypotheses_md, hint)
    spec = _task_spec_from_protocol(config, protocol, hypotheses_md)

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "experiment_protocol.json").write_text(
        protocol.to_json(),
        encoding="utf-8",
    )
    (stage_dir / "task_spec.yaml").write_text(spec.to_yaml(), encoding="utf-8")
    return StageResult(
        stage=Stage.EXPERIMENT_TASK_SPEC,
        status=StageStatus.DONE,
        artifacts=("experiment_protocol.json", "task_spec.yaml"),
        evidence_refs=("stage-09/experiment_protocol.json", "stage-09/task_spec.yaml"),
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


def _domain_metric_hint(config: RCConfig) -> tuple[str, str]:
    try:
        profile = detect_domain(topic=str(config.research.topic or ""))
        name = str(profile.default_metric_key or "").strip() or "primary_metric"
        direction = str(profile.default_metric_direction or "").strip() or "maximize"
        if direction not in {"maximize", "minimize"}:
            direction = "maximize"
        return name, direction
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stage 09: failed to resolve domain metric hint: %s", exc)
        return "primary_metric", "maximize"


def _protocol_from_llm(
    config: RCConfig,
    hypotheses_md: str,
    hint: tuple[str, str],
    llm: LLMClient | None,
) -> ExperimentProtocol | None:
    if llm is None:
        return None
    parsed_hypotheses = [item.to_dict() for item in parse_hypotheses_md(hypotheses_md)]
    schema = {
        "schema_version": "researchclaw.experiment_protocol.v1",
        "objective": "string",
        "hypotheses": [
            {
                "id": "H1",
                "statement": "string",
                "prediction": "string",
                "falsification": "string",
                "rationale": "string",
                "baselines": ["baseline"],
            }
        ],
        "metrics": [
            {
                "name": "designed_metric_name",
                "direction": "maximize|minimize",
                "unit": "string",
                "description": "string",
                "hypothesis_ids": ["H1"],
                "is_primary": True,
            }
        ],
        "comparisons": [
            {
                "id": "C1",
                "kind": "baseline_vs_treatment|conditions|ablation",
                "baseline": "string",
                "treatment": "string",
                "conditions": ["control", "treatment"],
                "metric": "designed_metric_name",
                "hypothesis_ids": ["H1"],
            }
        ],
        "decision_rules": [
            {
                "hypothesis_id": "H1",
                "metric": "designed_metric_name",
                "comparator": "gt|gte|lt|lte|delta_gt|delta_lt|abs_delta_lt|within_pct",
                "threshold": 0.0,
                "baseline_metric": "",
                "supported_if": "pass",
                "description": "string",
            }
        ],
    }
    prompt = (
        "Design a ResearchClaw experiment protocol as JSON only.\n"
        "The protocol must translate the hypotheses into executable metrics, "
        "comparisons, and decision rules. Design the metric from the hypothesis; "
        "the domain metric hint is only a starting point and may be overridden "
        "if a more scientific metric is needed.\n\n"
        f"Topic: {config.research.topic}\n"
        f"Domain metric hint: {hint[0]} ({hint[1]})\n"
        "Parsed hypotheses JSON:\n"
        f"{json.dumps(parsed_hypotheses, indent=2, sort_keys=True)}\n\n"
        "Return a JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2, sort_keys=True)}"
    )
    try:
        response = _chat_with_prompt(
            llm,
            "You output only valid JSON for an experiment protocol.",
            prompt,
            json_mode=True,
            max_tokens=2048,
        )
        payload = _safe_json_loads(response.content, None)
        if not isinstance(payload, dict):
            return None
        protocol = ExperimentProtocol.from_dict(payload)
        warnings = protocol.validate()
        for warning in warnings:
            logger.warning("Stage 09 protocol validation warning: %s", warning)
        return protocol
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stage 09: failed to parse LLM protocol: %s", exc)
        return None


def _template_protocol(
    config: RCConfig,
    hypotheses_md: str,
    hint: tuple[str, str],
) -> ExperimentProtocol:
    hypotheses = parse_hypotheses_md(hypotheses_md)
    metric = MetricSpec(
        name=hint[0],
        direction=hint[1],
        description="Primary metric selected from the detected domain profile.",
        hypothesis_ids=tuple(item.id for item in hypotheses),
        is_primary=True,
    )
    comparison = ComparisonSpec(
        id="C1",
        baseline="baseline",
        treatment="proposed",
        conditions=("baseline", "proposed"),
        metric=metric.name,
        hypothesis_ids=tuple(item.id for item in hypotheses),
    )
    comparator = "gt" if metric.direction == "maximize" else "lt"
    return ExperimentProtocol(
        objective=f"Design and execute an experiment for topic: {config.research.topic}",
        hypotheses=hypotheses,
        metrics=(metric,),
        comparisons=(comparison,),
        decision_rules=tuple(
            DecisionRule(
                hypothesis_id=hypothesis.id,
                metric=metric.name,
                comparator=comparator,
                threshold=0.0,
                supported_if="pass",
                description=(
                    f"Support {hypothesis.id} when {metric.name} satisfies "
                    f"{comparator} 0.0 under the planned comparison."
                ),
            )
            for hypothesis in hypotheses
        ),
    )


def _task_spec_from_protocol(
    config: RCConfig,
    protocol: ExperimentProtocol,
    hypotheses_md: str,
) -> TaskSpec:
    primary = protocol.primary_metric()
    objective = (
        f"Implement and evaluate the Experiment Protocol for topic: {config.research.topic}."
    )
    if protocol.objective.strip():
        objective += "\n\nProtocol objective:\n" + protocol.objective.strip()
    if hypotheses_md.strip():
        objective += "\n\nHypotheses:\n" + hypotheses_md.strip()
    objective += "\n\nProtocol artifact: stage-09/experiment_protocol.json"
    return TaskSpec(
        workspace=_workspace_path(config),
        objective=objective,
        constraints=[
            "Use the existing workspace; do not create a fresh project elsewhere.",
            f"Respect the configured time budget: {config.experiment.time_budget_sec} seconds.",
            "Commit all code changes before writing run_manifest.json.",
            f"Report the protocol primary metric `{primary.name}` with direction `{primary.direction}`.",
        ],
        primary_metric=primary.name,
        metric_direction=primary.direction,
        allowed_scope=["."],
        forbidden_scope=[".git/", ".researchclaw/"],
        expected_outputs=["outputs/metrics.json"],
        execution_contract=default_contract(
            primary_metric=primary.name,
            metric_direction=primary.direction,
            expected_outputs=["outputs/metrics.json"],
        ),
    )


def _workspace_path(config: RCConfig) -> str:
    workspace_agent = getattr(config.experiment, "workspace_agent", None)
    return str(getattr(workspace_agent, "workspace_path", "."))
