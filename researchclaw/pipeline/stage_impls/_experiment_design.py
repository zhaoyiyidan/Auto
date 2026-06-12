"""Stage 9: workspace-aware experiment planning."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
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

EXPECTED_OUTPUTS_SCHEMA_VERSION = "researchclaw.expected_outputs.v1"


@dataclass(frozen=True)
class PlanningContext:
    hypotheses_md: str
    topic: str
    workspace_path: str
    workspace_tree: str
    readme: str
    dependencies: str
    human_guidance: str
    hardware_profile: str


def validate_expected_outputs(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["expected_outputs.json must contain an object"]
    if data.get("schema_version") != EXPECTED_OUTPUTS_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {EXPECTED_OUTPUTS_SCHEMA_VERSION}"
        )
    outputs = data.get("outputs")
    if not isinstance(outputs, list):
        errors.append("outputs must be a list")
        return errors
    if not outputs:
        errors.append("outputs must be a non-empty list")
        return errors
    for index, item in enumerate(outputs):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"outputs[{index}] must be a non-empty string")
            continue
        errors.extend(_validate_output_path(item.strip(), index))
    return errors


def _execute_experiment_design(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    _ = adapters, prompts
    if llm is None:
        return _failed("E09_PLAN_INVALID: planning agent unavailable")

    context = _collect_planning_context(config, run_dir)
    try:
        plan_md = _generate_plan_md(llm, context)
        expected_outputs = _generate_expected_outputs(llm, context, plan_md)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stage 09 planning agent failed: %s", exc)
        return _failed(f"E09_PLAN_INVALID: planning agent failed: {exc}")

    output_errors = validate_expected_outputs(expected_outputs)
    if output_errors:
        return _failed("E09_PLAN_INVALID: " + "; ".join(output_errors))

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "plan.md").write_text(plan_md.strip() + "\n", encoding="utf-8")
    (stage_dir / "expected_outputs.json").write_text(
        json.dumps(expected_outputs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.EXPERIMENT_TASK_SPEC,
        status=StageStatus.DONE,
        artifacts=("plan.md", "expected_outputs.json"),
        evidence_refs=("stage-09/plan.md", "stage-09/expected_outputs.json"),
    )


def _collect_planning_context(config: RCConfig, run_dir: Path) -> PlanningContext:
    workspace = Path(config.experiment.workspace_agent.workspace_path)
    return PlanningContext(
        hypotheses_md=_read_prior_artifact(run_dir, "hypotheses.md") or "",
        topic=str(config.research.topic or ""),
        workspace_path=str(workspace),
        workspace_tree=_workspace_tree(workspace),
        readme=_first_existing_text(workspace, ("README.md", "README.rst", "README.txt")),
        dependencies=_dependency_text(workspace),
        human_guidance=_collect_human_guidance(run_dir),
        hardware_profile=_read_prior_artifact(run_dir, "hardware_profile.json") or "",
    )


def _generate_plan_md(llm: LLMClient, context: PlanningContext) -> str:
    prompt = (
        "You are a read-only experiment planning agent. Inspect the supplied "
        "workspace context and research context, then write only plan.md content. "
        "Do not write code. Do not propose editing files yourself.\n\n"
        "The plan must include sections for Hypotheses, Baselines, Ablations, "
        "Metrics, Decision Criteria, The details plans (more specifical, more better) to  Validate the Hypothesis, and "
        "Expected Outputs. Be specific enough "
        "for a later code agent to implement the experiment without inventing "
        "a new design.\n\n"
        f"Topic:\n{context.topic}\n\n"
        f"Hypotheses:\n{context.hypotheses_md}\n\n"
        f"Human guidance:\n{context.human_guidance}\n\n"
        f"Hardware profile:\n{context.hardware_profile}\n\n"
        f"Workspace path:\n{context.workspace_path}\n\n"
        f"Workspace tree:\n{context.workspace_tree}\n\n"
        f"README:\n{context.readme}\n\n"
        f"Dependencies:\n{context.dependencies}\n"
    )
    response = _chat_with_prompt(
        llm,
        "You write concise, actionable experiment plans in Markdown.",
        prompt,
        json_mode=False,
        max_tokens=4000,
    )
    return str(getattr(response, "content", "") or "").strip()


def _generate_expected_outputs(
    llm: LLMClient,
    context: PlanningContext,
    plan_md: str,
) -> dict[str, Any]:
    prompt = (
        "Based on the experiment plan below, return JSON only. The JSON must "
        "have schema_version 'researchclaw.expected_outputs.v1' and an outputs "
        "array listing relative workspace paths that the later experiment run "
        "must produce. Do not include metric schemas.\n\n"
        f"Workspace path: {context.workspace_path}\n\n"
        f"Plan:\n{plan_md}\n"
    )
    response = _chat_with_prompt(
        llm,
        "You output only valid JSON for expected experiment output paths.",
        prompt,
        json_mode=True,
        max_tokens=1000,
    )
    payload = _safe_json_loads(str(getattr(response, "content", "") or ""), None)
    if not isinstance(payload, dict):
        raise ValueError("planning agent did not return a JSON object")
    return payload


def _validate_output_path(path: str, index: int) -> list[str]:
    errors: list[str] = []
    pure = PurePosixPath(path.replace("\\", "/"))
    if pure.is_absolute() or Path(path).is_absolute():
        errors.append(f"outputs[{index}] must be a relative path, not absolute")
    if ".." in pure.parts:
        errors.append(f"outputs[{index}] must not contain ..")
    if pure.parts and pure.parts[0] in {".git", ".researchclaw", "__pycache__"}:
        errors.append(f"outputs[{index}] must not target {pure.parts[0]}")
    if ".git" in pure.parts:
        errors.append(f"outputs[{index}] must not target .git")
    return errors


def _workspace_tree(workspace: Path, *, max_entries: int = 200) -> str:
    if not workspace.is_dir():
        return "(workspace not found)"
    entries: list[str] = []
    for path in sorted(workspace.rglob("*")):
        rel = path.relative_to(workspace)
        if any(part in {".git", ".researchclaw", "__pycache__"} for part in rel.parts):
            continue
        entries.append(rel.as_posix() + ("/" if path.is_dir() else ""))
        if len(entries) >= max_entries:
            entries.append("(truncated)")
            break
    return "\n".join(entries) if entries else "(empty workspace)"


def _first_existing_text(workspace: Path, names: tuple[str, ...]) -> str:
    for name in names:
        path = workspace / name
        if path.is_file():
            return _read_text_limited(path)
    return ""


def _dependency_text(workspace: Path) -> str:
    parts: list[str] = []
    for name in ("pyproject.toml", "requirements.txt", "setup.py", "environment.yml"):
        path = workspace / name
        if path.is_file():
            parts.append(f"## {name}\n{_read_text_limited(path)}")
    return "\n\n".join(parts)


def _collect_human_guidance(run_dir: Path) -> str:
    parts: list[str] = []
    for path in sorted(run_dir.glob("stage-*/hitl_guidance.md")):
        parts.append(f"## {path.parent.name}\n{_read_text_limited(path)}")
    return "\n\n".join(parts)


def _read_text_limited(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except OSError:
        return ""


def _failed(error: str) -> StageResult:
    return StageResult(
        stage=Stage.EXPERIMENT_TASK_SPEC,
        status=StageStatus.FAILED,
        artifacts=(),
        error=error,
    )
