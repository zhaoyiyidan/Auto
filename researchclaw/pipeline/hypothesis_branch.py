"""Helpers for running one hypothesis validation branch."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any

from researchclaw.pipeline.hypothesis_tree import (
    _atomic_write_json,
    _atomic_write_text,
)
from researchclaw.pipeline.runner import execute_pipeline
from researchclaw.pipeline.stages import Stage, StageStatus


def _single_hypothesis_md(node: Any) -> str:
    lines = [
        "## H1",
        f"Statement: {getattr(node, 'statement', '')}",
        f"Prediction: {getattr(node, 'prediction', '')}",
        f"Falsification: {getattr(node, 'falsification', '')}",
        f"Rationale: {getattr(node, 'rationale', '')}",
    ]
    baselines = tuple(getattr(node, "baselines", ()) or ())
    if baselines:
        lines.append(f"Baselines: {', '.join(str(item) for item in baselines)}")
    return "\n".join(lines) + "\n"


def seed_branch_dir(branch_run_dir: Path, shared_run_dir: Path, node: Any) -> None:
    branch_run_dir = Path(branch_run_dir)
    shared_run_dir = Path(shared_run_dir)
    branch_run_dir.mkdir(parents=True, exist_ok=True)

    for stage_number in range(1, 8):
        link = branch_run_dir / f"stage-{stage_number:02d}"
        target = shared_run_dir / f"stage-{stage_number:02d}"
        relative_target = Path(os.path.relpath(target, start=branch_run_dir))
        if link.exists() or link.is_symlink():
            if link.is_symlink() and os.readlink(link) == relative_target.as_posix():
                continue
            raise FileExistsError(f"Branch stage path already exists: {link}")
        link.symlink_to(relative_target, target_is_directory=True)

    stage8 = branch_run_dir / "stage-08"
    if stage8.is_symlink():
        raise FileExistsError(f"Branch stage-08 must be a real directory: {stage8}")
    stage8.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(stage8 / "hypotheses.md", _single_hypothesis_md(node))


def _branch_run_id(branch_run_dir: Path, node: Any, attempt: Any) -> str:
    attempt_name = str(getattr(attempt, "attempt_id", "attempt")).split("/")[-1]
    run_name = (
        branch_run_dir.parents[2].name
        if len(branch_run_dir.parents) > 2
        else "branch"
    )
    return f"{run_name}-{getattr(node, 'id', 'hypothesis')}-{attempt_name}"


def validate_branch(
    *,
    branch_run_dir: Path,
    node: Any,
    attempt: Any,
    config: Any,
    adapters: Any,
) -> dict[str, Any]:
    branch_run_dir = Path(branch_run_dir)
    results = execute_pipeline(
        run_dir=branch_run_dir,
        run_id=_branch_run_id(branch_run_dir, node, attempt),
        config=config,
        adapters=adapters,
        from_stage=Stage.EXPERIMENT_TASK_SPEC,
        to_stage=Stage.RESEARCH_DECISION,
    )
    final_result = next(
        (
            result
            for result in reversed(results)
            if result.stage == Stage.RESEARCH_DECISION
        ),
        results[-1] if results else None,
    )
    succeeded = (
        final_result is not None
        and final_result.stage == Stage.RESEARCH_DECISION
        and final_result.status == StageStatus.DONE
    )
    payload = {
        "attempt_id": getattr(attempt, "attempt_id", ""),
        "node_id": getattr(node, "id", ""),
        "status": "succeeded" if succeeded else "failed",
        "decision": getattr(final_result, "decision", None) if final_result else None,
        "artifacts": list(getattr(final_result, "artifacts", ()) or ())
        if final_result
        else [],
        "error": getattr(final_result, "error", None) if final_result else "no results",
    }
    _atomic_write_json(branch_run_dir / "attempt_result.json", payload)
    return payload


def branch_config(
    config: Any,
    *,
    workspace_path: str | Path,
    session_name: str,
) -> Any:
    """Return a config copy with branch-local workspace agent settings."""
    workspace_agent = replace(
        config.experiment.workspace_agent,
        workspace_path=str(workspace_path),
        session_name=session_name,
    )
    experiment = replace(config.experiment, workspace_agent=workspace_agent)
    return replace(config, experiment=experiment)
