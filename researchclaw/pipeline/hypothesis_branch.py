"""Helpers for running one hypothesis validation branch."""

from __future__ import annotations

from dataclasses import replace
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from researchclaw.pipeline.hypothesis_tree import (
    _atomic_write_json,
    _atomic_write_text,
)
from researchclaw.pipeline.branch_checkpoint import (
    resolve_branch_resume_stage,
    write_branch_stage_done,
)
from researchclaw.pipeline.executor import StageResult
from researchclaw.pipeline.runner import _promote_best_stage14, execute_pipeline
from researchclaw.pipeline.stage15_verdict import read_stage15_verdict
from researchclaw.pipeline.stages import Stage, StageStatus

logger = logging.getLogger(__name__)


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
    branch_run_config = branch_config(config)
    workspace_path = getattr(attempt, "workspace_path", None)
    session_name = getattr(attempt, "agent_session_name", None)
    if workspace_path or session_name:
        branch_run_config = branch_config(
            branch_run_config,
            workspace_path=workspace_path,
            session_name=session_name,
        )
    branch_run_adapters = branch_adapters(adapters)
    resume_stage = resolve_branch_resume_stage(branch_run_dir)
    if resume_stage is None:
        return finalize_attempt_result(
            branch_run_dir,
            node,
            attempt,
            [
                StageResult(
                    stage=Stage.RESEARCH_DECISION,
                    status=StageStatus.DONE,
                    artifacts=(),
                )
            ],
        )

    def record_stage_done(stage: Stage) -> None:
        write_branch_stage_done(
            branch_run_dir,
            stage,
            attempt_id=str(getattr(attempt, "attempt_id", "")),
            node_id=str(getattr(node, "id", "")),
            workspace_path=workspace_path or branch_run_dir,
        )

    results = execute_pipeline(
        run_dir=branch_run_dir,
        run_id=_branch_run_id(branch_run_dir, node, attempt),
        config=branch_run_config,
        adapters=branch_run_adapters,
        from_stage=resume_stage,
        to_stage=Stage.RESEARCH_DECISION,
        auto_approve_gates=True,
        initialize_run_globals=False,
        on_stage_complete=record_stage_done,
    )
    return finalize_attempt_result(branch_run_dir, node, attempt, results)


def finalize_attempt_result(
    branch_run_dir: Path,
    node: Any,
    attempt: Any,
    results: list[Any],
) -> dict[str, Any]:
    branch_run_dir = Path(branch_run_dir)
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
    verdict_path = branch_run_dir / "stage-15" / "verdict.json"
    verdict: dict[str, Any] | None = None
    if verdict_path.exists():
        verdict = read_stage15_verdict(verdict_path)

    decision = (
        verdict.get("decision")
        if verdict is not None
        else getattr(final_result, "decision", None) if final_result else None
    )
    artifacts = list(getattr(final_result, "artifacts", ()) or ()) if final_result else []
    if verdict is not None and "verdict.json" not in artifacts:
        artifacts.append("verdict.json")

    payload = {
        "attempt_id": getattr(attempt, "attempt_id", ""),
        "node_id": getattr(node, "id", ""),
        "status": "succeeded" if succeeded else "failed",
        "decision": decision,
        "artifacts": artifacts,
        "error": getattr(final_result, "error", None) if final_result else "no results",
    }
    if verdict is not None:
        payload.update(
            {
                "next_hypothesis": verdict.get("next_hypothesis"),
                "confidence": verdict.get("confidence"),
                "evidence_summary": verdict.get("evidence_summary"),
                "metrics": verdict.get("key_metrics") or {},
            }
        )
    _atomic_write_json(branch_run_dir / "attempt_result.json", payload)
    return payload


def branch_adapters(adapters: Any) -> Any:
    """Return adapters for branch execution with HITL disabled."""
    if getattr(adapters, "hitl", None) is None:
        return adapters
    try:
        return replace(adapters, hitl=None)
    except Exception:  # noqa: BLE001
        logger.debug("Branch adapters could not disable HITL", exc_info=True)
        return adapters


def promote_best_stage14_for_branch(branch_run_dir: Path, config: Any) -> None:
    _promote_best_stage14(Path(branch_run_dir), config)


def _attempt_workspace_name(attempt: Any) -> str:
    attempt_id = str(getattr(attempt, "attempt_id", "attempt") or "attempt")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", attempt_id).strip("-")
    return name or "attempt"


def _is_git_worktree(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return completed.stdout.strip() == "true"


def _run_source_git(source_workspace: Path, *args: str, check: bool = True) -> None:
    subprocess.run(
        ["git", "-C", str(source_workspace), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def provision_workspace(
    attempt: Any,
    *,
    source_workspace: Path,
    workspace_root: Path | None = None,
) -> Any:
    source_workspace = Path(source_workspace).resolve()
    workspace_root = (
        Path(workspace_root)
        if workspace_root
        else source_workspace.parent / ".worktrees"
    ).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    recorded_workspace = getattr(attempt, "workspace_path", None)
    target = (
        Path(recorded_workspace).resolve()
        if recorded_workspace
        else workspace_root / _attempt_workspace_name(attempt)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if not _is_git_worktree(target):
        _run_source_git(source_workspace, "worktree", "prune")
        _run_source_git(
            source_workspace,
            "worktree",
            "repair",
            str(target),
            check=False,
        )
    if not _is_git_worktree(target):
        if target.exists():
            shutil.rmtree(target)
        _run_source_git(
            source_workspace,
            "worktree",
            "add",
            "--detach",
            str(target),
            "HEAD",
        )
    return replace(attempt, workspace_path=str(target.resolve()))


def release_workspace(
    attempt: Any,
    *,
    source_workspace: Path | None = None,
) -> None:
    if getattr(attempt, "status", "") not in {"succeeded", "failed", "abandoned"}:
        return
    workspace_path = getattr(attempt, "workspace_path", None)
    if not workspace_path:
        return
    target = Path(workspace_path)
    if not target.exists():
        return
    if source_workspace is None:
        raise ValueError("source_workspace is required to remove a git worktree")
    subprocess.run(
        [
            "git",
            "-C",
            str(Path(source_workspace)),
            "worktree",
            "remove",
            "--force",
            str(target),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def branch_config(
    config: Any,
    *,
    workspace_path: str | Path | None = None,
    session_name: str | None = None,
) -> Any:
    """Return a config copy with branch-local workspace and gate settings."""
    branch = config
    try:
        branch = replace(
            branch,
            security=replace(branch.security, hitl_required_stages=()),
        )
    except Exception:  # noqa: BLE001
        logger.debug("Branch config could not override HITL gates", exc_info=True)

    try:
        if session_name is not None:
            llm = branch.llm
            branch = replace(
                branch,
                llm=replace(
                    llm,
                    acp=replace(llm.acp, session_name=str(session_name)),
                ),
            )
    except Exception:  # noqa: BLE001
        if session_name is not None:
            logger.debug("Branch config could not override LLM session", exc_info=True)

    try:
        workspace_agent = branch.experiment.workspace_agent
        result_analysis_agent = branch.experiment.result_analysis_agent
        if workspace_path is not None:
            workspace_agent = replace(
                workspace_agent,
                workspace_path=str(workspace_path),
            )
        if session_name is not None:
            workspace_agent = replace(
                workspace_agent,
                session_name=str(session_name),
            )
            result_analysis_agent = replace(
                result_analysis_agent,
                session_name=f"{session_name}-analysis",
            )
        experiment = replace(
            branch.experiment,
            workspace_agent=workspace_agent,
            result_analysis_agent=result_analysis_agent,
        )
        branch = replace(branch, experiment=experiment)
    except Exception:  # noqa: BLE001
        if workspace_path is not None or session_name is not None:
            logger.debug("Branch config could not override workspace", exc_info=True)
    return branch
