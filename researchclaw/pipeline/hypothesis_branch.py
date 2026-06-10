"""Helpers for running one hypothesis validation branch."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any

from researchclaw.pipeline.hypothesis_tree import _atomic_write_text


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
