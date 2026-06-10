"""Helpers for running one hypothesis validation branch."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any


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
