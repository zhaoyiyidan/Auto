"""Workspace-native factory and wrappers for ACP code agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from researchclaw.experiment.workspace import WorkspaceAgentResult


@runtime_checkable
class WorkspaceAgentProvider(Protocol):
    """Protocol for agents that operate directly in an existing workspace."""

    name: str

    def generate_in_workspace(
        self,
        workspace_path: Path,
        prompt: str,
        workdir: Path | None = None,
        timeout_sec: int = 600,
    ) -> WorkspaceAgentResult:
        """Modify a git workspace and leave an agent run manifest."""
        ...


class GitWorkspaceAgent:
    """Thin wrapper around a workspace-native agent provider."""

    def __init__(
        self,
        inner: WorkspaceAgentProvider,
        workspace_path: Path,
        *,
        manifest_filename: str = "run_manifest.json",
    ) -> None:
        if not hasattr(inner, "generate_in_workspace"):
            raise TypeError("Workspace agent inner must define generate_in_workspace")
        self.inner = inner
        self.workspace_path = workspace_path
        self.manifest_filename = manifest_filename

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def session_name(self) -> str:
        return getattr(self.inner, "session_name", "")

    def generate_in_workspace(
        self,
        workspace_path: Path,
        prompt: str,
        workdir: Path | None = None,
        timeout_sec: int = 600,
    ) -> WorkspaceAgentResult:
        workspace = workspace_path.resolve()
        return self.inner.generate_in_workspace(
            workspace,
            prompt,
            workdir or workspace,
            timeout_sec,
        )


def create_workspace_agent(
    config: Any,
    llm: Any | None = None,
    prompts: Any | None = None,
) -> WorkspaceAgentProvider:
    """Create a configured workspace-native agent."""
    from researchclaw.experiment.acp_workspace_agent import AcpWorkspaceAgent
    from researchclaw.experiment.acp_workspace_session import AcpWorkspaceSession

    workspace_cfg = config.experiment.workspace_agent
    transport = getattr(workspace_cfg, "transport", "acp")
    if transport != "acp":
        raise ValueError(f"Unsupported workspace agent transport: {transport}")
    session_name = workspace_cfg.session_name or "researchclaw-code"
    acp_cfg = getattr(getattr(config, "llm", None), "acp", None)
    base_url = getattr(acp_cfg, "base_url", "") or getattr(config.llm, "base_url", "")
    api_key_env = getattr(acp_cfg, "api_key_env", "") or getattr(config.llm, "api_key_env", "")
    session = AcpWorkspaceSession(
        agent=workspace_cfg.agent,
        cwd=Path(workspace_cfg.workspace_path),
        acpx_command=workspace_cfg.acpx_command,
        session_name=session_name,
        timeout_sec=workspace_cfg.timeout_sec,
        max_turns=workspace_cfg.max_turns,
        base_url=base_url,
        api_key_env=api_key_env,
        model=getattr(config.llm, "primary_model", ""),
    )
    agent = AcpWorkspaceAgent(
        session,
        manifest_filename=workspace_cfg.manifest_filename,
    )
    return GitWorkspaceAgent(
        agent,
        Path(workspace_cfg.workspace_path),
        manifest_filename=workspace_cfg.manifest_filename,
    )
