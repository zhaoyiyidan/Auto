from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from researchclaw.experiment.acp_workspace_agent import AcpWorkspaceAgent
from researchclaw.experiment.workspace import WorkspaceAgentResult
from researchclaw.experiment.workspace_agent import (
    GitWorkspaceAgent,
    WorkspaceAgentProvider,
    create_workspace_agent,
)


class DummyWorkspaceAgent:
    name = "dummy"
    session_name = "dummy-session"

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, Path | None, int]] = []

    def generate_in_workspace(
        self,
        workspace_path: Path,
        prompt: str,
        workdir: Path | None = None,
        timeout_sec: int = 600,
    ) -> WorkspaceAgentResult:
        self.calls.append((workspace_path, prompt, workdir, timeout_sec))
        return WorkspaceAgentResult(
            base_sha="base",
            agent_commit_sha="head",
            manifest_path="run_manifest.json",
            diff_stat=" train.py | 1 +",
            raw_log="done",
            provider_name=self.name,
            elapsed_sec=0.01,
        )


class TestWorkspaceAgentProtocol:
    def test_protocol_can_be_subclassed(self) -> None:
        class MinimalWorkspaceAgent:
            name = "minimal"

            def generate_in_workspace(
                self,
                workspace_path: Path,
                prompt: str,
                workdir: Path | None = None,
                timeout_sec: int = 600,
            ):
                raise NotImplementedError

        assert isinstance(MinimalWorkspaceAgent(), WorkspaceAgentProvider)

    def test_name_property_required(self) -> None:
        assert "name" in WorkspaceAgentProvider.__annotations__

    def test_generate_in_workspace_signature(self) -> None:
        assert hasattr(WorkspaceAgentProvider, "generate_in_workspace")


class TestGitWorkspaceAgent:
    def test_delegates_to_workspace_native_inner(self, tmp_path: Path) -> None:
        inner = DummyWorkspaceAgent()
        agent = GitWorkspaceAgent(inner, tmp_path)

        result = agent.generate_in_workspace(
            tmp_path,
            "modify workspace",
            workdir=tmp_path,
            timeout_sec=123,
        )

        assert result.ok is True
        assert result.provider_name == "dummy"
        assert inner.calls == [(tmp_path.resolve(), "modify workspace", tmp_path, 123)]

    def test_requires_workspace_native_inner(self, tmp_path: Path) -> None:
        class OneShotOnly:
            name = "old-cli"

        with pytest.raises(TypeError, match="generate_in_workspace"):
            GitWorkspaceAgent(OneShotOnly(), tmp_path)  # type: ignore[arg-type]


class TestCreateWorkspaceAgent:
    def test_factory_creates_acp_workspace_agent(self, tmp_path: Path) -> None:
        cfg = _config(workspace_path=str(tmp_path), agent="claude")

        agent = create_workspace_agent(cfg)

        assert isinstance(agent, GitWorkspaceAgent)
        assert agent.name == "acp"
        assert isinstance(agent.inner, AcpWorkspaceAgent)
        assert agent.inner.session.session_name == "researchclaw-code-run-1"
        assert agent.inner.session.cwd == tmp_path

    def test_factory_supports_codex_acp_agent(self, tmp_path: Path) -> None:
        cfg = _config(workspace_path=str(tmp_path), agent="codex")

        agent = create_workspace_agent(cfg)

        assert isinstance(agent, GitWorkspaceAgent)
        assert agent.name == "acp"
        assert agent.inner.session.agent == "codex"

    def test_factory_rejects_non_acp_transport(self, tmp_path: Path) -> None:
        cfg = _config(workspace_path=str(tmp_path), transport="oneshot")

        with pytest.raises(ValueError, match="Unsupported workspace agent transport"):
            create_workspace_agent(cfg)


def _config(
    *,
    workspace_path: str,
    agent: str = "claude",
    transport: str = "acp",
) -> Any:
    from researchclaw.config import (
        ExperimentConfig,
        RCConfig,
        WorkspaceAgentConfig,
    )

    return RCConfig(
        experiment=ExperimentConfig(
            workspace_agent=WorkspaceAgentConfig(
                enabled=True,
                transport=transport,
                workspace_path=workspace_path,
                session_name="researchclaw-code-run-1",
                agent=agent,
                acpx_command="acpx",
                timeout_sec=1200,
                max_turns=40,
            ),
        )
    )
