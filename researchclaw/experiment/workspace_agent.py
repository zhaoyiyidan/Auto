"""Workspace-native wrappers for CLI code agents."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from researchclaw.experiment.code_agent import CodeAgentProvider, CodeAgentResult
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
    """Wrap a CodeAgentProvider with git provenance checks."""

    def __init__(
        self,
        inner: CodeAgentProvider,
        workspace_path: Path,
        *,
        manifest_filename: str = "run_manifest.json",
    ) -> None:
        self.inner = inner
        self.workspace_path = workspace_path
        self.manifest_filename = manifest_filename

    @property
    def name(self) -> str:
        return self.inner.name

    def generate_in_workspace(
        self,
        workspace_path: Path,
        prompt: str,
        workdir: Path | None = None,
        timeout_sec: int = 600,
    ) -> WorkspaceAgentResult:
        workspace = workspace_path.resolve()
        base_sha = _git(workspace, "rev-parse", "HEAD")
        start = time.monotonic()
        raw_log = ""
        error: str | None = None
        try:
            if hasattr(self.inner, "generate_in_workspace"):
                result = self.inner.generate_in_workspace(
                    workspace,
                    prompt,
                    workdir or workspace,
                    timeout_sec,
                )
                return result
            result = self.inner.generate(
                exp_plan=prompt,
                topic="workspace-native experiment",
                metric_key="primary_metric",
                pkg_hint="",
                compute_budget="",
                extra_guidance=prompt,
                workdir=workdir or workspace,
                timeout_sec=timeout_sec,
            )
            raw_log = result.raw_output
            if result.error:
                error = result.error
        except Exception as exc:  # noqa: BLE001
            raw_log = str(exc)
            error = str(exc)

        elapsed = time.monotonic() - start
        head_sha = _git(workspace, "rev-parse", "HEAD")
        agent_commit_sha = head_sha if head_sha != base_sha else None
        diff_stat = (
            _git(workspace, "diff", "--stat", base_sha, "HEAD")
            if agent_commit_sha
            else _git(workspace, "diff", "--stat")
        )
        manifest_path = self._find_manifest(workspace)
        if agent_commit_sha is None and error is None:
            error = "Agent did not create a new git commit"
        if manifest_path is None and error is None:
            error = f"Agent manifest not found: {self.manifest_filename}"
        return WorkspaceAgentResult(
            base_sha=base_sha,
            agent_commit_sha=agent_commit_sha,
            manifest_path=manifest_path,
            diff_stat=diff_stat,
            raw_log=raw_log,
            provider_name=self.name,
            elapsed_sec=elapsed,
            error=error,
        )

    def generate(self, **kwargs: Any) -> CodeAgentResult:
        return self.inner.generate(**kwargs)

    def refine(self, **kwargs: Any) -> CodeAgentResult:
        return self.inner.refine(**kwargs)

    def repair(self, **kwargs: Any) -> CodeAgentResult:
        return self.inner.repair(**kwargs)

    def _find_manifest(self, workspace: Path) -> str | None:
        candidates = [
            workspace / self.manifest_filename,
            workspace / ".researchclaw" / self.manifest_filename,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.relative_to(workspace).as_posix()
        found = sorted(workspace.glob(f"**/{self.manifest_filename}"))
        for candidate in found:
            if ".git" not in candidate.parts:
                return candidate.relative_to(workspace).as_posix()
        return None


def create_workspace_agent(
    config: Any,
    llm: Any | None = None,
    prompts: Any | None = None,
) -> WorkspaceAgentProvider:
    """Create a configured workspace-native agent."""
    from researchclaw.experiment.code_agent import create_code_agent

    agent = create_code_agent(config, llm=llm, prompts=prompts)
    if isinstance(agent, GitWorkspaceAgent):
        return agent
    workspace_cfg = config.experiment.workspace_agent
    return GitWorkspaceAgent(
        agent,
        Path(workspace_cfg.workspace_path),
        manifest_filename=workspace_cfg.manifest_filename,
    )


def _git(workspace: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()
