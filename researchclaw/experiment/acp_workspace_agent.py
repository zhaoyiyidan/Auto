"""Workspace-native code agent backed by a persistent ACP session."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Protocol

from researchclaw.experiment.acp_workspace_session import AcpWorkspaceSession
from researchclaw.experiment.manifest_validation import validate_manifest
from researchclaw.experiment.workspace import RunManifest, WorkspaceAgentResult


class WorkspaceSession(Protocol):
    """Minimal session contract used by AcpWorkspaceAgent."""

    session_name: str

    def run_task(self, prompt: str) -> str:
        """Run one task in the persistent workspace code session."""
        ...


class AcpWorkspaceAgent:
    """Invoke a persistent ACP code session and verify workspace provenance."""

    name = "acp"

    def __init__(
        self,
        session: WorkspaceSession | AcpWorkspaceSession,
        *,
        manifest_filename: str = "run_manifest.json",
    ) -> None:
        self.session = session
        self.manifest_filename = manifest_filename

    @property
    def session_name(self) -> str:
        return getattr(self.session, "session_name", "")

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
            raw_log = self.session.run_task(prompt)
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
        workspace_dirty = bool(_git(workspace, "status", "--porcelain"))
        manifest_path = self._find_manifest(workspace)
        if (
            error is not None
            and agent_commit_sha is not None
            and manifest_path is not None
            and _recoverable_session_error(error)
            and _manifest_valid_after_error(workspace, manifest_path, base_sha, head_sha)
        ):
            raw_log = (
                raw_log
                + "\n[ResearchClaw] ACP session ended with a recoverable error, "
                "but the workspace contains a clean committed update and valid "
                "run_manifest.json."
            )
            error = None
        if agent_commit_sha is None and error is None:
            error = "Agent did not create a new git commit"
        if manifest_path is None and error is None:
            error = f"Agent manifest not found: {self.manifest_filename}"
        if workspace_dirty and error is None:
            error = "Agent left workspace dirty"
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


def _recoverable_session_error(error: str) -> bool:
    lower = str(error or "").lower()
    return (
        "timed out" in lower
        or "timeout" in lower
        or "stream disconnected" in lower
        or "transient" in lower
    )


def _manifest_valid_after_error(
    workspace: Path,
    manifest_path: str,
    base_sha: str,
    head_sha: str,
) -> bool:
    try:
        manifest = RunManifest.from_path(workspace / manifest_path)
    except Exception:  # noqa: BLE001
        return False
    validation = validate_manifest(manifest, workspace, allow_dirty=False)
    if not validation.ok or manifest.code_commit == base_sha:
        return False
    return _is_ancestor(workspace, base_sha, manifest.code_commit) and _is_ancestor(
        workspace,
        manifest.code_commit,
        head_sha,
    )


def _is_ancestor(workspace: Path, ancestor: str, descendant: str) -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _git(workspace: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()
