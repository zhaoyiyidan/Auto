from __future__ import annotations

import subprocess
from pathlib import Path

from researchclaw.experiment.acp_workspace_agent import AcpWorkspaceAgent
from researchclaw.experiment.workspace import LaunchCommand, RunManifest
from researchclaw.experiment.workspace_agent import (
    SmokeWorkspaceAgent,
    WorkspaceAgentProvider,
)


class DummyAcpSession:
    session_name = "researchclaw-code-run-1"

    def __init__(
        self,
        *,
        commit: bool = True,
        manifest: bool = True,
        commit_manifest: bool = True,
        error: Exception | None = None,
        error_after_side_effects: Exception | None = None,
    ) -> None:
        self.commit = commit
        self.manifest = manifest
        self.commit_manifest = commit_manifest
        self.error = error
        self.error_after_side_effects = error_after_side_effects
        self.prompts: list[str] = []

    def run_task(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        repo = Path(prompt.split("WORKSPACE=")[1].splitlines()[0])
        (repo / "train.py").write_text("print('trained')\n", encoding="utf-8")
        subprocess.run(["git", "add", "train.py"], cwd=repo, check=True)
        if self.commit:
            subprocess.run(["git", "commit", "-m", "agent update"], cwd=repo, check=True)
        if self.manifest:
            sha = _git(repo, "rev-parse", "HEAD")
            manifest = RunManifest(
                code_commit=sha,
                launch=LaunchCommand(command="python train.py"),
                result_paths=["outputs/metrics.json"],
            )
            (repo / "run_manifest.json").write_text(
                manifest.to_json(), encoding="utf-8"
            )
            if self.commit and self.commit_manifest:
                subprocess.run(["git", "add", "run_manifest.json"], cwd=repo, check=True)
                subprocess.run(
                    ["git", "commit", "-m", "agent manifest"],
                    cwd=repo,
                    check=True,
                )
        if self.error_after_side_effects:
            raise self.error_after_side_effects
        return "agent completed"


def test_acp_workspace_agent_implements_protocol(tmp_path: Path) -> None:
    agent = AcpWorkspaceAgent(DummyAcpSession(), manifest_filename="run_manifest.json")

    assert isinstance(agent, WorkspaceAgentProvider)
    assert agent.name == "acp"


def test_generate_in_workspace_detects_commit_and_manifest(tmp_path: Path) -> None:
    repo = _tmp_git_repo(tmp_path)
    session = DummyAcpSession()
    agent = AcpWorkspaceAgent(session, manifest_filename="run_manifest.json")
    base_sha = _git(repo, "rev-parse", "HEAD")

    result = agent.generate_in_workspace(
        repo,
        f"WORKSPACE={repo}\nImplement experiment.",
    )

    assert result.ok is True
    assert result.base_sha == base_sha
    assert result.agent_commit_sha == _git(repo, "rev-parse", "HEAD")
    assert result.agent_commit_sha != base_sha
    assert result.manifest_path == "run_manifest.json"
    assert "train.py" in result.diff_stat
    assert result.raw_log == "agent completed"
    assert session.prompts == [f"WORKSPACE={repo}\nImplement experiment."]


def test_generate_in_workspace_fails_without_new_commit(tmp_path: Path) -> None:
    repo = _tmp_git_repo(tmp_path)
    agent = AcpWorkspaceAgent(
        DummyAcpSession(commit=False),
        manifest_filename="run_manifest.json",
    )

    result = agent.generate_in_workspace(repo, f"WORKSPACE={repo}\nModify.")

    assert result.ok is False
    assert result.agent_commit_sha is None
    assert "commit" in (result.error or "").lower()


def test_generate_in_workspace_fails_without_manifest(tmp_path: Path) -> None:
    repo = _tmp_git_repo(tmp_path)
    agent = AcpWorkspaceAgent(
        DummyAcpSession(manifest=False),
        manifest_filename="run_manifest.json",
    )

    result = agent.generate_in_workspace(repo, f"WORKSPACE={repo}\nModify.")

    assert result.ok is False
    assert result.agent_commit_sha is not None
    assert result.manifest_path is None
    assert "manifest" in (result.error or "").lower()


def test_generate_in_workspace_fails_when_workspace_left_dirty(tmp_path: Path) -> None:
    repo = _tmp_git_repo(tmp_path)
    agent = AcpWorkspaceAgent(
        DummyAcpSession(commit_manifest=False),
        manifest_filename="run_manifest.json",
    )

    result = agent.generate_in_workspace(repo, f"WORKSPACE={repo}\nModify.")

    assert result.ok is False
    assert result.agent_commit_sha is not None
    assert result.manifest_path == "run_manifest.json"
    assert "dirty" in (result.error or "").lower()


def test_generate_in_workspace_reports_session_error(tmp_path: Path) -> None:
    repo = _tmp_git_repo(tmp_path)
    agent = AcpWorkspaceAgent(
        DummyAcpSession(error=RuntimeError("acpx failed")),
        manifest_filename="run_manifest.json",
    )

    result = agent.generate_in_workspace(repo, f"WORKSPACE={repo}\nModify.")

    assert result.ok is False
    assert result.error == "acpx failed"
    assert result.raw_log == "acpx failed"


def test_generate_in_workspace_recovers_timeout_after_commit_and_manifest(
    tmp_path: Path,
) -> None:
    repo = _tmp_git_repo(tmp_path)
    agent = AcpWorkspaceAgent(
        DummyAcpSession(
            error_after_side_effects=RuntimeError(
                "ACP workspace task timed out after 1s"
            )
        ),
        manifest_filename="run_manifest.json",
    )
    base_sha = _git(repo, "rev-parse", "HEAD")

    result = agent.generate_in_workspace(repo, f"WORKSPACE={repo}\nModify.")

    assert result.ok is True
    assert result.error is None
    assert result.agent_commit_sha != base_sha
    assert result.manifest_path == "run_manifest.json"
    assert "recoverable error" in result.raw_log


def test_generate_in_workspace_does_not_recover_timeout_with_dirty_workspace(
    tmp_path: Path,
) -> None:
    repo = _tmp_git_repo(tmp_path)
    agent = AcpWorkspaceAgent(
        DummyAcpSession(
            commit_manifest=False,
            error_after_side_effects=RuntimeError(
                "ACP workspace task timed out after 1s"
            ),
        ),
        manifest_filename="run_manifest.json",
    )

    result = agent.generate_in_workspace(repo, f"WORKSPACE={repo}\nModify.")

    assert result.ok is False
    assert "timed out" in (result.error or "").lower()


def test_smoke_workspace_agent_commits_manifest_and_leaves_clean_tree(
    tmp_path: Path,
) -> None:
    repo = _tmp_git_repo(tmp_path)
    agent = SmokeWorkspaceAgent(manifest_filename="run_manifest.json")
    base_sha = _git(repo, "rev-parse", "HEAD")

    result = agent.generate_in_workspace(repo, "Implement smoke experiment.")

    assert result.ok is True
    assert result.base_sha == base_sha
    assert result.agent_commit_sha == _git(repo, "rev-parse", "HEAD")
    assert result.agent_commit_sha != base_sha
    assert result.manifest_path == "run_manifest.json"
    assert _git(repo, "status", "--porcelain") == ""
    assert (repo / "run_manifest.json").is_file()
    manifest = RunManifest.from_path(repo / "run_manifest.json")
    assert manifest.code_commit != base_sha
    assert _git(repo, "cat-file", "-t", manifest.code_commit) == "commit"


def _tmp_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)
    return repo


def _git(workspace: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()
