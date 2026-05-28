from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from researchclaw.experiment.code_agent import CodeAgentResult
from researchclaw.experiment.workspace import LaunchCommand, RunManifest
from researchclaw.experiment.workspace_agent import (
    GitWorkspaceAgent,
    WorkspaceAgentProvider,
    create_workspace_agent,
)


@pytest.fixture()
def tmp_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)
    return repo


class DummyInnerAgent:
    name = "dummy"

    def __init__(
        self,
        *,
        commit: bool = True,
        manifest: bool = True,
        error: str | None = None,
    ) -> None:
        self.commit = commit
        self.manifest = manifest
        self.error = error
        self.prompts: list[str] = []

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        return self._run(extra_guidance, workdir)

    def refine(
        self,
        *,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        return self._run(extra_hints, workdir)

    def repair(
        self,
        *,
        files: dict[str, str],
        issues: str,
        workdir: Path,
        timeout_sec: int = 300,
    ) -> CodeAgentResult:
        return self._run(issues, workdir)

    def _run(self, prompt: str, workdir: Path) -> CodeAgentResult:
        self.prompts.append(prompt)
        (workdir / "train.py").write_text("print('trained')\n", encoding="utf-8")
        subprocess.run(["git", "add", "train.py"], cwd=workdir, check=True)
        if self.commit:
            subprocess.run(["git", "commit", "-m", "agent update"], cwd=workdir, check=True)
        if self.manifest:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=workdir,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            manifest = RunManifest(
                code_commit=sha,
                launch=LaunchCommand(command="python train.py"),
                result_paths=["outputs/metrics.json"],
            )
            (workdir / "run_manifest.json").write_text(
                manifest.to_json(), encoding="utf-8"
            )
        return CodeAgentResult(
            files={"train.py": "print('trained')\n"},
            provider_name=self.name,
            elapsed_sec=0.01,
            raw_output="done",
            error=self.error,
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
    def test_records_base_sha(self, tmp_git_repo: Path) -> None:
        base_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tmp_git_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        result = GitWorkspaceAgent(DummyInnerAgent(), tmp_git_repo).generate_in_workspace(
            tmp_git_repo, "improve experiment"
        )

        assert result.base_sha == base_sha

    def test_detects_agent_commit(self, tmp_git_repo: Path) -> None:
        result = GitWorkspaceAgent(DummyInnerAgent(), tmp_git_repo).generate_in_workspace(
            tmp_git_repo, "improve experiment"
        )

        assert result.agent_commit_sha is not None
        assert result.agent_commit_sha != result.base_sha
        assert result.ok is True

    def test_detects_missing_commit(self, tmp_git_repo: Path) -> None:
        result = GitWorkspaceAgent(
            DummyInnerAgent(commit=False), tmp_git_repo
        ).generate_in_workspace(tmp_git_repo, "improve experiment")

        assert result.agent_commit_sha is None
        assert result.ok is False
        assert "commit" in (result.error or "").lower()

    def test_collects_diff_stat(self, tmp_git_repo: Path) -> None:
        result = GitWorkspaceAgent(DummyInnerAgent(), tmp_git_repo).generate_in_workspace(
            tmp_git_repo, "improve experiment"
        )

        assert "train.py" in result.diff_stat

    def test_manifest_exists(self, tmp_git_repo: Path) -> None:
        result = GitWorkspaceAgent(DummyInnerAgent(), tmp_git_repo).generate_in_workspace(
            tmp_git_repo, "improve experiment"
        )

        assert result.manifest_path == "run_manifest.json"
        assert (tmp_git_repo / "run_manifest.json").exists()

    def test_manifest_missing(self, tmp_git_repo: Path) -> None:
        result = GitWorkspaceAgent(
            DummyInnerAgent(manifest=False), tmp_git_repo
        ).generate_in_workspace(tmp_git_repo, "improve experiment")

        assert result.manifest_path is None
        assert result.ok is False
        assert "manifest" in (result.error or "").lower()


class TestCreateWorkspaceAgent:
    def test_factory_claude(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cfg = _config(provider="claude_code", workspace_path=str(tmp_path))
        monkeypatch.setattr("researchclaw.experiment.code_agent.shutil.which", lambda _: "claude")

        agent = create_workspace_agent(cfg)

        assert agent.name == "claude_code"

    def test_factory_codex(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cfg = _config(provider="codex", workspace_path=str(tmp_path))
        monkeypatch.setattr("researchclaw.experiment.code_agent.shutil.which", lambda _: "codex")

        agent = create_workspace_agent(cfg)

        assert agent.name == "codex"

    def test_factory_llm(self, tmp_path: Path) -> None:
        cfg = _config(provider="llm", workspace_path=str(tmp_path))

        with pytest.raises(RuntimeError):
            create_workspace_agent(cfg)

    def test_factory_invalid(self, tmp_path: Path) -> None:
        cfg = _config(provider="invalid", workspace_path=str(tmp_path))

        with pytest.raises(ValueError):
            create_workspace_agent(cfg)


def _config(*, provider: str, workspace_path: str) -> Any:
    from researchclaw.config import (
        ExperimentConfig,
        RCConfig,
        CliAgentConfig,
        WorkspaceAgentConfig,
    )

    return RCConfig(
        experiment=ExperimentConfig(
            cli_agent=CliAgentConfig(provider=provider),
            workspace_agent=WorkspaceAgentConfig(
                enabled=True,
                workspace_path=workspace_path,
            ),
        )
    )
