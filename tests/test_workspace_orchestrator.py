from __future__ import annotations

import json
import subprocess
from pathlib import Path

from researchclaw.experiment.workspace import (
    LaunchCommand,
    RunManifest,
    SubmitRequest,
    SubmitResult,
    WorkspaceAgentResult,
)
from researchclaw.experiment.workspace_agent_ledger import WorkspaceAgentLedger
from researchclaw.pipeline.workspace_orchestrator import (
    run_workspace_agent_task,
    wait_for_completion,
)


def test_legacy_workspace_pipeline_alias_removed() -> None:
    import researchclaw.pipeline.workspace_orchestrator as orchestrator

    assert not hasattr(orchestrator, "run_workspace_pipeline")


class DummySession:
    def __init__(self) -> None:
        self.exports: list[Path] = []
        self.closed = False

    def export_session(self, output_path: Path) -> None:
        self.exports.append(output_path)
        output_path.write_bytes(b"session")

    def close(self) -> None:
        self.closed = True


class DummyWorkspaceAgent:
    name = "acp"
    session_name = "researchclaw-code-run-1"

    def __init__(self) -> None:
        self.session = DummySession()
        self.prompts: list[str] = []

    def generate_in_workspace(
        self,
        workspace_path: Path,
        prompt: str,
        workdir: Path | None = None,
        timeout_sec: int = 600,
    ) -> WorkspaceAgentResult:
        self.prompts.append(prompt)
        base_sha = _git(workspace_path, "rev-parse", "HEAD")
        (workspace_path / "train.py").write_text("print('train')\n", encoding="utf-8")
        subprocess.run(["git", "add", "train.py"], cwd=workspace_path, check=True)
        subprocess.run(["git", "commit", "-m", "agent update"], cwd=workspace_path, check=True)
        head_sha = _git(workspace_path, "rev-parse", "HEAD")
        manifest = RunManifest(
            code_commit=head_sha,
            launch=LaunchCommand(command="python train.py"),
            result_paths=["outputs/metrics.json"],
        )
        (workspace_path / "run_manifest.json").write_text(
            manifest.to_json(),
            encoding="utf-8",
        )
        return WorkspaceAgentResult(
            base_sha=base_sha,
            agent_commit_sha=head_sha,
            manifest_path="run_manifest.json",
            diff_stat=_git(workspace_path, "diff", "--stat", base_sha, "HEAD"),
            raw_log="agent done",
            provider_name=self.name,
            elapsed_sec=0.1,
        )


class DummySubmitter:
    name = "dummy"

    def __init__(self) -> None:
        self.requests: list[SubmitRequest] = []

    def submit(self, request: SubmitRequest) -> SubmitResult:
        self.requests.append(request)
        result_path = request.workspace_path / "outputs" / "metrics.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text('{"accuracy": 0.9}\n', encoding="utf-8")
        return SubmitResult(
            job_id="job-1",
            submitter_name=self.name,
            status="submitted",
            metadata={"command": request.manifest.launch.command},
        )


class PollingDummySubmitter:
    name = "polling"

    def __init__(self, states: list[str]) -> None:
        self.states = states

    def poll(self, result: SubmitResult) -> str:
        if self.states:
            return self.states.pop(0)
        return "running"


def test_wait_for_completion_polls_until_terminal_state() -> None:
    submitter = PollingDummySubmitter(["running", "running", "completed"])
    result = SubmitResult(job_id="job-1", submitter_name="polling", status="submitted")

    final_status = wait_for_completion(
        submitter,
        result,
        timeout_sec=5,
        poll_interval_sec=0,
    )

    assert final_status == "completed"


def test_wait_for_completion_times_out() -> None:
    submitter = PollingDummySubmitter(["running"])
    result = SubmitResult(job_id="job-1", submitter_name="polling", status="submitted")

    final_status = wait_for_completion(
        submitter,
        result,
        timeout_sec=0,
        poll_interval_sec=0,
    )

    assert final_status == "timeout"


def test_wait_for_completion_without_poll_returns_unknown() -> None:
    result = SubmitResult(job_id="job-1", submitter_name="dummy", status="submitted")

    final_status = wait_for_completion(
        DummySubmitter(),
        result,
        timeout_sec=5,
        poll_interval_sec=0,
    )

    assert final_status == "unknown"


def test_run_workspace_agent_task_writes_ledger_registry_and_hashes(tmp_path: Path) -> None:
    workspace = _tmp_git_repo(tmp_path)
    run_dir = tmp_path / "run"
    agent = DummyWorkspaceAgent()
    submitter = DummySubmitter()
    ledger = WorkspaceAgentLedger(run_dir)

    result = run_workspace_agent_task(
        workspace_path=workspace,
        run_dir=run_dir,
        stage=13,
        agent=agent,
        submitter=submitter,
        prompt="improve the experiment",
        timeout_sec=120,
        ledger=ledger,
        close_policy="keep",
    )

    assert result.ok is True
    assert agent.prompts == ["improve the experiment"]
    assert len(submitter.requests) == 1
    stage_dir = run_dir / ".researchclaw" / "workspace-agent" / "stage-13"
    assert (stage_dir / "prompt.md").read_text(encoding="utf-8") == "improve the experiment"
    assert (stage_dir / "base_sha.txt").read_text(encoding="utf-8") == f"{result.base_sha}\n"
    assert json.loads((stage_dir / "agent_result.json").read_text(encoding="utf-8"))[
        "agent_commit_sha"
    ] == result.agent_commit_sha
    assert json.loads((stage_dir / "run_manifest.json").read_text(encoding="utf-8"))[
        "code_commit"
    ] == result.agent_commit_sha
    assert json.loads((stage_dir / "submit_result.json").read_text(encoding="utf-8"))[
        "job_id"
    ] == "job-1"
    registry_record = json.loads(
        (stage_dir / "registry_record.json").read_text(encoding="utf-8")
    )
    assert registry_record["session_name"] == "researchclaw-code-run-1"
    assert registry_record["job_id"] == "job-1"
    assert registry_record["result_hashes"]["outputs/metrics.json"]
    assert (stage_dir / "session_export.tar.gz").read_bytes() == b"session"
    assert agent.session.exports == [stage_dir / "session_export.tar.gz"]

    registry_lines = (run_dir / "workspace_experiment_registry.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(registry_lines) == 1
    assert json.loads(registry_lines[0])["session_name"] == "researchclaw-code-run-1"


def test_run_workspace_agent_task_does_not_submit_failed_agent(tmp_path: Path) -> None:
    workspace = _tmp_git_repo(tmp_path)
    run_dir = tmp_path / "run"
    submitter = DummySubmitter()

    class FailingAgent(DummyWorkspaceAgent):
        def generate_in_workspace(
            self,
            workspace_path: Path,
            prompt: str,
            workdir: Path | None = None,
            timeout_sec: int = 600,
        ) -> WorkspaceAgentResult:
            base_sha = _git(workspace_path, "rev-parse", "HEAD")
            return WorkspaceAgentResult(
                base_sha=base_sha,
                agent_commit_sha=None,
                manifest_path=None,
                diff_stat="",
                raw_log="failed",
                provider_name=self.name,
                elapsed_sec=0.1,
                error="Agent did not create a new git commit",
            )

    result = run_workspace_agent_task(
        workspace_path=workspace,
        run_dir=run_dir,
        stage=10,
        agent=FailingAgent(),
        submitter=submitter,
        prompt="generate",
        timeout_sec=120,
    )

    assert result.ok is False
    assert submitter.requests == []
    assert not (run_dir / "workspace_experiment_registry.jsonl").exists()


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
