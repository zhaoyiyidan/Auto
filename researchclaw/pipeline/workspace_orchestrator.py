"""Orchestration for workspace-native code agents and training submitters."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from researchclaw.experiment.record import (
    ExperimentRegistry,
    compute_result_hashes,
)
from researchclaw.experiment.submitter import TrainingSubmitter
from researchclaw.experiment.workspace import (
    ExperimentRecord,
    RunManifest,
    SubmitRequest,
    SubmitResult,
    WorkspaceAgentResult,
)
from researchclaw.experiment.workspace_agent_ledger import WorkspaceAgentLedger
from researchclaw.experiment.workspace_agent import WorkspaceAgentProvider
from researchclaw.pipeline._helpers import _utcnow_iso


def record_base_sha(workspace_path: Path) -> str:
    return _git(workspace_path, "rev-parse", "HEAD")


def invoke_workspace_agent(
    agent: WorkspaceAgentProvider,
    workspace_path: Path,
    workdir: Path,
    prompt: str,
    timeout_sec: int,
) -> WorkspaceAgentResult:
    return agent.generate_in_workspace(
        workspace_path=workspace_path,
        prompt=prompt,
        workdir=workdir,
        timeout_sec=timeout_sec,
    )


def verify_agent_commit(result: WorkspaceAgentResult, base_sha: str) -> bool:
    return result.agent_commit_sha is not None and result.agent_commit_sha != base_sha


def read_agent_manifest(workdir: Path) -> RunManifest | None:
    for candidate in (
        workdir / "run_manifest.json",
        workdir / ".researchclaw" / "run_manifest.json",
    ):
        if candidate.is_file():
            return RunManifest.from_path(candidate)
    return None


def submit_job(
    submitter: TrainingSubmitter,
    request: SubmitRequest,
) -> SubmitResult:
    return submitter.submit(request)


def wait_for_completion(
    submitter: TrainingSubmitter,
    result: SubmitResult,
    *,
    timeout_sec: int,
    poll_interval_sec: int,
) -> str:
    poll = getattr(submitter, "poll", None)
    if not callable(poll):
        return "unknown"

    deadline = time.monotonic() + max(timeout_sec, 0)
    while True:
        status = str(poll(result))
        if status in {"completed", "failed"}:
            return status
        if status == "unknown":
            return "unknown"
        if time.monotonic() >= deadline:
            return "timeout"
        if poll_interval_sec > 0:
            time.sleep(min(poll_interval_sec, max(deadline - time.monotonic(), 0)))


def run_workspace_agent_task(
    workspace_path: Path,
    run_dir: Path,
    stage: int,
    agent: WorkspaceAgentProvider,
    submitter: TrainingSubmitter,
    prompt: str,
    timeout_sec: int,
    *,
    iteration: int | None = None,
    ledger: WorkspaceAgentLedger | None = None,
    close_policy: str = "keep",
) -> WorkspaceAgentResult:
    """Run one workspace-native ACP agent task and record provenance."""
    workspace = workspace_path.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = ledger or WorkspaceAgentLedger(run_dir)
    stage_ledger_dir = ledger.stage_dir(stage, iteration=iteration)
    base_sha = record_base_sha(workspace)
    ledger.write_prompt(stage_ledger_dir, prompt)
    ledger.write_base_sha(stage_ledger_dir, base_sha)
    ledger.save_session_meta(
        {
            "provider": getattr(agent, "name", ""),
            "session_name": _agent_session_name(agent),
            "workspace": str(workspace),
        }
    )
    result = invoke_workspace_agent(
        agent=agent,
        workspace_path=workspace,
        workdir=workspace,
        prompt=prompt,
        timeout_sec=timeout_sec,
    )
    ledger.write_agent_result(stage_ledger_dir, result)
    (run_dir / f"stage-{stage:02d}-workspace-agent-result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not result.ok or not verify_agent_commit(result, base_sha):
        _export_session_snapshot(agent, ledger, stage_ledger_dir)
        _close_session_if_requested(agent, close_policy)
        return result

    manifest = _manifest_from_result(workspace, result) or read_agent_manifest(workspace)
    if manifest is None:
        failed = WorkspaceAgentResult(
            base_sha=result.base_sha,
            agent_commit_sha=result.agent_commit_sha,
            manifest_path=result.manifest_path,
            diff_stat=result.diff_stat,
            raw_log=result.raw_log,
            provider_name=result.provider_name,
            elapsed_sec=result.elapsed_sec,
            error="Agent manifest could not be read",
        )
        (run_dir / f"stage-{stage:02d}-workspace-agent-result.json").write_text(
            json.dumps(asdict(failed), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        ledger.write_agent_result(stage_ledger_dir, failed)
        _export_session_snapshot(agent, ledger, stage_ledger_dir)
        _close_session_if_requested(agent, close_policy)
        return failed
    if result.manifest_path:
        ledger.copy_manifest(stage_ledger_dir, workspace / result.manifest_path)

    submit_result = submit_job(
        submitter,
        SubmitRequest(
            manifest=manifest,
            workspace_path=workspace,
            run_dir=run_dir,
            stage=stage,
        ),
    )
    ledger.write_submit_result(stage_ledger_dir, submit_result)
    (run_dir / f"stage-{stage:02d}-submit-result.json").write_text(
        json.dumps(asdict(submit_result), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        result_hashes = compute_result_hashes(manifest.result_paths, workspace)
    except FileNotFoundError:
        result_hashes = {}
    record = ExperimentRecord(
        workspace=str(workspace),
        stage=stage,
        base_sha=result.base_sha,
        agent_commit_sha=result.agent_commit_sha or "",
        provider=result.provider_name,
        session_name=_agent_session_name(agent),
        agent_manifest=result.manifest_path or "",
        submitter=submit_result.submitter_name,
        job_id=submit_result.job_id,
        result_paths=manifest.result_paths,
        result_hashes=result_hashes,
        recorded_at=_utcnow_iso(),
    )
    ExperimentRegistry(run_dir / "workspace_experiment_registry.jsonl").append(record)
    ledger.write_registry_record(stage_ledger_dir, record)
    _export_session_snapshot(agent, ledger, stage_ledger_dir)
    _close_session_if_requested(agent, close_policy)
    return result


def _manifest_from_result(
    workspace: Path,
    result: WorkspaceAgentResult,
) -> RunManifest | None:
    if not result.manifest_path:
        return None
    path = workspace / result.manifest_path
    if not path.is_file():
        return None
    return RunManifest.from_path(path)


def _agent_session_name(agent: WorkspaceAgentProvider) -> str:
    session_name = getattr(agent, "session_name", "")
    if session_name:
        return str(session_name)
    inner = getattr(agent, "inner", None)
    if inner is not None:
        return str(getattr(inner, "session_name", ""))
    return ""


def _agent_session(agent: WorkspaceAgentProvider) -> object | None:
    session = getattr(agent, "session", None)
    if session is not None:
        return session
    inner = getattr(agent, "inner", None)
    if inner is not None:
        return getattr(inner, "session", None)
    return None


def _export_session_snapshot(
    agent: WorkspaceAgentProvider,
    ledger: WorkspaceAgentLedger,
    stage_ledger_dir: Path,
) -> None:
    session = _agent_session(agent)
    if session is None or not hasattr(session, "export_session"):
        return
    ledger.write_session_export(stage_ledger_dir, session)


def _close_session_if_requested(
    agent: WorkspaceAgentProvider,
    close_policy: str,
) -> None:
    if close_policy != "close":
        return
    session = _agent_session(agent)
    if session is not None and hasattr(session, "close"):
        session.close()


def _git(workspace: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()
