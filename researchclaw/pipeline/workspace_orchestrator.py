"""Orchestration for workspace-native code agents and training submitters."""

from __future__ import annotations

import json
import hashlib
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
    ExecutionRecord,
    ExperimentRecord,
    ResultArtifact,
    ResultArtifacts,
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


def run_workspace_agent_implement(
    workspace_path: Path,
    run_dir: Path,
    stage: int,
    agent: WorkspaceAgentProvider,
    prompt: str,
    timeout_sec: int,
    *,
    iteration: int | None = None,
    ledger: WorkspaceAgentLedger | None = None,
    close_policy: str = "keep",
) -> WorkspaceAgentResult:
    """Run a workspace code agent and record implementation provenance."""
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
    _write_workspace_agent_result(run_dir, stage, result)
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
        _write_workspace_agent_result(run_dir, stage, failed)
        ledger.write_agent_result(stage_ledger_dir, failed)
        _export_session_snapshot(agent, ledger, stage_ledger_dir)
        _close_session_if_requested(agent, close_policy)
        return failed
    if result.manifest_path:
        ledger.copy_manifest(stage_ledger_dir, workspace / result.manifest_path)
    _export_session_snapshot(agent, ledger, stage_ledger_dir)
    _close_session_if_requested(agent, close_policy)
    return result


def submit_and_collect(
    manifest: RunManifest,
    submitter: TrainingSubmitter,
    workspace_path: Path,
    run_dir: Path,
    stage: int,
    *,
    wait: bool,
    timeout_sec: int,
    poll_interval_sec: int,
    base_sha: str,
    agent_commit_sha: str,
    provider: str,
    session_name: str,
) -> ExecutionRecord:
    """Submit a manifest launch command and record result provenance."""
    started = time.monotonic()
    workspace = workspace_path.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    submit_result = submit_job(
        submitter,
        SubmitRequest(
            manifest=manifest,
            workspace_path=workspace,
            run_dir=run_dir,
            stage=stage,
        ),
    )
    final_status = (
        wait_for_completion(
            submitter,
            submit_result,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
        )
        if wait
        else submit_result.status
    )
    result_artifacts = _collect_result_artifacts(manifest, workspace)
    result_hashes = {
        artifact.path: artifact.sha256
        for artifact in result_artifacts.artifacts
        if artifact.exists and artifact.sha256
    }
    execution = ExecutionRecord(
        stage=stage,
        code_commit=manifest.code_commit,
        submitter=submit_result.submitter_name,
        job_id=submit_result.job_id,
        submit_status=submit_result.status,
        final_status=final_status,
        log_path=str(submit_result.metadata.get("log_path", "")),
        result_paths=manifest.result_paths,
        result_hashes=result_hashes,
        metrics=_collect_metrics(manifest.result_paths, workspace),
        elapsed_sec=round(time.monotonic() - started, 6),
        waited=wait,
        recorded_at=_utcnow_iso(),
    )
    (run_dir / "submit_result.json").write_text(
        json.dumps(asdict(submit_result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "execution_record.json").write_text(
        json.dumps(execution.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "result_artifacts.json").write_text(
        json.dumps(result_artifacts.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    record = ExperimentRecord(
        workspace=str(workspace),
        stage=stage,
        base_sha=base_sha,
        agent_commit_sha=agent_commit_sha,
        provider=provider,
        session_name=session_name,
        agent_manifest="run_manifest.json",
        submitter=submit_result.submitter_name,
        job_id=submit_result.job_id,
        result_paths=manifest.result_paths,
        result_hashes=result_hashes,
        recorded_at=_utcnow_iso(),
    )
    ExperimentRegistry(run_dir / "workspace_experiment_registry.jsonl").append(record)
    return execution


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


def _write_workspace_agent_result(
    run_dir: Path,
    stage: int,
    result: WorkspaceAgentResult,
) -> None:
    (run_dir / f"stage-{stage:02d}-workspace-agent-result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _collect_result_artifacts(
    manifest: RunManifest,
    workspace: Path,
) -> ResultArtifacts:
    artifacts: list[ResultArtifact] = []
    root = workspace.resolve()
    for rel in manifest.result_paths:
        path = (root / rel).resolve()
        if not path.exists():
            artifacts.append(
                ResultArtifact(path=rel, sha256="", size_bytes=0, exists=False)
            )
            continue
        artifacts.append(
            ResultArtifact(
                path=rel,
                sha256=_sha256_path(path),
                size_bytes=_path_size(path),
                exists=True,
            )
        )
    return ResultArtifacts(
        code_commit=manifest.code_commit,
        artifacts=artifacts,
        collected_at=_utcnow_iso(),
    )


def _collect_metrics(result_paths: list[str], workspace: Path) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for rel in result_paths:
        path = (workspace / rel).resolve()
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            metrics.update(payload)
    return metrics


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        _update_digest_from_file(digest, path)
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        _update_digest_from_file(digest, child)
    return digest.hexdigest()


def _update_digest_from_file(digest: "hashlib._Hash", path: Path) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


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
