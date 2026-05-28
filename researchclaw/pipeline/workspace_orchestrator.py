"""Orchestration for workspace-native code agents and training submitters."""

from __future__ import annotations

import json
import subprocess
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


def run_workspace_pipeline(
    workspace_path: Path,
    run_dir: Path,
    stage: int,
    agent: WorkspaceAgentProvider,
    submitter: TrainingSubmitter,
    prompt: str,
    timeout_sec: int,
) -> WorkspaceAgentResult:
    """Run the opt-in workspace-native agent path end to end."""
    workspace = workspace_path.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    base_sha = record_base_sha(workspace)
    result = invoke_workspace_agent(
        agent=agent,
        workspace_path=workspace,
        workdir=workspace,
        prompt=prompt,
        timeout_sec=timeout_sec,
    )
    (run_dir / f"stage-{stage:02d}-workspace-agent-result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not result.ok or not verify_agent_commit(result, base_sha):
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
        return failed

    submit_result = submit_job(
        submitter,
        SubmitRequest(
            manifest=manifest,
            workspace_path=workspace,
            run_dir=run_dir,
            stage=stage,
        ),
    )
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
        agent_manifest=result.manifest_path or "",
        submitter=submit_result.submitter_name,
        job_id=submit_result.job_id,
        result_paths=manifest.result_paths,
        result_hashes=result_hashes,
        recorded_at=_utcnow_iso(),
    )
    ExperimentRegistry(run_dir / "workspace_experiment_registry.jsonl").append(record)
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


def _git(workspace: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()
