"""Validation for workspace-native run manifests."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from researchclaw.experiment.workspace import ManifestValidation, RunManifest


def validate_manifest(
    manifest: RunManifest,
    workspace: Path,
    *,
    allow_dirty: bool = False,
) -> ManifestValidation:
    """Validate the agent-authored manifest against git/workspace invariants."""
    workspace = Path(workspace).resolve()
    errors: list[str] = []
    code_commit = str(getattr(manifest, "code_commit", ""))
    schema_version = str(
        getattr(manifest, "schema_version", "researchclaw.run_manifest.v1")
    )
    launch = getattr(manifest, "launch", None)
    launch_command = str(getattr(launch, "command", ""))
    launch_cwd = str(getattr(launch, "cwd", "."))
    result_paths = list(getattr(manifest, "result_paths", []) or [])

    if schema_version != "researchclaw.run_manifest.v1":
        errors.append("schema_version must be researchclaw.run_manifest.v1")
    if not code_commit.strip():
        errors.append("code_commit is required")
    if not launch_command.strip():
        errors.append("launch.command is required")
    if not result_paths:
        errors.append("result_paths must contain at least one path")

    commit_exists = _commit_exists(workspace, code_commit)
    if code_commit.strip() and not commit_exists:
        errors.append(f"code_commit does not exist in workspace git: {code_commit}")

    workspace_dirty = _workspace_dirty(workspace, result_paths=result_paths)
    if workspace_dirty and not allow_dirty:
        errors.append("workspace has uncommitted changes")

    return ManifestValidation(
        ok=not errors,
        schema_version=schema_version,
        code_commit=code_commit,
        commit_exists=commit_exists,
        workspace_dirty=workspace_dirty,
        launch_command=launch_command,
        launch_cwd=launch_cwd,
        result_paths=result_paths,
        errors=errors,
        checked_at=_utcnow_iso(),
    )


def _commit_exists(workspace: Path, commit: str) -> bool:
    if not commit.strip():
        return False
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _workspace_dirty(workspace: Path, *, result_paths: list[str] | tuple[str, ...]) -> bool:
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty_paths = [
        _status_path(line)
        for line in proc.stdout.splitlines()
        if line.strip()
    ]
    return any(
        path and not _is_result_artifact_path(path, result_paths)
        for path in dirty_paths
    )


def _status_path(line: str) -> str:
    path = line[3:] if len(line) > 3 else line
    if " -> " in path:
        path = path.rsplit(" -> ", 1)[-1]
    return path.strip().strip("/")


def _is_result_artifact_path(
    path: str,
    result_paths: list[str] | tuple[str, ...],
) -> bool:
    rel = path.replace("\\", "/").strip("/")
    if not rel:
        return False
    for result_path in result_paths:
        result = str(result_path).replace("\\", "/").strip("/")
        if not result:
            continue
        if "/" not in result:
            if rel == result:
                return True
            continue
        root = result.split("/", 1)[0]
        if rel == root or rel.startswith(root + "/"):
            return True
    return False


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
