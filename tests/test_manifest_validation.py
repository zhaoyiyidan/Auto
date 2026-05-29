from __future__ import annotations

from pathlib import Path

from researchclaw.experiment.workspace import LaunchCommand, RunManifest
from researchclaw.experiment.manifest_validation import validate_manifest
from tests.test_workspace_orchestrator import _git, _tmp_git_repo


def test_validate_manifest_accepts_clean_head(tmp_path: Path) -> None:
    workspace = _tmp_git_repo(tmp_path)
    head = _git(workspace, "rev-parse", "HEAD")
    manifest = RunManifest(
        code_commit=head,
        launch=LaunchCommand(command="python train.py"),
        result_paths=["outputs/metrics.json"],
    )

    validation = validate_manifest(manifest, workspace, allow_dirty=False)

    assert validation.ok is True
    assert validation.commit_exists is True
    assert validation.workspace_dirty is False
    assert validation.errors == []


def test_validate_manifest_rejects_empty_launch_command(tmp_path: Path) -> None:
    workspace = _tmp_git_repo(tmp_path)
    manifest = _unchecked_manifest(
        code_commit=_git(workspace, "rev-parse", "HEAD"),
        launch_command="",
        result_paths=["outputs/metrics.json"],
    )

    validation = validate_manifest(manifest, workspace)

    assert validation.ok is False
    assert any("launch.command" in error for error in validation.errors)


def test_validate_manifest_rejects_empty_result_paths(tmp_path: Path) -> None:
    workspace = _tmp_git_repo(tmp_path)
    manifest = RunManifest(
        code_commit=_git(workspace, "rev-parse", "HEAD"),
        launch=LaunchCommand(command="python train.py"),
        result_paths=[],
    )

    validation = validate_manifest(manifest, workspace)

    assert validation.ok is False
    assert any("result_paths" in error for error in validation.errors)


def test_validate_manifest_rejects_missing_commit(tmp_path: Path) -> None:
    workspace = _tmp_git_repo(tmp_path)
    manifest = RunManifest(
        code_commit="deadbeef",
        launch=LaunchCommand(command="python train.py"),
        result_paths=["outputs/metrics.json"],
    )

    validation = validate_manifest(manifest, workspace)

    assert validation.ok is False
    assert validation.commit_exists is False


def test_validate_manifest_dirty_workspace_policy(tmp_path: Path) -> None:
    workspace = _tmp_git_repo(tmp_path)
    (workspace / "scratch.txt").write_text("dirty\n", encoding="utf-8")
    manifest = RunManifest(
        code_commit=_git(workspace, "rev-parse", "HEAD"),
        launch=LaunchCommand(command="python train.py"),
        result_paths=["outputs/metrics.json"],
    )

    strict = validate_manifest(manifest, workspace, allow_dirty=False)
    allowed = validate_manifest(manifest, workspace, allow_dirty=True)

    assert strict.ok is False
    assert strict.workspace_dirty is True
    assert allowed.ok is True
    assert allowed.workspace_dirty is True


def _unchecked_manifest(
    *,
    code_commit: str,
    launch_command: str,
    result_paths: list[str],
) -> RunManifest:
    manifest = object.__new__(RunManifest)
    launch = object.__new__(LaunchCommand)
    object.__setattr__(launch, "command", launch_command)
    object.__setattr__(launch, "cwd", ".")
    object.__setattr__(launch, "env", {})
    object.__setattr__(launch, "resources", None)
    object.__setattr__(manifest, "code_commit", code_commit)
    object.__setattr__(manifest, "launch", launch)
    object.__setattr__(manifest, "result_paths", result_paths)
    object.__setattr__(manifest, "schema_version", "researchclaw.run_manifest.v1")
    return manifest
