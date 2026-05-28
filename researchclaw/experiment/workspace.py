"""Workspace-native experiment agent data models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResourceSpec:
    """Requested training resources for a launch command."""

    gpus: int = 0
    time: str = "01:00:00"
    partition: str = ""
    mem_gb: int = 16


@dataclass(frozen=True)
class LaunchCommand:
    """Command prepared by a workspace agent for a submitter to run."""

    command: str
    cwd: str = "."
    env: dict[str, str] = field(default_factory=dict)
    resources: ResourceSpec = field(default_factory=ResourceSpec)

    def __post_init__(self) -> None:
        if not isinstance(self.command, str) or not self.command.strip():
            raise TypeError("LaunchCommand.command must be a non-empty string")
        if not isinstance(self.cwd, str):
            raise TypeError("LaunchCommand.cwd must be a string")
        if not isinstance(self.env, dict):
            raise TypeError("LaunchCommand.env must be a dict")
        if not isinstance(self.resources, ResourceSpec):
            raise TypeError("LaunchCommand.resources must be a ResourceSpec")


@dataclass(frozen=True)
class RunManifest:
    """Agent-authored manifest describing how to launch and collect a run."""

    code_commit: str
    launch: LaunchCommand
    result_paths: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.code_commit, str) or not self.code_commit.strip():
            raise TypeError("RunManifest.code_commit must be a non-empty string")
        if not isinstance(self.launch, LaunchCommand):
            raise TypeError("RunManifest.launch must be a LaunchCommand")
        if not isinstance(self.result_paths, list) or not all(
            isinstance(path, str) for path in self.result_paths
        ):
            raise TypeError("RunManifest.result_paths must be a list[str]")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        data = json.loads(text)
        launch_data = data.get("launch")
        if not isinstance(launch_data, dict):
            raise TypeError("RunManifest.launch must be an object")
        resources_data = launch_data.get("resources") or {}
        if not isinstance(resources_data, dict):
            raise TypeError("RunManifest.launch.resources must be an object")
        launch = LaunchCommand(
            command=launch_data.get("command", ""),
            cwd=launch_data.get("cwd", "."),
            env=dict(launch_data.get("env") or {}),
            resources=ResourceSpec(
                gpus=int(resources_data.get("gpus", 0)),
                time=str(resources_data.get("time", "01:00:00")),
                partition=str(resources_data.get("partition", "")),
                mem_gb=int(resources_data.get("mem_gb", 16)),
            ),
        )
        return cls(
            code_commit=str(data.get("code_commit", "")),
            launch=launch,
            result_paths=list(data.get("result_paths") or []),
        )

    @classmethod
    def from_path(cls, path: Path) -> RunManifest:
        return cls.from_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class WorkspaceAgentResult:
    """Result of invoking a code agent in an existing git workspace."""

    base_sha: str
    agent_commit_sha: str | None
    manifest_path: str | None
    diff_stat: str
    raw_log: str
    provider_name: str
    elapsed_sec: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.agent_commit_sha is not None


@dataclass(frozen=True)
class SubmitRequest:
    """Request passed from ResearchClaw to a training submitter."""

    manifest: RunManifest
    workspace_path: Path
    run_dir: Path
    stage: int


@dataclass(frozen=True)
class SubmitResult:
    """Result returned by a training submitter."""

    job_id: str
    submitter_name: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentRecord:
    """ResearchClaw-owned provenance record for a workspace experiment."""

    workspace: str
    stage: int
    base_sha: str
    agent_commit_sha: str
    provider: str
    agent_manifest: str
    submitter: str
    job_id: str
    result_paths: list[str]
    recorded_at: str
    result_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        return cls(
            workspace=str(data["workspace"]),
            stage=int(data["stage"]),
            base_sha=str(data["base_sha"]),
            agent_commit_sha=str(data["agent_commit_sha"]),
            provider=str(data["provider"]),
            agent_manifest=str(data["agent_manifest"]),
            submitter=str(data["submitter"]),
            job_id=str(data["job_id"]),
            result_paths=list(data.get("result_paths") or []),
            result_hashes=dict(data.get("result_hashes") or {}),
            recorded_at=str(data["recorded_at"]),
        )
