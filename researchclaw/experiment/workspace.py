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
class MetricsSpec:
    """Primary metric metadata declared by a workspace agent manifest."""

    primary: str = "primary_metric"
    direction: str = "maximize"

    def __post_init__(self) -> None:
        if not isinstance(self.primary, str) or not self.primary.strip():
            raise TypeError("MetricsSpec.primary must be a non-empty string")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("MetricsSpec.direction must be 'maximize' or 'minimize'")


@dataclass(frozen=True)
class RunManifest:
    """Agent-authored manifest describing how to launch and collect a run."""

    code_commit: str
    launch: LaunchCommand
    result_paths: list[str]
    schema_version: str = "researchclaw.run_manifest.v1"
    metrics: MetricsSpec = field(default_factory=MetricsSpec)

    def __post_init__(self) -> None:
        if not isinstance(self.code_commit, str) or not self.code_commit.strip():
            raise TypeError("RunManifest.code_commit must be a non-empty string")
        if not isinstance(self.launch, LaunchCommand):
            raise TypeError("RunManifest.launch must be a LaunchCommand")
        if not isinstance(self.result_paths, list) or not all(
            isinstance(path, str) for path in self.result_paths
        ):
            raise TypeError("RunManifest.result_paths must be a list[str]")
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise TypeError("RunManifest.schema_version must be a non-empty string")
        if not isinstance(self.metrics, MetricsSpec):
            raise TypeError("RunManifest.metrics must be a MetricsSpec")

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
        metrics_data = data.get("metrics") or {}
        if not isinstance(metrics_data, dict):
            raise TypeError("RunManifest.metrics must be an object")
        return cls(
            code_commit=str(data.get("code_commit", "")),
            launch=launch,
            result_paths=list(data.get("result_paths") or []),
            schema_version=str(
                data.get("schema_version") or "researchclaw.run_manifest.v1"
            ),
            metrics=MetricsSpec(
                primary=str(metrics_data.get("primary") or "primary_metric"),
                direction=str(metrics_data.get("direction") or "maximize"),
            ),
        )

    @classmethod
    def from_path(cls, path: Path) -> RunManifest:
        return cls.from_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ManifestValidation:
    """ResearchClaw validation result for an agent-authored run manifest."""

    ok: bool
    schema_version: str
    code_commit: str
    commit_exists: bool
    workspace_dirty: bool
    launch_command: str
    launch_cwd: str
    result_paths: list[str]
    errors: list[str] = field(default_factory=list)
    checked_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestValidation:
        return cls(
            ok=bool(data["ok"]),
            schema_version=str(data["schema_version"]),
            code_commit=str(data["code_commit"]),
            commit_exists=bool(data["commit_exists"]),
            workspace_dirty=bool(data["workspace_dirty"]),
            launch_command=str(data["launch_command"]),
            launch_cwd=str(data["launch_cwd"]),
            result_paths=list(data.get("result_paths") or []),
            errors=list(data.get("errors") or []),
            checked_at=str(data.get("checked_at", "")),
        )


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
class ExecutionRecord:
    """Stage 12 execution summary produced by the harness."""

    stage: int
    code_commit: str
    submitter: str
    job_id: str
    submit_status: str
    final_status: str
    log_path: str
    result_paths: list[str]
    result_hashes: dict[str, str]
    metrics: dict[str, Any]
    elapsed_sec: float
    waited: bool
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionRecord:
        return cls(
            stage=int(data["stage"]),
            code_commit=str(data["code_commit"]),
            submitter=str(data["submitter"]),
            job_id=str(data["job_id"]),
            submit_status=str(data["submit_status"]),
            final_status=str(data["final_status"]),
            log_path=str(data.get("log_path", "")),
            result_paths=list(data.get("result_paths") or []),
            result_hashes=dict(data.get("result_hashes") or {}),
            metrics=dict(data.get("metrics") or {}),
            elapsed_sec=float(data.get("elapsed_sec", 0.0)),
            waited=bool(data.get("waited", False)),
            recorded_at=str(data["recorded_at"]),
        )


@dataclass(frozen=True)
class ResultArtifact:
    """Hashed result artifact collected after a submitted run."""

    path: str
    sha256: str
    size_bytes: int
    exists: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResultArtifact:
        return cls(
            path=str(data["path"]),
            sha256=str(data.get("sha256", "")),
            size_bytes=int(data.get("size_bytes", 0)),
            exists=bool(data.get("exists", False)),
        )


@dataclass(frozen=True)
class ResultArtifacts:
    """Collection of result artifact hashes for one code commit."""

    code_commit: str
    artifacts: list[ResultArtifact]
    collected_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResultArtifacts:
        return cls(
            code_commit=str(data["code_commit"]),
            artifacts=[
                ResultArtifact.from_dict(item)
                for item in list(data.get("artifacts") or [])
            ],
            collected_at=str(data["collected_at"]),
        )


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
    session_name: str = ""

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
            session_name=str(data.get("session_name", "")),
        )
