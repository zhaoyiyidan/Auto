from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from researchclaw.experiment.workspace import (
    ExperimentRecord,
    LaunchCommand,
    ManifestValidation,
    MetricsSpec,
    ResourceSpec,
    RunManifest,
    SubmitRequest,
    SubmitResult,
    WorkspaceAgentResult,
)


class TestLaunchCommand:
    def test_defaults(self) -> None:
        launch = LaunchCommand(command="python train.py")

        assert launch.command == "python train.py"
        assert launch.cwd == "."
        assert launch.env == {}
        assert launch.resources == ResourceSpec()

    def test_all_fields(self) -> None:
        resources = ResourceSpec(gpus=4, time="04:00:00", partition="a100", mem_gb=64)
        launch = LaunchCommand(
            command="bash scripts/train.sh",
            cwd="experiments/demo",
            env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            resources=resources,
        )

        assert launch.command == "bash scripts/train.sh"
        assert launch.cwd == "experiments/demo"
        assert launch.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert launch.resources == resources

    def test_immutability(self) -> None:
        launch = LaunchCommand(command="python train.py")

        with pytest.raises(FrozenInstanceError):
            launch.command = "python other.py"  # type: ignore[misc]


class TestResourceSpec:
    def test_defaults(self) -> None:
        spec = ResourceSpec()

        assert spec.gpus == 0
        assert spec.time == "01:00:00"
        assert spec.partition == ""
        assert spec.mem_gb == 16

    def test_custom_values(self) -> None:
        spec = ResourceSpec(gpus=2, time="12:00:00", partition="gpu", mem_gb=128)

        assert spec.gpus == 2
        assert spec.time == "12:00:00"
        assert spec.partition == "gpu"
        assert spec.mem_gb == 128


class TestRunManifest:
    def test_construction(self) -> None:
        manifest = RunManifest(
            code_commit="abc123",
            launch=LaunchCommand(command="python train.py"),
            result_paths=["outputs/metrics.json"],
        )

        assert manifest.code_commit == "abc123"
        assert manifest.launch.command == "python train.py"
        assert manifest.result_paths == ["outputs/metrics.json"]

    def test_type_enforcement(self) -> None:
        with pytest.raises(TypeError):
            RunManifest(
                code_commit="abc123",
                launch="python train.py",  # type: ignore[arg-type]
                result_paths=["outputs/metrics.json"],
            )

        with pytest.raises(TypeError):
            RunManifest(
                code_commit="abc123",
                launch=LaunchCommand(command="python train.py"),
                result_paths="outputs/metrics.json",  # type: ignore[arg-type]
            )

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        manifest = RunManifest(
            code_commit="abc123",
            launch=LaunchCommand(
                command="bash run.sh",
                cwd=".",
                env={"SEED": "7"},
                resources=ResourceSpec(gpus=1, time="02:00:00", mem_gb=32),
            ),
            result_paths=["results/metrics.json"],
        )
        path = tmp_path / "run_manifest.json"

        path.write_text(manifest.to_json(), encoding="utf-8")
        loaded = RunManifest.from_json(path.read_text(encoding="utf-8"))

        assert loaded == manifest

    def test_round_trips_schema_version_and_metrics(self) -> None:
        manifest = RunManifest(
            code_commit="abc123",
            launch=LaunchCommand(command="python train.py"),
            result_paths=["outputs/metrics.json"],
            metrics=MetricsSpec(primary="accuracy", direction="maximize"),
        )

        loaded = RunManifest.from_json(manifest.to_json())

        assert loaded.schema_version == "researchclaw.run_manifest.v1"
        assert loaded.metrics.primary == "accuracy"
        assert loaded.metrics.direction == "maximize"

    def test_from_json_defaults_legacy_payload(self) -> None:
        legacy = (
            '{"code_commit":"sha","launch":{"command":"python x.py"},'
            '"result_paths":[]}'
        )

        manifest = RunManifest.from_json(legacy)

        assert manifest.schema_version == "researchclaw.run_manifest.v1"
        assert manifest.metrics.primary == "primary_metric"

    def test_rejects_bad_metric_direction(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            RunManifest(
                code_commit="s",
                launch=LaunchCommand(command="x"),
                result_paths=[],
                metrics=MetricsSpec(primary="a", direction="sideways"),
            )


class TestWorkspaceAgentResult:
    def test_success(self) -> None:
        result = WorkspaceAgentResult(
            base_sha="base",
            agent_commit_sha="agent",
            manifest_path="run_manifest.json",
            diff_stat="1 file changed",
            raw_log="ok",
            provider_name="codex",
            elapsed_sec=1.2,
        )

        assert result.ok is True

    def test_error_scenario(self) -> None:
        result = WorkspaceAgentResult(
            base_sha="base",
            agent_commit_sha=None,
            manifest_path=None,
            diff_stat="",
            raw_log="failed",
            provider_name="codex",
            elapsed_sec=1.2,
            error="agent failed",
        )

        assert result.ok is False

    def test_ok_property(self) -> None:
        assert WorkspaceAgentResult(
            base_sha="base",
            agent_commit_sha=None,
            manifest_path=None,
            diff_stat="",
            raw_log="",
            provider_name="codex",
            elapsed_sec=0.1,
        ).ok is False

        assert WorkspaceAgentResult(
            base_sha="base",
            agent_commit_sha="agent",
            manifest_path=None,
            diff_stat="",
            raw_log="",
            provider_name="codex",
            elapsed_sec=0.1,
            error="bad manifest",
        ).ok is False


class TestSubmitRequest:
    def test_construction(self, tmp_path: Path) -> None:
        manifest = RunManifest(
            code_commit="abc123",
            launch=LaunchCommand(command="python train.py"),
            result_paths=["outputs/metrics.json"],
        )
        request = SubmitRequest(
            manifest=manifest,
            workspace_path=tmp_path / "workspace",
            run_dir=tmp_path / "run",
            stage=13,
        )

        assert request.manifest == manifest
        assert request.workspace_path == tmp_path / "workspace"
        assert request.run_dir == tmp_path / "run"
        assert request.stage == 13


class TestSubmitResult:
    def test_construction(self) -> None:
        result = SubmitResult(
            job_id="12345",
            submitter_name="slurm",
            status="submitted",
            metadata={"queue": "gpu"},
        )

        assert result.job_id == "12345"
        assert result.submitter_name == "slurm"
        assert result.status == "submitted"
        assert result.metadata == {"queue": "gpu"}


class TestExperimentRecord:
    def test_required_fields(self) -> None:
        record = ExperimentRecord(
            workspace="/tmp/workspace",
            stage=10,
            base_sha="base",
            agent_commit_sha="agent",
            provider="codex",
            agent_manifest=".researchclaw/run_manifest.json",
            submitter="local",
            job_id="local_1",
            result_paths=["outputs/metrics.json"],
            result_hashes={"outputs/metrics.json": "sha256"},
            recorded_at="2026-05-28T00:00:00Z",
        )

        assert record.workspace == "/tmp/workspace"
        assert record.stage == 10
        assert record.job_id == "local_1"

    def test_json_roundtrip(self) -> None:
        record = ExperimentRecord(
            workspace="/tmp/workspace",
            stage=13,
            base_sha="base",
            agent_commit_sha="agent",
            provider="claude_code",
            agent_manifest=".researchclaw/run_manifest.json",
            submitter="manual",
            job_id="manual_1",
            result_paths=["outputs/metrics.json"],
            result_hashes={"outputs/metrics.json": "abc"},
            recorded_at="2026-05-28T00:00:00Z",
        )

        payload = json.loads(json.dumps(asdict(record)))
        loaded = ExperimentRecord.from_dict(payload)

        assert loaded == record
        assert loaded.to_dict() == asdict(record)

    def test_result_hashes_default(self) -> None:
        record = ExperimentRecord(
            workspace="/tmp/workspace",
            stage=14,
            base_sha="base",
            agent_commit_sha="agent",
            provider="opencode",
            agent_manifest=".researchclaw/run_manifest.json",
            submitter="local",
            job_id="local_1",
            result_paths=[],
            recorded_at="2026-05-28T00:00:00Z",
        )

        assert record.result_hashes == {}


class TestManifestValidation:
    def test_dict_roundtrip(self) -> None:
        validation = ManifestValidation(
            ok=False,
            schema_version="researchclaw.run_manifest.v1",
            code_commit="abc123",
            commit_exists=True,
            workspace_dirty=True,
            launch_command="python train.py",
            launch_cwd=".",
            result_paths=["outputs/metrics.json"],
            errors=["workspace has uncommitted changes"],
            checked_at="2026-05-29T00:00:00Z",
        )

        loaded = ManifestValidation.from_dict(validation.to_dict())

        assert loaded == validation
        assert loaded.errors == ["workspace has uncommitted changes"]
