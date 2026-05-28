from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from researchclaw.experiment.submitter import (
    CustomPythonSubmitter,
    LocalSubmitter,
    ManualSubmitter,
    SlurmSubmitter,
    SshSlurmSubmitter,
    TrainingSubmitter,
    create_submitter,
)
from researchclaw.experiment.workspace import (
    LaunchCommand,
    ResourceSpec,
    RunManifest,
    SubmitRequest,
    SubmitResult,
)


class TestTrainingSubmitterProtocol:
    def test_protocol(self) -> None:
        class MinimalSubmitter:
            name = "minimal"

            def submit(self, request: SubmitRequest) -> SubmitResult:
                return SubmitResult(
                    job_id="minimal_1",
                    submitter_name=self.name,
                    status="submitted",
                )

        assert isinstance(MinimalSubmitter(), TrainingSubmitter)

    def test_name_required(self) -> None:
        assert "name" in TrainingSubmitter.__annotations__

    def test_submit_signature(self) -> None:
        assert hasattr(TrainingSubmitter, "submit")


class TestLocalSubmitter:
    def test_returns_job_id(self, tmp_path: Path) -> None:
        request = _request(tmp_path, command="python -c 'print(42)'")

        result = LocalSubmitter().submit(request)

        assert result.job_id.startswith("local_")
        assert result.submitter_name == "local"
        assert result.status == "submitted"

    def test_writes_log_file(self, tmp_path: Path) -> None:
        request = _request(tmp_path, command="python -c 'print(42)'")

        result = LocalSubmitter().submit(request)

        log_path = Path(result.metadata["log_path"])
        assert log_path.exists()

    def test_preserves_command(self, tmp_path: Path) -> None:
        command = "python -c 'print(42)'"
        request = _request(tmp_path, command=command)

        result = LocalSubmitter().submit(request)

        assert result.metadata["command"] == command

    def test_name(self) -> None:
        assert LocalSubmitter().name == "local"


class TestSlurmSubmitter:
    def test_generates_sbatch(self, tmp_path: Path) -> None:
        request = _request(tmp_path, command="python train.py")

        script = SlurmSubmitter().generate_sbatch(request)

        assert "#SBATCH --job-name=researchclaw-stage-13" in script
        assert "python train.py" in script

    def test_respects_resources(self, tmp_path: Path) -> None:
        request = _request(
            tmp_path,
            command="python train.py",
            resources=ResourceSpec(gpus=2, time="08:00:00", partition="gpu", mem_gb=64),
        )

        script = SlurmSubmitter().generate_sbatch(request)

        assert "#SBATCH --gres=gpu:2" in script
        assert "#SBATCH --time=08:00:00" in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --mem=64G" in script

    def test_name(self) -> None:
        assert SlurmSubmitter().name == "slurm"


class TestSshSlurmSubmitter:
    def test_ssh_wrapped(self, tmp_path: Path) -> None:
        submitter = SshSlurmSubmitter(host="cluster.example.com", user="alice", port=2222)
        request = _request(tmp_path, command="python train.py")

        cmd = submitter.build_ssh_command(request)

        assert cmd[:5] == ["ssh", "-p", "2222", "alice@cluster.example.com", "bash -lc"]
        assert "sbatch" in cmd[-1]

    def test_name(self) -> None:
        assert SshSlurmSubmitter(host="cluster.example.com").name == "ssh_slurm"


class TestManualSubmitter:
    def test_placeholder_id(self, tmp_path: Path) -> None:
        result = ManualSubmitter().submit(_request(tmp_path, command="python train.py"))

        assert result.job_id.startswith("manual_")
        assert result.status == "manual"

    def test_prints_instructions(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        ManualSubmitter().submit(_request(tmp_path, command="python train.py"))

        assert "python train.py" in capsys.readouterr().out

    def test_name(self) -> None:
        assert ManualSubmitter().name == "manual"


class TestCustomPythonSubmitter:
    def test_invokes_callable(self, tmp_path: Path) -> None:
        seen: dict[str, SubmitRequest] = {}

        def submit(request: SubmitRequest) -> SubmitResult:
            seen["request"] = request
            return SubmitResult("custom_1", "custom_python", "submitted")

        result = CustomPythonSubmitter(submit).submit(_request(tmp_path, command="echo ok"))

        assert seen["request"].stage == 13
        assert result.job_id == "custom_1"

    def test_name(self) -> None:
        assert CustomPythonSubmitter(lambda request: SubmitResult("1", "x", "ok")).name == "custom_python"

    def test_exception_propagated(self, tmp_path: Path) -> None:
        def submit(_request: SubmitRequest) -> SubmitResult:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            CustomPythonSubmitter(submit).submit(_request(tmp_path, command="echo ok"))


class TestCreateSubmitter:
    def test_local(self) -> None:
        assert create_submitter({"type": "local"}).name == "local"

    def test_slurm(self) -> None:
        assert create_submitter({"type": "slurm"}).name == "slurm"

    def test_manual(self) -> None:
        assert create_submitter({"type": "manual"}).name == "manual"

    def test_unknown(self) -> None:
        with pytest.raises(ValueError):
            create_submitter({"type": "unknown"})


def _request(
    tmp_path: Path,
    *,
    command: str,
    resources: ResourceSpec | None = None,
) -> SubmitRequest:
    workspace = tmp_path / "workspace"
    run_dir = tmp_path / "run"
    workspace.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=True)
    manifest = RunManifest(
        code_commit="abc123",
        launch=LaunchCommand(
            command=command,
            cwd=".",
            resources=resources or ResourceSpec(),
        ),
        result_paths=["outputs/metrics.json"],
    )
    return SubmitRequest(
        manifest=manifest,
        workspace_path=workspace,
        run_dir=run_dir,
        stage=13,
    )
