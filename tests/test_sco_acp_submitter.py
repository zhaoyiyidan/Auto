from __future__ import annotations

import json
import subprocess
from pathlib import Path

from researchclaw.experiment import sco_acp_submitter
from researchclaw.experiment.sco_acp_submitter import (
    _map_state,
    _parse_job_id,
    _select_worker_spec,
    submit,
)
from researchclaw.experiment.workspace import (
    LaunchCommand,
    ResourceSpec,
    RunManifest,
    SubmitRequest,
    SubmitResult,
)


def test_submit_builds_sco_acp_create_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "job pt-test123 submitted successfully\n", "")

    monkeypatch.setattr(sco_acp_submitter.subprocess, "run", fake_run)
    monkeypatch.setenv("SCO_ACP_WORKSPACE_NAME", "workspace-a")
    monkeypatch.setenv("SCO_ACP_AEC2_NAME", "cluster-a")
    monkeypatch.setenv("SCO_ACP_IMAGE", "registry.example/image:tag")
    monkeypatch.setenv("SCO_ACP_STORAGE_MOUNT", "vol-1:/data")
    monkeypatch.setenv("SCO_ACP_JOB_PREFIX", "rc-test")

    result = submit(_request(tmp_path))

    assert result.job_id == "pt-test123"
    assert result.submitter_name == "sco_acp"
    cmd = calls[0]
    assert cmd[:4] == ["sco", "acp", "jobs", "create"]
    assert cmd[cmd.index("--workspace-name") + 1] == "workspace-a"
    assert cmd[cmd.index("--aec2-name") + 1] == "cluster-a"
    assert cmd[cmd.index("--container-image-url") + 1] == "registry.example/image:tag"
    assert cmd[cmd.index("--worker-spec") + 1] == "n6ls.iu.i40.2c4g"
    assert cmd[cmd.index("--storage-mount") + 1] == "vol-1:/data"
    launch_script = cmd[cmd.index("--command") + 1]
    assert "cd " in launch_script
    assert "export TOKEN=abc" in launch_script
    assert "python train.py" in launch_script


def test_poll_maps_describe_state_and_writes_snapshot(tmp_path: Path, monkeypatch) -> None:
    describe = {"state": "SUCCEEDED", "name": "pt-test123"}
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs):
        calls.append(cmd)
        if "describe" in cmd:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(describe), "")
        return subprocess.CompletedProcess(cmd, 0, "logs\n", "")

    monkeypatch.setattr(sco_acp_submitter.subprocess, "run", fake_run)
    snapshot = tmp_path / "describe.json"
    result = SubmitResult(
        "pt-test123",
        "sco_acp",
        "submitted",
        metadata={
            "workspace_name": "workspace-a",
            "describe_path": str(snapshot),
            "log_path": str(tmp_path / "job.log"),
        },
    )

    assert sco_acp_submitter.poll(result) == "completed"
    assert json.loads(snapshot.read_text(encoding="utf-8"))["state"] == "SUCCEEDED"
    assert any("stream-logs" in cmd for cmd in calls)


def test_submit_reuses_existing_job_metadata(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    metadata_path = request.run_dir / "sco_acp_submitter_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "submit_cmd": [
                    "sco",
                    "acp",
                    "jobs",
                    "create",
                    "--workspace-name",
                    "workspace-a",
                    "--aec2-name",
                    "cluster-a",
                    "--job-name",
                    "researchclaw-s12",
                    "--worker-spec",
                    "n6ls.iu.i40.2c4g",
                    "--worker-nodes",
                    "1",
                    "--container-image-url",
                    "registry.example/image:tag",
                ],
                "stdout": "job pt-existing submitted successfully\n",
                "stderr": "",
                "returncode": 0,
                "script_path": str(request.run_dir / "stage-12-sco-acp.sh"),
                "job_id": "pt-existing",
            }
        ),
        encoding="utf-8",
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("submit should not create a second SCO job")

    monkeypatch.setattr(sco_acp_submitter.subprocess, "run", fail_run)

    result = submit(request)

    assert result.job_id == "pt-existing"
    assert result.metadata["workspace_name"] == "workspace-a"
    assert result.metadata["resumed_existing_submission"] is True


def test_parse_job_id_from_sco_create_output() -> None:
    assert _parse_job_id("job pt-ghdz0dsq submitted successfully, please wait!") == "pt-ghdz0dsq"
    assert _parse_job_id('{"id": "/workspaces/ws/trainingJobs/pt-json"}') == "pt-json"


def test_worker_spec_mapping() -> None:
    assert _select_worker_spec(0, 4) == "n6ls.iu.i40.2c4g"
    assert _select_worker_spec(0, 16) == "n6ls.iu.i40.4c16g"
    assert _select_worker_spec(1, 16) == "n6ls.iu.i40.1"
    assert _select_worker_spec(8, 16) == "n6ls.iu.i40.8"


def test_state_mapping() -> None:
    assert _map_state("SUCCEEDED") == "completed"
    assert _map_state("FAILED") == "failed"
    assert _map_state("SUSPENDED") == "failed"
    assert _map_state("RUNNING") == "running"


def _request(tmp_path: Path) -> SubmitRequest:
    workspace = tmp_path / "workspace"
    run_dir = tmp_path / "stage-12"
    workspace.mkdir()
    run_dir.mkdir()
    return SubmitRequest(
        manifest=RunManifest(
            code_commit="abc123def456",
            launch=LaunchCommand(
                command="python train.py",
                cwd=".",
                env={"TOKEN": "abc"},
                resources=ResourceSpec(gpus=0, mem_gb=4),
            ),
            result_paths=["outputs/metrics.json"],
        ),
        workspace_path=workspace,
        run_dir=run_dir,
        stage=12,
    )
