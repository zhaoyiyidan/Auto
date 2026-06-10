"""SCO ACP training submitter integration."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from researchclaw.experiment.workspace import SubmitRequest, SubmitResult

METADATA_FILE = "sco_acp_submitter_metadata.json"


def submit(request: SubmitRequest) -> SubmitResult:
    request.run_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_submission(request.run_dir)
    if existing is not None:
        return existing

    script_path = request.run_dir / f"stage-{request.stage:02d}-sco-acp.sh"
    script = _build_launch_script(request)
    script_path.write_text(script, encoding="utf-8")

    workspace_name = _env_required("SCO_ACP_WORKSPACE_NAME")
    aec2_name = _env_required("SCO_ACP_AEC2_NAME")
    image = _env_required("SCO_ACP_IMAGE")
    job_name = _job_name(request.stage)
    cmd = [
        "sco",
        "acp",
        "jobs",
        "create",
        "--workspace-name",
        workspace_name,
        "--aec2-name",
        aec2_name,
        "--job-name",
        job_name,
        "--worker-spec",
        _select_worker_spec(
            request.manifest.launch.resources.gpus,
            request.manifest.launch.resources.mem_gb,
        ),
        "--worker-nodes",
        "1",
        "--container-image-url",
        image,
        "--command",
        script,
    ]
    storage_mount = os.environ.get("SCO_ACP_STORAGE_MOUNT", "")
    if storage_mount:
        cmd.extend(["--storage-mount", storage_mount])

    proc = subprocess.run(
        cmd,
        cwd=request.workspace_path,
        capture_output=True,
        text=True,
        check=False,
    )
    metadata = {
        "submit_cmd": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
        "script_path": str(script_path),
        "workspace_name": workspace_name,
        "aec2_name": aec2_name,
        "job_name": job_name,
    }
    if proc.returncode != 0:
        _write_metadata(request.run_dir, metadata)
        raise RuntimeError(proc.stderr.strip() or "sco acp job create failed")

    job_id = _parse_job_id(proc.stdout)
    metadata["job_id"] = job_id
    metadata["describe_path"] = str(request.run_dir / "sco_acp_describe.json")
    metadata["log_path"] = str(request.run_dir / "sco_acp_job.log")
    _write_metadata(request.run_dir, metadata)
    return SubmitResult(
        job_id=job_id,
        submitter_name="sco_acp",
        status="submitted",
        metadata=metadata,
    )


def poll(result: SubmitResult) -> str:
    workspace_name = str(result.metadata.get("workspace_name") or "")
    if not workspace_name:
        workspace_name = os.environ.get("SCO_ACP_WORKSPACE_NAME", "")
    if not workspace_name:
        return "unknown"

    describe_cmd = [
        "sco",
        "acp",
        "jobs",
        "describe",
        "--workspace-name",
        workspace_name,
        "--name",
        result.job_id,
    ]
    describe = subprocess.run(
        describe_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if describe.returncode != 0:
        return "unknown"

    describe_path = result.metadata.get("describe_path")
    if describe_path:
        Path(str(describe_path)).write_text(describe.stdout, encoding="utf-8")

    log_cmd = [
        "sco",
        "acp",
        "jobs",
        "stream-logs",
        "--workspace-name",
        workspace_name,
        "--name",
        result.job_id,
    ]
    logs = subprocess.run(
        log_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path = result.metadata.get("log_path")
    if log_path and logs.stdout:
        Path(str(log_path)).write_text(logs.stdout, encoding="utf-8")

    try:
        payload = json.loads(describe.stdout or "{}")
    except ValueError:
        payload = {}
    return _map_state(str(payload.get("state") or payload.get("status") or ""))


def _parse_job_id(output: str) -> str:
    try:
        payload = json.loads(output)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        for key in ("job_id", "id", "name"):
            value = payload.get(key)
            if isinstance(value, str):
                match = re.search(r"pt-[A-Za-z0-9-]+", value)
                if match:
                    return match.group(0)

    match = re.search(r"\bpt-[A-Za-z0-9-]+\b", output)
    if match:
        return match.group(0)
    raise RuntimeError(f"Could not parse SCO ACP job id from output: {output.strip()}")


def _select_worker_spec(gpus: int, mem_gb: int) -> str:
    if gpus > 0:
        return f"n6ls.iu.i40.{gpus}"
    if mem_gb <= 4:
        return "n6ls.iu.i40.2c4g"
    return "n6ls.iu.i40.4c16g"


def _map_state(state: str) -> str:
    normalized = state.strip().upper()
    if normalized in {"SUCCEEDED", "SUCCESS", "COMPLETED", "COMPLETE"}:
        return "completed"
    if normalized in {
        "FAILED",
        "FAILURE",
        "CANCELLED",
        "CANCELED",
        "SUSPENDED",
        "STOPPED",
        "ERROR",
    }:
        return "failed"
    if normalized in {"RUNNING", "PENDING", "QUEUED", "CREATING", "STARTING"}:
        return "running"
    return "unknown"


def _load_existing_submission(run_dir: Path) -> SubmitResult | None:
    metadata_path = run_dir / METADATA_FILE
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if int(metadata.get("returncode", 1)) != 0:
        return None
    job_id = str(metadata.get("job_id") or "")
    if not job_id:
        stdout = str(metadata.get("stdout") or "")
        job_id = _parse_job_id(stdout)

    resumed = dict(metadata)
    resumed["resumed_existing_submission"] = True
    resumed.setdefault("workspace_name", _arg_after(resumed.get("submit_cmd"), "--workspace-name"))
    resumed.setdefault("describe_path", str(run_dir / "sco_acp_describe.json"))
    resumed.setdefault("log_path", str(run_dir / "sco_acp_job.log"))
    return SubmitResult(
        job_id=job_id,
        submitter_name="sco_acp",
        status="submitted",
        metadata=resumed,
    )


def _build_launch_script(request: SubmitRequest) -> str:
    cwd = request.workspace_path / request.manifest.launch.cwd
    lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(cwd.resolve()))}",
    ]
    for key, value in sorted(request.manifest.launch.env.items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append(request.manifest.launch.command)
    return "\n".join(lines)


def _job_name(stage: int) -> str:
    prefix = os.environ.get("SCO_ACP_JOB_PREFIX", "researchclaw")
    return f"{prefix}-s{stage:02d}"


def _env_required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is required for SCO ACP submission")
    return value


def _write_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    (run_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _arg_after(cmd: object, flag: str) -> str:
    if not isinstance(cmd, list):
        return ""
    try:
        return str(cmd[cmd.index(flag) + 1])
    except (ValueError, IndexError):
        return ""
