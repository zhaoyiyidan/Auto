"""Pluggable submitters for workspace-native experiment runs."""

from __future__ import annotations

import importlib
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from researchclaw.experiment.workspace import SubmitRequest, SubmitResult


@runtime_checkable
class TrainingSubmitter(Protocol):
    """Protocol implemented by training job submitters."""

    name: str

    def submit(self, request: SubmitRequest) -> SubmitResult:
        """Submit a workspace launch request."""
        ...


class LocalSubmitter:
    """Run the launch command locally in the background."""

    name = "local"

    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen[bytes]] = {}

    def submit(self, request: SubmitRequest) -> SubmitResult:
        run_dir = request.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / f"stage-{request.stage:02d}-local.log"
        cwd = (request.workspace_path / request.manifest.launch.cwd).resolve()
        env = None
        if request.manifest.launch.env:
            import os

            env = dict(os.environ)
            env.update(request.manifest.launch.env)
        log_handle = log_path.open("ab")
        try:
            proc = subprocess.Popen(
                request.manifest.launch.command,
                cwd=cwd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=True,
            )
        finally:
            log_handle.close()
        job_id = f"local_{proc.pid}_{int(time.time())}"
        self._procs[job_id] = proc
        return SubmitResult(
            job_id=job_id,
            submitter_name=self.name,
            status="submitted",
            metadata={
                "pid": proc.pid,
                "command": request.manifest.launch.command,
                "cwd": str(cwd),
                "log_path": str(log_path),
            },
        )

    def poll(self, result: SubmitResult) -> str:
        proc = self._procs.get(result.job_id)
        if proc is None:
            return "unknown"
        returncode = proc.poll()
        if returncode is None:
            return "running"
        if returncode == 0:
            return "completed"
        return "failed"


class SlurmSubmitter:
    """Submit the launch command with ``sbatch``."""

    name = "slurm"

    def generate_sbatch(self, request: SubmitRequest) -> str:
        resources = request.manifest.launch.resources
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=researchclaw-stage-{request.stage:02d}",
            f"#SBATCH --mem={resources.mem_gb}G",
            f"#SBATCH --time={resources.time}",
            "#SBATCH --output=slurm-%j.out",
            "#SBATCH --error=slurm-%j.err",
        ]
        if resources.gpus:
            lines.append(f"#SBATCH --gres=gpu:{resources.gpus}")
        if resources.partition:
            lines.append(f"#SBATCH --partition={resources.partition}")
        lines.extend(["", f"cd {shlex.quote(request.manifest.launch.cwd)}"])
        for key, value in sorted(request.manifest.launch.env.items()):
            lines.append(f"export {key}={shlex.quote(value)}")
        lines.append(request.manifest.launch.command)
        return "\n".join(lines)

    def submit(self, request: SubmitRequest) -> SubmitResult:
        request.run_dir.mkdir(parents=True, exist_ok=True)
        script_path = request.run_dir / f"stage-{request.stage:02d}.sbatch"
        script_path.write_text(self.generate_sbatch(request), encoding="utf-8")
        proc = subprocess.run(
            ["sbatch", str(script_path)],
            cwd=request.workspace_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "sbatch failed")
        job_id = _parse_slurm_job_id(proc.stdout)
        return SubmitResult(
            job_id=job_id,
            submitter_name=self.name,
            status="submitted",
            metadata={
                "script_path": str(script_path),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
        )

    def poll(self, result: SubmitResult) -> str:
        try:
            proc = subprocess.run(
                ["sacct", "-j", result.job_id, "-n", "-o", "State"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            proc = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        if proc.returncode == 0 and proc.stdout.strip():
            return _map_slurm_state(proc.stdout)
        try:
            queue_proc = subprocess.run(
                ["squeue", "-j", result.job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return "unknown"
        if queue_proc.returncode == 0 and queue_proc.stdout.strip():
            return _map_slurm_state(queue_proc.stdout)
        return "unknown"


class SshSlurmSubmitter:
    """Submit a Slurm job through SSH."""

    name = "ssh_slurm"

    def __init__(
        self,
        *,
        host: str,
        user: str = "",
        port: int = 22,
        key_path: str = "",
    ) -> None:
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path

    def build_ssh_command(self, request: SubmitRequest) -> list[str]:
        target = f"{self.user}@{self.host}" if self.user else self.host
        remote_dir = shlex.quote(str(request.workspace_path))
        script = SlurmSubmitter().generate_sbatch(request)
        remote = (
            f"cd {remote_dir} && "
            f"cat <<'EOFSCRIPT' > _researchclaw_job.sbatch\n{script}\nEOFSCRIPT\n"
            "sbatch _researchclaw_job.sbatch"
        )
        cmd = ["ssh", "-p", str(self.port)]
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        cmd.extend([target, "bash -lc", shlex.quote(remote)])
        return cmd

    def submit(self, request: SubmitRequest) -> SubmitResult:
        proc = subprocess.run(
            self.build_ssh_command(request),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ssh slurm submission failed")
        return SubmitResult(
            job_id=_parse_slurm_job_id(proc.stdout),
            submitter_name=self.name,
            status="submitted",
            metadata={"stdout": proc.stdout, "stderr": proc.stderr},
        )

    def poll(self, result: SubmitResult) -> str:
        target = f"{self.user}@{self.host}" if self.user else self.host
        remote = (
            f"sacct -j {shlex.quote(result.job_id)} -n -o State || "
            f"squeue -j {shlex.quote(result.job_id)} -h -o %T"
        )
        cmd = ["ssh", "-p", str(self.port)]
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        cmd.extend([target, "bash -lc", shlex.quote(remote)])
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return "unknown"
        if proc.returncode != 0 or not proc.stdout.strip():
            return "unknown"
        return _map_slurm_state(proc.stdout)


class ManualSubmitter:
    """Write a runnable script and print instructions without submitting it."""

    name = "manual"

    def submit(self, request: SubmitRequest) -> SubmitResult:
        request.run_dir.mkdir(parents=True, exist_ok=True)
        script_path = request.run_dir / f"stage-{request.stage:02d}-manual.sh"
        command = request.manifest.launch.command
        script_path.write_text(
            "#!/bin/bash\n"
            f"cd {shlex.quote(str((request.workspace_path / request.manifest.launch.cwd).resolve()))}\n"
            f"{command}\n",
            encoding="utf-8",
        )
        print(f"Run manually: {command}")
        return SubmitResult(
            job_id=f"manual_{request.stage}_{int(time.time())}",
            submitter_name=self.name,
            status="manual",
            metadata={"script_path": str(script_path), "command": command},
        )

    def poll(self, result: SubmitResult) -> str:
        return "unknown"


class CustomPythonSubmitter:
    """Submitter backed by a user-provided Python callable."""

    name = "custom_python"

    def __init__(self, callable_or_path: Callable[[SubmitRequest], SubmitResult] | str) -> None:
        self._callable = (
            _load_callable(callable_or_path)
            if isinstance(callable_or_path, str)
            else callable_or_path
        )

    def submit(self, request: SubmitRequest) -> SubmitResult:
        return self._callable(request)

    def poll(self, result: SubmitResult) -> str:
        poll = getattr(self._callable, "poll", None)
        if callable(poll):
            return str(poll(result))
        return "unknown"


def create_submitter(config: Any) -> TrainingSubmitter:
    """Create a submitter from a dict, SubmitterConfig, or RCConfig."""
    cfg = _coerce_submitter_config(config)
    kind = str(_cfg_get(cfg, "type", "local"))
    if kind == "local":
        return LocalSubmitter()
    if kind == "slurm":
        return SlurmSubmitter()
    if kind == "ssh_slurm":
        return SshSlurmSubmitter(
            host=str(_cfg_get(cfg, "ssh_host", "")),
            user=str(_cfg_get(cfg, "ssh_user", "")),
            port=int(_cfg_get(cfg, "ssh_port", 22)),
            key_path=str(_cfg_get(cfg, "ssh_key_path", "")),
        )
    if kind == "manual":
        return ManualSubmitter()
    if kind == "custom_python":
        callable_path = str(_cfg_get(cfg, "custom_callable", ""))
        if not callable_path:
            raise ValueError("custom_python submitter requires custom_callable")
        return CustomPythonSubmitter(callable_path)
    raise ValueError(f"Unknown submitter type: {kind}")


def _parse_slurm_job_id(output: str) -> str:
    parts = output.strip().split()
    if parts and parts[-1].isdigit():
        return parts[-1]
    raise RuntimeError(f"Could not parse sbatch output: {output.strip()}")


def _map_slurm_state(output: str) -> str:
    states = {
        line.strip().split()[0].upper()
        for line in output.splitlines()
        if line.strip()
    }
    if not states:
        return "unknown"
    if states & {"COMPLETED"}:
        return "completed"
    if states & {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }:
        return "failed"
    if states & {
        "CONFIGURING",
        "COMPLETING",
        "PENDING",
        "RESIZING",
        "RUNNING",
        "SUSPENDED",
    }:
        return "running"
    return "unknown"


def _load_callable(path: str) -> Callable[[SubmitRequest], SubmitResult]:
    module_name, sep, attr = path.partition(":")
    if not sep:
        module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid callable path: {path}")
    module = importlib.import_module(module_name)
    func = getattr(module, attr)
    if not callable(func):
        raise TypeError(f"Configured submitter is not callable: {path}")
    return func


def _coerce_submitter_config(config: Any) -> Any:
    if isinstance(config, dict):
        return config
    experiment = getattr(config, "experiment", None)
    if experiment is not None and hasattr(experiment, "submitter"):
        return experiment.submitter
    return config


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
