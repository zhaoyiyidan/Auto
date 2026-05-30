---
name: custom-python-submitter
description: >
  Write and configure ResearchClaw Stage 12 custom_python submitters. Use when a
  project needs a custom cluster, queue, cloud, container, SSH, or lab-specific
  experiment submission backend instead of local/slurm/ssh_slurm/manual.
metadata:
  category: researchclaw
  trigger-keywords: "custom submitter,custom_python,stage 12,cluster submitter,submitter,SubmitRequest,SubmitResult"
  applicable-stages: "12"
  priority: "2"
  version: "1.1"
  author: researchclaw
---

# ResearchClaw Custom Python Submitter

Use this skill when implementing a project-specific Stage 12 submitter for
ResearchClaw workspace-native experiment runs.

Stage 12 reads the validated `run_manifest.json`, creates a submitter from
`experiment.submitter`, calls `submitter.submit(SubmitRequest)`, optionally
polls for completion, then collects result artifacts and metrics from the
workspace paths declared in the manifest.

## Current Interface

Configure the submitter as:

```yaml
experiment:
  submitter:
    type: custom_python
    custom_callable: "my_project.submitters.arc_cluster:submit"
    wait_for_completion: true
    poll_interval_sec: 300
    wait_timeout_sec: 259200  # seconds; 72 hours
```

`custom_callable` supports both import formats:

```text
module.submodule:function_name
module.submodule.function_name
```

The imported object must be callable and accept exactly one argument:

```python
from researchclaw.experiment.workspace import SubmitRequest, SubmitResult

def submit(request: SubmitRequest) -> SubmitResult:
    ...
```

Return a `SubmitResult`:

```python
SubmitResult(
    job_id="cluster-job-id",
    submitter_name="my_cluster",
    status="submitted",
    metadata={
        "log_path": "/path/or/relative/log/file",
        "anything_else": "...",
    },
)
```

If submission cannot be started, raise an exception. Stage 12 will fail with
`E12_HARNESS_FAIL: <exception>`.

## SubmitRequest Fields

`request.manifest` is the validated `RunManifest` from Stage 11.

Important manifest fields:

```python
request.manifest.code_commit          # git commit produced by Stage 10
request.manifest.launch.command       # command the agent wants to run
request.manifest.launch.cwd           # cwd relative to workspace, usually "."
request.manifest.launch.env           # env vars from manifest
request.manifest.launch.resources.gpus
request.manifest.launch.resources.time
request.manifest.launch.resources.partition
request.manifest.launch.resources.mem_gb
request.manifest.result_paths         # artifacts Stage 12 will collect later
request.manifest.metrics.primary
request.manifest.metrics.direction
```

Other request fields:

```python
request.workspace_path  # pathlib.Path to the git workspace
request.run_dir         # pathlib.Path to stage-12 output directory
request.stage           # int, normally 12
```

## Stage 12 Behavior You Must Design For

After your callable returns, ResearchClaw:

1. Polls completion if `wait_for_completion: true`.
2. Collects every path in `request.manifest.result_paths` from
   `request.workspace_path`.
3. Hashes existing result artifacts.
4. Reads metrics only from result paths that are JSON files.
5. Merges top-level JSON object keys into `execution_record.json.metrics`.

Therefore the submitted job must write results into the workspace at the exact
relative paths declared by `manifest.result_paths`, for example:

```json
{
  "accuracy": 0.91,
  "loss": 0.23,
  "aggregates": {
    "baseline/seed_0": {
      "accuracy": {"mean": 0.88, "n": 3}
    }
  }
}
```

## Minimal Local Wrapper Template

Use this when a site-specific wrapper command must be used but execution is
still local to the workspace machine.

```python
# my_project/submitters/local_wrapper.py
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from researchclaw.experiment.workspace import SubmitRequest, SubmitResult


def submit(request: SubmitRequest) -> SubmitResult:
    request.run_dir.mkdir(parents=True, exist_ok=True)

    cwd = (request.workspace_path / request.manifest.launch.cwd).resolve()
    log_path = request.run_dir / f"stage-{request.stage:02d}-custom.log"

    env = dict(os.environ)
    env.update(request.manifest.launch.env)

    command = request.manifest.launch.command
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    return SubmitResult(
        job_id=f"local_custom_{proc.pid}_{int(time.time())}",
        submitter_name="local_custom",
        status="submitted",
        metadata={
            "pid": proc.pid,
            "command": command,
            "cwd": str(cwd),
            "log_path": str(log_path),
        },
    )
```

## Cluster Submitter Template

Use this when submitting to a lab scheduler or custom cluster CLI.

```python
# my_project/submitters/my_cluster.py
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from researchclaw.experiment.workspace import SubmitRequest, SubmitResult


def submit(request: SubmitRequest) -> SubmitResult:
    request.run_dir.mkdir(parents=True, exist_ok=True)

    launch = request.manifest.launch
    resources = launch.resources
    cwd = (request.workspace_path / launch.cwd).resolve()

    env_lines = [
        f"export {key}={shlex.quote(str(value))}"
        for key, value in sorted(launch.env.items())
    ]
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(cwd))}",
            *env_lines,
            launch.command,
        ]
    )

    script_path = request.run_dir / f"stage-{request.stage:02d}-custom-submit.sh"
    script_path.write_text(script + "\n", encoding="utf-8")

    submit_cmd = [
        "myqueue",
        "submit",
        "--name", f"researchclaw-stage-{request.stage:02d}",
        "--gpus", str(resources.gpus),
        "--mem-gb", str(resources.mem_gb),
        "--time", resources.time,
        "--script", str(script_path),
    ]
    if resources.partition:
        submit_cmd.extend(["--partition", resources.partition])

    proc = subprocess.run(
        submit_cmd,
        cwd=request.workspace_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "custom cluster submission failed")

    job_id = parse_job_id(proc.stdout)

    metadata_path = request.run_dir / "custom_submitter_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "submit_cmd": submit_cmd,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "script_path": str(script_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return SubmitResult(
        job_id=job_id,
        submitter_name="my_cluster",
        status="submitted",
        metadata={
            "script_path": str(script_path),
            "metadata_path": str(metadata_path),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        },
    )


def parse_job_id(stdout: str) -> str:
    # Replace this with the scheduler's real output parser.
    # Prefer raising over returning a fake id.
    parts = stdout.strip().split()
    if not parts:
        raise RuntimeError("could not parse job id from empty submit output")
    return parts[-1]
```

## Stage 12 to Stage 13 Status Contract

The current ResearchClaw architecture uses a synchronous Stage 12 contract:
Stage 12 must not hand an in-progress job to Stage 13. It should submit the
job, poll until a terminal outcome or a configured timeout, then write
`execution_record.json`.

The only final statuses a custom submitter should allow Stage 13 to consume are:

```text
completed  # job finished successfully and results should be available
failed     # job failed, was cancelled, crashed, hit OOM, or cannot be trusted
timeout    # ResearchClaw wait_timeout_sec elapsed before completion
```

Stage 13 treats them as:

```text
completed -> validate metrics/artifacts/contract, then continue or fix_code
failed    -> fix_code, returning to Stage 10
timeout   -> stop the pipeline for manual debugging; no repair_request.json
```

Use `wait_for_completion: true`. If `wait_for_completion: false`, Stage 12 can
record `submitted`, which is not compatible with this terminal-status contract.

Configure the maximum experiment wait in seconds. If the project thinks in hours,
convert manually:

```text
wait_timeout_sec = max_experiment_hours * 3600
```

For example, a 72 hour maximum is `259200`.

## Polling Status

ResearchClaw's current `custom_python` wrapper calls:

```python
poll = getattr(the_callable, "poll", None)
```

This means polling works only if the imported callable object itself has a
`poll(result: SubmitResult) -> str` attribute. A plain function imported by
`"module:function"` usually does not carry durable scheduler state, but it can
still have an attribute assigned at module import time:

```python
from researchclaw.experiment.workspace import SubmitResult


def submit(request):
    ...


def poll(result: SubmitResult) -> str:
    proc = subprocess.run(
        ["myqueue", "status", result.job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    text = proc.stdout.lower()
    if "completed" in text or "succeeded" in text:
        return "completed"
    if (
        "failed" in text
        or "cancelled" in text
        or "out_of_memory" in text
        or "oom" in text
        or "time_limit" in text
        or "timeout" in text
    ):
        return "failed"
    if "running" in text or "queued" in text or "pending" in text or "submitted" in text:
        return "running"
    # Transient scheduler lookup issues should not leak to Stage 13.
    # Return "running" to keep polling, or "failed" if the job is unrecoverable.
    return "running"


submit.poll = poll
```

`poll()` should return only:

```text
running    # still queued/submitted/running; Stage 12 keeps polling
completed  # terminal success; Stage 13 can validate outputs
failed     # terminal failure; Stage 13 routes to fix_code
```

Do not return scheduler-specific strings such as `PENDING`, `RUNNING`,
`COMPLETED`, `FAILED`, `CANCELLED`, or `OUT_OF_MEMORY` directly. Map them to the
canonical statuses above.

Do not return `unknown` for transient scheduler lookup failures under the current
Stage 13 contract. Returning `unknown` causes Stage 12 to stop waiting and write
`final_status: unknown`, which Stage 13 treats as an execution failure. Prefer:

```text
temporary status lookup failure -> running
permanent job/status loss       -> failed
```

Do not return `timeout` from `poll()`. ResearchClaw generates
`final_status: timeout` when `wait_timeout_sec` is exceeded. If the scheduler
itself reports a time-limit failure, map it to `failed` and preserve the raw
status in logs or submitter metadata.

Typical scheduler mapping:

```text
PENDING / QUEUED / SUBMITTED -> running
RUNNING                      -> running
COMPLETED / SUCCEEDED        -> completed
FAILED / CANCELLED / OOM     -> failed
TIME_LIMIT / DEADLINE        -> failed
```

## Practical Checklist

Before using a custom submitter:

1. Put the module on `PYTHONPATH` so `importlib.import_module()` can import it.
2. Confirm the config path imports:
   ```bash
   python -c "from my_project.submitters.my_cluster import submit; print(submit)"
   ```
3. Do not fabricate success. If submission fails, raise `RuntimeError`.
4. Always return a real scheduler `job_id` or a traceable placeholder for manual systems.
5. Write logs or submit metadata under `request.run_dir`.
6. Ensure the job writes all `manifest.result_paths` under `request.workspace_path`.
7. If metrics are needed, ensure at least one `result_paths` entry is a JSON file
   containing top-level metric keys.
8. Keep `wait_for_completion: true` for the current Stage 12/13 contract and
   attach `submit.poll = poll`.
9. Keep secrets out of config and logs; read tokens from environment variables.

## Quick Unit Test Template

```python
from pathlib import Path

from my_project.submitters.my_cluster import submit
from researchclaw.experiment.workspace import (
    LaunchCommand,
    RunManifest,
    SubmitRequest,
)


def test_custom_submitter_smoke(tmp_path: Path):
    workspace = tmp_path / "workspace"
    run_dir = tmp_path / "stage-12"
    workspace.mkdir()
    run_dir.mkdir()

    request = SubmitRequest(
        manifest=RunManifest(
            code_commit="abc123",
            launch=LaunchCommand(command="python train.py", cwd="."),
            result_paths=["outputs/metrics.json"],
        ),
        workspace_path=workspace,
        run_dir=run_dir,
        stage=12,
    )

    result = submit(request)

    assert result.job_id
    assert result.submitter_name
    assert result.status in {"submitted", "manual"}
    if hasattr(submit, "poll"):
        assert submit.poll(result) in {"running", "completed", "failed"}
```

## Source Files

Relevant implementation points:

- `researchclaw/experiment/submitter.py`
- `researchclaw/experiment/workspace.py`
- `researchclaw/pipeline/workspace_orchestrator.py`
- `tests/test_training_submitter.py`
