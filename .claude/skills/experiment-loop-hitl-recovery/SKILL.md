---
name: experiment-loop-hitl-recovery
description: >
  Recover ResearchClaw Stage 10-13 experiment loops after automatic repair
  attempts are exhausted, timeout stops the pipeline, or a human manually fixes
  the workspace and wants to rerun validation/execution.
metadata:
  category: researchclaw
  trigger-keywords: "experiment loop exhausted,manual fix,stage 10,stage 13,retry count,experiment_loop_history,max_cycles,HITL recovery"
  applicable-stages: "10,11,12,13"
  priority: "2"
  version: "1.0"
  author: researchclaw
---

# Experiment Loop HITL Recovery

Use this skill when the Stage 10-13 experiment loop has stopped or exhausted
automatic repair attempts, and a human has manually fixed the workspace or wants
to allow more retries.

## Mental Model

The experiment loop is:

```text
Stage 10: code agent implements or repairs code
Stage 11: validate run_manifest.json
Stage 12: submit/wait/collect experiment results
Stage 13: route based on status, metrics, artifacts, and contract
```

Automatic retries are counted in:

```text
<run_dir>/experiment_loop_history.json
```

The configured retry budget is:

```yaml
experiment:
  repair:
    max_cycles: 3
```

The loop can retry only while:

```text
len(experiment_loop_history.json["iterations"]) < experiment.repair.max_cycles
```

If the history already has 3 iterations and `max_cycles: 3`, ResearchClaw will
not spend another automatic repair attempt. To let it try again after human
intervention, either increase `max_cycles` or trim/reset the history file.

## Preferred Recovery: Increase max_cycles

Use this when you want to preserve the full repair history.

1. Edit the config used for the run:

   ```yaml
   experiment:
     repair:
       enabled: true
       max_cycles: 6
   ```

2. Resume from the appropriate stage.

   If the human already fixed code and updated `run_manifest.json`, resume at
   Stage 11:

   ```bash
   researchclaw run --config config.arc.yaml --output <run_dir> --from-stage MANIFEST_VALIDATE_AND_PREPARE
   ```

   If the agent should repair again using the existing `repair_request.json`,
   resume at Stage 10:

   ```bash
   researchclaw run --config config.arc.yaml --output <run_dir> --from-stage CODE_AGENT_IMPLEMENT_OR_REPAIR
   ```

   If Stage 12 timed out and the experiment was fixed externally, resume at
   Stage 12 only when the validated manifest is still correct:

   ```bash
   researchclaw run --config config.arc.yaml --output <run_dir> --from-stage HARNESS_SUBMIT_AND_COLLECT
   ```

## Manual Code Fix Checklist

When the human fixes the workspace manually, make the manual fix look like a
valid Stage 10 output before resuming at Stage 11.

1. Edit the configured workspace.
2. Commit the fix:

   ```bash
   git add <files>
   git commit -m "fix: manual experiment repair"
   git rev-parse HEAD
   ```

3. Update the workspace `run_manifest.json`:

   ```json
   {
     "code_commit": "<new_commit_sha>",
     "launch": {
       "command": "python train.py --output outputs/metrics.json",
       "cwd": ".",
       "env": {},
       "resources": {"gpus": 0, "time": "01:00:00", "partition": "", "mem_gb": 16}
     },
     "result_paths": ["outputs/metrics.json"],
     "metrics": {"primary": "accuracy", "direction": "maximize"}
   }
   ```

4. Copy the updated manifest into the current Stage 10 artifact location:

   ```bash
   cp <workspace>/run_manifest.json <run_dir>/stage-10/run_manifest.json
   ```

5. Archive stale downstream stage directories before resuming at Stage 11.
   ResearchClaw searches prior artifacts by stage directory, so an old
   `stage-11/run_manifest.json` can shadow the updated `stage-10/run_manifest.json`.

   ```bash
   ts=$(date +%Y%m%d-%H%M%S)
   for stage in 11 12 13; do
     if [ -d "<run_dir>/stage-${stage}" ]; then
       mv "<run_dir>/stage-${stage}" "<run_dir>/stage-${stage}_manual_backup_${ts}"
     fi
   done
   ```

6. If the old `repair_request.json` has already been resolved by the human,
   archive it so a later Stage 10 agent run does not consume stale instructions:

   ```bash
   mv <run_dir>/repair_request.json <run_dir>/repair_request_manual_resolved.json
   ```

7. Resume at Stage 11:

   ```bash
   researchclaw run --config config.arc.yaml --output <run_dir> --from-stage MANIFEST_VALIDATE_AND_PREPARE
   ```

## Resetting the Retry Counter

Use this only when you intentionally want to ignore old failed attempts for the
next automatic loop. Prefer increasing `max_cycles` when possible.

Back up the history first:

```bash
cp <run_dir>/experiment_loop_history.json <run_dir>/experiment_loop_history.before_manual_recovery.json
```

To reset all automatic retry count:

```bash
python - <<'PY'
import json
from pathlib import Path

run_dir = Path("<run_dir>")
path = run_dir / "experiment_loop_history.json"
payload = {
    "schema_version": "researchclaw.experiment_loop_history.v1",
    "iterations": [],
}
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
```

To allow exactly one more automatic retry while preserving earlier attempts in
the backup, trim the live file so it has `max_cycles - 1` entries.

## Which Stage Should I Resume From?

Use Stage 10 when:

```text
You want the code agent to repair again.
There is a current repair_request.json.
The workspace fix has not been manually committed.
```

Use Stage 11 when:

```text
A human already fixed and committed code.
run_manifest.json has been updated with the new commit.
Stale stage-11/stage-12/stage-13 directories have been archived or removed.
You want ResearchClaw to validate the manifest before submitting.
```

Use Stage 12 when:

```text
The manifest is already validated.
You only need to submit/run/collect again.
```

Use Stage 13 when:

```text
Stage 12 artifacts already exist and you only changed metadata or collected
missing result files manually.
```

## Important Files

```text
<run_dir>/experiment_loop_history.json       # retry counter
<run_dir>/repair_request.json                # Stage 10 repair input
<run_dir>/refine_request.json                # Stage 9 task-spec revision input
<run_dir>/stage-09/task_spec.yaml            # task and execution contract
<run_dir>/stage-10/run_manifest.json         # launch manifest consumed by Stage 11
<run_dir>/stage-11/manifest_validation.json  # validated manifest evidence
<run_dir>/stage-12/execution_record.json     # status and metrics
<run_dir>/stage-12/result_artifacts.json      # collected artifacts
<run_dir>/stage-13/experiment_decision.json   # route and reason
```

## Do Not

- Do not edit `experiment_loop_history.json` without making a backup.
- Do not resume at Stage 12 after manual code changes unless Stage 11 has already
  validated the updated manifest.
- Do not resume at Stage 11 while stale `stage-11/run_manifest.json` is still
  present; it can shadow the updated `stage-10/run_manifest.json`.
- Do not leave a stale `repair_request.json` if a human already fixed the issue
  and you are resuming at Stage 11.
- Do not change old `stage-10_v*` or `stage-12_v*` directories unless you are
  intentionally rewriting history for a local debugging run.
