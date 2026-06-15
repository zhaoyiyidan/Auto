from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from researchclaw.pipeline.stages import Stage

BRANCH_STATE_FILENAME = "branch_state.json"
BRANCH_STATE_SCHEMA_VERSION = "researchclaw.branch_state.v1"
BRANCH_STAGE_MIN = Stage.EXPERIMENT_TASK_SPEC
BRANCH_STAGE_MAX = Stage.RESEARCH_DECISION


class BranchStateError(RuntimeError):
    """Raised when branch resume state is unsafe to interpret silently."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _branch_state_path(branch_run_dir: Path) -> Path:
    return branch_run_dir / BRANCH_STATE_FILENAME


def _stage_name(stage_number: int) -> str:
    try:
        return Stage(stage_number).name
    except ValueError as exc:
        raise BranchStateError(f"Stage {stage_number} is outside the pipeline") from exc


def _resolve_from_last_completed(last_completed: object) -> Stage | None:
    last_completed_number = _coerce_int(last_completed)
    if last_completed_number is None:
        return BRANCH_STAGE_MIN
    if last_completed_number < int(BRANCH_STAGE_MIN):
        return BRANCH_STAGE_MIN
    if last_completed_number == int(BRANCH_STAGE_MAX):
        return None
    if last_completed_number > int(BRANCH_STAGE_MAX):
        raise BranchStateError(
            f"Branch checkpoint completed stage {last_completed_number}, "
            f"outside {int(BRANCH_STAGE_MIN)}-{int(BRANCH_STAGE_MAX)}"
        )

    next_stage_number = last_completed_number + 1
    if not int(BRANCH_STAGE_MIN) <= next_stage_number <= int(BRANCH_STAGE_MAX):
        raise BranchStateError(
            f"Resolved branch resume stage {next_stage_number}, "
            f"outside {int(BRANCH_STAGE_MIN)}-{int(BRANCH_STAGE_MAX)}"
        )
    return Stage(next_stage_number)


def _read_last_completed(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return _coerce_int(data.get("last_completed_stage"))


def _has_branch_stage_artifacts(branch_run_dir: Path) -> bool:
    if not branch_run_dir.exists():
        return False
    for child in branch_run_dir.iterdir():
        if not child.name.startswith("stage-"):
            continue
        stage_text = child.name.removeprefix("stage-")
        try:
            stage_number = int(stage_text)
        except ValueError:
            continue
        if int(BRANCH_STAGE_MIN) <= stage_number <= int(BRANCH_STAGE_MAX):
            return True
    return False


def write_branch_stage_done(
    branch_run_dir: Path | str,
    stage: Stage,
    *,
    attempt_id: str,
    node_id: str,
    workspace_path: Path | str,
) -> None:
    branch_run_path = Path(branch_run_dir)
    branch_run_path.mkdir(parents=True, exist_ok=True)
    stage = Stage(stage)
    if not int(BRANCH_STAGE_MIN) <= int(stage) <= int(BRANCH_STAGE_MAX):
        raise BranchStateError(
            f"Cannot record non-branch stage {int(stage)} in branch state"
        )

    existing = read_branch_state(branch_run_path) or {}
    existing_status = existing.get("stage_status")
    if isinstance(existing_status, dict):
        stage_status = {
            str(key): str(value) for key, value in existing_status.items()
        }
    else:
        stage_status = {}
    stage_status[str(int(stage))] = "done"

    previous_last = _coerce_int(existing.get("last_completed_stage"))
    if previous_last is None:
        last_completed = int(stage)
    else:
        last_completed = max(previous_last, int(stage))

    state: dict[str, Any] = {
        "schema_version": BRANCH_STATE_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "node_id": node_id,
        "last_completed_stage": last_completed,
        "last_completed_name": _stage_name(last_completed),
        "stage_status": stage_status,
        "workspace_path": str(workspace_path),
        "updated_at": _utcnow_iso(),
    }

    target = _branch_state_path(branch_run_path)
    fd, tmp_path = tempfile.mkstemp(
        dir=branch_run_path,
        suffix=".tmp",
        prefix="branch_state_",
    )
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(state, indent=2))
        Path(tmp_path).replace(target)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def read_branch_state(branch_run_dir: Path | str) -> dict[str, Any] | None:
    state_path = _branch_state_path(Path(branch_run_dir))
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def resolve_branch_resume_stage(branch_run_dir: Path | str) -> Stage | None:
    branch_run_path = Path(branch_run_dir)
    state_path = _branch_state_path(branch_run_path)
    state = read_branch_state(branch_run_path)
    if state is not None:
        return _resolve_from_last_completed(state.get("last_completed_stage"))

    checkpoint_last_completed = _read_last_completed(
        branch_run_path / "checkpoint.json"
    )
    if checkpoint_last_completed is not None:
        return _resolve_from_last_completed(checkpoint_last_completed)

    if state_path.exists() and _has_branch_stage_artifacts(branch_run_path):
        raise BranchStateError(
            f"{BRANCH_STATE_FILENAME} is corrupt and branch stage artifacts exist"
        )

    return BRANCH_STAGE_MIN

