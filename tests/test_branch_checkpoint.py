from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.pipeline.branch_checkpoint import (
    BRANCH_STATE_FILENAME,
    BranchStateError,
    read_branch_state,
    resolve_branch_resume_stage,
    write_branch_stage_done,
)
from researchclaw.pipeline.stages import Stage


def test_write_and_read_branch_state_round_trip(tmp_path: Path) -> None:
    branch_run_dir = tmp_path / "branch"
    workspace_path = tmp_path / "workspace"

    write_branch_stage_done(
        branch_run_dir,
        Stage.HARNESS_SUBMIT_AND_COLLECT,
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        workspace_path=workspace_path,
    )

    state = read_branch_state(branch_run_dir)

    assert state is not None
    assert state["schema_version"] == "researchclaw.branch_state.v1"
    assert state["attempt_id"] == "h-001/attempt-001"
    assert state["node_id"] == "h-001"
    assert state["last_completed_stage"] == int(Stage.HARNESS_SUBMIT_AND_COLLECT)
    assert state["last_completed_name"] == Stage.HARNESS_SUBMIT_AND_COLLECT.name
    assert state["stage_status"] == {
        str(int(Stage.HARNESS_SUBMIT_AND_COLLECT)): "done"
    }
    assert state["workspace_path"] == str(workspace_path)
    assert isinstance(state["updated_at"], str)


def test_write_branch_stage_done_is_monotonic(tmp_path: Path) -> None:
    branch_run_dir = tmp_path / "branch"

    write_branch_stage_done(
        branch_run_dir,
        Stage.RESULT_ANALYSIS,
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        workspace_path=tmp_path / "workspace",
    )
    write_branch_stage_done(
        branch_run_dir,
        Stage.EXPERIMENT_TASK_SPEC,
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        workspace_path=tmp_path / "workspace",
    )

    state = read_branch_state(branch_run_dir)

    assert state is not None
    assert state["last_completed_stage"] == int(Stage.RESULT_ANALYSIS)
    assert state["last_completed_name"] == Stage.RESULT_ANALYSIS.name
    assert state["stage_status"][str(int(Stage.RESULT_ANALYSIS))] == "done"
    assert state["stage_status"][str(int(Stage.EXPERIMENT_TASK_SPEC))] == "done"


@pytest.mark.parametrize(
    ("last_completed", "expected_stage"),
    [
        (None, Stage.EXPERIMENT_TASK_SPEC),
        (8, Stage.EXPERIMENT_TASK_SPEC),
        (9, Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR),
        (12, Stage.EXPERIMENT_ROUTE_DECISION),
        (14, Stage.RESEARCH_DECISION),
        (15, None),
    ],
)
def test_resolve_branch_resume_stage_from_branch_state(
    tmp_path: Path,
    last_completed: int | None,
    expected_stage: Stage | None,
) -> None:
    branch_run_dir = tmp_path / "branch"
    branch_run_dir.mkdir()
    state = {
        "schema_version": "researchclaw.branch_state.v1",
        "attempt_id": "h-001/attempt-001",
        "node_id": "h-001",
        "stage_status": {},
        "workspace_path": str(tmp_path / "workspace"),
        "updated_at": "2026-06-15T00:00:00+00:00",
    }
    if last_completed is not None:
        state["last_completed_stage"] = last_completed
    (branch_run_dir / BRANCH_STATE_FILENAME).write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    assert resolve_branch_resume_stage(branch_run_dir) == expected_stage


def test_resolve_branch_resume_stage_falls_back_to_checkpoint_json(
    tmp_path: Path,
) -> None:
    branch_run_dir = tmp_path / "branch"
    branch_run_dir.mkdir()
    (branch_run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "last_completed_stage": int(Stage.MANIFEST_VALIDATE_AND_PREPARE),
                "last_completed_name": Stage.MANIFEST_VALIDATE_AND_PREPARE.name,
            }
        ),
        encoding="utf-8",
    )

    assert (
        resolve_branch_resume_stage(branch_run_dir)
        == Stage.HARNESS_SUBMIT_AND_COLLECT
    )


def test_read_branch_state_returns_none_for_missing_or_corrupt_file(
    tmp_path: Path,
) -> None:
    branch_run_dir = tmp_path / "branch"
    branch_run_dir.mkdir()

    assert read_branch_state(branch_run_dir) is None

    (branch_run_dir / BRANCH_STATE_FILENAME).write_text("{broken", encoding="utf-8")

    assert read_branch_state(branch_run_dir) is None
    assert resolve_branch_resume_stage(branch_run_dir) == Stage.EXPERIMENT_TASK_SPEC


def test_resolve_branch_resume_stage_raises_when_state_is_corrupt_with_stage_artifacts(
    tmp_path: Path,
) -> None:
    branch_run_dir = tmp_path / "branch"
    branch_run_dir.mkdir()
    (branch_run_dir / BRANCH_STATE_FILENAME).write_text("{broken", encoding="utf-8")
    (branch_run_dir / "stage-12").mkdir()

    with pytest.raises(BranchStateError):
        resolve_branch_resume_stage(branch_run_dir)
