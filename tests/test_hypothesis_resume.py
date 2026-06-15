from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline import runner as rc_runner
from researchclaw.pipeline import hypothesis_branch
from researchclaw.pipeline.branch_checkpoint import (
    read_branch_state,
    write_branch_stage_done,
)
from researchclaw.pipeline.executor import StageResult
from researchclaw.pipeline.hypothesis_branch import validate_branch
from researchclaw.pipeline.hypothesis_coordinator import HypothesisValidationCoordinator
from researchclaw.pipeline.hypothesis_store import HypothesisNode, ValidationAttempt
from researchclaw.pipeline.stages import Stage, StageStatus


def _branch_dir(tmp_path: Path) -> Path:
    return tmp_path / "run" / "hypothesis_branches" / "h-001" / "attempt-001"


def _node(node_id: str = "h-001") -> HypothesisNode:
    return HypothesisNode(
        id=node_id,
        statement="Treatment improves accuracy.",
        prediction="Accuracy improves.",
        falsification="Accuracy does not improve.",
    )


def _attempt(branch_run_dir: Path, *, node_id: str = "h-001") -> ValidationAttempt:
    return ValidationAttempt(
        attempt_id=f"{node_id}/attempt-001",
        node_id=node_id,
        branch_run_dir=str(branch_run_dir),
        workspace_path=str(branch_run_dir.parents[2] / ".worktrees" / f"{node_id}-attempt-001"),
    )


def _config(tmp_path: Path) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "branch-resume-test", "mode": "docs-first"},
            "research": {"topic": "branch resume"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _done_stage15(decision: str = "proceed") -> StageResult:
    return StageResult(
        stage=Stage.RESEARCH_DECISION,
        status=StageStatus.DONE,
        artifacts=("decision.md",),
        decision=decision,
    )


@pytest.mark.parametrize(
    ("last_completed", "expected_resume"),
    [
        (Stage.EXPERIMENT_TASK_SPEC, Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR),
        (Stage.RESULT_ANALYSIS, Stage.RESEARCH_DECISION),
    ],
)
def test_completed_branch_stage_resumes_at_next_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    last_completed: Stage,
    expected_resume: Stage,
) -> None:
    branch_run_dir = _branch_dir(tmp_path)
    attempt = _attempt(branch_run_dir)
    write_branch_stage_done(
        branch_run_dir,
        last_completed,
        attempt_id=attempt.attempt_id,
        node_id=attempt.node_id,
        workspace_path=attempt.workspace_path or branch_run_dir,
    )
    recorded: dict[str, Any] = {}

    def fake_execute_pipeline(**kwargs: Any) -> list[Any]:
        recorded.update(kwargs)
        return [_done_stage15()]

    monkeypatch.setattr(hypothesis_branch, "execute_pipeline", fake_execute_pipeline)

    validate_branch(
        branch_run_dir=branch_run_dir,
        node=_node(),
        attempt=attempt,
        config=object(),
        adapters=object(),
    )

    assert recorded["from_stage"] is expected_resume


def test_killed_during_stage10_reruns_stage10_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch_run_dir = _branch_dir(tmp_path)
    attempt = _attempt(branch_run_dir)
    starts: list[Stage] = []

    def fake_execute_pipeline(**kwargs: Any) -> list[Any]:
        starts.append(kwargs["from_stage"])
        if len(starts) == 1:
            kwargs["on_stage_complete"](Stage.EXPERIMENT_TASK_SPEC)
            raise RuntimeError("container killed during stage 10")
        return [_done_stage15()]

    monkeypatch.setattr(hypothesis_branch, "execute_pipeline", fake_execute_pipeline)

    with pytest.raises(RuntimeError, match="container killed"):
        validate_branch(
            branch_run_dir=branch_run_dir,
            node=_node(),
            attempt=attempt,
            config=object(),
            adapters=object(),
        )

    state = read_branch_state(branch_run_dir)
    assert state is not None
    assert state["last_completed_stage"] == int(Stage.EXPERIMENT_TASK_SPEC)

    validate_branch(
        branch_run_dir=branch_run_dir,
        node=_node(),
        attempt=attempt,
        config=object(),
        adapters=object(),
    )

    assert starts == [
        Stage.EXPERIMENT_TASK_SPEC,
        Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
    ]


def test_branch_stage13_abort_keeps_branch_state_at_stage12(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch_run_dir = _branch_dir(tmp_path)
    attempt = _attempt(branch_run_dir)
    write_branch_stage_done(
        branch_run_dir,
        Stage.HARNESS_SUBMIT_AND_COLLECT,
        attempt_id=attempt.attempt_id,
        node_id=attempt.node_id,
        workspace_path=attempt.workspace_path or branch_run_dir,
    )

    def fake_execute_stage(stage: Stage, **kwargs: Any) -> StageResult:
        _ = kwargs
        stage_dir = branch_run_dir / f"stage-{int(stage):02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        if stage is Stage.EXPERIMENT_ROUTE_DECISION:
            return StageResult(
                stage=stage,
                status=StageStatus.DONE,
                artifacts=("experiment_decision.json",),
                decision="abort",
            )
        return StageResult(stage=stage, status=StageStatus.DONE, artifacts=("out.md",))

    monkeypatch.setattr(rc_runner, "execute_stage", fake_execute_stage)

    validate_branch(
        branch_run_dir=branch_run_dir,
        node=_node(),
        attempt=attempt,
        config=_config(tmp_path),
        adapters=AdapterBundle(),
    )

    state = read_branch_state(branch_run_dir)
    assert state is not None
    assert state["last_completed_stage"] == int(Stage.HARNESS_SUBMIT_AND_COLLECT)


def test_stage15_done_reconstructs_attempt_result_without_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch_run_dir = _branch_dir(tmp_path)
    attempt = _attempt(branch_run_dir)
    stage15 = branch_run_dir / "stage-15"
    stage15.mkdir(parents=True)
    (stage15 / "verdict.json").write_text(
        json.dumps(
            {
                "decision": "proceed",
                "confidence": 0.82,
                "evidence_summary": "Already decided.",
                "key_metrics": {"score": 0.8},
            }
        ),
        encoding="utf-8",
    )
    write_branch_stage_done(
        branch_run_dir,
        Stage.RESEARCH_DECISION,
        attempt_id=attempt.attempt_id,
        node_id=attempt.node_id,
        workspace_path=attempt.workspace_path or branch_run_dir,
    )

    def fail_execute_pipeline(**kwargs: Any) -> list[Any]:
        _ = kwargs
        raise AssertionError("execute_pipeline must not rerun completed stage 15")

    monkeypatch.setattr(hypothesis_branch, "execute_pipeline", fail_execute_pipeline)

    result = validate_branch(
        branch_run_dir=branch_run_dir,
        node=_node(),
        attempt=attempt,
        config=object(),
        adapters=object(),
    )

    assert result["status"] == "succeeded"
    assert result["decision"] == "proceed"
    assert result["artifacts"] == ["verdict.json"]
    assert result["metrics"] == {"score": 0.8}
    assert (branch_run_dir / "attempt_result.json").is_file()


def test_mixed_queue_skips_completed_resumes_interrupted_and_runs_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    coordinator = HypothesisValidationCoordinator(run_dir)
    coordinator.split_and_queue(
        """
## H1
Statement: Completed hypothesis.

## H2
Statement: Interrupted hypothesis.

## H3
Statement: Pending hypothesis.
""",
        created_at="2026-06-15T00:00:00+00:00",
    )

    completed_dir = run_dir / "hypothesis_branches" / "h-001" / "attempt-001"
    (completed_dir / "attempt_result.json").write_text(
        json.dumps(
            {
                "attempt_id": "h-001/attempt-001",
                "node_id": "h-001",
                "status": "succeeded",
                "decision": "proceed",
                "artifacts": ["stage-15/verdict.json"],
                "metrics": {"score": 0.9},
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    interrupted_dir = run_dir / "hypothesis_branches" / "h-002" / "attempt-001"
    interrupted_workspace = run_dir / ".worktrees" / "h-002-attempt-001"
    for stage in (
        Stage.EXPERIMENT_TASK_SPEC,
        Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
        Stage.MANIFEST_VALIDATE_AND_PREPARE,
        Stage.HARNESS_SUBMIT_AND_COLLECT,
    ):
        write_branch_stage_done(
            interrupted_dir,
            stage,
            attempt_id="h-002/attempt-001",
            node_id="h-002",
            workspace_path=interrupted_workspace,
        )

    seen: list[tuple[str, dict[int, str], str | None]] = []

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> dict[str, Any]:
        _ = config, adapters
        seen.append((node.id, dict(attempt.stage_status), attempt.workspace_path))
        return {
            "attempt_id": attempt.attempt_id,
            "node_id": node.id,
            "status": "succeeded",
            "decision": "proceed",
            "artifacts": ["stage-15/verdict.json"],
            "metrics": {"score": 0.7},
            "error": None,
        }

    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    completed = coordinator.run_pending_work_concurrent(
        config=object(),
        adapters=object(),
        max_concurrent=1,
        created_at="2026-06-15T00:01:00+00:00",
    )

    assert [attempt.attempt_id for attempt in completed] == [
        "h-001/attempt-001",
        "h-002/attempt-001",
        "h-003/attempt-001",
    ]
    assert seen == [
        (
            "h-002",
            {
                int(Stage.EXPERIMENT_TASK_SPEC): "done",
                int(Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR): "done",
                int(Stage.MANIFEST_VALIDATE_AND_PREPARE): "done",
                int(Stage.HARNESS_SUBMIT_AND_COLLECT): "done",
            },
            str(interrupted_workspace),
        ),
        ("h-003", {}, None),
    ]
