# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, cast

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline import executor as rc_executor
from researchclaw.pipeline import runner as rc_runner
from researchclaw.pipeline.executor import StageResult
from researchclaw.pipeline.stages import STAGE_SEQUENCE, Stage, StageStatus


@pytest.fixture()
def rc_config(tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "rc-runner-test", "mode": "docs-first"},
        "research": {"topic": "pipeline testing"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline",
        },
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


@pytest.fixture()
def adapters() -> AdapterBundle:
    return AdapterBundle()


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path


def _with_hypothesis_validation(config: RCConfig, *, enabled: bool) -> RCConfig:
    return replace(
        config,
        hypothesis_validation=replace(config.hypothesis_validation, enabled=enabled),
    )


def _done(stage: Stage, artifacts: tuple[str, ...] = ("out.md",)) -> StageResult:
    return StageResult(stage=stage, status=StageStatus.DONE, artifacts=artifacts)


def _failed(stage: Stage, msg: str = "boom") -> StageResult:
    return StageResult(stage=stage, status=StageStatus.FAILED, artifacts=(), error=msg)


def _paused(stage: Stage, msg: str = "resume needed") -> StageResult:
    return StageResult(
        stage=stage,
        status=StageStatus.PAUSED,
        artifacts=("refinement_log.json",),
        error=msg,
        decision="resume",
    )


def _blocked(stage: Stage) -> StageResult:
    return StageResult(
        stage=stage,
        status=StageStatus.BLOCKED_APPROVAL,
        artifacts=("gate.md",),
        decision="block",
    )


def _rejected(stage: Stage) -> StageResult:
    return StageResult(
        stage=stage,
        status=StageStatus.REJECTED,
        artifacts=("review.md",),
        decision="reject",
        error="reviewer rejected",
    )


def _with_hitl_required_stages(
    config: RCConfig, stages: tuple[int, ...]
) -> RCConfig:
    return replace(
        config,
        security=replace(config.security, hitl_required_stages=stages),
    )


def test_is_forward_progress_predicate() -> None:
    from researchclaw.pipeline.runner import is_forward_progress

    def result(stage: Stage, status: StageStatus, decision: str) -> StageResult:
        return StageResult(
            stage=stage,
            status=status,
            artifacts=(),
            decision=decision,
        )

    assert not is_forward_progress(
        result(Stage.HARNESS_SUBMIT_AND_COLLECT, StageStatus.FAILED, "retry")
    )
    for route in ("fix_code", "rerun", "hitl", "abort"):
        assert not is_forward_progress(
            result(Stage.EXPERIMENT_ROUTE_DECISION, StageStatus.DONE, route)
        )
    for route in ("continue", "proceed"):
        assert is_forward_progress(
            result(Stage.EXPERIMENT_ROUTE_DECISION, StageStatus.DONE, route)
        )
    assert not is_forward_progress(
        result(
            Stage.MANIFEST_VALIDATE_AND_PREPARE,
            StageStatus.DONE,
            "fix_code",
        )
    )
    assert is_forward_progress(
        result(
            Stage.MANIFEST_VALIDATE_AND_PREPARE,
            StageStatus.DONE,
            "proceed",
        )
    )
    assert is_forward_progress(
        result(Stage.RESULT_ANALYSIS, StageStatus.DONE, "proceed")
    )


def test_run_experiment_diagnosis_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    _ = adapters
    called: list[int] = []
    monkeypatch.setattr(
        "researchclaw.experiment.workspace_agent.create_workspace_agent",
        lambda *args, **kwargs: called.append(1),
    )
    s14 = run_dir / "stage-14"
    s14.mkdir()
    (s14 / "experiment_summary.json").write_text(
        json.dumps(
            {
                "condition_summaries": {"A": {"metrics": {"accuracy": 0.1}}},
                "best_run": {"metrics": {}},
            }
        ),
        encoding="utf-8",
    )
    s12 = run_dir / "stage-12"
    s12.mkdir()
    (s12 / "execution_record.json").write_text(
        json.dumps({"stdout": "", "stderr": "", "final_status": "completed"}),
        encoding="utf-8",
    )

    rc_runner._run_experiment_diagnosis(run_dir, rc_config, "rid")

    assert called == []
    assert (run_dir / "experiment_diagnosis.json").is_file()
    assert not (run_dir / "repair_prompt.txt").exists()


def test_execute_pipeline_runs_stages_in_sequence(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-seq",
        config=rc_config,
        adapters=adapters,
    )
    assert seen == list(STAGE_SEQUENCE)
    assert len(results) == 23
    assert all(r.status == StageStatus.DONE for r in results)


def test_execute_pipeline_stops_on_failed_stage(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    fail_stage = Stage.SEARCH_STRATEGY

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == fail_stage:
            return _failed(stage, "forced failure")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-fail",
        config=rc_config,
        adapters=adapters,
    )
    assert results[-1].stage == fail_stage
    assert results[-1].status == StageStatus.FAILED
    assert len(results) == int(fail_stage)


def test_terminal_failure_notifies_on_critical_stage(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    fail_stage = Stage.SEARCH_STRATEGY
    calls: list[dict[str, Any]] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == fail_stage:
            return _failed(stage, "forced failure")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(
        rc_runner,
        "notify_terminal_failure",
        lambda **kwargs: calls.append(kwargs),
    )

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-terminal-fail",
        config=rc_config,
        adapters=adapters,
    )

    assert len(calls) == 1
    assert calls[0]["config"] is rc_config
    assert calls[0]["run_id"] == "run-terminal-fail"
    assert calls[0]["stage_name"] == fail_stage.name
    assert calls[0]["stage_num"] == int(fail_stage)
    assert calls[0]["error"] == "forced failure"
    assert calls[0]["run_dir"] == run_dir


def test_noncritical_skip_does_not_notify(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    fail_stage = Stage.KNOWLEDGE_ARCHIVE
    calls: list[dict[str, Any]] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == fail_stage:
            return _failed(stage, "archive failed")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(
        rc_runner,
        "notify_terminal_failure",
        lambda **kwargs: calls.append(kwargs),
    )

    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-noncritical-skip",
        config=rc_config,
        adapters=adapters,
        skip_noncritical=True,
    )

    assert calls == []
    assert results[-1].stage == Stage.CITATION_VERIFY


def test_execute_pipeline_stops_on_paused_stage(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    pause_stage = Stage.HARNESS_SUBMIT_AND_COLLECT

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == pause_stage:
            return _paused(stage, "ACP prompt timed out after 1800s")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-paused",
        config=rc_config,
        adapters=adapters,
    )
    assert results[-1].stage == pause_stage
    assert results[-1].status == StageStatus.PAUSED
    assert len(results) == int(pause_stage)
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["last_completed_stage"] == int(Stage.MANIFEST_VALIDATE_AND_PREPARE)
    summary = json.loads((run_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["stages_paused"] == 1
    assert summary["final_status"] == "paused"


def test_paused_stage_does_not_notify(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    pause_stage = Stage.SEARCH_STRATEGY
    calls: list[dict[str, Any]] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == pause_stage:
            return _paused(stage, "waiting for operator")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(
        rc_runner,
        "notify_terminal_failure",
        lambda **kwargs: calls.append(kwargs),
    )

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-paused-no-notify",
        config=rc_config,
        adapters=adapters,
    )

    assert calls == []


def test_rejected_stage_does_not_notify(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rejected_stage = Stage.LITERATURE_SCREEN
    calls: list[dict[str, Any]] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == rejected_stage:
            return _rejected(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(
        rc_runner,
        "notify_terminal_failure",
        lambda **kwargs: calls.append(kwargs),
    )

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-rejected-no-notify",
        config=rc_config,
        adapters=adapters,
    )

    assert calls == []


def test_execute_pipeline_stops_on_gate_when_stop_on_gate_enabled(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    gate_stage = Stage.LITERATURE_SCREEN

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == gate_stage:
            return _blocked(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-gate-stop",
        config=rc_config,
        adapters=adapters,
        stop_on_gate=True,
    )
    assert results[-1].stage == gate_stage
    assert results[-1].status == StageStatus.BLOCKED_APPROVAL
    assert len(results) == int(gate_stage)


def test_execute_pipeline_continues_after_gate_when_stop_on_gate_disabled(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    gate_stage = Stage.LITERATURE_SCREEN

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == gate_stage:
            return _blocked(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-gate-continue",
        config=rc_config,
        adapters=adapters,
        stop_on_gate=False,
    )
    assert len(results) == 23
    assert any(item.status == StageStatus.BLOCKED_APPROVAL for item in results)


def test_execute_pipeline_writes_pipeline_summary_json(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-summary",
        config=rc_config,
        adapters=adapters,
    )
    summary_path = run_dir / "pipeline_summary.json"
    assert summary_path.exists()


def test_pipeline_summary_has_expected_fields_and_values(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == Stage.LITERATURE_SCREEN:
            return _blocked(stage)
        if stage == Stage.HYPOTHESIS_GEN:
            return _failed(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-summary-fields",
        config=rc_config,
        adapters=adapters,
    )
    summary = cast(
        dict[str, Any],
        json.loads((run_dir / "pipeline_summary.json").read_text(encoding="utf-8")),
    )
    assert summary["run_id"] == "run-summary-fields"
    assert summary["stages_executed"] == len(results)
    assert summary["stages_done"] == sum(
        1 for r in results if r.status == StageStatus.DONE
    )
    assert summary["stages_paused"] == 0
    assert summary["stages_blocked"] == 1
    assert summary["stages_failed"] == 1
    assert summary["from_stage"] == 1
    assert summary["final_stage"] == int(Stage.HYPOTHESIS_GEN)
    assert summary["final_status"] == "failed"
    assert "generated" in summary


def test_execute_pipeline_from_stage_skips_earlier_stages(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-from-stage",
        config=rc_config,
        adapters=adapters,
        from_stage=Stage.PAPER_OUTLINE,
    )
    assert seen[0] == Stage.PAPER_OUTLINE
    assert len(seen) == len(STAGE_SEQUENCE) - (int(Stage.PAPER_OUTLINE) - 1)
    assert len(results) == len(seen)


def test_execute_pipeline_on_stage_complete_fires_for_done_normal_stages_in_order(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    completed: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        if stage == Stage.SEARCH_STRATEGY:
            return _failed(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-stage-complete-callback-normal",
        config=rc_config,
        adapters=adapters,
        to_stage=Stage.SEARCH_STRATEGY,
        on_stage_complete=completed.append,
    )

    assert completed == [Stage.TOPIC_INIT, Stage.PROBLEM_DECOMPOSE]
    assert results[-1].stage == Stage.SEARCH_STRATEGY
    assert results[-1].status == StageStatus.FAILED


def test_execute_pipeline_on_stage_complete_fires_for_experiment_loop_stages(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    completed: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, "continue")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-stage-complete-callback-loop",
        config=rc_config,
        adapters=adapters,
        from_stage=Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
        to_stage=Stage.RESEARCH_DECISION,
        on_stage_complete=completed.append,
    )

    assert completed == [
        Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
        Stage.MANIFEST_VALIDATE_AND_PREPARE,
        Stage.HARNESS_SUBMIT_AND_COLLECT,
        Stage.EXPERIMENT_ROUTE_DECISION,
        Stage.RESULT_ANALYSIS,
        Stage.RESEARCH_DECISION,
    ]


def test_execute_pipeline_on_stage_complete_error_does_not_abort_run(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        return _done(stage)

    def failing_callback(stage: Stage) -> None:
        _ = stage
        raise RuntimeError("callback failed")

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-stage-complete-callback-fail",
        config=rc_config,
        adapters=adapters,
        to_stage=Stage.PROBLEM_DECOMPOSE,
        on_stage_complete=failing_callback,
    )

    assert [result.stage for result in results] == [
        Stage.TOPIC_INIT,
        Stage.PROBLEM_DECOMPOSE,
    ]


def test_execute_pipeline_writes_kb_entries_when_kb_root_provided(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
    tmp_path: Path,
) -> None:
    calls: list[tuple[int, str, str]] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        stage_dir = run_dir / f"stage-{int(stage):02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "out.md").write_text(f"stage {int(stage)}", encoding="utf-8")
        return _done(stage)

    def mock_write_stage_to_kb(
        kb_root: Path,
        stage_id: int,
        stage_name: str,
        run_id: str,
        artifacts: list[str],
        stage_dir: Path,
        **kwargs,
    ):
        _ = kb_root, artifacts, stage_dir, kwargs
        calls.append((stage_id, stage_name, run_id))
        return []

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(rc_runner, "write_stage_to_kb", mock_write_stage_to_kb)

    kb_root = tmp_path / "kb-out"
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-kb",
        config=rc_config,
        adapters=adapters,
        kb_root=kb_root,
    )
    assert len(results) == 23
    assert len(calls) == 23
    assert calls[0] == (1, "topic_init", "run-kb")


def test_execute_pipeline_passes_auto_approve_flag_to_execute_stage(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    received: list[bool] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        received.append(kwargs["auto_approve_gates"])
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-auto-approve",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )
    assert received
    assert all(received)


@pytest.mark.parametrize(
    ("stage", "started", "expected"),
    [
        (Stage.TOPIC_INIT, False, True),
        (Stage.PROBLEM_DECOMPOSE, False, False),
        (Stage.PAPER_DRAFT, True, True),
    ],
)
def test_should_start_logic(stage: Stage, started: bool, expected: bool) -> None:
    assert rc_runner._should_start(stage, Stage.TOPIC_INIT, started) is expected


@pytest.mark.parametrize(
    ("results", "expected_status", "expected_final_stage"),
    [
        ([], "no_stages", int(Stage.TOPIC_INIT)),
        ([_done(Stage.TOPIC_INIT)], "done", int(Stage.TOPIC_INIT)),
        (
            [_done(Stage.TOPIC_INIT), _paused(Stage.PROBLEM_DECOMPOSE)],
            "paused",
            int(Stage.PROBLEM_DECOMPOSE),
        ),
        (
            [_done(Stage.TOPIC_INIT), _failed(Stage.PROBLEM_DECOMPOSE)],
            "failed",
            int(Stage.PROBLEM_DECOMPOSE),
        ),
    ],
)
def test_build_pipeline_summary_core_fields(
    results, expected_status: str, expected_final_stage: int
) -> None:
    summary = rc_runner._build_pipeline_summary(
        run_id="run-core",
        results=results,
        from_stage=Stage.TOPIC_INIT,
    )
    assert summary["run_id"] == "run-core"
    assert summary["final_status"] == expected_status
    assert summary["final_stage"] == expected_final_stage


def test_pipeline_prints_stage_progress(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_results = [
        StageResult(
            stage=Stage.TOPIC_INIT, status=StageStatus.DONE, artifacts=("topic.json",)
        ),
        StageResult(
            stage=Stage.PROBLEM_DECOMPOSE,
            status=StageStatus.DONE,
            artifacts=("tree.json",),
        ),
        StageResult(
            stage=Stage.SEARCH_STRATEGY,
            status=StageStatus.FAILED,
            artifacts=(),
            error="LLM timeout",
        ),
    ]

    call_idx = 0

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = stage, kwargs
        nonlocal call_idx
        idx = call_idx
        call_idx += 1
        return mock_results[min(idx, len(mock_results) - 1)]

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(rc_runner, "write_stage_to_kb", lambda *args, **kwargs: [])

    _ = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="rc-test-001",
        config=rc_config,
        adapters=adapters,
    )

    captured = capsys.readouterr()
    assert "TOPIC_INIT — running..." in captured.out
    assert "TOPIC_INIT — done" in captured.out
    assert "SEARCH_STRATEGY — FAILED" in captured.out
    assert "LLM timeout" in captured.out


def test_pipeline_prints_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_result = StageResult(
        stage=Stage.TOPIC_INIT,
        status=StageStatus.DONE,
        artifacts=("topic.json",),
    )
    mock_fail = StageResult(
        stage=Stage.PROBLEM_DECOMPOSE,
        status=StageStatus.FAILED,
        artifacts=(),
        error="test",
    )
    results_iter = iter([mock_result, mock_fail])

    monkeypatch.setattr(
        rc_runner, "execute_stage", lambda *args, **kwargs: next(results_iter)
    )
    monkeypatch.setattr(rc_runner, "write_stage_to_kb", lambda *args, **kwargs: [])

    _ = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="rc-test-002",
        config=rc_config,
        adapters=adapters,
    )

    captured = capsys.readouterr()
    import re

    assert re.search(r"\d+\.\d+s\)", captured.out), (
        f"No elapsed time found in: {captured.out}"
    )


# ── PIVOT/PROCEED/EXTEND decision loop tests ──


def _pivot_result(stage: Stage) -> StageResult:
    return StageResult(
        stage=stage, status=StageStatus.DONE, artifacts=("decision.md",), decision="pivot"
    )


def _route_result(stage: Stage, route: str) -> StageResult:
    return StageResult(
        stage=stage,
        status=StageStatus.DONE,
        artifacts=("experiment_decision.json",),
        decision=route,
    )


def _extend_result(stage: Stage) -> StageResult:
    return StageResult(
        stage=stage, status=StageStatus.DONE, artifacts=("decision.md",), decision="extend"
    )


def _touch_stage_dir(run_dir: Path, stage: Stage) -> None:
    stage_dir = run_dir / f"stage-{int(stage):02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "marker.txt").write_text(stage.name, encoding="utf-8")


def test_pivot_decision_triggers_rollback_to_hypothesis_gen(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = _with_hypothesis_validation(rc_config, enabled=False)
    seen: list[Stage] = []
    pivot_count = 0

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        nonlocal pivot_count
        if stage == Stage.RESEARCH_DECISION and pivot_count == 0:
            pivot_count += 1
            return _pivot_result(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-pivot",
        config=rc_config,
        adapters=adapters,
    )
    # Should have seen HYPOTHESIS_GEN at least twice (original + rollback)
    hyp_gen_count = sum(1 for s in seen if s == Stage.HYPOTHESIS_GEN)
    assert hyp_gen_count >= 2
    # Decision history should be recorded
    history_path = run_dir / "decision_history.json"
    assert history_path.exists()
    history = json.loads(history_path.read_text())
    assert len(history) == 1
    assert history[0]["decision"] == "pivot"


def test_pivot_recursion_forwards_to_stage_and_callback(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = _with_hypothesis_validation(rc_config, enabled=False)
    seen: list[Stage] = []
    notified: list[Stage] = []
    pivot_count = 0

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        _touch_stage_dir(run_dir, stage)
        nonlocal pivot_count
        if stage == Stage.RESEARCH_DECISION and pivot_count == 0:
            pivot_count += 1
            return _pivot_result(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-pivot-forward-kwargs",
        config=rc_config,
        adapters=adapters,
        from_stage=Stage.HYPOTHESIS_GEN,
        to_stage=Stage.PAPER_OUTLINE,
        on_stage_complete=notified.append,
    )

    assert max(int(stage) for stage in seen) == int(Stage.PAPER_OUTLINE)
    assert seen.count(Stage.HYPOTHESIS_GEN) == 2
    assert notified.count(Stage.HYPOTHESIS_GEN) == 2


def test_default_hypothesis_validation_skips_legacy_decision_recursion(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        if stage == Stage.RESEARCH_DECISION:
            return _pivot_result(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-default-hypothesis-validation",
        config=rc_config,
        adapters=adapters,
    )

    assert rc_config.hypothesis_validation.enabled is True
    assert seen.count(Stage.HYPOTHESIS_GEN) == 1
    assert seen.count(Stage.RESEARCH_DECISION) == 1
    assert not (run_dir / "decision_history.json").exists()


def test_default_hypothesis_validation_skips_cycle_archive_wiring(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    from researchclaw.pipeline import hypothesis_cycle_archive as cycle_archive

    calls: list[str | None] = []

    def archive(run_dir: Path, *, decision: str | None = None) -> Path | None:
        _ = run_dir
        calls.append(decision)
        return None

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        return StageResult(
            stage=stage,
            status=StageStatus.DONE,
            artifacts=("decision.md",),
            decision="proceed",
        )

    monkeypatch.setattr(cycle_archive, "archive_current_hypothesis_cycle", archive)
    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-default-no-cycle-archive",
        config=rc_config,
        adapters=adapters,
        from_stage=Stage.RESEARCH_DECISION,
        to_stage=Stage.RESEARCH_DECISION,
    )

    assert rc_config.hypothesis_validation.enabled is True
    assert calls == []


def test_per_hypothesis_mode_bypasses_single_line_stage9_to_15(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    from researchclaw.pipeline.hypothesis_coordinator import (
        HypothesisValidationCoordinator,
    )

    seen: list[Stage] = []
    coordinator_calls: list[Path] = []

    def mock_execute_stage(stage: Stage, **kwargs: Any) -> StageResult:
        stage_dir = run_dir / f"stage-{int(stage):02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        seen.append(stage)
        if stage is Stage.HYPOTHESIS_GEN:
            (stage_dir / "hypotheses.md").write_text(
                """
## H1
Statement: First branch hypothesis.
Prediction: First prediction.
Falsification: First falsification.

## H2
Statement: Second branch hypothesis.
Prediction: Second prediction.
Falsification: Second falsification.
""",
                encoding="utf-8",
            )
            return _done(stage, artifacts=("hypotheses.md",))
        return _done(stage)

    def fake_run_until_queue_empty(self: Any, **kwargs: Any) -> list[Any]:
        _ = kwargs
        coordinator_calls.append(self.run_dir)
        (self.run_dir / "hypothesis_aggregate.json").write_text(
            json.dumps({"validation_summary": [{}, {}]}),
            encoding="utf-8",
        )
        return []

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(
        HypothesisValidationCoordinator,
        "run_until_queue_empty",
        fake_run_until_queue_empty,
    )

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-per-hypothesis",
        config=rc_config,
        adapters=adapters,
    )

    assert coordinator_calls == [run_dir]
    assert not any(
        Stage.EXPERIMENT_TASK_SPEC <= stage <= Stage.RESEARCH_DECISION
        for stage in seen
    )
    assert Stage.PAPER_OUTLINE in seen
    assert (run_dir / "hypothesis_aggregate.json").is_file()
    assert not (run_dir / "hypothesis_tree" / "current_node.txt").exists()


def test_resume_stage16_drains_per_hypothesis_queue_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    from researchclaw.pipeline.hypothesis_coordinator import (
        HypothesisValidationCoordinator,
    )

    (run_dir / "hypothesis_tree").mkdir()
    seen: list[str] = []

    def fake_run_until_queue_empty(self: Any, **kwargs: Any) -> list[Any]:
        _ = self, kwargs
        seen.append("coordinator")
        (run_dir / "hypothesis_aggregate.json").write_text(
            json.dumps({"validation_summary": []}),
            encoding="utf-8",
        )
        return []

    def mock_execute_stage(stage: Stage, **kwargs: Any) -> StageResult:
        _ = kwargs
        seen.append(stage.name)
        return _done(stage)

    monkeypatch.setattr(
        HypothesisValidationCoordinator,
        "run_until_queue_empty",
        fake_run_until_queue_empty,
    )
    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-resume-per-hypothesis",
        config=rc_config,
        adapters=adapters,
        from_stage=Stage.PAPER_OUTLINE,
        to_stage=Stage.PAPER_OUTLINE,
    )

    assert seen[:2] == ["coordinator", "PAPER_OUTLINE"]


def test_experiment_loop_continue_runs_10_11_12_13_then_14(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, "continue")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-loop-continue",
        config=rc_config,
        adapters=adapters,
    )
    start = seen.index(Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR)
    assert seen[start : start + 5] == [
        Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
        Stage.MANIFEST_VALIDATE_AND_PREPARE,
        Stage.HARNESS_SUBMIT_AND_COLLECT,
        Stage.EXPERIMENT_ROUTE_DECISION,
        Stage.RESULT_ANALYSIS,
    ]


def test_stage11_to_stage_stops_after_stage11(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        _touch_stage_dir(run_dir, stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-stage11-only",
        config=rc_config,
        adapters=adapters,
        from_stage=Stage.MANIFEST_VALIDATE_AND_PREPARE,
        to_stage=Stage.MANIFEST_VALIDATE_AND_PREPARE,
    )

    assert seen == [Stage.MANIFEST_VALIDATE_AND_PREPARE]


def test_experiment_loop_stage11_invalid_rejumps_to_10(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []
    invalid_once = True

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        _touch_stage_dir(run_dir, stage)
        nonlocal invalid_once
        if stage == Stage.MANIFEST_VALIDATE_AND_PREPARE and invalid_once:
            invalid_once = False
            return StageResult(
                stage=stage,
                status=StageStatus.DONE,
                artifacts=("manifest_validation.json",),
                decision="fix_code",
                error="invalid manifest",
            )
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, "continue")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-stage11-fix",
        config=rc_config,
        adapters=adapters,
    )
    start = seen.index(Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR)
    assert seen[start : start + 6] == [
        Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
        Stage.MANIFEST_VALIDATE_AND_PREPARE,
        Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR,
        Stage.MANIFEST_VALIDATE_AND_PREPARE,
        Stage.HARNESS_SUBMIT_AND_COLLECT,
        Stage.EXPERIMENT_ROUTE_DECISION,
    ]
    assert (run_dir / "stage-10_v1").is_dir()


def test_experiment_loop_recoverable_failure_does_not_notify(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    calls: list[dict[str, Any]] = []
    invalid_once = True

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        _touch_stage_dir(run_dir, stage)
        nonlocal invalid_once
        if stage == Stage.MANIFEST_VALIDATE_AND_PREPARE and invalid_once:
            invalid_once = False
            return StageResult(
                stage=stage,
                status=StageStatus.DONE,
                artifacts=("manifest_validation.json",),
                decision="fix_code",
                error="invalid manifest",
            )
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, "continue")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(
        rc_runner,
        "notify_terminal_failure",
        lambda **kwargs: calls.append(kwargs),
    )

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-stage11-fix-no-notify",
        config=rc_config,
        adapters=adapters,
    )

    assert calls == []


def test_experiment_loop_route_fix_code_rejumps_to_10(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []
    routes = iter(["fix_code", "continue"])

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, next(routes))
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-fix-code",
        config=rc_config,
        adapters=adapters,
    )
    assert seen.count(Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR) == 2
    assert (run_dir / "stage-10_v1").is_dir()


def test_stage13_fix_code_does_not_checkpoint_stage13(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    route_calls = 0

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        _touch_stage_dir(run_dir, stage)
        nonlocal route_calls
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            route_calls += 1
            route = "fix_code" if route_calls == 1 else "continue"
            return _route_result(stage, route)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-stage13-fix-code-checkpoint",
        config=rc_config,
        adapters=adapters,
    )

    checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
    assert checkpoint["last_completed_stage"] >= int(Stage.RESULT_ANALYSIS)
    assert (run_dir / "stage-10_v1").is_dir()


def test_stage13_abort_does_not_advance_checkpoint_or_callback(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    completed: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, "abort")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-stage13-abort-checkpoint",
        config=rc_config,
        adapters=adapters,
        from_stage=Stage.HARNESS_SUBMIT_AND_COLLECT,
        on_stage_complete=completed.append,
    )

    checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
    assert checkpoint["last_completed_stage"] == int(Stage.HARNESS_SUBMIT_AND_COLLECT)
    assert completed == [Stage.HARNESS_SUBMIT_AND_COLLECT]


def test_experiment_loop_route_rerun_rejumps_to_12(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []
    routes = iter(["rerun", "continue"])

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, next(routes))
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-rerun",
        config=rc_config,
        adapters=adapters,
    )
    assert seen.count(Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR) == 1
    assert seen.count(Stage.HARNESS_SUBMIT_AND_COLLECT) == 2
    assert (run_dir / "stage-12_v1").is_dir()


def test_experiment_loop_unknown_route_stops_without_stage9_reentry(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []
    routes = iter(["unknown_legacy_route", "continue"])

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, next(routes))
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-unknown-route",
        config=rc_config,
        adapters=adapters,
    )
    assert seen.count(Stage.EXPERIMENT_TASK_SPEC) == 1
    assert seen.count(Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR) == 1
    assert seen.count(Stage.EXPERIMENT_ROUTE_DECISION) == 1


def test_experiment_budget_exhausted_pauses_for_hitl(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    from researchclaw.pipeline.stages import MAX_EXPERIMENT_ITERATIONS

    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, "fix_code")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-max-experiment",
        config=rc_config,
        adapters=adapters,
    )
    assert seen.count(Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR) == (
        MAX_EXPERIMENT_ITERATIONS + 1
    )
    assert Stage.RESULT_ANALYSIS not in seen
    assert (run_dir / "experiment_hitl_required.json").is_file()
    summary = json.loads((run_dir / "pipeline_summary.json").read_text())
    assert summary["final_status"] == "paused"


def test_experiment_budget_exhaustion_uses_configured_repair_cycles(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = replace(
        rc_config,
        experiment=replace(
            rc_config.experiment,
            repair=replace(rc_config.experiment.repair, max_cycles=1),
        ),
    )
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, "fix_code")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-configured-max-experiment",
        config=rc_config,
        adapters=adapters,
    )

    assert seen.count(Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR) == 2
    history = json.loads((run_dir / "experiment_loop_history.json").read_text())
    assert len(history["iterations"]) == 1
    assert Stage.RESULT_ANALYSIS not in seen
    assert (run_dir / "experiment_hitl_required.json").is_file()
    summary = json.loads((run_dir / "pipeline_summary.json").read_text())
    assert summary["final_status"] == "paused"


def test_experiment_loop_abort_stops(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, "abort")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-abort",
        config=rc_config,
        adapters=adapters,
    )
    assert Stage.RESULT_ANALYSIS not in seen


def test_experiment_loop_abort_does_not_notify(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    calls: list[dict[str, Any]] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, "abort")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(
        rc_runner,
        "notify_terminal_failure",
        lambda **kwargs: calls.append(kwargs),
    )

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-abort-no-notify",
        config=rc_config,
        adapters=adapters,
    )

    assert calls == []


def test_experiment_loop_terminal_stop_notifies(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    calls: list[dict[str, Any]] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.HARNESS_SUBMIT_AND_COLLECT:
            return _failed(stage, "harness crashed")
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    monkeypatch.setattr(
        rc_runner,
        "notify_terminal_failure",
        lambda **kwargs: calls.append(kwargs),
    )

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-loop-terminal-fail",
        config=rc_config,
        adapters=adapters,
    )

    assert len(calls) == 1
    assert calls[0]["run_id"] == "run-loop-terminal-fail"
    assert calls[0]["stage_name"] == Stage.HARNESS_SUBMIT_AND_COLLECT.name
    assert calls[0]["stage_num"] == int(Stage.HARNESS_SUBMIT_AND_COLLECT)
    assert calls[0]["error"] == "harness crashed"
    assert calls[0]["run_dir"] == run_dir


def test_experiment_loop_history_persisted(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    routes = iter(["fix_code", "continue"])

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        _touch_stage_dir(run_dir, stage)
        if stage == Stage.EXPERIMENT_ROUTE_DECISION:
            return _route_result(stage, next(routes))
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-history",
        config=rc_config,
        adapters=adapters,
    )
    history = json.loads((run_dir / "experiment_loop_history.json").read_text())
    assert len(history["iterations"]) == 1
    assert history["iterations"][0]["route"] == "fix_code"


def test_extend_decision_triggers_rollback_to_hypothesis_gen(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = _with_hypothesis_validation(rc_config, enabled=False)
    seen: list[Stage] = []
    extend_count = 0

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        nonlocal extend_count
        if stage == Stage.RESEARCH_DECISION and extend_count == 0:
            extend_count += 1
            return _extend_result(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-extend",
        config=rc_config,
        adapters=adapters,
    )

    hyp_gen_count = sum(1 for s in seen if s == Stage.HYPOTHESIS_GEN)
    assert hyp_gen_count >= 2


def test_extend_writes_extension_context_file(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = _with_hypothesis_validation(rc_config, enabled=False)
    extend_count = 0

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        stage_dir = run_dir / f"stage-{int(stage):02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        nonlocal extend_count
        if stage == Stage.HYPOTHESIS_GEN and extend_count == 0:
            (stage_dir / "hypotheses.md").write_text(
                "# Hypotheses\nH1: current hypothesis.",
                encoding="utf-8",
            )
        if stage == Stage.RESULT_ANALYSIS and extend_count == 0:
            (stage_dir / "analysis.md").write_text(
                "# Analysis\nA follow-up mechanism emerged.",
                encoding="utf-8",
            )
            (stage_dir / "experiment_summary.json").write_text(
                json.dumps({"metrics_summary": {"accuracy": 0.8}}),
                encoding="utf-8",
            )
        if stage == Stage.RESEARCH_DECISION and extend_count == 0:
            extend_count += 1
            (stage_dir / "decision.md").write_text(
                "## Decision\nEXTEND\n## Next Actions\nProbe the follow-up mechanism.",
                encoding="utf-8",
            )
            return _extend_result(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-extend-context",
        config=rc_config,
        adapters=adapters,
    )

    context_path = run_dir / "hypothesis_extension_context.md"
    assert context_path.exists()
    context = context_path.read_text(encoding="utf-8")
    assert "current hypothesis" in context
    assert "follow-up mechanism" in context
    assert "accuracy" in context


def test_pivot_cleans_old_extension_context(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = _with_hypothesis_validation(rc_config, enabled=False)
    (run_dir / "hypothesis_extension_context.md").write_text(
        "stale extension context",
        encoding="utf-8",
    )
    pivot_count = 0

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        nonlocal pivot_count
        if stage == Stage.RESEARCH_DECISION and pivot_count == 0:
            pivot_count += 1
            return _pivot_result(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-pivot-clean-context",
        config=rc_config,
        adapters=adapters,
    )

    assert not (run_dir / "hypothesis_extension_context.md").exists()


def test_max_pivot_count_prevents_infinite_loop(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = _with_hypothesis_validation(rc_config, enabled=False)
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        # Always PIVOT — should be limited by MAX_DECISION_PIVOTS
        if stage == Stage.RESEARCH_DECISION:
            return _pivot_result(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-max-pivot",
        config=rc_config,
        adapters=adapters,
    )
    # RESEARCH_DECISION should appear at most MAX_DECISION_PIVOTS + 1 times
    from researchclaw.pipeline.stages import MAX_DECISION_PIVOTS
    decision_count = sum(1 for s in seen if s == Stage.RESEARCH_DECISION)
    assert decision_count <= MAX_DECISION_PIVOTS + 1


def test_proceed_decision_does_not_trigger_rollback(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-proceed",
        config=rc_config,
        adapters=adapters,
    )
    # Should be exactly 23 stages, no rollback
    assert len(seen) == 23
    assert not (run_dir / "decision_history.json").exists()


def test_read_pivot_count_returns_zero_for_no_history(run_dir: Path) -> None:
    assert rc_runner._read_pivot_count(run_dir) == 0


def test_record_decision_history_appends(run_dir: Path) -> None:
    rc_runner._record_decision_history(run_dir, "pivot", Stage.HYPOTHESIS_GEN, 1)
    rc_runner._record_decision_history(run_dir, "extend", Stage.HYPOTHESIS_GEN, 2)
    history = json.loads((run_dir / "decision_history.json").read_text())
    assert len(history) == 2
    assert history[0]["decision"] == "pivot"
    assert history[1]["decision"] == "extend"


def test_decision_history_records_extend(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = _with_hypothesis_validation(rc_config, enabled=False)
    extend_count = 0

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        nonlocal extend_count
        if stage == Stage.RESEARCH_DECISION and extend_count == 0:
            extend_count += 1
            return _extend_result(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-extend-history",
        config=rc_config,
        adapters=adapters,
    )

    history = json.loads((run_dir / "decision_history.json").read_text())
    assert history[0]["decision"] == "extend"
    assert history[0]["rollback_target"] == Stage.HYPOTHESIS_GEN.name


def _write_human_gate_artifacts(
    run_dir: Path, decision: str, rationale: str
) -> None:
    s8 = run_dir / "stage-08"
    s8.mkdir(parents=True, exist_ok=True)
    (s8 / "hypotheses.md").write_text(
        "# Hypotheses\nH1: original useful hypothesis.",
        encoding="utf-8",
    )
    s14 = run_dir / "stage-14"
    s14.mkdir(parents=True, exist_ok=True)
    (s14 / "analysis.md").write_text(
        "# Analysis\nThe evidence supports a human-gated decision.",
        encoding="utf-8",
    )
    (s14 / "experiment_summary.json").write_text(
        json.dumps({"metrics_summary": {"accuracy": {"mean": 0.84}}}),
        encoding="utf-8",
    )
    s15 = run_dir / "stage-15"
    s15.mkdir(parents=True, exist_ok=True)
    (s15 / "decision.md").write_text(
        f"## Decision\n{decision}\n## Justification\n{rationale}",
        encoding="utf-8",
    )
    (s15 / ".gate_proposal.json").write_text("{}", encoding="utf-8")


def _mock_pipeline_after_human_gate(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
) -> list[Stage]:
    seen: list[Stage] = []
    stage15_actual_calls = 0

    def mock_execute_stage(stage: Stage, **kwargs: object) -> StageResult:
        nonlocal stage15_actual_calls
        seen.append(stage)
        stage_dir = run_dir / f"stage-{int(stage):02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        if stage == Stage.RESEARCH_DECISION and stage15_actual_calls == 0:
            stage15_actual_calls += 1
            return rc_executor.execute_stage(stage, **kwargs)  # type: ignore[arg-type]
        if stage == Stage.RESEARCH_DECISION:
            (stage_dir / "decision.md").write_text(
                "## Decision\nPROCEED\n## Justification\nSecond pass proceeds.",
                encoding="utf-8",
            )
            return StageResult(
                stage=stage,
                status=StageStatus.DONE,
                artifacts=("decision.md",),
                decision="proceed",
            )
        if stage == Stage.HYPOTHESIS_GEN:
            (stage_dir / "hypotheses.md").write_text(
                "# Hypotheses\nH2: follow-up hypothesis.",
                encoding="utf-8",
            )
        if stage == Stage.RESULT_ANALYSIS:
            (stage_dir / "analysis.md").write_text("# Analysis\nUpdated.", encoding="utf-8")
            (stage_dir / "experiment_summary.json").write_text(
                json.dumps({"metrics_summary": {"accuracy": {"mean": 0.85}}}),
                encoding="utf-8",
            )
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    return seen


def test_checkpoint_after_gated_stage15_returns_stage15_on_resume(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    def mock_execute_stage(stage: Stage, **kwargs: object) -> StageResult:
        _ = kwargs
        if stage == Stage.RESEARCH_DECISION:
            return _blocked(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-stage15-gate-checkpoint",
        config=rc_config,
        adapters=adapters,
        stop_on_gate=True,
    )

    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["last_completed_stage"] == int(Stage.RESULT_ANALYSIS)
    assert rc_runner.read_checkpoint(run_dir) is Stage.RESEARCH_DECISION


def test_extend_after_human_gate_uses_edited_decision(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = _with_hypothesis_validation(rc_config, enabled=False)
    rc_config = _with_hitl_required_stages(rc_config, (5, 9, 15, 20))
    _write_human_gate_artifacts(
        run_dir,
        "EXTEND",
        "Human final extend rationale should drive routing.",
    )
    rc_runner._write_checkpoint(run_dir, Stage.RESULT_ANALYSIS, "run-human-extend")
    seen = _mock_pipeline_after_human_gate(monkeypatch, run_dir)

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-human-extend",
        config=rc_config,
        adapters=adapters,
        from_stage=Stage.RESEARCH_DECISION,
    )

    assert seen[0] is Stage.RESEARCH_DECISION
    assert seen.count(Stage.HYPOTHESIS_GEN) >= 1
    history = json.loads((run_dir / "decision_history.json").read_text())
    assert history[0]["decision"] == "extend"


def test_pivot_after_human_gate_uses_edited_decision(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = _with_hypothesis_validation(rc_config, enabled=False)
    rc_config = _with_hitl_required_stages(rc_config, (5, 9, 15, 20))
    _write_human_gate_artifacts(
        run_dir,
        "PIVOT",
        "Human final pivot rationale should drive routing.",
    )
    rc_runner._write_checkpoint(run_dir, Stage.RESULT_ANALYSIS, "run-human-pivot")
    seen = _mock_pipeline_after_human_gate(monkeypatch, run_dir)

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-human-pivot",
        config=rc_config,
        adapters=adapters,
        from_stage=Stage.RESEARCH_DECISION,
    )

    assert seen[0] is Stage.RESEARCH_DECISION
    assert seen.count(Stage.HYPOTHESIS_GEN) >= 1
    assert not (run_dir / "hypothesis_extension_context.md").exists()
    history = json.loads((run_dir / "decision_history.json").read_text())
    assert history[0]["decision"] == "pivot"


def test_full_auto_no_sentinel_created(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = _with_hitl_required_stages(rc_config, (5, 9, 15, 20))
    _write_human_gate_artifacts(run_dir, "PROCEED", "Existing sentinel should be absent.")
    (run_dir / "stage-15" / ".gate_proposal.json").unlink()

    def stage15_executor(
        stage_dir: Path,
        _run_dir: Path,
        _config: RCConfig,
        _adapters: AdapterBundle,
        *,
        llm: object = None,
        **_kwargs: object,
    ) -> StageResult:
        _ = llm
        (stage_dir / "decision.md").write_text(
            "## Decision\nPROCEED\n## Justification\nAuto approved.",
            encoding="utf-8",
        )
        (stage_dir / "decision_review.md").write_text(
            "# Decision Review\n\nDecision Reviewed: PROCEED\n",
            encoding="utf-8",
        )
        return StageResult(
            stage=Stage.RESEARCH_DECISION,
            status=StageStatus.DONE,
            artifacts=("decision.md", "decision_review.md"),
            decision="proceed",
        )

    monkeypatch.setitem(
        rc_executor._STAGE_EXECUTORS, Stage.RESEARCH_DECISION, stage15_executor
    )

    result = rc_executor.execute_stage(
        Stage.RESEARCH_DECISION,
        run_dir=run_dir,
        run_id="run-full-auto",
        config=rc_config,
        adapters=adapters,
        auto_approve_gates=True,
    )

    assert result.status == StageStatus.DONE
    assert not (run_dir / "stage-15" / ".gate_proposal.json").exists()


def test_extension_context_uses_human_edited_rationale_on_extend(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
) -> None:
    rc_config = _with_hypothesis_validation(rc_config, enabled=False)
    rc_config = _with_hitl_required_stages(rc_config, (5, 9, 15, 20))
    rationale = "Human edited rationale must appear in extension context."
    _write_human_gate_artifacts(run_dir, "EXTEND", rationale)
    rc_runner._write_checkpoint(run_dir, Stage.RESULT_ANALYSIS, "run-human-context")
    _mock_pipeline_after_human_gate(monkeypatch, run_dir)

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-human-context",
        config=rc_config,
        adapters=adapters,
        from_stage=Stage.RESEARCH_DECISION,
    )

    context = (run_dir / "hypothesis_extension_context.md").read_text(
        encoding="utf-8"
    )
    assert rationale in context


# ── Deliverables packaging tests ──


def _setup_stage_artifacts(run_dir: Path) -> None:
    """Create typical stage-22 and stage-23 output files for testing."""
    s22 = run_dir / "stage-22"
    s22.mkdir(parents=True, exist_ok=True)
    (s22 / "paper_final.md").write_text("# My Paper\nContent here.", encoding="utf-8")
    (s22 / "paper.tex").write_text("\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}", encoding="utf-8")
    (s22 / "references.bib").write_text("@article{smith2024,\n  title={Test}\n}", encoding="utf-8")
    code_dir = s22 / "code"
    code_dir.mkdir()
    (code_dir / "main.py").write_text("print('hello')", encoding="utf-8")
    (code_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (code_dir / "README.md").write_text("# Code\n", encoding="utf-8")

    s23 = run_dir / "stage-23"
    s23.mkdir(parents=True, exist_ok=True)
    (s23 / "paper_final_verified.md").write_text("# My Paper (verified)\nContent.", encoding="utf-8")
    (s23 / "references_verified.bib").write_text("@article{smith2024,\n  title={Test}\n}", encoding="utf-8")
    (s23 / "verification_report.json").write_text(
        json.dumps({"summary": {"total": 5, "verified": 4}}), encoding="utf-8"
    )


def test_package_deliverables_collects_all_artifacts(
    run_dir: Path, rc_config: RCConfig
) -> None:
    _setup_stage_artifacts(run_dir)
    dest = rc_runner._package_deliverables(run_dir, "run-pkg-test", rc_config)
    assert dest is not None
    assert dest == run_dir / "deliverables"
    assert (dest / "paper_final.md").exists()
    assert (dest / "paper.tex").exists()
    assert (dest / "references.bib").exists()
    assert (dest / "code" / "main.py").exists()
    assert (dest / "verification_report.json").exists()
    assert (dest / "manifest.json").exists()
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["run_id"] == "run-pkg-test"
    assert "paper_final.md" in manifest["files"]


def test_package_deliverables_prefers_verified_versions(
    run_dir: Path, rc_config: RCConfig
) -> None:
    _setup_stage_artifacts(run_dir)
    rc_runner._package_deliverables(run_dir, "run-verified", rc_config)
    dest = run_dir / "deliverables"
    # Should contain verified content (from stage 23), not base (from stage 22)
    paper = (dest / "paper_final.md").read_text(encoding="utf-8")
    assert "verified" in paper
    bib = (dest / "references.bib").read_text(encoding="utf-8")
    assert "smith2024" in bib


def test_package_deliverables_falls_back_to_stage22(
    run_dir: Path, rc_config: RCConfig
) -> None:
    """When stage 23 outputs are missing, falls back to stage 22 versions."""
    s22 = run_dir / "stage-22"
    s22.mkdir(parents=True, exist_ok=True)
    (s22 / "paper_final.md").write_text("# Base Paper", encoding="utf-8")
    (s22 / "references.bib").write_text("@article{a,title={A}}", encoding="utf-8")

    dest = rc_runner._package_deliverables(run_dir, "run-fallback", rc_config)
    assert dest is not None
    paper = (dest / "paper_final.md").read_text(encoding="utf-8")
    assert "Base Paper" in paper


def test_package_deliverables_returns_none_when_no_stage_artifacts(
    run_dir: Path, tmp_path: Path,
) -> None:
    """Returns None when no stage artifacts exist and no style files found."""
    # Use a config with an unknown conference so style files aren't bundled
    data = {
        "project": {"name": "empty-test", "mode": "docs-first"},
        "research": {"topic": "empty"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline",
        },
        "export": {"target_conference": "unknown_conf_9999"},
    }
    cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
    result = rc_runner._package_deliverables(run_dir, "run-empty", cfg)
    assert result is None
    assert not (run_dir / "deliverables").exists()


def test_package_deliverables_includes_style_files(
    run_dir: Path, rc_config: RCConfig
) -> None:
    """Style files (.sty, .bst) for the target conference are bundled."""
    _setup_stage_artifacts(run_dir)
    dest = rc_runner._package_deliverables(run_dir, "run-styles", rc_config)
    assert dest is not None
    # Default config uses neurips_2025 → should have neurips_2025.sty
    assert (dest / "neurips_2025.sty").exists()
    manifest = json.loads((dest / "manifest.json").read_text())
    assert "neurips_2025.sty" in manifest["files"]


# ── Atomic checkpoint write tests ──


def test_write_checkpoint_uses_atomic_rename(run_dir: Path) -> None:
    """Checkpoint must be written via temp file + rename, not direct write"""
    rc_runner._write_checkpoint(run_dir, Stage.TOPIC_INIT, "run-atomic")
    cp = run_dir / "checkpoint.json"
    assert cp.exists()
    data = json.loads(cp.read_text(encoding="utf-8"))
    assert data["last_completed_stage"] == int(Stage.TOPIC_INIT)
    assert data["run_id"] == "run-atomic"


def test_write_checkpoint_leaves_no_temp_files(run_dir: Path) -> None:
    """Atomic write must clean up temp files on success"""
    rc_runner._write_checkpoint(run_dir, Stage.TOPIC_INIT, "run-clean")
    temps = list(run_dir.glob("*.tmp"))
    assert temps == [], f"Leftover temp files: {temps}"


def test_write_checkpoint_preserves_old_on_write_failure(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the temp-file write fails, the existing checkpoint must survive"""
    import builtins

    rc_runner._write_checkpoint(run_dir, Stage.TOPIC_INIT, "run-ok")

    original_open = builtins.open

    def _exploding_open(path, *args, **kwargs):
        # After os.close(fd), _write_checkpoint opens via path string —
        # intercept temp-file opens (checkpoint_*.tmp)
        if isinstance(path, (str, Path)) and "checkpoint_" in str(path):
            raise OSError("disk full")
        if isinstance(path, int):
            raise OSError("disk full")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _exploding_open)
    with pytest.raises(OSError):
        rc_runner._write_checkpoint(run_dir, Stage.PROBLEM_DECOMPOSE, "run-ok")

    # Original checkpoint must be intact
    data = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert data["last_completed_stage"] == int(Stage.TOPIC_INIT)
    # Temp file must be cleaned up
    assert list(run_dir.glob("checkpoint_*.tmp")) == []


def test_write_checkpoint_overwrites_previous(run_dir: Path) -> None:
    """A second checkpoint call must fully replace the first"""
    rc_runner._write_checkpoint(run_dir, Stage.TOPIC_INIT, "run-1")
    rc_runner._write_checkpoint(run_dir, Stage.PROBLEM_DECOMPOSE, "run-1")
    data = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert data["last_completed_stage"] == int(Stage.PROBLEM_DECOMPOSE)
    assert data["last_completed_name"] == Stage.PROBLEM_DECOMPOSE.name


def _degraded(stage: Stage) -> StageResult:
    return StageResult(
        stage=stage,
        status=StageStatus.DONE,
        artifacts=("quality_report.json",),
        decision="degraded",
    )


def test_degraded_quality_gate_continues_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When quality gate returns decision='degraded', pipeline continues to completion."""
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        _ = kwargs
        seen.append(stage)
        if stage == Stage.QUALITY_GATE:
            return _degraded(stage)
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-degraded",
        config=rc_config,
        adapters=adapters,
    )
    # All 23 stages should execute (not stopped at quality gate)
    assert len(results) == 23
    assert seen == list(STAGE_SEQUENCE)
    # Quality gate result should have decision="degraded"
    qg_result = [r for r in results if r.stage == Stage.QUALITY_GATE][0]
    assert qg_result.decision == "degraded"
    assert qg_result.status == StageStatus.DONE
    # Pipeline summary should have degraded=True
    summary = json.loads((run_dir / "pipeline_summary.json").read_text())
    assert summary["degraded"] is True
    # Output should show DEGRADED message
    captured = capsys.readouterr()
    assert "DEGRADED" in captured.out


def test_package_deliverables_called_after_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    rc_config: RCConfig,
    adapters: AdapterBundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Deliverables packaging is called at end of execute_pipeline."""
    _setup_stage_artifacts(run_dir)

    def mock_execute_stage(stage: Stage, **kwargs) -> StageResult:
        return _done(stage)

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)
    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-with-deliverables",
        config=rc_config,
        adapters=adapters,
    )
    captured = capsys.readouterr()
    assert "Deliverables packaged" in captured.out
    assert (run_dir / "deliverables" / "manifest.json").exists()


# ---------------------------------------------------------------------------
# BUG-223: _promote_best_stage14 must always write experiment_summary_best.json
# ---------------------------------------------------------------------------

def _make_stage14_summary(run_dir: Path, suffix: str, pm_value: float) -> None:
    """Helper: create a stage-14{suffix}/experiment_summary.json."""
    d = run_dir / f"stage-14{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    data = {
        "metrics_summary": {
            "primary_metric": {"min": pm_value, "max": pm_value, "mean": pm_value, "count": 1}
        },
        "condition_summaries": {"cond_a": {"metrics": {"primary_metric": pm_value}}},
    }
    (d / "experiment_summary.json").write_text(json.dumps(data), encoding="utf-8")


def _write_stage9_metric_protocol(run_dir: Path, *, direction: str) -> None:
    d = run_dir / "stage-09"
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.md").write_text(
        "# Experiment Plan\n\n"
        "## Hypotheses\nH1 evaluates the candidate experiment.\n\n"
        "## Baselines\nCompare against the baseline condition.\n\n"
        "## Ablations\nRemove the candidate intervention.\n\n"
        "## Metrics\nUse the experiment output values; direction: "
        f"{direction}.\n\n"
        "## Decision Criteria\nChoose the run that satisfies the plan.\n\n"
        "## Expected Outputs\noutputs/results.json\n",
        encoding="utf-8",
    )
    (d / "expected_outputs.json").write_text(
        json.dumps(
            {
                "schema_version": "researchclaw.expected_outputs.v1",
                "outputs": ["outputs/results.json"],
            }
        ),
        encoding="utf-8",
    )


class TestPromoteBestStage14BestJson:
    """BUG-223: experiment_summary_best.json must be written even when
    stage-14/ already has the best result (early-return path)."""

    @pytest.fixture()
    def max_config(self, rc_config: RCConfig) -> RCConfig:
        """Compatibility fixture; artifact resolver defaults to maximize."""
        return rc_config

    def test_best_json_written_when_current_is_best(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """stage-14/ already best → should still write best.json."""
        _make_stage14_summary(run_dir, "", 90.0)
        _make_stage14_summary(run_dir, "_v1", 80.0)
        _make_stage14_summary(run_dir, "_v2", 70.0)

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        assert best_path.exists(), "experiment_summary_best.json must always be written"
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 90.0

    def test_best_json_written_when_promotion_needed(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """stage-14/ is NOT best → promote + write best.json."""
        _make_stage14_summary(run_dir, "", 70.0)
        _make_stage14_summary(run_dir, "_v1", 95.0)

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        assert best_path.exists()
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 95.0

    def test_best_json_written_with_equal_values(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """BUG-223 exact scenario: stage-14 and stage-14_v1 have equal
        metrics, stage-14_v2 is regressed."""
        _make_stage14_summary(run_dir, "", 64.46)
        _make_stage14_summary(run_dir, "_v1", 64.46)
        _make_stage14_summary(run_dir, "_v2", 26.80)

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        assert best_path.exists(), "BUG-223: best.json missing when current is tied-best"
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 64.46


class TestPromoteBestStage14AnalysisBest:
    """BUG-225: analysis_best.md must be written from best stage-14 iteration."""

    @pytest.fixture()
    def max_config(self, rc_config: RCConfig) -> RCConfig:
        return rc_config

    def test_analysis_best_written_from_best_iteration(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """analysis_best.md should come from the best stage-14 iteration."""
        _make_stage14_summary(run_dir, "", 70.0)
        _make_stage14_summary(run_dir, "_v1", 95.0)
        # Write analysis.md in each
        (run_dir / "stage-14" / "analysis.md").write_text("Degenerate analysis", encoding="utf-8")
        (run_dir / "stage-14_v1" / "analysis.md").write_text("Best analysis v1", encoding="utf-8")

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        best_analysis = run_dir / "analysis_best.md"
        assert best_analysis.exists(), "BUG-225: analysis_best.md must be written"
        assert best_analysis.read_text(encoding="utf-8") == "Best analysis v1"

    def test_analysis_best_written_when_current_is_best(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """Even when stage-14 is already best, analysis_best.md should be written."""
        _make_stage14_summary(run_dir, "", 90.0)
        _make_stage14_summary(run_dir, "_v1", 80.0)
        (run_dir / "stage-14" / "analysis.md").write_text("Best analysis current", encoding="utf-8")
        (run_dir / "stage-14_v1" / "analysis.md").write_text("Worse analysis", encoding="utf-8")

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        best_analysis = run_dir / "analysis_best.md"
        assert best_analysis.exists()
        assert best_analysis.read_text(encoding="utf-8") == "Best analysis current"

    def test_no_analysis_best_when_no_analysis_md(
        self, run_dir: Path, max_config: RCConfig
    ) -> None:
        """If best stage-14 has no analysis.md, no analysis_best.md is written."""
        _make_stage14_summary(run_dir, "", 90.0)

        rc_runner._promote_best_stage14(run_dir, max_config)  # type: ignore[attr-defined]

        assert not (run_dir / "analysis_best.md").exists()


class TestPromoteBestStage14DegenerateDetection:
    """BUG-226: Degenerate near-zero metrics must not be promoted as best."""

    def test_degenerate_minimize_skipped(self, run_dir: Path, rc_config: RCConfig) -> None:
        """When minimize, a value 1000x smaller than second-best is degenerate."""
        _write_stage9_metric_protocol(run_dir, direction="minimize")
        _make_stage14_summary(run_dir, "", 7.26e-8)   # degenerate (broken normalization)
        _make_stage14_summary(run_dir, "_v2", 0.37)   # valid

        rc_runner._promote_best_stage14(run_dir, rc_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        assert best_path.exists()
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 0.37, "Degenerate value should be skipped, valid v2 promoted"

    def test_legitimate_minimize_not_skipped(self, run_dir: Path, rc_config: RCConfig) -> None:
        """When values are within normal range, smaller is legitimately best."""
        _write_stage9_metric_protocol(run_dir, direction="minimize")
        _make_stage14_summary(run_dir, "", 0.15)
        _make_stage14_summary(run_dir, "_v1", 0.37)

        rc_runner._promote_best_stage14(run_dir, rc_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 0.15, "Legitimate lower value should be promoted"

    def test_single_candidate_not_affected(self, run_dir: Path, rc_config: RCConfig) -> None:
        """Single candidate is never skipped regardless of value."""
        _make_stage14_summary(run_dir, "", 1e-10)

        rc_runner._promote_best_stage14(run_dir, rc_config)  # type: ignore[attr-defined]

        best_path = run_dir / "experiment_summary_best.json"
        data = json.loads(best_path.read_text(encoding="utf-8"))
        pm = data["metrics_summary"]["primary_metric"]
        assert pm["mean"] == 1e-10
