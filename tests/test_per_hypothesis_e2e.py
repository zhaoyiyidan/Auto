from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline import runner as rc_runner
from researchclaw.pipeline.executor import StageResult
from researchclaw.pipeline.hypothesis_coordinator import HypothesisValidationCoordinator
from researchclaw.pipeline.stages import Stage, StageStatus


def _config(tmp_path: Path) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "per-hypothesis-e2e", "mode": "docs-first"},
            "research": {"topic": "per hypothesis e2e"},
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
            "hypothesis_validation": {
                "enabled": True,
                "max_concurrent_branches": 2,
                "max_attempts_per_node": 1,
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _hypotheses_md() -> str:
    return """
## H1
Statement: Learning-rate warmup improves training stability.
Prediction: Validation loss variance decreases with warmup.
Falsification: Variance is unchanged.

## H2
Statement: Longer warmup improves final accuracy.
Prediction: Accuracy increases after longer warmup.
Falsification: Accuracy is unchanged.
"""


def test_two_hypotheses_run_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    coordinator = HypothesisValidationCoordinator(run_dir)
    coordinator.split_and_queue(_hypotheses_md())
    events: list[tuple[str, str, float]] = []
    lock = threading.Lock()

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        _ = attempt, config, adapters
        with lock:
            events.append((node.id, "start", time.monotonic()))
        time.sleep(0.1)
        with lock:
            events.append((node.id, "end", time.monotonic()))
        return SimpleNamespace(
            decision="proceed",
            artifacts=("stage-15/verdict.json",),
            metrics={"score": 0.9},
        )

    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    coordinator.run_until_queue_empty(
        config=_config(tmp_path),
        adapters=AdapterBundle(),
        max_concurrent=2,
    )

    starts = [item for item in events if item[1] == "start"]
    ends = [item for item in events if item[1] == "end"]
    assert len(starts) == 2
    assert len(ends) == 2
    assert max(timestamp for _, _, timestamp in starts) < min(
        timestamp for _, _, timestamp in ends
    )
    for node_id in ("h-001", "h-002"):
        assert (run_dir / "hypothesis_branches" / node_id / "attempt-001").is_dir()
    aggregate = json.loads((run_dir / "hypothesis_aggregate.json").read_text())
    assert len(aggregate["validation_summary"]) == 2


def test_extend_creates_child_and_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    coordinator = HypothesisValidationCoordinator(run_dir)
    coordinator.split_and_queue(_hypotheses_md())
    seen: list[str] = []

    def fake_validate_branch(
        node: Any,
        attempt: Any,
        config: Any,
        adapters: Any,
    ) -> Any:
        _ = attempt, config, adapters
        seen.append(node.id)
        if node.id == "h-001":
            return SimpleNamespace(
                decision="extend",
                artifacts=("stage-15/verdict.json",),
                metrics={"score": 0.7},
                next_hypothesis={
                    "statement": "Adaptive warmup improves training stability.",
                    "prediction": "Adaptive warmup lowers loss variance.",
                    "falsification": "Loss variance is unchanged.",
                    "rationale": "Follow-up from branch evidence.",
                },
            )
        return SimpleNamespace(
            decision="proceed",
            artifacts=("stage-15/verdict.json",),
            metrics={"score": 0.9},
        )

    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    coordinator.run_until_queue_empty(
        config=_config(tmp_path),
        adapters=AdapterBundle(),
        max_concurrent=1,
    )

    assert seen == ["h-001", "h-002", "h-003"]
    aggregate = json.loads((run_dir / "hypothesis_aggregate.json").read_text())
    assert aggregate["hypothesis_tree"]["total_nodes"] == 3
    assert len(aggregate["validation_summary"]) == 3


def test_no_legacy_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    coordinator = HypothesisValidationCoordinator(run_dir)
    coordinator.split_and_queue(_hypotheses_md())

    def fake_validate_branch(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        return SimpleNamespace(
            decision="proceed",
            artifacts=("stage-15/verdict.json",),
            metrics={"score": 0.9},
        )

    monkeypatch.setattr(coordinator, "validate_branch", fake_validate_branch)

    coordinator.run_until_queue_empty(
        config=_config(tmp_path),
        adapters=AdapterBundle(),
        max_concurrent=2,
    )

    assert not (run_dir / "current_node.txt").exists()
    assert not (run_dir / "pending_transition.json").exists()
    assert not (run_dir / "hypothesis_tree" / "current_node.txt").exists()
    events_path = run_dir / "hypothesis_tree" / "events.jsonl"
    assert "pivot_rollback" not in events_path.read_text(encoding="utf-8")


def test_no_root_recursive_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    seen: list[Stage] = []

    def mock_execute_stage(stage: Stage, **kwargs: Any) -> StageResult:
        _ = kwargs
        seen.append(stage)
        stage_dir = run_dir / f"stage-{int(stage):02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        if stage is Stage.HYPOTHESIS_GEN:
            (stage_dir / "hypotheses.md").write_text(
                _hypotheses_md(),
                encoding="utf-8",
            )
            return StageResult(
                stage=stage,
                status=StageStatus.DONE,
                artifacts=("hypotheses.md",),
            )
        if Stage.EXPERIMENT_TASK_SPEC <= stage <= Stage.RESEARCH_DECISION:
            raise AssertionError("root Stage 9-15 path should be bypassed")
        return StageResult(stage=stage, status=StageStatus.DONE, artifacts=("out.md",))

    def fake_run_until_queue_empty(self: Any, **kwargs: Any) -> list[Any]:
        _ = self, kwargs
        (run_dir / "hypothesis_aggregate.json").write_text(
            json.dumps(
                {
                    "validation_summary": [
                        {"node_id": "h-001", "decision": "pivot"},
                        {"node_id": "h-002", "decision": "proceed"},
                    ]
                }
            ),
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
        run_id="run-no-root-recursive-rollback",
        config=_config(tmp_path),
        adapters=AdapterBundle(),
    )

    assert Stage.HYPOTHESIS_GEN in seen
    assert Stage.EXPERIMENT_TASK_SPEC not in seen
    assert Stage.RESEARCH_DECISION not in seen
    assert not (run_dir / "decision_history.json").exists()


def test_runner_splits_and_executes_branch_stage9_to_15(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline.stage15_verdict import write_stage15_verdict

    run_dir = tmp_path / "run"
    stage_calls: list[tuple[Path, Stage]] = []

    def mock_execute_stage(stage: Stage, **kwargs: Any) -> StageResult:
        stage_run_dir = Path(kwargs["run_dir"])
        stage_calls.append((stage_run_dir, stage))
        stage_dir = stage_run_dir / f"stage-{int(stage):02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        if stage is Stage.HYPOTHESIS_GEN:
            (stage_dir / "hypotheses.md").write_text(
                _hypotheses_md(),
                encoding="utf-8",
            )
            return StageResult(
                stage=stage,
                status=StageStatus.DONE,
                artifacts=("hypotheses.md",),
            )
        if stage is Stage.EXPERIMENT_TASK_SPEC:
            (stage_dir / "plan.md").write_text("branch plan\n", encoding="utf-8")
            (stage_dir / "expected_outputs.json").write_text(
                json.dumps(
                    {
                        "schema_version": "researchclaw.expected_outputs.v1",
                        "outputs": ["outputs/metrics.json"],
                    }
                ),
                encoding="utf-8",
            )
            return StageResult(
                stage=stage,
                status=StageStatus.DONE,
                artifacts=("plan.md", "expected_outputs.json"),
            )
        if stage is Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR:
            (stage_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
            return StageResult(
                stage=stage,
                status=StageStatus.DONE,
                artifacts=("run_manifest.json",),
            )
        if stage is Stage.RESEARCH_DECISION:
            decision_md = "# Decision\n\nPROCEED\n"
            (stage_dir / "decision.md").write_text(decision_md, encoding="utf-8")
            write_stage15_verdict(
                stage_dir,
                decision="proceed",
                decision_md=decision_md,
                confidence=0.8,
                key_metrics={"score": 0.8},
            )
            return StageResult(
                stage=stage,
                status=StageStatus.DONE,
                artifacts=("decision.md", "verdict.json"),
                decision="proceed",
            )
        return StageResult(
            stage=stage,
            status=StageStatus.DONE,
            artifacts=(f"stage-{int(stage):02d}.json",),
        )

    monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-branch-stage9-to-15",
        config=_config(tmp_path),
        adapters=AdapterBundle(),
        to_stage=Stage.RESEARCH_DECISION,
    )

    branch_stage15_calls = [
        path
        for path, stage in stage_calls
        if stage is Stage.RESEARCH_DECISION
        and "hypothesis_branches" in path.parts
    ]
    assert len(branch_stage15_calls) == 2
    assert not any(
        stage is Stage.RESEARCH_DECISION and path == run_dir
        for path, stage in stage_calls
    )
    aggregate = json.loads((run_dir / "hypothesis_aggregate.json").read_text())
    assert len(aggregate["validation_summary"]) == 2
    for node_id in ("h-001", "h-002"):
        branch_run_dir = run_dir / "hypothesis_branches" / node_id / "attempt-001"
        assert (branch_run_dir / "stage-15" / "verdict.json").is_file()
        assert (branch_run_dir / "attempt_result.json").is_file()
    assert not (run_dir / "current_node.txt").exists()
    assert not (run_dir / "hypothesis_tree" / "current_node.txt").exists()
