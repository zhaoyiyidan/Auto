# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline import runner as rc_runner
from researchclaw.pipeline.executor import StageResult
from researchclaw.pipeline.stages import Stage, StageStatus


FIXED_NOW = "2026-01-01T00:00:00+00:00"


def _config(tmp_path: Path) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "runner-state-characterization", "mode": "docs-first"},
            "research": {"topic": "pipeline state characterization"},
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
            "experiment": {},
            "hypothesis_validation": {"enabled": False},
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _done(stage: Stage, *, decision: str = "proceed") -> StageResult:
    return StageResult(
        stage=stage,
        status=StageStatus.DONE,
        artifacts=(f"stage-{int(stage):02d}.txt",),
        decision=decision,
    )


def _write_stage14_candidate(run_dir: Path, dirname: str, score: float, analysis: str) -> None:
    stage_dir = run_dir / dirname
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "experiment_summary.json").write_text(
        json.dumps(
            {
                "metrics_summary": {
                    "primary_metric": {
                        "min": score,
                        "max": score,
                        "mean": score,
                        "count": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (stage_dir / "analysis.md").write_text(analysis, encoding="utf-8")


def _run_mocked_legacy_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[StageResult], list[Stage], str]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_id = "run-root-state"
    executed: list[Stage] = []

    monkeypatch.setattr(rc_runner, "_utcnow_iso", lambda: FIXED_NOW)

    def fake_execute_stage(stage: Stage, **kwargs: Any) -> StageResult:
        _ = kwargs
        executed.append(stage)
        stage_dir = run_dir / f"stage-{int(stage):02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / f"stage-{int(stage):02d}.txt").write_text(
            stage.name,
            encoding="utf-8",
        )
        if stage == Stage.HYPOTHESIS_GEN:
            (stage_dir / "hypotheses.md").write_text(
                "# Hypotheses\nH1: state files remain run-root scoped.",
                encoding="utf-8",
            )
        if stage == Stage.RESULT_ANALYSIS:
            _write_stage14_candidate(
                run_dir,
                "stage-14",
                0.82,
                "# Analysis\nCurrent stage 14 is the best.",
            )
        if stage == Stage.RESEARCH_DECISION:
            (stage_dir / "decision.md").write_text(
                "## Decision\nPROCEED",
                encoding="utf-8",
            )
            (stage_dir / "decision_structured.json").write_text(
                json.dumps({"decision": "proceed"}),
                encoding="utf-8",
            )
            return _done(stage, decision="proceed")
        return _done(stage)

    def fake_experiment_loop(**kwargs: Any) -> tuple[list[StageResult], str]:
        _ = kwargs
        loop_results = [
            _done(Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR),
            _done(Stage.MANIFEST_VALIDATE_AND_PREPARE),
            _done(Stage.HARNESS_SUBMIT_AND_COLLECT),
            _done(Stage.EXPERIMENT_ROUTE_DECISION),
        ]
        for result in loop_results:
            stage_dir = run_dir / f"stage-{int(result.stage):02d}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            (stage_dir / f"stage-{int(result.stage):02d}.txt").write_text(
                result.stage.name,
                encoding="utf-8",
            )
        return loop_results, "continue"

    monkeypatch.setattr(rc_runner, "execute_stage", fake_execute_stage)
    monkeypatch.setattr(rc_runner, "_run_experiment_loop", fake_experiment_loop)

    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id=run_id,
        config=_config(tmp_path),
        adapters=AdapterBundle(),
    )
    return run_dir, results, executed, run_id


def test_execute_pipeline_writes_run_root_state_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # characterization
    run_dir, results, executed, run_id = _run_mocked_legacy_pipeline(
        tmp_path,
        monkeypatch,
    )

    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    heartbeat = json.loads((run_dir / "heartbeat.json").read_text(encoding="utf-8"))
    best_summary = json.loads(
        (run_dir / "experiment_summary_best.json").read_text(encoding="utf-8")
    )

    assert len(results) == 23
    assert Stage.PAPER_OUTLINE in executed
    assert checkpoint == {
        "last_completed_stage": int(Stage.CITATION_VERIFY),
        "last_completed_name": Stage.CITATION_VERIFY.name,
        "run_id": run_id,
        "timestamp": FIXED_NOW,
    }
    assert heartbeat["last_stage"] == int(Stage.CITATION_VERIFY)
    assert heartbeat["last_stage_name"] == Stage.CITATION_VERIFY.name
    assert heartbeat["run_id"] == run_id
    assert heartbeat["timestamp"] == FIXED_NOW
    assert best_summary["metrics_summary"]["primary_metric"]["mean"] == 0.82
    assert (
        run_dir / "analysis_best.md"
    ).read_text(encoding="utf-8") == "# Analysis\nCurrent stage 14 is the best."


def test_promote_best_stage14_writes_run_root_best_artifacts(tmp_path: Path) -> None:
    # characterization
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_stage14_candidate(run_dir, "stage-14", 0.10, "# Analysis\nWorse current.")
    _write_stage14_candidate(run_dir, "stage-14_v1", 0.91, "# Analysis\nBest branch.")

    rc_runner._promote_best_stage14(run_dir, _config(tmp_path))

    best_summary = json.loads(
        (run_dir / "experiment_summary_best.json").read_text(encoding="utf-8")
    )

    assert best_summary["metrics_summary"]["primary_metric"]["mean"] == 0.91
    assert (run_dir / "analysis_best.md").read_text(
        encoding="utf-8"
    ) == "# Analysis\nBest branch."
    promoted_current = json.loads(
        (run_dir / "stage-14" / "experiment_summary.json").read_text(encoding="utf-8")
    )
    assert promoted_current["metrics_summary"]["primary_metric"]["mean"] == 0.91


def test_legacy_run_root_inventory_is_characterized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # characterization
    run_dir, results, _executed, _run_id = _run_mocked_legacy_pipeline(
        tmp_path,
        monkeypatch,
    )

    stage_dirs = {f"stage-{idx:02d}" for idx in range(1, 24)}
    run_root_state = {
        "analysis_best.md",
        "checkpoint.json",
        "deliverables",
        "experiment_memory",
        "experiment_summary_best.json",
        "heartbeat.json",
        "hypothesis_tree",
        "pipeline_summary.json",
    }

    assert len(results) == 23
    assert {path.name for path in run_dir.iterdir()} == stage_dirs | run_root_state
    assert {
        path.name
        for path in (run_dir / "hypothesis_tree").iterdir()
    } == {
        "current_node.txt",
        "events.jsonl",
        "node_tree",
        "nodes",
        "tree.json",
    }
    assert {
        str(path.relative_to(run_dir / "hypothesis_tree"))
        for path in (run_dir / "hypothesis_tree").rglob("*")
        if path.is_file()
    } >= {
        "current_node.txt",
        "events.jsonl",
        "node_tree/index.json",
        "node_tree/root/_node.json",
        "node_tree/root/h-1/_hypothesis.md",
        "node_tree/root/h-1/_node.json",
        "nodes/h-1/hypothesis.md",
        "nodes/h-1/node.json",
        "nodes/root/node.json",
        "tree.json",
    }
    assert {
        path.name
        for path in (run_dir / "deliverables").iterdir()
    } == {
        "manifest.json",
        "neurips_2025.sty",
    }
