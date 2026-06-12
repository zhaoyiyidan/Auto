from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.stages import StageStatus


VALID_PLAN_MD = """
# Experiment Plan

## Hypotheses
H1 predicts that corrected accumulation semantics improve convergence.

## Baselines
Use true large-batch training and fixed accumulation depths as baselines.

## Ablations
Run without corrected clipping, without corrected scheduler stepping, and without
adaptive accumulation so each mechanism is isolated.

## Metrics
Report time-to-target loss, final validation loss, update cosine similarity, and
wall-clock runtime.

## Decision Criteria
Support H1 only when corrected accumulation improves time-to-target loss by at
least 10% over the best baseline without worse validation loss.

## Expected Outputs
Stage 10 must produce outputs/results.json and outputs/summary.md.
"""


class QueueLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        self.calls.append(messages)
        from researchclaw.llm.client import LLMResponse

        if not self.responses:
            return LLMResponse(content="", model="queue")
        return LLMResponse(content=self.responses.pop(0), model="queue")


def _config(tmp_path: Path) -> RCConfig:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("# Workspace\nRun experiments here.\n", encoding="utf-8")
    return RCConfig.from_dict(
        {
            "project": {"name": "stage9-plan-agent-test", "mode": "docs-first"},
            "research": {"topic": "gradient accumulation semantics"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": False, "use_message": False},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "test",
                "primary_model": "fake-model",
            },
            "experiment": {
                "mode": "workspace",
                "workspace_agent": {
                    "enabled": True,
                    "transport": "acp",
                    "workspace_path": str(workspace),
                    "session_name": "stage9-plan",
                    "agent": "codex",
                    "timeout_sec": 300,
                },
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _write_hypotheses(run_dir: Path) -> None:
    stage8 = run_dir / "stage-08"
    stage8.mkdir(parents=True)
    (stage8 / "hypotheses.md").write_text(
        "## H1\nStatement: Correct accumulation matches true large batch.\n",
        encoding="utf-8",
    )


def test_stage9_fails_without_planning_agent_instead_of_fallback(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        _execute_experiment_design,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_hypotheses(run_dir)
    stage_dir = run_dir / "stage-09"
    cfg = _config(tmp_path)

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        cfg,
        AdapterBundle(),
        llm=None,
    )

    assert result.status is StageStatus.FAILED
    assert not (stage_dir / "plan.md").exists()
    assert not (stage_dir / "expected_outputs.json").exists()
    assert not (stage_dir / "experiment_protocol.json").exists()
    assert not (stage_dir / "task_spec.yaml").exists()


def test_stage9_accepts_freeform_plan_md(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        _execute_experiment_design,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_hypotheses(run_dir)
    stage_dir = run_dir / "stage-09"
    llm = QueueLLM(
        [
            "# Plan\nHypothesis: H1 only. Baseline and metric details are prose.\n",
            json.dumps(
                {
                    "schema_version": "researchclaw.expected_outputs.v1",
                    "outputs": ["outputs/results.json"],
                }
            ),
        ]
    )

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(tmp_path),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status is StageStatus.DONE
    assert "H1 only" in (stage_dir / "plan.md").read_text(encoding="utf-8")


def test_stage9_fails_if_expected_outputs_invalid(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        _execute_experiment_design,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_hypotheses(run_dir)
    stage_dir = run_dir / "stage-09"
    llm = QueueLLM(
        [
            VALID_PLAN_MD,
            json.dumps(
                {
                    "schema_version": "researchclaw.expected_outputs.v1",
                    "outputs": ["/tmp/results.json"],
                }
            ),
        ]
    )

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(tmp_path),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status is StageStatus.FAILED
    assert "absolute" in (result.error or "").lower()


def test_stage9_succeeds_with_valid_outputs(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        _execute_experiment_design,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_hypotheses(run_dir)
    stage_dir = run_dir / "stage-09"
    expected = {
        "schema_version": "researchclaw.expected_outputs.v1",
        "outputs": ["outputs/results.json", "outputs/summary.md"],
    }
    llm = QueueLLM([VALID_PLAN_MD, json.dumps(expected)])

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(tmp_path),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status is StageStatus.DONE
    assert result.artifacts == ("plan.md", "expected_outputs.json")
    assert (stage_dir / "plan.md").read_text(encoding="utf-8").strip()
    assert json.loads((stage_dir / "expected_outputs.json").read_text(encoding="utf-8")) == expected


def test_stage9_does_not_write_old_artifacts(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        _execute_experiment_design,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_hypotheses(run_dir)
    stage_dir = run_dir / "stage-09"
    llm = QueueLLM(
        [
            VALID_PLAN_MD,
            json.dumps(
                {
                    "schema_version": "researchclaw.expected_outputs.v1",
                    "outputs": ["outputs/results.json"],
                }
            ),
        ]
    )

    _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(tmp_path),
        AdapterBundle(),
        llm=llm,
    )

    assert not (stage_dir / "experiment_protocol.json").exists()
    assert not (stage_dir / "task_spec.yaml").exists()
    assert not (stage_dir / "experiment_design_intent.md").exists()
