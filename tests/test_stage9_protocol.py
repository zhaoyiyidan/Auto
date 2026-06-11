from __future__ import annotations

from pathlib import Path

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.stages import StageStatus


def _config(tmp_path: Path, *, topic: str = "dark matter direct detection") -> RCConfig:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return RCConfig.from_dict(
        {
            "project": {"name": "stage9-protocol-test", "mode": "docs-first"},
            "research": {"topic": topic, "domains": ["hep_ph"]},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "experiment": {
                "mode": "sandbox",
                "workspace_agent": {
                    "enabled": True,
                    "transport": "acp",
                    "workspace_path": str(workspace),
                    "session_name": "researchclaw-code-test",
                    "agent": "claude",
                    "timeout_sec": 300,
                    "max_turns": 20,
                    "close_policy": "close",
                },
                "submitter": {"type": "manual"},
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _write_hypotheses(run_dir: Path) -> None:
    stage8 = run_dir / "stage-08"
    stage8.mkdir(parents=True)
    (stage8 / "hypotheses.md").write_text(
        """
## H1
Hypothesis statement: A collider-aware selection improves exclusion reach.
Measurable prediction: exclusion_95cl increases over baseline.
Failure condition: exclusion_95cl is unchanged or lower.

## H2
Hypothesis statement: The same selection remains stable under detector smearing.
""",
        encoding="utf-8",
    )


def test_stage9_deterministic_writes_protocol_and_derived_task_spec(
    tmp_path: Path,
) -> None:
    from researchclaw.experiment.protocol import ExperimentProtocol
    from researchclaw.experiment.workspace import TaskSpec
    from researchclaw.pipeline.stage_impls._experiment_design import (
        _execute_experiment_design,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_hypotheses(run_dir)
    stage_dir = run_dir / "stage-09"
    stage_dir.mkdir()
    cfg = _config(tmp_path)

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        cfg,
        AdapterBundle(),
        llm=None,
    )

    assert result.status is StageStatus.DONE
    assert result.artifacts == (
        "experiment_protocol.json",
        "task_spec.yaml",
        "experiment_design_intent.md",
    )
    assert result.evidence_refs == (
        "stage-09/experiment_protocol.json",
        "stage-09/task_spec.yaml",
        "stage-09/experiment_design_intent.md",
    )

    protocol = ExperimentProtocol.from_path(stage_dir / "experiment_protocol.json")
    spec = TaskSpec.from_path(stage_dir / "task_spec.yaml")
    primary = protocol.primary_metric()

    assert [hypothesis.id for hypothesis in protocol.hypotheses] == ["H1", "H2"]
    assert len(protocol.decision_rules) == 2
    assert primary.name == "exclusion_95cl"
    assert primary.direction == "maximize"
    assert spec.primary_metric == primary.name
    assert spec.metric_direction == primary.direction
    assert any(
        "final git commit" in constraint and "run_manifest.json" in constraint
        for constraint in spec.constraints
    )
    assert any(
        "code_commit" in constraint and "clean git status" in constraint
        for constraint in spec.constraints
    )
    assert spec.execution_contract is not None
    assert spec.execution_contract.metrics.primary.name == primary.name
    assert spec.execution_contract.metrics.primary.direction == primary.direction
    assert "Experiment Protocol" in spec.objective
