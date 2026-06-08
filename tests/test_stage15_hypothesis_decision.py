from __future__ import annotations

import json
from pathlib import Path

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.experiment.protocol import (
    DecisionRule,
    ExperimentProtocol,
    HypothesisSpec,
    MetricSpec,
)


class FakeLLM:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, "kwargs": kwargs})
        from researchclaw.llm.client import LLMResponse

        return LLMResponse(content=self.response_text, model="fake")


def _config(tmp_path: Path) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "stage15-protocol-test", "mode": "docs-first"},
            "research": {"topic": "hypothesis routing"},
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
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _write_protocol(run_dir: Path, *, comparator: str = "gt", threshold: float = 0.0) -> None:
    stage9 = run_dir / "stage-09"
    stage9.mkdir(parents=True, exist_ok=True)
    (stage9 / "experiment_protocol.json").write_text(
        ExperimentProtocol(
            hypotheses=(HypothesisSpec(id="H1", statement="Metric should pass."),),
            metrics=(MetricSpec(name="score", direction="maximize", is_primary=True),),
            decision_rules=(
                DecisionRule(
                    hypothesis_id="H1",
                    metric="score",
                    comparator=comparator,
                    threshold=threshold,
                ),
            ),
        ).to_json(),
        encoding="utf-8",
    )


def _write_summary(run_dir: Path, summary: dict[str, object]) -> None:
    stage14 = run_dir / "stage-14"
    stage14.mkdir(parents=True, exist_ok=True)
    (stage14 / "experiment_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (stage14 / "analysis.md").write_text("# Analysis\nEvidence summary.", encoding="utf-8")


def _run_decision(tmp_path: Path, run_dir: Path, llm=None):
    from researchclaw.pipeline.stage_impls._analysis import _execute_research_decision

    stage_dir = run_dir / "stage-15"
    stage_dir.mkdir(parents=True, exist_ok=True)
    return _execute_research_decision(
        stage_dir,
        run_dir,
        _config(tmp_path),
        AdapterBundle(),
        llm=llm,
    )


def test_protocol_supported_hypothesis_proceeds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_protocol(run_dir, threshold=0.5)
    _write_summary(run_dir, {"primary_metric": "score", "best_metric": 0.8})

    result = _run_decision(tmp_path, run_dir, llm=None)

    payload = json.loads((run_dir / "stage-15" / "decision_structured.json").read_text())
    verdict = json.loads((run_dir / "hypothesis_verdict.json").read_text())
    assert result.decision == "proceed"
    assert payload["source"] == "hypothesis_judge"
    assert payload["per_hypothesis"]["H1"]["verdict"] == "supported"
    assert verdict["decision"] == "proceed"
    assert "## Decision: PROCEED" in (run_dir / "stage-15" / "decision.md").read_text()


def test_protocol_refuted_hypothesis_pivots(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_protocol(run_dir, threshold=0.5)
    _write_summary(run_dir, {"primary_metric": "score", "best_metric": 0.1})

    result = _run_decision(tmp_path, run_dir, llm=None)

    assert result.decision == "pivot"
    payload = json.loads((run_dir / "stage-15" / "decision_structured.json").read_text())
    assert payload["counts"]["refuted"] == 1


def test_protocol_inconclusive_hypothesis_extends(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_protocol(run_dir, threshold=0.5)
    _write_summary(run_dir, {"metrics_summary": {}})

    result = _run_decision(tmp_path, run_dir, llm=None)

    assert result.decision == "extend"
    payload = json.loads((run_dir / "stage-15" / "decision_structured.json").read_text())
    assert payload["counts"]["inconclusive"] == 1


def test_no_protocol_falls_back_to_legacy_decision_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_summary(run_dir, {"primary_metric": "score", "best_metric": 0.8})
    llm = FakeLLM("## Decision\nPIVOT\n## Justification\nLegacy path.")

    result = _run_decision(tmp_path, run_dir, llm=llm)

    payload = json.loads((run_dir / "stage-15" / "decision_structured.json").read_text())
    assert result.decision == "pivot"
    assert payload.get("source") != "hypothesis_judge"


def test_requirements_gate_precedes_protocol_judge(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_protocol(run_dir, threshold=0.5)
    _write_summary(run_dir, {"primary_metric": "score", "best_metric": 0.8})
    (run_dir / "stage-09" / "requirements.json").write_text(
        json.dumps([{"id": "req1", "description": "must pass", "must_pass": True}]),
        encoding="utf-8",
    )
    llm = FakeLLM(
        json.dumps(
            {
                "per_requirement": [
                    {
                        "id": "req1",
                        "must_pass": True,
                        "met": True,
                        "evidence": "score present",
                        "missing": "",
                    }
                ],
                "verdict": "proceed",
                "delta_feedback": "",
            }
        )
    )

    result = _run_decision(tmp_path, run_dir, llm=llm)

    payload = json.loads((run_dir / "stage-15" / "decision_structured.json").read_text())
    assert result.decision == "proceed"
    assert payload["source"] == "agent_requirements_gate"
    assert not (run_dir / "hypothesis_verdict.json").exists()


def test_inconclusive_protocol_rule_uses_llm_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_protocol(run_dir, threshold=0.5)
    _write_summary(run_dir, {"metrics_summary": {}, "analysis": "Supports H1."})
    llm = FakeLLM(json.dumps({"verdict": "supported", "rationale": "analysis supports H1"}))

    result = _run_decision(tmp_path, run_dir, llm=llm)

    assert result.decision == "proceed"
    assert llm.calls
    payload = json.loads((run_dir / "stage-15" / "decision_structured.json").read_text())
    assert payload["per_hypothesis"]["H1"]["verdict"] == "supported"
