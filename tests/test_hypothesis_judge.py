from __future__ import annotations

import json

from researchclaw.experiment.protocol import DecisionRule, ExperimentProtocol, HypothesisSpec


class FakeLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        _ = messages, kwargs
        from researchclaw.llm.client import LLMResponse

        return LLMResponse(content=json.dumps(self.payload), model="fake")


def test_evaluate_decision_rule_gt_and_lt() -> None:
    from researchclaw.pipeline.hypothesis_judge import evaluate_decision_rule

    summary = {
        "primary_metric": "accuracy",
        "best_metric": 0.8,
        "metrics_summary": {"loss": {"mean": 0.2}},
    }

    assert (
        evaluate_decision_rule(
            DecisionRule(hypothesis_id="H1", metric="accuracy", comparator="gt", threshold=0.7),
            summary,
        )
        == "supported"
    )
    assert (
        evaluate_decision_rule(
            DecisionRule(hypothesis_id="H2", metric="loss", comparator="lt", threshold=0.3),
            summary,
        )
        == "supported"
    )


def test_evaluate_decision_rule_missing_metric_inconclusive() -> None:
    from researchclaw.pipeline.hypothesis_judge import evaluate_decision_rule

    assert (
        evaluate_decision_rule(
            DecisionRule(hypothesis_id="H1", metric="missing"),
            {"metrics_summary": {}},
        )
        == "inconclusive"
    )


def test_evaluate_decision_rule_delta_and_within_pct() -> None:
    from researchclaw.pipeline.hypothesis_judge import evaluate_decision_rule

    summary = {
        "metrics_summary": {
            "treatment": {"mean": 0.65},
            "baseline": {"mean": 0.60},
            "calibrated": {"mean": 102.0},
            "target": {"mean": 100.0},
        }
    }

    assert (
        evaluate_decision_rule(
            DecisionRule(
                hypothesis_id="H1",
                metric="treatment",
                comparator="delta_gt",
                threshold=0.03,
                baseline_metric="baseline",
            ),
            summary,
        )
        == "supported"
    )
    assert (
        evaluate_decision_rule(
            DecisionRule(
                hypothesis_id="H2",
                metric="calibrated",
                comparator="within_pct",
                threshold=3.0,
                baseline_metric="target",
            ),
            summary,
        )
        == "supported"
    )


def test_evaluate_decision_rule_supported_if_fail_inverts() -> None:
    from researchclaw.pipeline.hypothesis_judge import evaluate_decision_rule

    assert (
        evaluate_decision_rule(
            DecisionRule(
                hypothesis_id="H1",
                metric="accuracy",
                comparator="gt",
                threshold=0.7,
                supported_if="fail",
            ),
            {"primary_metric": "accuracy", "best_metric": 0.8},
        )
        == "refuted"
    )


def test_decision_from_verdicts_mapping() -> None:
    from researchclaw.pipeline.hypothesis_judge import decision_from_verdicts

    assert decision_from_verdicts({"H1": {"verdict": "supported"}}) == "proceed"
    assert decision_from_verdicts({"H1": {"verdict": "refuted"}}) == "pivot"
    assert decision_from_verdicts({"H1": {"verdict": "inconclusive"}}) == "extend"


def test_judge_hypotheses_counts_and_decision() -> None:
    from researchclaw.pipeline.hypothesis_judge import judge_hypotheses

    protocol = ExperimentProtocol(
        hypotheses=(
            HypothesisSpec(id="H1", statement="Accuracy improves."),
            HypothesisSpec(id="H2", statement="Loss falls."),
        ),
        decision_rules=(
            DecisionRule(hypothesis_id="H1", metric="accuracy", comparator="gt", threshold=0.7),
            DecisionRule(hypothesis_id="H2", metric="loss", comparator="lt", threshold=0.1),
        ),
    )

    verdict = judge_hypotheses(
        protocol,
        {
            "primary_metric": "accuracy",
            "best_metric": 0.8,
            "metrics_summary": {"loss": {"mean": 0.2}},
        },
    )

    assert verdict["decision"] == "proceed"
    assert verdict["counts"] == {"supported": 1, "refuted": 1, "inconclusive": 0}


def test_inconclusive_rule_uses_llm_fallback() -> None:
    from researchclaw.pipeline.hypothesis_judge import evaluate_decision_rule

    llm = FakeLLM({"verdict": "supported", "rationale": "Analysis supports H1."})

    verdict = evaluate_decision_rule(
        DecisionRule(hypothesis_id="H1", metric="missing"),
        {"analysis": "The observed trace supports H1."},
        llm=llm,
    )

    assert verdict == "supported"
    assert llm.calls == 1
