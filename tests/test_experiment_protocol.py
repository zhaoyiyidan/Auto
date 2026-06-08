from __future__ import annotations

import json
from pathlib import Path


def test_protocol_round_trips_json(tmp_path: Path) -> None:
    from researchclaw.experiment.protocol import (
        ComparisonSpec,
        DecisionRule,
        ExperimentProtocol,
        HypothesisSpec,
        MetricSpec,
    )

    protocol = ExperimentProtocol(
        objective="Test a robustness claim",
        hypotheses=(
            HypothesisSpec(
                id="H1",
                statement="The treatment improves robustness.",
                prediction="Accuracy increases under shift.",
                falsification="No measurable gain.",
                rationale="Prior work suggests regularization helps.",
                baselines=("baseline",),
            ),
        ),
        metrics=(
            MetricSpec(
                name="ood_accuracy_delta",
                direction="maximize",
                unit="points",
                description="Treatment minus baseline OOD accuracy.",
                hypothesis_ids=("H1",),
                is_primary=True,
            ),
        ),
        comparisons=(
            ComparisonSpec(
                id="C1",
                baseline="baseline",
                treatment="regularized",
                metric="ood_accuracy_delta",
                hypothesis_ids=("H1",),
            ),
        ),
        decision_rules=(
            DecisionRule(
                hypothesis_id="H1",
                metric="ood_accuracy_delta",
                comparator="delta_gt",
                threshold=0.02,
                baseline_metric="baseline_ood_accuracy",
            ),
        ),
    )
    path = tmp_path / "experiment_protocol.json"

    path.write_text(protocol.to_json(), encoding="utf-8")
    loaded = ExperimentProtocol.from_path(path)

    assert loaded == protocol
    assert json.loads(protocol.to_json())["schema_version"] == (
        "researchclaw.experiment_protocol.v1"
    )


def test_from_dict_is_lenient() -> None:
    from researchclaw.experiment.protocol import ExperimentProtocol

    assert ExperimentProtocol.from_dict("not a mapping").primary_metric().name == (
        "primary_metric"
    )
    assert ExperimentProtocol.from_json("{bad json").primary_metric().direction == "maximize"


def test_from_dict_coerces_partial() -> None:
    from researchclaw.experiment.protocol import ExperimentProtocol

    protocol = ExperimentProtocol.from_dict(
        {
            "objective": 123,
            "hypotheses": [{"id": "1", "statement": "Claim", "baselines": "base"}],
            "metrics": [{"name": "loss", "direction": "down", "is_primary": 1}],
            "comparisons": [{"id": 1, "conditions": "control", "metric": "loss"}],
            "decision_rules": [
                {"hypothesis_id": 1, "metric": "loss", "threshold": "0.5"}
            ],
        }
    )

    assert protocol.objective == "123"
    assert protocol.hypotheses[0].id == "H1"
    assert protocol.hypotheses[0].baselines == ("base",)
    assert protocol.metrics[0].direction == "minimize"
    assert protocol.comparisons[0].id == "C1"
    assert protocol.comparisons[0].conditions == ("control",)
    assert protocol.decision_rules[0].threshold == 0.5


def test_primary_metric_resolution() -> None:
    from researchclaw.experiment.protocol import ExperimentProtocol, MetricSpec

    assert ExperimentProtocol().primary_metric().name == "primary_metric"
    assert (
        ExperimentProtocol(metrics=(MetricSpec(name="secondary"),)).primary_metric().name
        == "secondary"
    )
    assert (
        ExperimentProtocol(
            metrics=(
                MetricSpec(name="secondary"),
                MetricSpec(name="primary", is_primary=True),
            )
        ).primary_metric().name
        == "primary"
    )


def test_metric_spec_direction_coerces() -> None:
    from researchclaw.experiment.protocol import MetricSpec

    assert MetricSpec(name="loss", direction="min").direction == "minimize"
    assert MetricSpec(name="accuracy", direction="higher").direction == "maximize"
    assert MetricSpec(name="score", direction="unexpected").direction == "maximize"


def test_validate_returns_warnings_not_raises() -> None:
    from researchclaw.experiment.protocol import DecisionRule, ExperimentProtocol, MetricSpec

    protocol = ExperimentProtocol(
        metrics=(MetricSpec(name="accuracy"),),
        decision_rules=(DecisionRule(hypothesis_id="H2", metric="missing"),),
    )

    warnings = protocol.validate()

    assert any("missing metric" in warning for warning in warnings)
    assert any("unknown hypothesis" in warning for warning in warnings)


def test_parse_hypotheses_md_h_headings_and_labels() -> None:
    from researchclaw.experiment.protocol import parse_hypotheses_md

    parsed = parse_hypotheses_md(
        """
# H1
**Hypothesis statement:** Regularization improves OOD accuracy.
**Measurable prediction:** Treatment beats baseline.
**Failure condition:** Difference is zero or negative.
**Rationale:** It should reduce overfitting.
**Required baselines:** ERM, dropout

## Hypothesis 2
Hypothesis statement: Compression preserves F1.
Falsification: F1 drops materially.

## Generated
metadata trailer
"""
    )

    assert [item.id for item in parsed] == ["H1", "H2"]
    assert parsed[0].statement == "Regularization improves OOD accuracy."
    assert parsed[0].prediction == "Treatment beats baseline."
    assert parsed[0].falsification == "Difference is zero or negative."
    assert parsed[0].rationale == "It should reduce overfitting."
    assert parsed[0].baselines == ("ERM", "dropout")
    assert parsed[1].falsification == "F1 drops materially."


def test_parse_hypotheses_md_no_headings_single() -> None:
    from researchclaw.experiment.protocol import parse_hypotheses_md

    parsed = parse_hypotheses_md("A compact falsifiable hypothesis.")

    assert len(parsed) == 1
    assert parsed[0].id == "H1"
    assert parsed[0].statement == "A compact falsifiable hypothesis."


def test_parse_hypotheses_md_empty() -> None:
    from researchclaw.experiment.protocol import parse_hypotheses_md

    assert parse_hypotheses_md("") == ()
