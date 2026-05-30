from __future__ import annotations


def test_default_contract_reproduces_current_statuses() -> None:
    from researchclaw.experiment.execution_contract import default_contract

    contract = default_contract(
        primary_metric="accuracy",
        metric_direction="maximize",
        expected_outputs=["outputs/metrics.json"],
    )

    assert contract.completion.success_statuses == (
        "completed",
        "complete",
        "succeeded",
        "success",
        "done",
        "passed",
    )
    assert contract.completion.incomplete_statuses == (
        "queued",
        "pending",
        "running",
        "submitted",
    )
    assert contract.completion.require_any_metric is True
    assert contract.completion.require_any_artifact is False


def test_contract_to_dict_from_dict_round_trip() -> None:
    from researchclaw.experiment.execution_contract import (
        AgentDeclaredChecks,
        ArtifactCheck,
        CompletionContract,
        ExecutionContract,
        MetricCheck,
        MetricsContract,
        PrimaryMetric,
    )

    contract = ExecutionContract(
        result_artifacts=(
            ArtifactCheck("outputs/metrics.json", type="json", required=True),
        ),
        metrics=MetricsContract(
            primary=PrimaryMetric("accuracy", direction="maximize", unit="fraction"),
            required=(MetricCheck("accuracy", type="number", required=True),),
            allow_extra=False,
        ),
        completion=CompletionContract(
            success_statuses=("done",),
            incomplete_statuses=("queued",),
            require_any_metric=True,
            require_any_artifact=True,
        ),
        agent_declared=AgentDeclaredChecks(
            enabled=True,
            location="outputs/metrics.json",
            field="hypothesis_checks",
            required=True,
        ),
    )

    assert ExecutionContract.from_dict(contract.to_dict()) == contract


def test_from_dict_lenient_missing_keys_uses_defaults() -> None:
    from researchclaw.experiment.execution_contract import ExecutionContract

    contract = ExecutionContract.from_dict({})

    assert contract.schema_version == "researchclaw.execution_contract.v1"
    assert contract.metrics.primary.name == "primary_metric"
    assert contract.metrics.primary.direction == "maximize"
    assert contract.result_artifacts == ()


def test_from_dict_ignores_unknown_keys() -> None:
    from researchclaw.experiment.execution_contract import ExecutionContract

    contract = ExecutionContract.from_dict(
        {
            "unknown": "ignored",
            "metrics": {
                "primary": {"name": "accuracy", "unknown": "ignored"},
                "unknown": "ignored",
            },
        }
    )

    assert contract.metrics.primary.name == "accuracy"
    assert not hasattr(contract, "unknown")


def test_default_contract_artifacts_not_required() -> None:
    from researchclaw.experiment.execution_contract import default_contract

    contract = default_contract(
        primary_metric="accuracy",
        metric_direction="maximize",
        expected_outputs=["outputs/metrics.json", "outputs/log.txt"],
    )

    assert [item.path for item in contract.result_artifacts] == [
        "outputs/metrics.json",
        "outputs/log.txt",
    ]
    assert {item.required for item in contract.result_artifacts} == {False}
