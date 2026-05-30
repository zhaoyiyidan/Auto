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


def _execution_record(
    *,
    final_status: str | None = "completed",
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "metrics": {"accuracy": 0.9} if metrics is None else metrics,
    }
    if final_status is not None:
        payload["final_status"] = final_status
    return payload


def _result_artifacts(*, exists: bool = True) -> dict[str, object]:
    return {
        "artifacts": [
            {
                "path": "outputs/metrics.json",
                "exists": exists,
                "sha256": "sha" if exists else "",
                "size_bytes": 12 if exists else 0,
            }
        ]
    }


def test_evaluate_completion_complete() -> None:
    from researchclaw.experiment.execution_contract import (
        ExecutionContract,
        evaluate_contract,
    )

    evidence = evaluate_contract(
        ExecutionContract(),
        _execution_record(final_status="Done"),
        {},
    )

    assert evidence.completion_status == "complete"
    assert evidence.ok is True


def test_evaluate_completion_incomplete() -> None:
    from researchclaw.experiment.execution_contract import (
        ExecutionContract,
        evaluate_contract,
    )

    evidence = evaluate_contract(
        ExecutionContract(),
        _execution_record(final_status="running"),
        {},
    )

    assert evidence.completion_status == "incomplete"
    assert evidence.ok is False
    assert evidence.violations == ("completion:incomplete:final_status=running",)


def test_evaluate_completion_failed() -> None:
    from researchclaw.experiment.execution_contract import (
        ExecutionContract,
        evaluate_contract,
    )

    evidence = evaluate_contract(
        ExecutionContract(),
        _execution_record(final_status="failed"),
        {},
    )

    assert evidence.completion_status == "failed"
    assert evidence.ok is False
    assert evidence.violations == ("completion:failed:final_status=failed",)


def test_evaluate_completion_missing_status() -> None:
    from researchclaw.experiment.execution_contract import (
        ExecutionContract,
        evaluate_contract,
    )

    evidence = evaluate_contract(
        ExecutionContract(),
        _execution_record(final_status=None),
        {},
    )

    assert evidence.completion_status == "failed"
    assert evidence.violations == ("completion:failed:final_status=missing",)


def test_evaluate_metrics_present_but_non_numeric() -> None:
    from researchclaw.experiment.execution_contract import (
        ExecutionContract,
        evaluate_contract,
    )

    evidence = evaluate_contract(
        ExecutionContract(),
        _execution_record(metrics={"status": "ok"}),
        {},
    )

    assert evidence.metrics_present is True
    assert evidence.metrics_have_numeric is False
    assert "metrics:empty" not in evidence.violations


def test_evaluate_artifacts_declared_all_missing_flags_violation() -> None:
    from researchclaw.experiment.execution_contract import (
        ArtifactCheck,
        ExecutionContract,
        evaluate_contract,
    )

    evidence = evaluate_contract(
        ExecutionContract(
            result_artifacts=(ArtifactCheck("outputs/metrics.json", required=False),)
        ),
        _execution_record(),
        _result_artifacts(exists=False),
    )

    assert evidence.artifacts_declared is True
    assert evidence.any_artifact_exists is False
    assert "artifacts:all_missing" in evidence.violations
    assert evidence.ok is False


def test_evaluate_required_metric_missing_and_wrong_type() -> None:
    from researchclaw.experiment.execution_contract import (
        ExecutionContract,
        MetricCheck,
        MetricsContract,
        PrimaryMetric,
        evaluate_contract,
    )

    contract = ExecutionContract(
        metrics=MetricsContract(
            primary=PrimaryMetric("accuracy"),
            required=(
                MetricCheck("accuracy", "number"),
                MetricCheck("epoch", "int"),
                MetricCheck("label", "string"),
                MetricCheck("converged", "bool"),
                MetricCheck("optional", "number", required=False),
            ),
        )
    )

    evidence = evaluate_contract(
        contract,
        _execution_record(
            metrics={
                "accuracy": "0.9",
                "epoch": True,
                "converged": "yes",
            }
        ),
        {},
    )

    assert evidence.metric_checks[0].present is True
    assert evidence.metric_checks[0].type_ok is False
    assert "metric:accuracy:wrong_type" in evidence.violations
    assert "metric:epoch:wrong_type" in evidence.violations
    assert "metric:label:missing" in evidence.violations
    assert "metric:converged:wrong_type" in evidence.violations
    assert "metric:optional:missing" not in evidence.violations


def test_evaluate_ok_true_on_clean_complete_run() -> None:
    from researchclaw.experiment.execution_contract import (
        ArtifactCheck,
        ExecutionContract,
        MetricCheck,
        MetricsContract,
        PrimaryMetric,
        evaluate_contract,
    )

    contract = ExecutionContract(
        result_artifacts=(ArtifactCheck("outputs/metrics.json", required=True),),
        metrics=MetricsContract(
            primary=PrimaryMetric("accuracy"),
            required=(MetricCheck("accuracy", "number"),),
        ),
    )

    evidence = evaluate_contract(contract, _execution_record(), _result_artifacts())

    assert evidence.ok is True
    assert evidence.violations == ()
    assert evidence.to_dict()["ok"] is True


def test_evaluate_agent_declared_checks_recorded_when_enabled() -> None:
    from researchclaw.experiment.execution_contract import (
        AgentDeclaredChecks,
        ContractEvidence,
        ExecutionContract,
        evaluate_contract,
    )

    evidence = evaluate_contract(
        ExecutionContract(
            agent_declared=AgentDeclaredChecks(
                enabled=True,
                location="outputs/metrics.json",
                field="hypothesis_checks",
                required=True,
            )
        ),
        _execution_record(metrics={"accuracy": 0.9, "hypothesis_checks": []}),
        {},
    )

    assert evidence.agent_checks.requested is True
    assert evidence.agent_checks.found is True
    assert ContractEvidence.from_dict(evidence.to_dict()) == evidence
