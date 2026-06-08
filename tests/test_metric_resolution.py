from __future__ import annotations

from pathlib import Path


def test_resolve_metric_prefers_protocol(tmp_path: Path) -> None:
    from researchclaw.experiment.metric_resolution import resolve_experiment_metric
    from researchclaw.experiment.protocol import ExperimentProtocol, MetricSpec
    from researchclaw.experiment.workspace import TaskSpec

    stage9 = tmp_path / "stage-09"
    stage9.mkdir()
    (stage9 / "experiment_protocol.json").write_text(
        ExperimentProtocol(
            metrics=(MetricSpec(name="protocol_score", direction="minimize", is_primary=True),)
        ).to_json(),
        encoding="utf-8",
    )
    (stage9 / "task_spec.yaml").write_text(
        TaskSpec(
            workspace="/repo",
            objective="Do it",
            constraints=[],
            primary_metric="task_score",
            metric_direction="maximize",
            allowed_scope=["."],
            forbidden_scope=[],
            expected_outputs=["outputs/metrics.json"],
        ).to_yaml(),
        encoding="utf-8",
    )

    assert resolve_experiment_metric(tmp_path) == ("protocol_score", "minimize")


def test_resolve_metric_falls_back_to_task_spec_contract(tmp_path: Path) -> None:
    from researchclaw.experiment.execution_contract import default_contract
    from researchclaw.experiment.metric_resolution import resolve_experiment_metric
    from researchclaw.experiment.workspace import TaskSpec

    stage9 = tmp_path / "stage-09"
    stage9.mkdir()
    (stage9 / "task_spec.yaml").write_text(
        TaskSpec(
            workspace="/repo",
            objective="Do it",
            constraints=[],
            primary_metric="task_score",
            metric_direction="maximize",
            allowed_scope=["."],
            forbidden_scope=[],
            expected_outputs=["outputs/metrics.json"],
            execution_contract=default_contract(
                primary_metric="contract_score",
                metric_direction="minimize",
                expected_outputs=["outputs/metrics.json"],
            ),
        ).to_yaml(),
        encoding="utf-8",
    )

    assert resolve_experiment_metric(tmp_path) == ("contract_score", "minimize")


def test_resolve_metric_falls_back_to_task_spec_fields(tmp_path: Path) -> None:
    from researchclaw.experiment.metric_resolution import resolve_experiment_metric
    from researchclaw.experiment.workspace import TaskSpec

    stage9 = tmp_path / "stage-09"
    stage9.mkdir()
    (stage9 / "task_spec.yaml").write_text(
        TaskSpec(
            workspace="/repo",
            objective="Do it",
            constraints=[],
            primary_metric="task_score",
            metric_direction="maximize",
            allowed_scope=["."],
            forbidden_scope=[],
            expected_outputs=["outputs/metrics.json"],
        ).to_yaml(),
        encoding="utf-8",
    )

    assert resolve_experiment_metric(tmp_path) == ("task_score", "maximize")


def test_resolve_metric_missing_artifacts_safe_default(tmp_path: Path) -> None:
    from researchclaw.experiment.metric_resolution import resolve_experiment_metric

    assert resolve_experiment_metric(tmp_path) == ("primary_metric", "maximize")
