from __future__ import annotations

from pathlib import Path


def test_task_spec_round_trips_yaml(tmp_path: Path) -> None:
    from researchclaw.experiment.workspace import TaskSpec

    spec = TaskSpec(
        workspace="/repo",
        objective="improve X",
        constraints=["single GPU"],
        primary_metric="accuracy",
        metric_direction="maximize",
        allowed_scope=["src/"],
        forbidden_scope=["data/raw/"],
        expected_outputs=["outputs/metrics.json"],
    )
    path = tmp_path / "task_spec.yaml"

    path.write_text(spec.to_yaml(), encoding="utf-8")
    loaded = TaskSpec.from_path(path)

    assert loaded == spec
    assert "schema_version: researchclaw.task_spec.v1" in spec.to_yaml()


def test_task_spec_with_contract_round_trips(tmp_path: Path) -> None:
    from researchclaw.experiment.execution_contract import default_contract
    from researchclaw.experiment.workspace import TaskSpec

    spec = TaskSpec(
        workspace="/repo",
        objective="improve X",
        constraints=["single GPU"],
        primary_metric="accuracy",
        metric_direction="maximize",
        allowed_scope=["src/"],
        forbidden_scope=["data/raw/"],
        expected_outputs=["outputs/metrics.json"],
        execution_contract=default_contract(
            primary_metric="accuracy",
            metric_direction="maximize",
            expected_outputs=["outputs/metrics.json"],
        ),
    )
    path = tmp_path / "task_spec.yaml"

    path.write_text(spec.to_yaml(), encoding="utf-8")
    loaded = TaskSpec.from_path(path)

    assert loaded == spec
    assert loaded.execution_contract is not None
    assert loaded.execution_contract.metrics.primary.name == "accuracy"


def test_task_spec_without_contract_omits_key_in_yaml() -> None:
    from researchclaw.experiment.workspace import TaskSpec

    spec = TaskSpec(
        workspace="/repo",
        objective="improve X",
        constraints=[],
        primary_metric="accuracy",
        metric_direction="maximize",
        allowed_scope=["src/"],
        forbidden_scope=[],
        expected_outputs=["outputs/metrics.json"],
    )

    assert "execution_contract" not in spec.to_yaml()


def test_task_spec_from_yaml_with_contract_parses() -> None:
    from researchclaw.experiment.workspace import TaskSpec

    spec = TaskSpec.from_yaml(
        "\n".join(
            [
                "workspace: /repo",
                "objective: improve X",
                "constraints: []",
                "primary_metric: accuracy",
                "metric_direction: maximize",
                "allowed_scope: ['src/']",
                "forbidden_scope: []",
                "expected_outputs: ['outputs/metrics.json']",
                "execution_contract:",
                "  metrics:",
                "    primary:",
                "      name: accuracy",
                "      direction: maximize",
                "    required:",
                "      - name: accuracy",
                "        type: number",
            ]
        )
    )

    assert spec.execution_contract is not None
    assert spec.execution_contract.metrics.primary.name == "accuracy"
    assert spec.execution_contract.metrics.required[0].name == "accuracy"


def test_task_spec_from_yaml_lenient_partial_contract() -> None:
    from researchclaw.experiment.workspace import TaskSpec

    spec = TaskSpec.from_yaml(
        "\n".join(
            [
                "workspace: /repo",
                "objective: improve X",
                "constraints: []",
                "primary_metric: accuracy",
                "metric_direction: maximize",
                "allowed_scope: ['src/']",
                "forbidden_scope: []",
                "expected_outputs: ['outputs/metrics.json']",
                "execution_contract:",
                "  result_artifacts:",
                "    - path: outputs/metrics.json",
            ]
        )
    )

    assert spec.execution_contract is not None
    assert spec.execution_contract.metrics.primary.name == "primary_metric"
    assert spec.execution_contract.result_artifacts[0].required is True
