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
