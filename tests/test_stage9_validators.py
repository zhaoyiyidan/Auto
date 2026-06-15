from __future__ import annotations


def test_validate_expected_outputs_passes_minimal() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        validate_expected_outputs,
    )

    assert (
        validate_expected_outputs(
            {
                "schema_version": "researchclaw.expected_outputs.v1",
                "outputs": ["outputs/results.json"],
            }
        )
        == []
    )


def test_validate_expected_outputs_fails_wrong_schema_version() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        validate_expected_outputs,
    )

    errors = validate_expected_outputs(
        {"schema_version": "old", "outputs": ["outputs/results.json"]}
    )

    assert any("schema_version" in item for item in errors)


def test_validate_expected_outputs_fails_empty_outputs() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        validate_expected_outputs,
    )

    errors = validate_expected_outputs(
        {"schema_version": "researchclaw.expected_outputs.v1", "outputs": []}
    )

    assert any("outputs" in item and "non-empty" in item for item in errors)


def test_validate_expected_outputs_fails_absolute_path() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        validate_expected_outputs,
    )

    errors = validate_expected_outputs(
        {
            "schema_version": "researchclaw.expected_outputs.v1",
            "outputs": ["/tmp/results.json"],
        }
    )

    assert any("absolute" in item.lower() for item in errors)


def test_validate_expected_outputs_fails_dotdot_path() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        validate_expected_outputs,
    )

    errors = validate_expected_outputs(
        {
            "schema_version": "researchclaw.expected_outputs.v1",
            "outputs": ["../results.json"],
        }
    )

    assert any(".." in item for item in errors)


def test_validate_expected_outputs_fails_git_path() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        validate_expected_outputs,
    )

    errors = validate_expected_outputs(
        {
            "schema_version": "researchclaw.expected_outputs.v1",
            "outputs": [".git/results.json"],
        }
    )

    assert any(".git" in item for item in errors)


def test_validate_expected_outputs_fails_not_a_list() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        validate_expected_outputs,
    )

    errors = validate_expected_outputs(
        {
            "schema_version": "researchclaw.expected_outputs.v1",
            "outputs": "outputs/results.json",
        }
    )

    assert any("list" in item.lower() for item in errors)


def test_validate_expected_outputs_rejects_implementation_artifacts() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        validate_expected_outputs,
    )

    errors = validate_expected_outputs(
        {
            "schema_version": "researchclaw.expected_outputs.v1",
            "outputs": [
                "plan.md",
                "expected_outputs.json",
                "run_manifest.json",
                "experiment_config.yaml",
                "run_experiment.py",
                "scripts/run_experiment.py",
                "notebooks/analyze.ipynb",
                "run.sh",
                "results/hypothesis_decisions.md",
            ],
        }
    )

    joined = "\n".join(errors)
    assert "plan.md" in joined
    assert "expected_outputs.json" in joined
    assert "run_manifest.json" in joined
    assert "experiment_config.yaml" in joined
    assert "run_experiment.py" in joined
    assert "scripts/run_experiment.py" in joined
    assert "notebooks/analyze.ipynb" in joined
    assert "run.sh" in joined
    assert "results/hypothesis_decisions.md" not in joined


def test_expected_outputs_prompt_excludes_implementation_artifacts() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import (
        PlanningContext,
        _expected_outputs_prompt,
    )

    prompt = _expected_outputs_prompt(
        PlanningContext(
            hypotheses_md="H1",
            topic="topic",
            workspace_path="/tmp/workspace",
            workspace_tree="README.md",
            readme="",
            dependencies="",
            human_guidance="",
            hardware_profile="",
        ),
        "# Plan",
    )

    assert "formal experiment" in prompt
    assert "Exclude implementation files" in prompt
    assert "run_manifest.json" in prompt
    assert "plan.md" in prompt
