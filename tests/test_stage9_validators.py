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
