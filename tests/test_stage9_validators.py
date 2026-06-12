from __future__ import annotations


VALID_PLAN_MD = """
# Experiment Plan

## Hypotheses
H1 predicts the proposed training rule improves convergence.

## Baselines
Compare against fixed accumulation and true large-batch training.

## Ablations
Remove adaptive scheduling and optimizer-step corrections separately.

## Metrics
Report time-to-target loss and update cosine similarity.

## Decision Criteria
Support H1 only if the proposed method beats the best baseline by 10%.

## Expected Outputs
The experiment must write outputs/results.json and outputs/summary.md.
"""


def test_validate_plan_md_passes_with_all_sections() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import validate_plan_md

    assert validate_plan_md(VALID_PLAN_MD) == []


def test_validate_plan_md_fails_missing_baseline() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import validate_plan_md

    text = VALID_PLAN_MD.replace("## Baselines", "## Comparisons")

    assert any("baseline" in item.lower() for item in validate_plan_md(text))


def test_validate_plan_md_fails_missing_ablation() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import validate_plan_md

    text = VALID_PLAN_MD.replace("## Ablations", "## Sensitivity")

    assert any("ablation" in item.lower() for item in validate_plan_md(text))


def test_validate_plan_md_fails_missing_metrics() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import validate_plan_md

    text = VALID_PLAN_MD.replace("## Metrics", "## Measurements")

    assert any("metric" in item.lower() for item in validate_plan_md(text))


def test_validate_plan_md_fails_missing_decision() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import validate_plan_md

    text = VALID_PLAN_MD.replace("## Decision Criteria", "## Interpretation")

    assert any("decision" in item.lower() for item in validate_plan_md(text))


def test_validate_plan_md_fails_empty_section_body() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import validate_plan_md

    text = VALID_PLAN_MD.replace(
        "Compare against fixed accumulation and true large-batch training.",
        "",
    )

    assert any(
        "baseline" in item.lower() and "empty" in item.lower()
        for item in validate_plan_md(text)
    )


def test_validate_plan_md_fails_too_short() -> None:
    from researchclaw.pipeline.stage_impls._experiment_design import validate_plan_md

    text = """
## Hypotheses
H1.
## Baselines
B.
## Ablations
A.
## Metrics
M.
## Decision
D.
## Expected Outputs
O.
"""

    assert any("too short" in item.lower() for item in validate_plan_md(text))


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
