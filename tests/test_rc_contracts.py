import re

import pytest

from researchclaw.pipeline.contracts import CONTRACTS, StageContract
from researchclaw.pipeline.stages import GATE_STAGES, STAGE_SEQUENCE, Stage


def test_contracts_dict_has_exactly_23_entries():
    assert len(CONTRACTS) == 23


def test_every_stage_has_matching_contract_entry():
    assert set(CONTRACTS.keys()) == set(Stage)


@pytest.mark.parametrize("stage", STAGE_SEQUENCE)
def test_each_stage_member_resolves_to_stage_contract(stage: Stage):
    assert isinstance(CONTRACTS[stage], StageContract)


@pytest.mark.parametrize("stage,contract", tuple(CONTRACTS.items()))
def test_contract_stage_field_matches_dict_key(stage: Stage, contract: StageContract):
    assert contract.stage is stage


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_output_files_is_non_empty_for_all_contracts(contract: StageContract):
    assert contract.output_files


@pytest.mark.parametrize("stage,contract", tuple(CONTRACTS.items()))
def test_error_code_starts_with_e_and_contains_stage_number(
    stage: Stage, contract: StageContract
):
    assert contract.error_code.startswith("E")
    assert f"{int(stage):02d}" in contract.error_code
    assert re.match(r"^E\d{2}_[A-Z0-9_]+$", contract.error_code)


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_max_retries_is_non_negative_for_all_contracts(contract: StageContract):
    assert contract.max_retries >= 0


def test_gate_stages_have_expected_max_retries():
    assert CONTRACTS[Stage.LITERATURE_SCREEN].max_retries == 0
    assert CONTRACTS[Stage.EXPERIMENT_TASK_SPEC].max_retries == 0
    assert CONTRACTS[Stage.QUALITY_GATE].max_retries == 0


@pytest.mark.parametrize("stage", tuple(GATE_STAGES))
def test_gate_stage_contracts_are_never_retried(stage: Stage):
    assert CONTRACTS[stage].max_retries == 0


def test_topic_init_contract_has_expected_input_output_files():
    contract = CONTRACTS[Stage.TOPIC_INIT]

    assert contract.input_files == ()
    assert contract.output_files == ("goal.md", "hardware_profile.json")


def test_export_publish_contract_has_expected_outputs():
    contract = CONTRACTS[Stage.EXPORT_PUBLISH]

    assert contract.output_files == ("paper_final.md", "code/")


def test_workspace_native_stage_contracts_are_exact():
    expected = {
        Stage.EXPERIMENT_TASK_SPEC: (
            ("hypotheses.md",),
            ("plan.md", "expected_outputs.json"),
            "E09_PLAN_INVALID",
            0,
        ),
        Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR: (
            ("plan.md", "expected_outputs.json"),
            ("stage-10-workspace-agent-result.json", "run_manifest.json"),
            "E10_CODE_AGENT_FAIL",
            2,
        ),
        Stage.MANIFEST_VALIDATE_AND_PREPARE: (
            ("run_manifest.json",),
            ("manifest_validation.json", "run_manifest.json"),
            "E11_MANIFEST_INVALID",
            0,
        ),
        Stage.HARNESS_SUBMIT_AND_COLLECT: (
            ("manifest_validation.json",),
            ("execution_record.json", "submit_result.json", "result_artifacts.json"),
            "E12_HARNESS_FAIL",
            2,
        ),
        Stage.EXPERIMENT_ROUTE_DECISION: (
            ("execution_record.json",),
            ("experiment_decision.json",),
            "E13_ROUTE_FAIL",
            0,
        ),
        Stage.RESULT_ANALYSIS: (
            ("execution_record.json",),
            ("analysis.md", "experiment_summary.json", "provenance.json"),
            "E14_ANALYSIS_ERR",
            1,
        ),
    }
    for stage, (inputs, outputs, error_code, max_retries) in expected.items():
        contract = CONTRACTS[stage]
        assert contract.input_files == inputs
        assert contract.output_files == outputs
        assert contract.error_code == error_code
        assert contract.max_retries == max_retries


def test_stage9_outputs_plan_and_expected_outputs_only():
    contract = CONTRACTS[Stage.EXPERIMENT_TASK_SPEC]
    assert contract.output_files == ("plan.md", "expected_outputs.json")


def test_stage9_max_retries_still_zero():
    assert CONTRACTS[Stage.EXPERIMENT_TASK_SPEC].max_retries == 0


def test_stage10_input_contract_uses_plan_artifacts():
    stage10 = CONTRACTS[Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR]
    assert stage10.input_files == ("plan.md", "expected_outputs.json")


def test_stage_contract_has_no_collider_output_files_field():
    assert "collider_output_files" not in StageContract.__dataclass_fields__


def test_select_output_files_returns_contract_outputs_unconditionally():
    from researchclaw.pipeline.executor import _select_output_files

    contract = CONTRACTS[Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR]

    assert _select_output_files(contract, object()) == contract.output_files
    assert _select_output_files(None, object()) == ()


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_dod_is_non_empty_string_for_all_contracts(contract: StageContract):
    assert isinstance(contract.dod, str)
    assert contract.dod.strip()


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_input_files_is_tuple_of_strings(contract: StageContract):
    assert isinstance(contract.input_files, tuple)
    assert all(isinstance(path, str) and path for path in contract.input_files)


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_output_files_is_tuple_of_strings(contract: StageContract):
    assert isinstance(contract.output_files, tuple)
    assert all(isinstance(path, str) and path for path in contract.output_files)


def test_error_codes_are_unique_across_contracts():
    all_codes = [contract.error_code for contract in CONTRACTS.values()]
    assert len(all_codes) == len(set(all_codes))


def test_contracts_follow_stage_sequence_order():
    assert tuple(CONTRACTS.keys()) == STAGE_SEQUENCE


@pytest.mark.parametrize("stage", STAGE_SEQUENCE)
def test_contract_stage_int_matches_stage_enum_value(stage: Stage):
    assert int(CONTRACTS[stage].stage) == int(stage)
