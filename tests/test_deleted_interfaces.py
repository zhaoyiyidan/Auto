from __future__ import annotations

import pytest

from researchclaw.pipeline.contracts import CONTRACTS
from researchclaw.pipeline.stages import Stage


def test_task_spec_class_deleted() -> None:
    with pytest.raises(ImportError):
        from researchclaw.experiment.workspace import TaskSpec  # noqa: F401


def test_experiment_protocol_class_deleted() -> None:
    with pytest.raises(ImportError):
        from researchclaw.experiment.protocol import ExperimentProtocol  # noqa: F401


def test_stage9_contract_has_new_outputs() -> None:
    contract = CONTRACTS[Stage.EXPERIMENT_TASK_SPEC]

    assert contract.output_files == ("plan.md", "expected_outputs.json")
    assert "task_spec.yaml" not in contract.output_files
    assert "experiment_protocol.json" not in contract.output_files
    assert "experiment_design_intent.md" not in contract.output_files


def test_stage10_contract_consumes_plan_and_expected_outputs() -> None:
    contract = CONTRACTS[Stage.CODE_AGENT_IMPLEMENT_OR_REPAIR]

    assert contract.input_files == ("plan.md", "expected_outputs.json")
    assert "task_spec.yaml" not in contract.input_files
