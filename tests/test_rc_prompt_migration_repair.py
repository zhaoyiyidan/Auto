"""Migration tests for experiment repair prompts."""

from __future__ import annotations

from researchclaw.pipeline.experiment_diagnosis import (
    Deficiency,
    DeficiencyType,
    ExperimentDiagnosis,
)
from researchclaw.pipeline.experiment_repair import build_repair_prompt
from researchclaw.prompts import PromptManager


def _diagnosis_with_time_guard() -> ExperimentDiagnosis:
    return ExperimentDiagnosis(
        deficiencies=[
            Deficiency(
                type=DeficiencyType.TIME_GUARD_DOMINANT,
                severity="major",
                description="Time guard killed 8/10 conditions",
                affected_conditions=["C3", "C4", "C5"],
                suggested_fix="Reduce conditions",
            )
        ],
        conditions_completed=["C1", "C2"],
        conditions_failed=["C3", "C4", "C5"],
        total_planned=10,
        completion_rate=0.2,
    )


def test_repair_catalog_entry_exists() -> None:
    pm = PromptManager()

    assert (
        pm.sub_prompt_meta("experiment_repair_instructions").prompt_id
        == "experiment_repair_instructions"
    )
    assert "EXPERIMENT DIAGNOSIS" in pm.block(
        "diagnosis_header",
        completion_rate="20%",
        completed_count="2",
        total_planned="10",
    )
    assert "SCOPE REDUCTION" in pm.block(
        "scope_reduction",
        n_planned="10",
        n_completed="2",
        time_budget_sec="2400",
        max_conditions="3",
    )


def test_repair_prompt_preserves_constraints() -> None:
    prompt = build_repair_prompt(
        diagnosis=_diagnosis_with_time_guard(),
        original_code={"main.py": "print('hello')"},
        time_budget_sec=2400,
    )

    assert "EXPERIMENT REPAIR TASK" in prompt
    assert "SCOPE REDUCTION" in prompt
    assert "CURRENT CODE" in prompt
    assert "Every condition MUST output" in prompt
    assert "WORKSPACE AGENT INSTRUCTIONS" in prompt
    assert "run_manifest.json" in prompt


def test_diagnosis_prompt_sources_header_from_catalog(monkeypatch) -> None:
    import researchclaw.pipeline.experiment_diagnosis as diagnosis_mod

    pm = PromptManager()
    pm._blocks["diagnosis_header"] = "SENTINEL DIAGNOSIS HEADER\n"  # type: ignore[attr-defined]
    monkeypatch.setattr(diagnosis_mod, "PromptManager", lambda: pm, raising=False)

    prompt = _diagnosis_with_time_guard().to_repair_prompt()

    assert "SENTINEL DIAGNOSIS HEADER" in prompt


def test_repair_helper_sources_instructions_from_catalog(monkeypatch) -> None:
    import researchclaw.pipeline.experiment_repair as repair_mod

    pm = PromptManager()
    pm._sub_prompts["experiment_repair_instructions"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL REPAIR SYSTEM",
        "user": "{diagnosis_prompt}\n{current_code}",
    }
    monkeypatch.setattr(repair_mod, "PromptManager", lambda: pm, raising=False)

    prompt = repair_mod.build_repair_prompt(
        diagnosis=_diagnosis_with_time_guard(),
        original_code={"main.py": "print('hello')"},
        time_budget_sec=2400,
    )

    assert "SENTINEL REPAIR SYSTEM" in prompt
