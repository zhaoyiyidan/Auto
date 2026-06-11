"""Migration tests for paper continuation prompts."""

from __future__ import annotations

from researchclaw.llm.client import LLMResponse
from researchclaw.prompts import PromptManager


class RecordingLLM:
    def __init__(self) -> None:
        self.user_prompts: list[str] = []
        self.systems: list[str] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.systems.append(str(kwargs.get("system", "")))
        self.user_prompts.append(messages[0]["content"])
        idx = len(self.user_prompts) - 1
        parts = [
            "## Title\nPaper\n\n## Abstract\nSummary.",
            "## Method\nMethod body.\n\n## Experiments\nExperiments body.",
            "## Results\nResults body.\n\n## Conclusion\nConclusion body.",
        ]
        return LLMResponse(content=parts[min(idx, len(parts) - 1)], model="fake")


def _write_sections(*, pm: PromptManager, is_hep: bool = False) -> RecordingLLM:
    from researchclaw.pipeline.stage_impls._paper_writing import _write_paper_sections

    llm = RecordingLLM()
    _write_paper_sections(
        llm=llm,
        pm=pm,
        preamble="Preamble",
        topic_constraint="Topic constraint\n",
        exp_metrics_instruction="Experiment metrics",
        citation_instruction="Citation instruction",
        outline="Outline",
        is_hep=is_hep,
    )
    return llm


def test_paper_continuation_catalog_entries_exist() -> None:
    pm = PromptManager()

    assert pm.sub_prompt_meta("paper_continuation_method").prompt_id == "paper_continuation_method"
    assert pm.sub_prompt_meta("paper_continuation_results").prompt_id == "paper_continuation_results"
    assert (
        pm.sub_prompt_meta("paper_continuation_hep_method").prompt_id
        == "paper_continuation_hep_method"
    )
    assert (
        pm.sub_prompt_meta("paper_continuation_hep_results").prompt_id
        == "paper_continuation_hep_results"
    )


def test_paper_continuation_prompts_preserve_constraints() -> None:
    llm = _write_sections(pm=PromptManager())

    assert "You are continuing a paper" in llm.user_prompts[1]
    assert "5. **Method**" in llm.user_prompts[1]
    assert "6. **Experiments**" in llm.user_prompts[1]
    assert "You are completing a paper" in llm.user_prompts[2]
    assert "7. **Results**" in llm.user_prompts[2]
    assert "Do NOT include a References section" in llm.user_prompts[2]


def test_hep_paper_continuation_prompts_preserve_constraints() -> None:
    llm = _write_sections(pm=PromptManager(), is_hep=True)

    assert "You are continuing an HEP phenomenology paper" in llm.user_prompts[1]
    assert "4. **Model / Theoretical framework**" in llm.user_prompts[1]
    assert "5. **Phenomenology / Computational setup**" in llm.user_prompts[1]
    assert "You are completing an HEP phenomenology paper" in llm.user_prompts[2]
    assert "6. **Results**" in llm.user_prompts[2]
    assert "95% CL exclusion contours" in llm.user_prompts[2]


def test_paper_continuation_helpers_source_from_catalog() -> None:
    pm = PromptManager()
    pm._sub_prompts["paper_continuation_method"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL METHOD SYSTEM",
        "user": "SENTINEL METHOD {previous_sections} {outline}",
    }
    pm._sub_prompts["paper_continuation_results"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL RESULTS SYSTEM",
        "user": "SENTINEL RESULTS {previous_sections} {outline}",
    }

    llm = _write_sections(pm=pm)

    assert "SENTINEL METHOD" in llm.user_prompts[1]
    assert "SENTINEL RESULTS" in llm.user_prompts[2]
    assert "SENTINEL METHOD SYSTEM" in llm.systems[1]
    assert "SENTINEL RESULTS SYSTEM" in llm.systems[2]


def test_hep_paper_continuation_helpers_source_from_catalog() -> None:
    pm = PromptManager()
    pm._sub_prompts["paper_continuation_hep_method"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL HEP METHOD SYSTEM",
        "user": "SENTINEL HEP METHOD {previous_sections} {outline}",
    }
    pm._sub_prompts["paper_continuation_hep_results"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL HEP RESULTS SYSTEM",
        "user": "SENTINEL HEP RESULTS {previous_sections} {outline}",
    }

    llm = _write_sections(pm=pm, is_hep=True)

    assert "SENTINEL HEP METHOD" in llm.user_prompts[1]
    assert "SENTINEL HEP RESULTS" in llm.user_prompts[2]
    assert "SENTINEL HEP METHOD SYSTEM" in llm.systems[1]
    assert "SENTINEL HEP RESULTS SYSTEM" in llm.systems[2]
