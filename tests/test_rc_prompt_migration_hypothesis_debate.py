"""Migration tests for Stage 8 hypothesis debate prompts."""

from __future__ import annotations

from researchclaw.prompts import PromptManager


def test_hypothesis_debate_catalog_entries_exist() -> None:
    pm = PromptManager()

    assert pm.sub_prompt_meta("hypothesis_judge").prompt_id == "hypothesis_judge"
    assert (
        pm.sub_prompt_meta("hypothesis_synthesizer").prompt_id
        == "hypothesis_synthesizer"
    )


def test_judge_prompt_preserves_constraints() -> None:
    from researchclaw.pipeline.stage_impls._hypothesis_debate import (
        _build_judge_prompt,
    )

    messages, system = _build_judge_prompt("candidate claim", "synthesis", "topic")

    assert "candidate claim" in messages[0]["content"]
    assert "Return exactly this JSON shape" in messages[0]["content"]
    assert "verdict" in messages[0]["content"]
    assert "Do not rewrite" in system


def test_synthesizer_prompt_preserves_constraints() -> None:
    from researchclaw.pipeline.stage_impls._hypothesis_debate import (
        _build_synthesizer_prompt,
    )

    messages, system = _build_synthesizer_prompt(
        {"innovator": "claim one", "contrarian": "claim two"},
        "synthesis",
        "topic",
    )

    content = messages[0]["content"]
    assert "Do not mention agent names" in content
    assert "debate rounds" in content
    assert "Candidate Set 1" in content
    assert "Output only the hypotheses document" in system


def test_judge_helper_sources_from_catalog(monkeypatch) -> None:
    from researchclaw.pipeline.stage_impls import _hypothesis_debate

    pm = PromptManager()
    pm._sub_prompts["hypothesis_judge"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL JUDGE SYSTEM",
        "user": "{topic}\n{synthesis}\n{candidate_claim}",
    }
    monkeypatch.setattr(_hypothesis_debate, "PromptManager", lambda: pm)

    messages, system = _hypothesis_debate._build_judge_prompt(
        "candidate claim",
        "synthesis",
        "topic",
    )

    assert system == "SENTINEL JUDGE SYSTEM"
    assert "candidate claim" in messages[0]["content"]


def test_synthesizer_helper_sources_from_catalog(monkeypatch) -> None:
    from researchclaw.pipeline.stage_impls import _hypothesis_debate

    pm = PromptManager()
    pm._sub_prompts["hypothesis_synthesizer"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL SYNTH SYSTEM",
        "user": "{topic}\n{synthesis}\n{claims_text}",
    }
    monkeypatch.setattr(_hypothesis_debate, "PromptManager", lambda: pm)

    messages, system = _hypothesis_debate._build_synthesizer_prompt(
        {"innovator": "claim one"},
        "synthesis",
        "topic",
    )

    assert system == "SENTINEL SYNTH SYSTEM"
    assert "claim one" in messages[0]["content"]
