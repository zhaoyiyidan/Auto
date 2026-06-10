"""Migration tests for the manual literature search-agent prompt."""

from __future__ import annotations

from researchclaw.prompts import PromptManager


def _kwargs() -> dict[str, object]:
    return {
        "topic": "retrieval augmented generation for code agents",
        "queries": ["retrieval augmented generation code agents"],
        "year_min": 2022,
        "plan_text": "queries:\n- retrieval augmented generation code agents",
        "goal_text": "# Goal\nStudy retrieval for code agents.",
        "problem_tree": "# Problems\n- literature coverage",
    }


def test_search_agent_catalog_entry_exists() -> None:
    pm = PromptManager()

    assert pm.sub_prompt_meta("search_agent").prompt_id == "search_agent"


def test_search_agent_prompt_preserves_constraints() -> None:
    from researchclaw.pipeline.stage_impls._literature import _render_search_agent_prompt

    prompt = _render_search_agent_prompt(**_kwargs())

    assert "Manual Literature Search Agent Prompt" in prompt
    assert "literature search agent" in prompt
    assert "Return only JSONL" in prompt
    assert "full_text_summary" in prompt
    assert "key_evidence" in prompt
    assert "Example JSONL Line" in prompt


def test_search_agent_helper_sources_from_catalog(monkeypatch) -> None:
    from researchclaw.pipeline.stage_impls import _literature

    pm = PromptManager()
    pm._sub_prompts["search_agent"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL SEARCH SYSTEM",
        "user": "{topic}\n{query_lines}\n{fields}\n{template}",
    }
    monkeypatch.setattr(_literature, "PromptManager", lambda: pm)

    prompt = _literature._render_search_agent_prompt(**_kwargs())

    assert "SENTINEL SEARCH SYSTEM" in prompt
    assert "retrieval augmented generation code agents" in prompt
