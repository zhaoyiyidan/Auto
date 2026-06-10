"""Migration tests for citation relevance prompts."""

from __future__ import annotations

import json
from types import SimpleNamespace

from researchclaw.llm.client import LLMResponse
from researchclaw.prompts import PromptManager


class RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, **kwargs})
        return LLMResponse(
            content=json.dumps({"smith2020": 0.9, "jones2019": 0.2}),
            model="fake",
        )


def _citations() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(cite_key="smith2020", title="Relevant Retrieval Paper"),
        SimpleNamespace(cite_key="jones2019", title="Unrelated System Paper"),
    ]


def test_citation_relevance_catalog_entry_exists() -> None:
    pm = PromptManager()

    assert pm.sub_prompt_meta("citation_relevance").prompt_id == "citation_relevance"


def test_citation_relevance_prompt_preserves_constraints() -> None:
    from researchclaw.pipeline.stage_impls._review_publish import _check_citation_relevance

    llm = RecordingLLM()
    scores = _check_citation_relevance(
        llm,
        "retrieval augmented agents",
        _citations(),
    )

    call = llm.calls[0]
    prompt = call["messages"][0]["content"]  # type: ignore[index]
    assert "Research topic: retrieval augmented agents" in prompt
    assert "Rate the relevance of each citation" in prompt
    assert "Return ONLY a JSON object" in prompt
    assert "[smith2020]" in prompt
    assert "You assess citation relevance" in str(call["system"])
    assert call["json_mode"] is True
    assert scores == {"smith2020": 0.9, "jones2019": 0.2}


def test_citation_relevance_sources_from_catalog(monkeypatch) -> None:
    from researchclaw.pipeline.stage_impls import _review_publish

    pm = PromptManager()
    pm._sub_prompts["citation_relevance"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL CITATION SYSTEM",
        "user": "SENTINEL CITATION {topic}\n{citations_text}",
        "json_mode": True,
    }
    monkeypatch.setattr(_review_publish, "PromptManager", lambda: pm)

    llm = RecordingLLM()
    _review_publish._check_citation_relevance(
        llm,
        "retrieval augmented agents",
        _citations(),
    )

    call = llm.calls[0]
    assert "SENTINEL CITATION SYSTEM" in str(call["system"])
    assert "SENTINEL CITATION retrieval augmented agents" in call["messages"][0]["content"]  # type: ignore[index]
    assert "[smith2020]" in call["messages"][0]["content"]  # type: ignore[index]
