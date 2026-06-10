"""Migration tests for compiled paper review prompts."""

from __future__ import annotations

import json
from pathlib import Path

from researchclaw.llm.client import LLMResponse
from researchclaw.prompts import PromptManager


class RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, **kwargs})
        return LLMResponse(
            content=json.dumps(
                {
                    "soundness": 6,
                    "presentation": 7,
                    "contribution": 6,
                    "originality": 6,
                    "clarity": 7,
                    "significance": 6,
                    "reproducibility": 7,
                    "overall_score": 6,
                    "confidence": 8,
                    "decision": "reject",
                    "strengths": ["clear setup"],
                    "weaknesses": ["limited evidence"],
                    "critical_issues": ["missing baseline"],
                    "minor_issues": [],
                    "summary": "Needs revision.",
                }
            ),
            model="fake",
        )


def _paper_files(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    pdf_path.with_suffix(".tex").write_text(
        "\\section{Method} The method is described.",
        encoding="utf-8",
    )
    return pdf_path


def test_compiled_pdf_review_catalog_entry_exists() -> None:
    pm = PromptManager()

    assert pm.sub_prompt_meta("compiled_pdf_review").prompt_id == "compiled_pdf_review"


def test_compiled_pdf_review_prompt_preserves_constraints(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._paper_writing import _review_compiled_pdf

    llm = RecordingLLM()
    result = _review_compiled_pdf(_paper_files(tmp_path), llm, "retrieval augmented agents")

    call = llm.calls[0]
    prompt = call["messages"][0]["content"]  # type: ignore[index]
    assert "senior Area Chair" in prompt
    assert "PAPER TOPIC: retrieval augmented agents" in prompt
    assert "DIMENSIONS" in prompt
    assert "overall_score" in prompt
    assert "critical academic reviewer" in str(call["system"])
    assert result["mean_score"] == 6.43


def test_compiled_pdf_review_sources_from_catalog(tmp_path: Path, monkeypatch) -> None:
    from researchclaw.pipeline.stage_impls import _paper_writing

    pm = PromptManager()
    pm._sub_prompts["compiled_pdf_review"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL PDF REVIEW SYSTEM",
        "user": "SENTINEL PDF REVIEW {topic} {tex_content}",
        "json_mode": True,
    }
    monkeypatch.setattr(_paper_writing, "PromptManager", lambda: pm)

    llm = RecordingLLM()
    _paper_writing._review_compiled_pdf(
        _paper_files(tmp_path),
        llm,
        "retrieval augmented agents",
    )

    call = llm.calls[0]
    assert "SENTINEL PDF REVIEW SYSTEM" in str(call["system"])
    assert "SENTINEL PDF REVIEW retrieval augmented agents" in call["messages"][0]["content"]  # type: ignore[index]
    assert "\\section{Method}" in call["messages"][0]["content"]  # type: ignore[index]
    assert call["json_mode"] is True
