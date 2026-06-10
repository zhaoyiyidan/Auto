"""Migration tests for framework diagram prompt generation."""

from __future__ import annotations

from pathlib import Path

from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMResponse
from researchclaw.prompts import PromptManager


class RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, **kwargs})
        return LLMResponse(
            content=(
                "Create a detailed left-to-right framework diagram with input data, "
                "retrieval memory, policy optimizer, and evaluation blocks."
            ),
            model="fake",
        )


def _config(tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "demo", "mode": "docs-first"},
        "research": {"topic": "retrieval augmented reinforcement learning", "domains": ["ml"]},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "none"},
        "knowledge_base": {"backend": "markdown", "root": "docs/kb"},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "OPENAI_API_KEY",
            "primary_model": "fake-model",
        },
        "security": {"hitl_required_stages": [5, 9, 20]},
        "experiment": {"mode": "workspace"},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def _paper_text() -> str:
    return (
        "## Title\nRetrieval Memory for Reinforcement Learning\n\n"
        "## Abstract\nWe study memory-augmented policy optimization.\n\n"
        "## Method\nThe approach uses an encoder, retrieval memory, policy decoder, "
        "training objective, and evaluation loop."
    )


def test_framework_diagram_catalog_entry_exists() -> None:
    pm = PromptManager()

    assert pm.sub_prompt_meta("framework_diagram_prompt").prompt_id == "framework_diagram_prompt"


def test_framework_diagram_prompt_preserves_constraints(tmp_path: Path) -> None:
    from researchclaw.pipeline._helpers import _generate_framework_diagram_prompt

    llm = RecordingLLM()
    result = _generate_framework_diagram_prompt(_paper_text(), _config(tmp_path), llm=llm)

    call = llm.calls[0]
    system = str(call["system"])
    user = call["messages"][0]["content"]  # type: ignore[index]
    assert "expert academic figure designer" in system
    assert "Output ONLY the prompt text" in system
    assert "Research topic: retrieval augmented reinforcement learning" in user
    assert "Method section excerpt" in user
    assert "Generate a detailed text-to-image prompt" in user
    assert "## Image Generation Prompt" in result


def test_framework_diagram_sources_from_catalog(tmp_path: Path, monkeypatch) -> None:
    from researchclaw.pipeline import _helpers

    pm = PromptManager()
    pm._sub_prompts["framework_diagram_prompt"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL FIGURE SYSTEM",
        "user": "SENTINEL FIGURE {title} {topic}\n{method_section}",
    }
    monkeypatch.setattr(_helpers, "PromptManager", lambda: pm)

    llm = RecordingLLM()
    _helpers._generate_framework_diagram_prompt(_paper_text(), _config(tmp_path), llm=llm)

    call = llm.calls[0]
    assert "SENTINEL FIGURE SYSTEM" in str(call["system"])
    assert "SENTINEL FIGURE" in call["messages"][0]["content"]  # type: ignore[index]
    assert "retrieval memory" in call["messages"][0]["content"]  # type: ignore[index]
