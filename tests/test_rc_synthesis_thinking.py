from __future__ import annotations

from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline.stage_impls._synthesis import _execute_synthesis


class CapturingLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return LLMResponse(content=self.response, model="fake-model")


def _config(tmp_path: Path) -> RCConfig:
    data: dict[str, Any] = {
        "project": {"name": "synthesis-thinking-test", "mode": "docs-first"},
        "research": {
            "topic": "retrieval augmented generation for code agents",
            "domains": ["ml"],
            "quality_threshold": 0.7,
        },
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
        },
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def test_synthesis_requests_thinking_stripping(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cards_dir = run_dir / "stage-06" / "cards"
    cards_dir.mkdir(parents=True)
    (cards_dir / "paper.md").write_text("# Paper\nUseful evidence", encoding="utf-8")
    stage_dir = run_dir / "stage-07"
    stage_dir.mkdir()
    llm = CapturingLLM("[thinking] hidden\n\n# Synthesis\nClean")

    _execute_synthesis(stage_dir, run_dir, _config(tmp_path), AdapterBundle(), llm=llm)

    assert llm.calls[0]["kwargs"]["strip_thinking"] is True
