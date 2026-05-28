from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline.stage_impls._topic import (
    _execute_problem_decompose,
    _execute_topic_init,
)


class CapturingLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return LLMResponse(content=self.response, model="fake-model")


@pytest.fixture()
def rc_config(tmp_path: Path) -> RCConfig:
    data: dict[str, Any] = {
        "project": {"name": "topic-thinking-test", "mode": "docs-first"},
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


def test_topic_init_requests_thinking_stripping(
    tmp_path: Path, rc_config: RCConfig
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-01"
    stage_dir.mkdir(parents=True)
    llm = CapturingLLM("[thinking] hidden\n\n# Goal\nClean")

    _execute_topic_init(stage_dir, run_dir, rc_config, AdapterBundle(), llm=llm)

    assert llm.calls[0]["kwargs"]["strip_thinking"] is True


def test_problem_decompose_requests_thinking_stripping(
    tmp_path: Path, rc_config: RCConfig
) -> None:
    run_dir = tmp_path / "run"
    stage1 = run_dir / "stage-01"
    stage1.mkdir(parents=True)
    (stage1 / "goal.md").write_text("# Goal\nClean", encoding="utf-8")
    stage_dir = run_dir / "stage-02"
    stage_dir.mkdir()
    llm = CapturingLLM("[thinking] hidden\n\n# Problem Decomposition\nClean")

    _execute_problem_decompose(stage_dir, run_dir, rc_config, AdapterBundle(), llm=llm)

    assert llm.calls[0]["kwargs"]["strip_thinking"] is True
