"""Migration tests for the topic quality evaluation prompt."""

from __future__ import annotations

import json
from pathlib import Path

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.prompts import PromptManager


class RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, **kwargs})
        from researchclaw.llm.client import LLMResponse

        if len(self.calls) == 1:
            return LLMResponse(content="# Problem tree", model="fake")
        return LLMResponse(
            content=json.dumps(
                {
                    "novelty": 8,
                    "specificity": 8,
                    "feasibility": 8,
                    "overall": 8,
                    "suggestion": "",
                }
            ),
            model="fake",
        )


def _config(tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "demo", "mode": "docs-first"},
        "research": {"topic": "retrieval augmented generation", "domains": ["ml"]},
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


def test_topic_quality_eval_catalog_entry_exists() -> None:
    pm = PromptManager()

    assert pm.sub_prompt_meta("topic_quality_eval").prompt_id == "topic_quality_eval"


def test_topic_quality_eval_sources_from_catalog(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._topic import _execute_problem_decompose

    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-02"
    (run_dir / "stage-01").mkdir(parents=True)
    stage_dir.mkdir(parents=True)
    (run_dir / "stage-01" / "goal.md").write_text("# Goal", encoding="utf-8")

    pm = PromptManager()
    pm._sub_prompts["topic_quality_eval"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL TOPIC SYSTEM {domain_label}",
        "user": "SENTINEL TOPIC USER {topic}",
        "json_mode": True,
    }
    llm = RecordingLLM()

    _execute_problem_decompose(
        stage_dir,
        run_dir,
        _config(tmp_path),
        AdapterBundle(),
        llm=llm,
        prompts=pm,
    )

    assert len(llm.calls) == 2
    assert "SENTINEL TOPIC SYSTEM" in str(llm.calls[1]["system"])
    assert "SENTINEL TOPIC USER retrieval augmented generation" in llm.calls[1]["messages"][0]["content"]  # type: ignore[index]
