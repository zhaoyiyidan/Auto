"""Tests for prompt audit capture."""

from __future__ import annotations

import json
from pathlib import Path

from researchclaw.config import RCConfig
from researchclaw.pipeline.executor import _prompt_audit_sink_for_run
from researchclaw.prompts import PromptManager, RenderedPrompt


def _valid_config_data(audit_capture: bool = False) -> dict[str, object]:
    return {
        "project": {"name": "demo", "mode": "docs-first"},
        "research": {"topic": "Test topic", "domains": ["ml"]},
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
        "prompts": {"audit_capture": audit_capture},
    }


def _topic_vars() -> dict[str, str]:
    return {
        "topic": "RL",
        "domains": "ml",
        "project_name": "test",
        "quality_threshold": "4.0",
    }


def test_audit_sink_invoked_per_for_stage() -> None:
    calls: list[tuple[str, RenderedPrompt]] = []
    pm = PromptManager(audit_sink=lambda prompt_id, prompt: calls.append((prompt_id, prompt)))

    sp = pm.for_stage("topic_init", **_topic_vars())

    assert calls == [("topic_init", sp)]


def test_audit_disabled_writes_nothing(tmp_path: Path) -> None:
    config = RCConfig.from_dict(
        _valid_config_data(audit_capture=False),
        project_root=tmp_path,
        check_paths=False,
    )

    sink = _prompt_audit_sink_for_run(config, tmp_path)

    assert config.prompts.audit_capture is False
    assert sink is None
    assert not (tmp_path / "prompts").exists()


def test_audit_enabled_writes_one_file_per_stage(tmp_path: Path) -> None:
    config = RCConfig.from_dict(
        _valid_config_data(audit_capture=True),
        project_root=tmp_path,
        check_paths=False,
    )
    sink = _prompt_audit_sink_for_run(config, tmp_path)
    assert sink is not None

    sp = PromptManager(audit_sink=sink).for_stage("topic_init", **_topic_vars())

    prompt_files = sorted((tmp_path / "prompts").glob("*.json"))
    assert [p.name for p in prompt_files] == ["topic_init.json"]
    data = json.loads(prompt_files[0].read_text(encoding="utf-8"))
    assert data == {
        "prompt_id": "topic_init",
        "version": "1.0.0",
        "system": sp.system,
        "user": sp.user,
        "json_mode": False,
        "max_tokens": None,
    }
