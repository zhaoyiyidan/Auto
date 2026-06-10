from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from researchclaw.config import LarkNotifyConfig, LarkTargetConfig, RCConfig
from researchclaw.notify.lark import LarkNotifyResult
from researchclaw.notify.pipeline import build_failure_message, notify_terminal_failure


@pytest.fixture()
def rc_config(tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "notify-pipeline-test", "mode": "docs-first"},
        "research": {"topic": "terminal failure alerts"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline",
        },
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def _target() -> LarkTargetConfig:
    return LarkTargetConfig(
        name="ops",
        receive_id_type="chat_id",
        receive_id="oc_ops",
    )


def _with_notifications(
    config: RCConfig,
    *,
    on_stage_fail: bool = True,
    lark_enabled: bool = True,
    dry_run: bool = True,
    targets: tuple[LarkTargetConfig, ...] | None = None,
) -> RCConfig:
    lark = LarkNotifyConfig(
        enabled=lark_enabled,
        app_id="app-id",
        app_secret="app-secret",
        targets=targets if targets is not None else (_target(),),
        dry_run=dry_run,
    )
    return replace(
        config,
        notifications=replace(
            config.notifications,
            on_stage_fail=on_stage_fail,
            lark=lark,
        ),
    )


def _arg_after(cmd: tuple[str, ...], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _text_from_result(result: LarkNotifyResult) -> str:
    data = json.loads(_arg_after(result.targets[0].command, "--data"))
    return json.loads(data["content"])["text"]


def test_disabled_when_on_stage_fail_false(rc_config: RCConfig, tmp_path: Path) -> None:
    config = _with_notifications(rc_config, on_stage_fail=False)

    with patch("researchclaw.notify.pipeline.LarkNotifier.from_rc_config") as factory:
        result = notify_terminal_failure(
            config=config,
            run_id="run-disabled",
            stage_name="SEARCH_STRATEGY",
            stage_num=3,
            error="boom",
            run_dir=tmp_path,
        )

    assert result is None
    factory.assert_not_called()


def test_noop_when_lark_disabled(rc_config: RCConfig, tmp_path: Path) -> None:
    config = _with_notifications(
        rc_config,
        on_stage_fail=True,
        lark_enabled=False,
        dry_run=False,
    )

    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        result = notify_terminal_failure(
            config=config,
            run_id="run-lark-disabled",
            stage_name="SEARCH_STRATEGY",
            stage_num=3,
            error="boom",
            run_dir=tmp_path,
        )

    mock_run.assert_not_called()
    assert isinstance(result, LarkNotifyResult)
    assert result.targets == ()


def test_sends_when_enabled(rc_config: RCConfig, tmp_path: Path) -> None:
    config = _with_notifications(rc_config, on_stage_fail=True, dry_run=True)

    result = notify_terminal_failure(
        config=config,
        run_id="run-send",
        stage_name="SEARCH_STRATEGY",
        stage_num=3,
        error="boom",
        run_dir=tmp_path,
    )

    assert isinstance(result, LarkNotifyResult)
    assert result.targets[0].status == "dry_run"


def test_message_contains_run_id_stage_error(
    rc_config: RCConfig,
    tmp_path: Path,
) -> None:
    config = _with_notifications(rc_config, on_stage_fail=True, dry_run=True)

    result = notify_terminal_failure(
        config=config,
        run_id="run-message",
        stage_name="SEARCH_STRATEGY",
        stage_num=3,
        error="LLM timeout",
        run_dir=tmp_path,
    )

    assert isinstance(result, LarkNotifyResult)
    text = _text_from_result(result)
    assert "run-message" in text
    assert "Stage 03" in text
    assert "SEARCH_STRATEGY" in text
    assert "LLM timeout" in text


def test_message_contains_run_dir(rc_config: RCConfig, tmp_path: Path) -> None:
    config = _with_notifications(rc_config, on_stage_fail=True, dry_run=True)
    run_dir = tmp_path / "run"

    result = notify_terminal_failure(
        config=config,
        run_id="run-dir",
        stage_name="SYNTHESIS",
        stage_num=7,
        error="failed",
        run_dir=run_dir,
    )

    assert isinstance(result, LarkNotifyResult)
    assert str(run_dir) in _text_from_result(result)


def test_best_effort_swallows_send_exception(
    rc_config: RCConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _with_notifications(rc_config, on_stage_fail=True, dry_run=True)

    with (
        patch(
            "researchclaw.notify.pipeline.LarkNotifier.send",
            side_effect=RuntimeError("send exploded"),
        ),
        caplog.at_level("WARNING", logger="researchclaw.notify.pipeline"),
    ):
        result = notify_terminal_failure(
            config=config,
            run_id="run-best-effort",
            stage_name="SEARCH_STRATEGY",
            stage_num=3,
            error="boom",
            run_dir=tmp_path,
        )

    assert result is None
    assert "Terminal-failure Lark notification failed" in caplog.text


def test_best_effort_swallows_construction_exception(
    rc_config: RCConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _with_notifications(rc_config, on_stage_fail=True, dry_run=True)

    with (
        patch(
            "researchclaw.notify.pipeline.LarkNotifier.from_rc_config",
            side_effect=RuntimeError("construction exploded"),
        ),
        caplog.at_level("WARNING", logger="researchclaw.notify.pipeline"),
    ):
        result = notify_terminal_failure(
            config=config,
            run_id="run-construction",
            stage_name="SEARCH_STRATEGY",
            stage_num=3,
            error="boom",
            run_dir=tmp_path,
        )

    assert result is None
    assert "Terminal-failure Lark notification failed" in caplog.text


def test_build_failure_message_shape(tmp_path: Path) -> None:
    title, body = build_failure_message(
        run_id="run-shape",
        stage_name="SYNTHESIS",
        stage_num=7,
        error=None,
        run_dir=tmp_path,
    )

    assert title == "ResearchClaw pipeline FAILED: run-shape"
    assert "Run: run-shape" in body
    assert "Stage 07: SYNTHESIS" in body
    assert "FAILED" in body
    assert "Error: unknown error" in body
    assert f"Run dir: {tmp_path}" in body
    assert f"researchclaw status {tmp_path}" in body
    assert "researchclaw run --resume" in body
    assert "Time (UTC):" in body


def test_returns_none_when_notifications_absent(tmp_path: Path) -> None:
    config = SimpleNamespace()

    result = notify_terminal_failure(
        config=config,
        run_id="run-no-notifications",
        stage_name="SEARCH_STRATEGY",
        stage_num=3,
        error="boom",
        run_dir=tmp_path,
    )

    assert result is None
