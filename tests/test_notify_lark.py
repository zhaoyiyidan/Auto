from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from researchclaw.config import LarkNotifyConfig, LarkTargetConfig, RCConfig
from researchclaw.notify.lark import (
    LarkNotifier,
    LarkNotifyResult,
    LarkTargetResult,
)


def _target(
    name: str = "me",
    *,
    kind: str = "user",
    receive_id_type: str = "open_id",
    receive_id: str = "ou_xxx",
) -> LarkTargetConfig:
    return LarkTargetConfig(
        name=name,
        kind=kind,
        receive_id_type=receive_id_type,
        receive_id=receive_id,
    )


def _config(
    *,
    enabled: bool = True,
    targets: tuple[LarkTargetConfig, ...] | None = None,
    command: str = "lark-cli",
    app_id: str = "config-app-id",
    app_secret: str = "config-app-secret",
    app_id_env: str = "LARK_APP_ID",
    app_secret_env: str = "LARK_APP_SECRET",
    timeout_sec: int = 15,
    dry_run: bool = False,
) -> LarkNotifyConfig:
    return LarkNotifyConfig(
        enabled=enabled,
        command=command,
        app_id=app_id,
        app_secret=app_secret,
        app_id_env=app_id_env,
        app_secret_env=app_secret_env,
        targets=targets if targets is not None else (_target(),),
        timeout_sec=timeout_sec,
        dry_run=dry_run,
    )


def _completed(
    *,
    returncode: int = 0,
    stdout: str = '{"code":0,"msg":"success"}',
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _expected_command(
    target: LarkTargetConfig,
    title: str = "Title",
    body: str = "Body",
    *,
    command: str = "lark-cli",
) -> list[str]:
    text = f"{title}\n\n{body}" if title else body
    content_obj = {"text": text}
    data_obj = {
        "receive_id": target.receive_id,
        "msg_type": "text",
        "content": json.dumps(content_obj, ensure_ascii=False),
    }
    return [
        command,
        "api",
        "POST",
        "/open-apis/im/v1/messages",
        "--params",
        json.dumps(
            {"receive_id_type": target.receive_id_type},
            ensure_ascii=False,
        ),
        "--data",
        json.dumps(data_obj, ensure_ascii=False),
        "--format",
        "json",
    ]


def _command_from_mock(mock_run) -> list[str]:
    return list(mock_run.call_args.args[0])


def _arg_after(cmd: list[str] | tuple[str, ...], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _content_from_command(cmd: list[str] | tuple[str, ...]) -> dict[str, str]:
    data = json.loads(_arg_after(cmd, "--data"))
    return json.loads(data["content"])


def test_send_success_invokes_subprocess_once():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        result = LarkNotifier(_config()).send("Title", "Body")

    assert mock_run.call_count == 1
    assert result.targets[0].status == "ok"


def test_send_builds_exact_argv():
    target = _target(receive_id_type="chat_id", receive_id="oc_xxx")
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(_config(targets=(target,))).send("Title", "Body")

    cmd = _command_from_mock(mock_run)
    assert cmd == _expected_command(target)
    assert "--as" not in cmd


def test_content_is_double_json_encoded():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(_config()).send("Title", "Body")

    data = json.loads(_arg_after(_command_from_mock(mock_run), "--data"))
    assert isinstance(data["content"], str)
    assert json.loads(data["content"]) == {"text": "Title\n\nBody"}


def test_params_carries_receive_id_type():
    target = _target(receive_id_type="chat_id", receive_id="oc_xxx")
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(_config(targets=(target,))).send("Title", "Body")

    params = json.loads(_arg_after(_command_from_mock(mock_run), "--params"))
    assert params == {"receive_id_type": "chat_id"}


def test_data_carries_receive_id_and_msg_type():
    target = _target(receive_id="ou_custom")
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(_config(targets=(target,))).send("Title", "Body")

    data = json.loads(_arg_after(_command_from_mock(mock_run), "--data"))
    assert data["receive_id"] == "ou_custom"
    assert data["msg_type"] == "text"


def test_title_and_body_composed_into_text():
    result = LarkNotifier(_config(dry_run=True)).send("Needs review", "Stage 9")
    assert _content_from_command(result.targets[0].command)["text"] == (
        "Needs review\n\nStage 9"
    )

    bare = LarkNotifier(_config(dry_run=True)).send("", "Only body")
    assert _content_from_command(bare.targets[0].command)["text"] == "Only body"


def test_multiple_targets_each_invoked_once():
    targets = (
        _target("me", receive_id="ou_1"),
        _target("group", receive_id_type="chat_id", receive_id="oc_1"),
        _target("ops", receive_id_type="chat_id", receive_id="oc_2"),
    )
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        result = LarkNotifier(_config(targets=targets)).send("Title", "Body")

    assert mock_run.call_count == 3
    receive_ids = [
        json.loads(_arg_after(call.args[0], "--data"))["receive_id"]
        for call in mock_run.call_args_list
    ]
    assert receive_ids == ["ou_1", "oc_1", "oc_2"]
    assert [target.status for target in result.targets] == ["ok", "ok", "ok"]


def test_one_target_failure_does_not_interrupt_others():
    targets = (
        _target("first", receive_id="ou_1"),
        _target("second", receive_id="ou_2"),
        _target("third", receive_id="ou_3"),
    )
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.side_effect = [_completed(), OSError("boom"), _completed()]

        result = LarkNotifier(_config(targets=targets)).send("Title", "Body")

    assert mock_run.call_count == 3
    assert [target.status for target in result.targets] == ["ok", "error", "ok"]


def test_nonzero_returncode_captured_as_error():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed(returncode=1, stderr="cli failed")

        result = LarkNotifier(_config()).send("Title", "Body")

    assert result.targets[0].status == "error"
    assert "cli failed" in result.targets[0].detail


def test_api_business_error_in_stdout_marked_error():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed(
            stdout='{"code":99991663,"msg":"receive id invalid"}'
        )

        result = LarkNotifier(_config()).send("Title", "Body")

    assert result.targets[0].status == "error"
    assert "99991663" in result.targets[0].detail


def test_timeout_expired_handled():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("lark-cli", timeout=15)

        result = LarkNotifier(_config()).send("Title", "Body")

    assert result.targets[0].status == "error"


def test_filenotfound_missing_binary_handled():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("missing")

        result = LarkNotifier(_config()).send("Title", "Body")

    assert result.targets[0].status == "error"


def test_oserror_handled():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.side_effect = OSError("os error")

        result = LarkNotifier(_config()).send("Title", "Body")

    assert result.targets[0].status == "error"


def test_generic_exception_handled():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.side_effect = Exception("unexpected")

        result = LarkNotifier(_config()).send("Title", "Body")

    assert result.targets[0].status == "error"


def test_dry_run_makes_no_subprocess_call():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        result = LarkNotifier(_config(dry_run=True)).send("Title", "Body")

    mock_run.assert_not_called()
    assert result.ok is True


def test_dry_run_returns_dry_run_status_per_target():
    targets = (_target("first"), _target("second", receive_id="ou_2"))

    result = LarkNotifier(_config(targets=targets, dry_run=True)).send("Title", "Body")

    assert [target.status for target in result.targets] == ["dry_run", "dry_run"]


def test_dry_run_still_builds_command():
    target = _target(receive_id_type="chat_id", receive_id="oc_xxx")

    result = LarkNotifier(_config(targets=(target,), dry_run=True)).send(
        "Title",
        "Body",
    )

    assert result.targets[0].command == tuple(_expected_command(target))


def test_disabled_is_noop():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        result = LarkNotifier(_config(enabled=False)).send("Title", "Body")

    mock_run.assert_not_called()
    assert result.targets == ()
    assert result.ok is True


def test_empty_targets_is_noop():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        result = LarkNotifier(_config(targets=())).send("Title", "Body")

    mock_run.assert_not_called()
    assert result.targets == ()
    assert result.ok is True


def test_env_var_overrides_config_plaintext(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LARK_APP_ID", "env-app-id")
    monkeypatch.setenv("LARK_APP_SECRET", "env-app-secret")
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(
            _config(app_id="config-id", app_secret="config-secret")
        ).send("Title", "Body")

    env = mock_run.call_args.kwargs["env"]
    assert env["LARK_APP_ID"] == "env-app-id"
    assert env["LARK_APP_SECRET"] == "env-app-secret"


def test_config_plaintext_used_when_env_absent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(
            _config(app_id="config-id", app_secret="config-secret")
        ).send("Title", "Body")

    env = mock_run.call_args.kwargs["env"]
    assert env["LARK_APP_ID"] == "config-id"
    assert env["LARK_APP_SECRET"] == "config-secret"


def test_resolved_creds_injected_into_subprocess_env_kwarg():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(
            _config(app_id="config-id", app_secret="config-secret")
        ).send("Title", "Body")

    env = mock_run.call_args.kwargs["env"]
    assert env["LARK_APP_ID"] == "config-id"
    assert env["LARK_APP_SECRET"] == "config-secret"


def test_custom_env_names_also_injected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUSTOM_LARK_ID", "custom-id")
    monkeypatch.setenv("CUSTOM_LARK_SECRET", "custom-secret")
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(
            _config(
                app_id_env="CUSTOM_LARK_ID",
                app_secret_env="CUSTOM_LARK_SECRET",
            )
        ).send("Title", "Body")

    env = mock_run.call_args.kwargs["env"]
    assert env["LARK_APP_ID"] == "custom-id"
    assert env["LARK_APP_SECRET"] == "custom-secret"
    assert env["CUSTOM_LARK_ID"] == "custom-id"
    assert env["CUSTOM_LARK_SECRET"] == "custom-secret"


def test_partial_credentials_not_injected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(_config(app_id="config-id", app_secret="")).send(
            "Title",
            "Body",
        )

    assert "env" not in mock_run.call_args.kwargs


def test_no_credentials_inherits_parent_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(_config(app_id="", app_secret="")).send("Title", "Body")

    assert "env" not in mock_run.call_args.kwargs


def test_secret_not_present_in_argv():
    secret = "super-secret-value"
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(_config(app_secret=secret)).send("Title", "Body")

    assert all(secret not in part for part in _command_from_mock(mock_run))


def test_secret_not_logged(caplog: pytest.LogCaptureFixture):
    secret = "logged-secret-value"
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed(returncode=1, stderr="failed")

        with caplog.at_level("DEBUG", logger="researchclaw.notify.lark"):
            LarkNotifier(_config(app_secret=secret)).send("Title", "Body")

    assert secret not in caplog.text


def test_timeout_sec_forwarded_to_subprocess():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(_config(timeout_sec=7)).send("Title", "Body")

    assert mock_run.call_args.kwargs["timeout"] == 7


def test_subprocess_called_with_repo_idiom_kwargs():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkNotifier(_config()).send("Title", "Body")

    kwargs = mock_run.call_args.kwargs
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["check"] is False


def test_empty_receive_id_handling():
    targets = (
        _target("empty", receive_id=""),
        _target("present", receive_id="ou_present"),
    )
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        result = LarkNotifier(_config(targets=targets)).send("Title", "Body")

    assert mock_run.call_count == 1
    assert [target.status for target in result.targets] == ["skipped", "ok"]
    assert "receive_id" in result.targets[0].detail


def test_chinese_utf8_text_round_trips():
    result = LarkNotifier(_config(dry_run=True)).send("阶段提醒", "需要确认 ✅")

    cmd = result.targets[0].command
    data_arg = _arg_after(cmd, "--data")
    assert "\\u" not in data_arg
    assert _content_from_command(cmd)["text"] == "阶段提醒\n\n需要确认 ✅"


def test_result_object_shape_and_aggregation():
    target = _target("first")

    result = LarkNotifier(_config(targets=(target,), dry_run=True)).send(
        "Title",
        "Body",
    )

    assert isinstance(result, LarkNotifyResult)
    assert isinstance(result.targets[0], LarkTargetResult)
    assert result.targets[0].name == "first"
    assert result.targets[0].status == "dry_run"
    assert result.ok is True


def test_result_aggregate_ok_false_when_any_error():
    targets = (_target("good", receive_id="ou_1"), _target("bad", receive_id="ou_2"))
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.side_effect = [_completed(), OSError("boom")]

        result = LarkNotifier(_config(targets=targets)).send("Title", "Body")

    assert result.ok is False


def test_from_rc_config_builds_notifier():
    data = _valid_rc_data()
    data["notifications"]["lark"] = {
        "enabled": True,
        "targets": {
            "me": {
                "receive_id_type": "open_id",
                "receive_id": "ou_xxx",
            },
        },
    }
    rc_config = RCConfig.from_dict(data, check_paths=False)
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        result = LarkNotifier.from_rc_config(rc_config).send("Title", "Body")

    assert result.targets[0].status == "ok"


def test_from_rc_config_disabled_when_lark_absent():
    rc_config = RCConfig.from_dict(_valid_rc_data(), check_paths=False)
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        result = LarkNotifier.from_rc_config(rc_config).send("Title", "Body")

    mock_run.assert_not_called()
    assert result.targets == ()
    assert result.ok is True


def _valid_rc_data() -> dict[str, dict[str, object]]:
    return {
        "project": {"name": "demo", "mode": "docs-first"},
        "research": {"topic": "Test topic", "domains": ["ml", "agents"]},
        "runtime": {"timezone": "America/New_York"},
        "notifications": {"channel": "discord"},
        "knowledge_base": {"backend": "markdown", "root": "docs/kb"},
        "openclaw_bridge": {
            "use_cron": True,
            "use_message": True,
            "use_memory": True,
            "use_sessions_spawn": True,
            "use_web_fetch": True,
            "use_browser": False,
        },
        "llm": {
            "provider": "openai-compatible",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "OPENAI_API_KEY",
            "primary_model": "gpt-4.1",
            "fallback_models": ["gpt-4o-mini", "gpt-4o"],
        },
        "security": {"hitl_required_stages": [5, 9, 20]},
        "experiment": {
            "mode": "simulated",
            "metric_direction": "minimize",
        },
    }
