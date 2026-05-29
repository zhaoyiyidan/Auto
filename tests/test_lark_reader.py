from __future__ import annotations

import json
import subprocess
from datetime import datetime
from unittest.mock import patch

import pytest

from researchclaw.config import LarkNotifyConfig
from researchclaw.notify.lark import LarkMessageReader


def _config(
    *,
    command: str = "lark-cli",
    app_id: str = "config-app-id",
    app_secret: str = "config-app-secret",
    app_id_env: str = "LARK_APP_ID",
    app_secret_env: str = "LARK_APP_SECRET",
    timeout_sec: int = 15,
) -> LarkNotifyConfig:
    return LarkNotifyConfig(
        enabled=True,
        command=command,
        app_id=app_id,
        app_secret=app_secret,
        app_id_env=app_id_env,
        app_secret_env=app_secret_env,
        timeout_sec=timeout_sec,
    )


def _completed(
    *,
    returncode: int = 0,
    stdout: str = '{"code":0,"data":{"items":[],"has_more":false}}',
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _command_from_mock(mock_run) -> list[str]:
    return list(mock_run.call_args.args[0])


def _arg_after(cmd: list[str] | tuple[str, ...], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _params_from_command(cmd: list[str] | tuple[str, ...]) -> dict[str, str]:
    return json.loads(_arg_after(cmd, "--params"))


def test_builds_exact_get_argv():
    since_iso = "1970-01-01T00:00:05+00:00"
    expected_params = {
        "container_id_type": "chat",
        "container_id": "oc_abc123",
        "start_time": "5",
        "sort_type": "ByCreateTimeAsc",
        "page_size": "50",
    }
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkMessageReader(_config(command="custom-lark")).list_messages(
            chat_id="oc_abc123",
            since_iso=since_iso,
        )

    assert _command_from_mock(mock_run) == [
        "custom-lark",
        "api",
        "GET",
        "/open-apis/im/v1/messages",
        "--params",
        json.dumps(expected_params, ensure_ascii=False),
        "--format",
        "json",
    ]


def test_start_time_from_since_iso():
    since_iso = "2026-05-29T12:00:30+00:00"
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkMessageReader(_config()).list_messages(
            chat_id="oc_abc123",
            since_iso=since_iso,
        )

    params = _params_from_command(_command_from_mock(mock_run))
    assert params["start_time"] == str(int(datetime.fromisoformat(since_iso).timestamp()))


def test_env_injected_via_build_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LARK_APP_ID", "env-app-id")
    monkeypatch.setenv("LARK_APP_SECRET", "env-app-secret")
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkMessageReader(
            _config(app_id="config-id", app_secret="config-secret")
        ).list_messages(chat_id="oc_abc123", since_iso="1970-01-01T00:00:00+00:00")

    env = mock_run.call_args.kwargs["env"]
    assert env["LARK_APP_ID"] == "env-app-id"
    assert env["LARK_APP_SECRET"] == "env-app-secret"


def test_uses_repo_subprocess_kwargs():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed()

        LarkMessageReader(_config(timeout_sec=7)).list_messages(
            chat_id="oc_abc123",
            since_iso="1970-01-01T00:00:00+00:00",
        )

    kwargs = mock_run.call_args.kwargs
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert kwargs["check"] is False
    assert kwargs["timeout"] == 7
