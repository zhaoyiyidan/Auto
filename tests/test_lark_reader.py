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


def _message_item(
    *,
    message_id: str = "om_1",
    msg_type: str = "text",
    text: str = "approve",
    sender_type: str = "user",
    sender_id: str = "ou_1",
    create_time: str | int = "5000",
    chat_id: str = "oc_abc123",
) -> dict[str, object]:
    return {
        "message_id": message_id,
        "msg_type": msg_type,
        "chat_id": chat_id,
        "create_time": create_time,
        "sender": {
            "sender_type": sender_type,
            "sender_id": {"open_id": sender_id},
        },
        "body": {"content": json.dumps({"text": text}, ensure_ascii=False)},
    }


def _stdout_with_items(
    items: list[dict[str, object]],
    *,
    has_more: bool = False,
    page_token: str = "",
) -> str:
    return json.dumps(
        {
            "code": 0,
            "data": {
                "items": items,
                "has_more": has_more,
                "page_token": page_token,
            },
        },
        ensure_ascii=False,
    )


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


def test_parses_items_into_lark_messages():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed(
            stdout=_stdout_with_items(
                [
                    _message_item(
                        message_id="om_1",
                        text="approve: go",
                        sender_id="ou_reviewer",
                        create_time="5000",
                    )
                ]
            )
        )

        messages = LarkMessageReader(_config()).list_messages(
            chat_id="oc_abc123",
            since_iso="1970-01-01T00:00:04+00:00",
        )

    assert len(messages) == 1
    assert messages[0].message_id == "om_1"
    assert messages[0].msg_type == "text"
    assert messages[0].text == "approve: go"
    assert messages[0].sender_id == "ou_reviewer"
    assert messages[0].sender_type == "user"
    assert messages[0].create_time_ms == 5000
    assert messages[0].chat_id == "oc_abc123"


def test_filters_non_user_senders():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed(
            stdout=_stdout_with_items(
                [
                    _message_item(message_id="om_bot", sender_type="app"),
                    _message_item(message_id="om_user", sender_type="user"),
                ]
            )
        )

        messages = LarkMessageReader(_config()).list_messages(
            chat_id="oc_abc123",
            since_iso="1970-01-01T00:00:00+00:00",
        )

    assert [message.message_id for message in messages] == ["om_user"]


def test_filters_before_since():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed(
            stdout=_stdout_with_items(
                [
                    _message_item(message_id="om_before", create_time="4000"),
                    _message_item(message_id="om_after", create_time="5000"),
                ]
            )
        )

        messages = LarkMessageReader(_config()).list_messages(
            chat_id="oc_abc123",
            since_iso="1970-01-01T00:00:05+00:00",
        )

    assert [message.message_id for message in messages] == ["om_after"]


def test_non_text_yields_empty_text():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed(
            stdout=_stdout_with_items([_message_item(msg_type="image", text="hidden")])
        )

        messages = LarkMessageReader(_config()).list_messages(
            chat_id="oc_abc123",
            since_iso="1970-01-01T00:00:00+00:00",
        )

    assert messages[0].msg_type == "image"
    assert messages[0].text == ""


def test_create_time_ms_string_coerced_int():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed(
            stdout=_stdout_with_items([_message_item(create_time="12345")])
        )

        messages = LarkMessageReader(_config()).list_messages(
            chat_id="oc_abc123",
            since_iso="1970-01-01T00:00:00+00:00",
        )

    assert messages[0].create_time_ms == 12345


def test_paginates_until_has_more_false():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _completed(
                stdout=_stdout_with_items(
                    [_message_item(message_id="om_1")],
                    has_more=True,
                    page_token="next-token",
                )
            ),
            _completed(
                stdout=_stdout_with_items(
                    [_message_item(message_id="om_2")],
                    has_more=False,
                )
            ),
        ]

        messages = LarkMessageReader(_config()).list_messages(
            chat_id="oc_abc123",
            since_iso="1970-01-01T00:00:00+00:00",
        )

    assert [message.message_id for message in messages] == ["om_1", "om_2"]
    assert mock_run.call_count == 2
    second_params = _params_from_command(mock_run.call_args_list[1].args[0])
    assert second_params["page_token"] == "next-token"


def test_pagination_capped_by_max_pages():
    with patch("researchclaw.notify.lark.subprocess.run") as mock_run:
        mock_run.return_value = _completed(
            stdout=_stdout_with_items(
                [_message_item(message_id="om_1")],
                has_more=True,
                page_token="next-token",
            )
        )

        messages = LarkMessageReader(_config()).list_messages(
            chat_id="oc_abc123",
            since_iso="1970-01-01T00:00:00+00:00",
            max_pages=1,
        )

    assert [message.message_id for message in messages] == ["om_1"]
    assert mock_run.call_count == 1
