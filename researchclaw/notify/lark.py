"""Feishu/Lark notification sender backed by the official lark-cli."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

from researchclaw.config import LarkNotifyConfig, LarkTargetConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LarkTargetResult:
    name: str
    status: str
    detail: str = ""
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class LarkNotifyResult:
    targets: tuple[LarkTargetResult, ...]

    @property
    def ok(self) -> bool:
        return all(target.status != "error" for target in self.targets)


@dataclass(frozen=True)
class LarkMessage:
    message_id: str
    msg_type: str
    text: str
    sender_id: str
    sender_type: str
    create_time_ms: int
    chat_id: str


class LarkNotifier:
    def __init__(self, config: LarkNotifyConfig) -> None:
        self.config = config

    @classmethod
    def from_rc_config(cls, rc_config: object) -> "LarkNotifier":
        notifications = getattr(rc_config, "notifications", None)
        config = getattr(notifications, "lark", LarkNotifyConfig())
        return cls(config)

    def send(self, title: str, body: str) -> LarkNotifyResult:
        if not self.config.enabled or not self.config.targets:
            return LarkNotifyResult(())

        results: list[LarkTargetResult] = []
        env = _build_env(self.config)
        for target in self.config.targets:
            if not target.receive_id:
                results.append(
                    LarkTargetResult(
                        name=target.name,
                        status="skipped",
                        detail="missing receive_id",
                    )
                )
                continue

            command = tuple(_build_command(self.config, target, title, body))
            if self.config.dry_run:
                results.append(
                    LarkTargetResult(
                        name=target.name,
                        status="dry_run",
                        command=command,
                    )
                )
            else:
                results.append(self._send_target(target, command, env))

        return LarkNotifyResult(tuple(results))

    def _send_target(
        self,
        target: LarkTargetConfig,
        command: tuple[str, ...],
        env: dict[str, str] | None,
    ) -> LarkTargetResult:
        try:
            kwargs = {
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "timeout": self.config.timeout_sec,
                "check": False,
            }
            if env is not None:
                kwargs["env"] = env

            completed = subprocess.run(
                list(command),
                **kwargs,
            )
            result = _result_from_completed(target.name, command, completed)
        except subprocess.TimeoutExpired as exc:
            result = LarkTargetResult(
                name=target.name,
                status="error",
                detail=f"lark-cli timed out after {exc.timeout}s",
                command=command,
            )
        except FileNotFoundError as exc:
            result = LarkTargetResult(
                name=target.name,
                status="error",
                detail=f"lark-cli executable not found: {exc}",
                command=command,
            )
        except OSError as exc:
            result = LarkTargetResult(
                name=target.name,
                status="error",
                detail=f"lark-cli OS error: {exc}",
                command=command,
            )
        except Exception as exc:
            result = LarkTargetResult(
                name=target.name,
                status="error",
                detail=f"lark-cli unexpected error: {exc}",
                command=command,
            )

        result = _redact_result(result, self.config, env)
        if result.status == "error":
            logger.warning(
                "Lark notification to %s failed: %s",
                target.name or target.receive_id,
                result.detail,
            )
        return result


def _compose_text(title: str, body: str) -> str:
    return f"{title}\n\n{body}" if title else body


def _build_command(
    config: LarkNotifyConfig,
    target: LarkTargetConfig,
    title: str,
    body: str,
) -> list[str]:
    content_obj = {"text": _compose_text(title, body)}
    data_obj = {
        "receive_id": target.receive_id,
        "msg_type": "text",
        "content": json.dumps(content_obj, ensure_ascii=False),
    }
    return [
        config.command,
        "api",
        "POST",
        "/open-apis/im/v1/messages",
        "--params",
        json.dumps({"receive_id_type": target.receive_id_type}, ensure_ascii=False),
        "--data",
        json.dumps(data_obj, ensure_ascii=False),
        "--format",
        "json",
    ]


def _build_list_command(
    config: LarkNotifyConfig,
    *,
    chat_id: str,
    start_time: int,
    page_size: int = 50,
    page_token: str = "",
) -> list[str]:
    params = {
        "container_id_type": "chat",
        "container_id": chat_id,
        "start_time": str(start_time),
        "sort_type": "ByCreateTimeAsc",
        "page_size": str(page_size),
    }
    if page_token:
        params["page_token"] = page_token

    return [
        config.command,
        "api",
        "GET",
        "/open-apis/im/v1/messages",
        "--params",
        json.dumps(params, ensure_ascii=False),
        "--format",
        "json",
    ]


class LarkMessageReader:
    def __init__(self, config: LarkNotifyConfig) -> None:
        self.config = config

    def list_messages(
        self,
        *,
        chat_id: str,
        since_iso: str,
        max_pages: int = 5,
    ) -> list[LarkMessage]:
        start_time = _iso_to_unix_seconds(since_iso)
        since_ms = start_time * 1000
        command = _build_list_command(
            self.config,
            chat_id=chat_id,
            start_time=start_time,
        )
        kwargs = {
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": self.config.timeout_sec,
            "check": False,
        }
        env = _build_env(self.config)
        if env is not None:
            kwargs["env"] = env

        completed = subprocess.run(command, **kwargs)
        return _messages_from_stdout(completed.stdout, since_ms=since_ms)


def _iso_to_unix_seconds(value: str) -> int:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _messages_from_stdout(stdout: str, *, since_ms: int) -> list[LarkMessage]:
    try:
        payload = json.loads(stdout)
    except (TypeError, ValueError):
        return []

    if not isinstance(payload, dict):
        return []

    data = payload.get("data")
    if not isinstance(data, dict):
        return []

    messages: list[LarkMessage] = []
    for item in data.get("items") or ():
        message = _message_from_item(item)
        if message is None:
            continue
        if message.sender_type != "user":
            continue
        if message.create_time_ms < since_ms:
            continue
        messages.append(message)
    return messages


def _message_from_item(item: object) -> LarkMessage | None:
    if not isinstance(item, dict):
        return None

    sender = item.get("sender") if isinstance(item.get("sender"), dict) else {}
    sender_type = str(sender.get("sender_type", "") or "")
    create_time_ms = _safe_int_value(item.get("create_time"), 0)
    msg_type = str(item.get("msg_type", "") or "")

    return LarkMessage(
        message_id=str(item.get("message_id", "") or ""),
        msg_type=msg_type,
        text=_extract_message_text(item, msg_type),
        sender_id=_extract_sender_id(sender.get("sender_id")),
        sender_type=sender_type,
        create_time_ms=create_time_ms,
        chat_id=str(item.get("chat_id", "") or ""),
    )


def _extract_sender_id(value: object) -> str:
    if isinstance(value, dict):
        for key in ("open_id", "user_id", "union_id", "email"):
            candidate = value.get(key)
            if candidate:
                return str(candidate)
        return ""
    return str(value or "")


def _extract_message_text(item: dict[object, object], msg_type: str) -> str:
    if msg_type != "text":
        return ""

    body = item.get("body") if isinstance(item.get("body"), dict) else {}
    content = body.get("content", "")
    if isinstance(content, str):
        try:
            content_obj = json.loads(content)
        except (TypeError, ValueError):
            return ""
    elif isinstance(content, dict):
        content_obj = content
    else:
        return ""

    if not isinstance(content_obj, dict):
        return ""
    return str(content_obj.get("text", "") or "")


def _safe_int_value(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_env(config: LarkNotifyConfig) -> dict[str, str] | None:
    app_id = os.environ.get(config.app_id_env) or config.app_id
    app_secret = os.environ.get(config.app_secret_env) or config.app_secret
    if not app_id or not app_secret:
        return None

    env = os.environ.copy()
    env["LARK_APP_ID"] = app_id
    env["LARK_APP_SECRET"] = app_secret
    if config.app_id_env:
        env[config.app_id_env] = app_id
    if config.app_secret_env:
        env[config.app_secret_env] = app_secret
    return env


def _redact_result(
    result: LarkTargetResult,
    config: LarkNotifyConfig,
    env: dict[str, str] | None,
) -> LarkTargetResult:
    detail = _redact_secrets(result.detail, config, env)
    if detail == result.detail:
        return result
    return LarkTargetResult(
        name=result.name,
        status=result.status,
        detail=detail,
        command=result.command,
    )


def _redact_secrets(
    text: str,
    config: LarkNotifyConfig,
    env: dict[str, str] | None,
) -> str:
    if not text:
        return text

    candidates = {
        config.app_secret,
        os.environ.get("LARK_APP_SECRET", ""),
    }
    if config.app_secret_env:
        candidates.add(os.environ.get(config.app_secret_env, ""))
    if env is not None:
        candidates.add(env.get("LARK_APP_SECRET", ""))
        if config.app_secret_env:
            candidates.add(env.get(config.app_secret_env, ""))

    redacted = text
    for secret in candidates:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def _result_from_completed(
    name: str,
    command: tuple[str, ...],
    completed: subprocess.CompletedProcess[str],
) -> LarkTargetResult:
    if completed.returncode != 0:
        detail = completed.stderr or completed.stdout or "lark-cli failed"
        return LarkTargetResult(
            name=name,
            status="error",
            detail=detail,
            command=command,
        )

    api_error = _api_error_detail(completed.stdout)
    if api_error:
        return LarkTargetResult(
            name=name,
            status="error",
            detail=api_error,
            command=command,
        )

    return LarkTargetResult(name=name, status="ok", command=command)


def _api_error_detail(stdout: str) -> str:
    try:
        payload = json.loads(stdout)
    except (TypeError, ValueError):
        return ""

    if not isinstance(payload, dict):
        return ""

    code = payload.get("code")
    if code in (None, 0, "0"):
        return ""

    msg = payload.get("msg") or payload.get("message") or "Lark API error"
    return f"Lark API error code={code}: {msg}"
