"""Feishu/Lark notification sender backed by the official lark-cli."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass

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
                results.append(self._send_target(target, command))

        return LarkNotifyResult(tuple(results))

    def _send_target(
        self,
        target: LarkTargetConfig,
        command: tuple[str, ...],
    ) -> LarkTargetResult:
        try:
            completed = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.timeout_sec,
                check=False,
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
