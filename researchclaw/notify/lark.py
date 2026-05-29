"""Feishu/Lark notification sender backed by the official lark-cli."""

from __future__ import annotations

import logging
import json
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
                results.append(
                    LarkTargetResult(
                        name=target.name,
                        status="error",
                        detail="lark-cli invocation not implemented",
                        command=command,
                    )
                )

        return LarkNotifyResult(tuple(results))


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
