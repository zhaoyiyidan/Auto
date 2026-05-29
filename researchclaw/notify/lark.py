"""Feishu/Lark notification sender backed by the official lark-cli."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

from researchclaw.config import LarkNotifyConfig

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
        raise NotImplementedError
