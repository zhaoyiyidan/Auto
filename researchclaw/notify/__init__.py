"""Notification integrations."""

from researchclaw.notify.lark import (
    LarkNotifier,
    LarkNotifyResult,
    LarkTargetResult,
)
from researchclaw.notify.pipeline import (
    build_failure_message,
    notify_terminal_failure,
)

__all__ = (
    "build_failure_message",
    "LarkNotifier",
    "LarkNotifyResult",
    "LarkTargetResult",
    "notify_terminal_failure",
)
