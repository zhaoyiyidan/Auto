"""Bridge Lark chat replies into the existing file-based HITL wait loop."""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any

from researchclaw.config import LarkHITLConfig
from researchclaw.hitl.file_wait import write_response
from researchclaw.hitl.intervention import WaitingState
from researchclaw.hitl.store import HITLStore
from researchclaw.notify.lark_reply import parse_reply

logger = logging.getLogger(__name__)


class PollResult(str, Enum):
    NO_WAITING = "no_waiting"
    ALREADY_HANDLED = "already_handled"
    NOTIFIED_ONLY = "notified_only"
    NO_REPLY = "no_reply"
    INVALID_ACTION = "invalid_action"
    RESPONDED = "responded"
    ERROR = "error"


class LarkHITLListener:
    def __init__(
        self,
        *,
        reader: Any,
        notifier: Any,
        run_dir: Path,
        config: LarkHITLConfig,
        run_id: str = "",
    ) -> None:
        self.reader = reader
        self.notifier = notifier
        self.run_dir = Path(run_dir)
        self.config = config
        self.run_id = run_id
        self.store = HITLStore(self.run_dir)
        self._handled_keys: set[tuple[str, int]] = set()
        self._notified_keys: set[tuple[str, int]] = set()
        self._reply_after_ms: dict[tuple[str, int], int] = {}
        self._active_key: tuple[str, int] | None = None
        self._feedback_sent: set[str] = set()

    def poll_once(self) -> PollResult:
        try:
            return self._poll_once()
        except Exception:
            logger.warning("Lark HITL listener poll failed", exc_info=True)
            return PollResult.ERROR

    def run(self, max_iterations: int | None = None) -> None:
        iterations = 0
        try:
            while max_iterations is None or iterations < max_iterations:
                self.poll_once()
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
                time.sleep(self.config.poll_interval_sec)
        except KeyboardInterrupt:
            return

    def _poll_once(self) -> PollResult:
        data = self.store.load_waiting()
        if data is None:
            if (self.run_dir / "hitl" / "waiting.json").exists():
                return PollResult.ERROR
            return PollResult.NO_WAITING

        waiting = WaitingState.from_dict(data)
        key = (waiting.since, waiting.stage)
        if key != self._active_key:
            self._active_key = key

        if key in self._handled_keys:
            return PollResult.ALREADY_HANDLED

        notified_this_tick = False
        if self.config.notify and key not in self._notified_keys:
            result = self.notifier.send(
                _prompt_title(self.run_id, waiting),
                _prompt_body(waiting),
            )
            if not getattr(result, "ok", True):
                logger.warning("Lark HITL notification failed")
            cutoff_ms = _notification_cutoff_ms(result)
            if cutoff_ms > 0:
                self._reply_after_ms[key] = cutoff_ms
            elif getattr(result, "ok", True):
                logger.warning("Lark HITL notification has no message timestamp")
            self._notified_keys.add(key)
            notified_this_tick = True

        if self.config.notify and key not in self._reply_after_ms:
            return PollResult.NOTIFIED_ONLY if notified_this_tick else PollResult.NO_REPLY

        messages = self.reader.list_messages(
            chat_id=self.config.chat_id,
            since_iso=waiting.since,
        )
        for message in messages:
            if not _message_is_after_notification(message, key, self._reply_after_ms):
                continue
            sender_id = str(getattr(message, "sender_id", "") or "")
            if self.config.allowed_senders and sender_id not in self.config.allowed_senders:
                continue

            parsed = parse_reply(getattr(message, "text", ""))
            if parsed is None:
                continue
            if not _action_allowed(parsed.action.value, waiting, self.config):
                self._send_invalid_feedback(message, waiting)
                return PollResult.INVALID_ACTION

            write_response(self.run_dir / "hitl", parsed.to_human_input())
            self._handled_keys.add(key)
            return PollResult.RESPONDED

        if notified_this_tick:
            return PollResult.NOTIFIED_ONLY
        return PollResult.NO_REPLY

    def _send_invalid_feedback(self, message: object, waiting: WaitingState) -> None:
        message_id = str(getattr(message, "message_id", "") or "")
        if message_id and message_id in self._feedback_sent:
            return

        self.notifier.send(
            "Invalid action",
            "Action is not allowed for this pause. Valid actions: "
            + ", ".join(_allowed_actions(waiting, self.config)),
        )
        if message_id:
            self._feedback_sent.add(message_id)


def _prompt_title(run_id: str, waiting: WaitingState) -> str:
    run_label = run_id or "current run"
    return f"ResearchClaw HITL review needed: {run_label}"


def _prompt_body(waiting: WaitingState) -> str:
    actions = ", ".join(waiting.available_actions)
    lines = [
        f"Stage {waiting.stage}: {waiting.stage_name}",
        f"Reason: {waiting.reason.value}",
        f"Available actions: {actions}",
    ]
    if waiting.context_summary:
        lines.append(f"Context: {waiting.context_summary}")
    if waiting.output_files:
        lines.append("Output files: " + ", ".join(waiting.output_files))
    return "\n".join(lines)


def _allowed_actions(
    waiting: WaitingState,
    config: LarkHITLConfig,
) -> tuple[str, ...]:
    if not config.allowed_actions:
        return waiting.available_actions
    configured = set(config.allowed_actions)
    return tuple(action for action in waiting.available_actions if action in configured)


def _action_allowed(
    action: str,
    waiting: WaitingState,
    config: LarkHITLConfig,
) -> bool:
    return action in _allowed_actions(waiting, config)


def _notification_cutoff_ms(result: object) -> int:
    targets = getattr(result, "targets", ()) or ()
    cutoffs = [
        int(getattr(target, "create_time_ms", 0) or 0)
        for target in targets
        if getattr(target, "status", "") == "ok"
    ]
    return max(cutoffs, default=0)


def _message_is_after_notification(
    message: object,
    key: tuple[str, int],
    reply_after_ms: dict[tuple[str, int], int],
) -> bool:
    cutoff_ms = reply_after_ms.get(key, 0)
    if cutoff_ms <= 0:
        return True
    return int(getattr(message, "create_time_ms", 0) or 0) > cutoff_ms
