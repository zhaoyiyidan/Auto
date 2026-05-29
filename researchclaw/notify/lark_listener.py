"""Bridge Lark chat replies into the existing file-based HITL wait loop."""

from __future__ import annotations

import logging
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
        self._active_key: tuple[str, int] | None = None
        self._feedback_sent: set[str] = set()

    def poll_once(self) -> PollResult:
        data = self.store.load_waiting()
        if data is None:
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
            self._notified_keys.add(key)
            notified_this_tick = True

        messages = self.reader.list_messages(
            chat_id=self.config.chat_id,
            since_iso=waiting.since,
        )
        for message in messages:
            parsed = parse_reply(getattr(message, "text", ""))
            if parsed is None:
                continue
            write_response(self.run_dir / "hitl", parsed.to_human_input())
            self._handled_keys.add(key)
            return PollResult.RESPONDED

        if notified_this_tick:
            return PollResult.NOTIFIED_ONLY
        return PollResult.NO_REPLY


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
