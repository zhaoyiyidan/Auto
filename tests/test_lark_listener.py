from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from researchclaw.config import LarkHITLConfig
from researchclaw.hitl.file_wait import write_waiting
from researchclaw.hitl.intervention import PauseReason, WaitingState
from researchclaw.notify.lark_listener import LarkHITLListener, PollResult


@dataclass(frozen=True)
class _FakeNotifyResult:
    ok: bool = True


class _FakeNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def send(self, title: str, body: str) -> _FakeNotifyResult:
        self.calls.append((title, body))
        return _FakeNotifyResult()


class _FakeReader:
    def __init__(self, batches: list[list[object]] | None = None) -> None:
        self.batches = list(batches or [])
        self.calls: list[tuple[str, str]] = []

    def list_messages(self, *, chat_id: str, since_iso: str):
        self.calls.append((chat_id, since_iso))
        if self.batches:
            return self.batches.pop(0)
        return []


def _config(*, notify: bool = True) -> LarkHITLConfig:
    return LarkHITLConfig(enabled=True, chat_id="oc_abc123", notify=notify)


def _write_waiting(run_dir: Path, *, since: str = "2026-05-29T12:00:00+00:00"):
    return write_waiting(
        run_dir / "hitl",
        WaitingState(
            stage=5,
            stage_name="Review",
            reason=PauseReason.GATE_APPROVAL,
            since=since,
            available_actions=("approve", "reject"),
            context_summary="Check the stage output.",
            output_files=("report.md",),
        ),
    )


def test_no_waiting_returns_no_waiting(tmp_path: Path):
    reader = _FakeReader()
    notifier = _FakeNotifier()
    listener = LarkHITLListener(
        reader=reader,
        notifier=notifier,
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.NO_WAITING
    assert notifier.calls == []
    assert reader.calls == []


def test_first_poll_notifies_once_returns_notified_only(tmp_path: Path):
    _write_waiting(tmp_path)
    reader = _FakeReader()
    notifier = _FakeNotifier()
    listener = LarkHITLListener(
        reader=reader,
        notifier=notifier,
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.NOTIFIED_ONLY
    assert len(notifier.calls) == 1
    title, body = notifier.calls[0]
    assert "run-1" in title
    assert "Review" in body
    assert "approve, reject" in body
    assert reader.calls == [("oc_abc123", "2026-05-29T12:00:00+00:00")]


def test_notification_not_resent_second_poll(tmp_path: Path):
    _write_waiting(tmp_path)
    reader = _FakeReader()
    notifier = _FakeNotifier()
    listener = LarkHITLListener(
        reader=reader,
        notifier=notifier,
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.NOTIFIED_ONLY
    assert listener.poll_once() is PollResult.NO_REPLY
    assert len(notifier.calls) == 1


def test_notify_false_skips_notification(tmp_path: Path):
    _write_waiting(tmp_path)
    reader = _FakeReader()
    notifier = _FakeNotifier()
    listener = LarkHITLListener(
        reader=reader,
        notifier=notifier,
        run_dir=tmp_path,
        config=_config(notify=False),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.NO_REPLY
    assert notifier.calls == []
    assert reader.calls == [("oc_abc123", "2026-05-29T12:00:00+00:00")]
