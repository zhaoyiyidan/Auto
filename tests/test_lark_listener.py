from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from researchclaw.config import LarkHITLConfig
from researchclaw.hitl.file_wait import clear_waiting, write_waiting
from researchclaw.hitl.intervention import PauseReason, WaitingState
from researchclaw.notify.lark import LarkMessage
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


def _write_waiting(
    run_dir: Path,
    *,
    since: str = "2026-05-29T12:00:00+00:00",
    stage: int = 5,
    stage_name: str = "Review",
):
    return write_waiting(
        run_dir / "hitl",
        WaitingState(
            stage=stage,
            stage_name=stage_name,
            reason=PauseReason.GATE_APPROVAL,
            since=since,
            available_actions=("approve", "reject"),
            context_summary="Check the stage output.",
            output_files=("report.md",),
        ),
    )


def _message(
    text: str,
    *,
    message_id: str = "om_1",
    sender_id: str = "ou_1",
) -> LarkMessage:
    return LarkMessage(
        message_id=message_id,
        msg_type="text",
        text=text,
        sender_id=sender_id,
        sender_type="user",
        create_time_ms=0,
        chat_id="oc_abc123",
    )


def _response_data(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "hitl" / "response.json").read_text())


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


def test_valid_approve_writes_response(tmp_path: Path):
    _write_waiting(tmp_path)
    reader = _FakeReader([[_message("approve")]])
    listener = LarkHITLListener(
        reader=reader,
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.RESPONDED
    assert _response_data(tmp_path)["action"] == "approve"


def test_reject_with_reason_written(tmp_path: Path):
    _write_waiting(tmp_path)
    reader = _FakeReader([[_message("reject: needs baselines")]])
    listener = LarkHITLListener(
        reader=reader,
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.RESPONDED
    response = _response_data(tmp_path)
    assert response["action"] == "reject"
    assert response["message"] == "needs baselines"


def test_responds_exactly_once(tmp_path: Path):
    _write_waiting(tmp_path)
    reader = _FakeReader([[_message("approve")], [_message("reject")]])
    listener = LarkHITLListener(
        reader=reader,
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.RESPONDED
    (tmp_path / "hitl" / "response.json").unlink()

    assert listener.poll_once() is PollResult.ALREADY_HANDLED
    assert not (tmp_path / "hitl" / "response.json").exists()


def test_new_pause_after_resume_handled(tmp_path: Path):
    _write_waiting(tmp_path)
    notifier = _FakeNotifier()
    reader = _FakeReader([[_message("approve")], [_message("reject")]])
    listener = LarkHITLListener(
        reader=reader,
        notifier=notifier,
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.RESPONDED
    clear_waiting(tmp_path / "hitl")
    (tmp_path / "hitl" / "response.json").unlink()
    _write_waiting(
        tmp_path,
        since="2026-05-29T12:01:00+00:00",
        stage=6,
        stage_name="Next Review",
    )

    assert listener.poll_once() is PollResult.RESPONDED
    assert _response_data(tmp_path)["action"] == "reject"
    assert len(notifier.calls) == 2


def test_same_second_different_stage_not_deduped(tmp_path: Path):
    same_since = "2026-05-29T12:00:00+00:00"
    _write_waiting(tmp_path, since=same_since, stage=5)
    reader = _FakeReader([[_message("approve")], [_message("reject")]])
    listener = LarkHITLListener(
        reader=reader,
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.RESPONDED
    (tmp_path / "hitl" / "response.json").unlink()
    _write_waiting(tmp_path, since=same_since, stage=6)

    assert listener.poll_once() is PollResult.RESPONDED
    assert _response_data(tmp_path)["action"] == "reject"
