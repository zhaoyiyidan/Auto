from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

from researchclaw.config import LarkHITLConfig
from researchclaw.hitl.file_wait import clear_waiting, poll_for_response, write_waiting
from researchclaw.hitl.intervention import HumanInput, PauseReason, WaitingState
from researchclaw.notify.lark import LarkMessage
from researchclaw.notify.lark_listener import LarkHITLListener, PollResult


@dataclass(frozen=True)
class _FakeNotifyResult:
    ok: bool = True
    targets: tuple[object, ...] = ()


@dataclass(frozen=True)
class _FakeTargetResult:
    status: str = "ok"
    create_time_ms: int = 1000


class _FakeNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.next_create_time_ms = 1000

    def send(self, title: str, body: str) -> _FakeNotifyResult:
        self.calls.append((title, body))
        result = _FakeNotifyResult(
            targets=(_FakeTargetResult(create_time_ms=self.next_create_time_ms),)
        )
        self.next_create_time_ms += 1000
        return result


class _FakeReader:
    def __init__(self, batches: list[list[object]] | None = None) -> None:
        self.batches = list(batches or [])
        self.calls: list[tuple[str, str]] = []

    def list_messages(self, *, chat_id: str, since_iso: str):
        self.calls.append((chat_id, since_iso))
        if self.batches:
            return self.batches.pop(0)
        return []


class _RaisingReader:
    def list_messages(self, *, chat_id: str, since_iso: str):
        raise RuntimeError("reader failed")


def _config(
    *,
    notify: bool = True,
    allowed_actions: tuple[str, ...] = (),
    allowed_senders: tuple[str, ...] = (),
) -> LarkHITLConfig:
    return LarkHITLConfig(
        enabled=True,
        chat_id="oc_abc123",
        notify=notify,
        allowed_actions=allowed_actions,
        allowed_senders=allowed_senders,
    )


def _write_waiting(
    run_dir: Path,
    *,
    since: str = "2026-05-29T12:00:00+00:00",
    stage: int = 5,
    stage_name: str = "Review",
    available_actions: tuple[str, ...] = ("approve", "reject"),
):
    return write_waiting(
        run_dir / "hitl",
        WaitingState(
            stage=stage,
            stage_name=stage_name,
            reason=PauseReason.GATE_APPROVAL,
            since=since,
            available_actions=available_actions,
            context_summary="Check the stage output.",
            output_files=("report.md",),
        ),
    )


def _message(
    text: str,
    *,
    message_id: str = "om_1",
    sender_id: str = "ou_1",
    create_time_ms: int = 10_000,
) -> LarkMessage:
    return LarkMessage(
        message_id=message_id,
        msg_type="text",
        text=text,
        sender_id=sender_id,
        sender_type="user",
        create_time_ms=create_time_ms,
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


def test_reply_before_notification_is_ignored(tmp_path: Path):
    _write_waiting(tmp_path)
    reader = _FakeReader([[_message("approve", create_time_ms=999)]])
    listener = LarkHITLListener(
        reader=reader,
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.NOTIFIED_ONLY
    assert not (tmp_path / "hitl" / "response.json").exists()


def test_old_reply_not_reused_for_next_pause(tmp_path: Path):
    _write_waiting(tmp_path)
    reader = _FakeReader(
        [
            [_message("approve", message_id="om_old", create_time_ms=1500)],
            [_message("approve", message_id="om_old", create_time_ms=1500)],
        ]
    )
    listener = LarkHITLListener(
        reader=reader,
        notifier=_FakeNotifier(),
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

    assert listener.poll_once() is PollResult.NOTIFIED_ONLY
    assert not (tmp_path / "hitl" / "response.json").exists()


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


def test_allowed_senders_filter(tmp_path: Path):
    _write_waiting(tmp_path)
    reader = _FakeReader(
        [
            [
                _message("approve", message_id="om_disallowed", sender_id="ou_bad"),
                _message("reject", message_id="om_allowed", sender_id="ou_allowed"),
            ]
        ]
    )
    listener = LarkHITLListener(
        reader=reader,
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(allowed_senders=("ou_allowed",)),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.RESPONDED
    assert _response_data(tmp_path)["action"] == "reject"


def test_unparseable_then_valid_used(tmp_path: Path):
    _write_waiting(tmp_path)
    reader = _FakeReader([[_message("lol"), _message("approve", message_id="om_2")]])
    listener = LarkHITLListener(
        reader=reader,
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.RESPONDED
    assert _response_data(tmp_path)["action"] == "approve"


def test_action_not_in_available_actions_no_write(tmp_path: Path):
    _write_waiting(tmp_path, available_actions=("approve", "reject"))
    notifier = _FakeNotifier()
    reader = _FakeReader([[_message("skip", message_id="om_skip")]])
    listener = LarkHITLListener(
        reader=reader,
        notifier=notifier,
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.INVALID_ACTION
    assert not (tmp_path / "hitl" / "response.json").exists()
    assert len(notifier.calls) == 2
    assert "Invalid action" in notifier.calls[1][0]
    assert "approve, reject" in notifier.calls[1][1]


def test_allowed_actions_config_restricts(tmp_path: Path):
    _write_waiting(tmp_path, available_actions=("approve", "reject"))
    reader = _FakeReader([[_message("reject", message_id="om_reject")]])
    listener = LarkHITLListener(
        reader=reader,
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(allowed_actions=("approve",)),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.INVALID_ACTION
    assert not (tmp_path / "hitl" / "response.json").exists()


def test_feedback_sent_once_per_message_id(tmp_path: Path):
    _write_waiting(tmp_path, available_actions=("approve",))
    invalid = _message("reject", message_id="om_same")
    notifier = _FakeNotifier()
    reader = _FakeReader([[invalid], [invalid]])
    listener = LarkHITLListener(
        reader=reader,
        notifier=notifier,
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.INVALID_ACTION
    assert listener.poll_once() is PollResult.INVALID_ACTION
    assert len(notifier.calls) == 2


def test_corrupt_waiting_returns_error(tmp_path: Path):
    hitl_dir = tmp_path / "hitl"
    hitl_dir.mkdir()
    (hitl_dir / "waiting.json").write_text("{bad", encoding="utf-8")
    listener = LarkHITLListener(
        reader=_FakeReader(),
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.ERROR


def test_reader_exception_caught(tmp_path: Path):
    _write_waiting(tmp_path)
    listener = LarkHITLListener(
        reader=_RaisingReader(),
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.ERROR


def test_run_stops_after_max_iterations(tmp_path: Path, monkeypatch):
    _write_waiting(tmp_path)
    reader = _FakeReader()
    listener = LarkHITLListener(
        reader=reader,
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )
    monkeypatch.setattr("researchclaw.notify.lark_listener.time.sleep", lambda _: None)

    listener.run(max_iterations=3)

    assert len(reader.calls) == 3


def test_listener_unblocks_poll_for_response(tmp_path: Path):
    responses: list[HumanInput] = []

    def wait_for_response() -> None:
        responses.append(
            poll_for_response(
                tmp_path / "hitl",
                poll_interval_sec=0.1,
                timeout_sec=5,
            )
        )

    thread = Thread(target=wait_for_response)
    thread.start()
    _write_waiting(tmp_path)
    listener = LarkHITLListener(
        reader=_FakeReader([[_message("approve")]]),
        notifier=_FakeNotifier(),
        run_dir=tmp_path,
        config=_config(),
        run_id="run-1",
    )

    assert listener.poll_once() is PollResult.RESPONDED
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert responses[0].action.value == "approve"
    assert not (tmp_path / "hitl" / "response.json").exists()
