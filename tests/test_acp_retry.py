"""Tests for the shared ACP transient-disconnect retry helper."""

from __future__ import annotations

import pytest

from researchclaw.llm.acp_retry import (
    TransientAcpDisconnect,
    is_transient_text,
    run_acp_with_retry,
)


class TestIsTransient:
    @pytest.mark.parametrize(
        "text",
        [
            "stream disconnected before completion",
            "stream closed before response.completed",
            "agent needs reconnect",
            "session not found",
            "Query closed",
            "Reconnecting... 1/5",
            "reconnecting...  3 / 5",
        ],
    )
    def test_matches_transient_signatures(self, text: str) -> None:
        assert is_transient_text(text) is True

    @pytest.mark.parametrize(
        "text",
        ["", "agent log: training complete", "accuracy=0.99", "done end_turn"],
    )
    def test_no_false_positive_on_clean_output(self, text: str) -> None:
        assert is_transient_text(text) is False


class TestRunAcpWithRetry:
    def test_retries_then_succeeds(self) -> None:
        calls = {"n": 0}
        resets: list[int] = []
        sleeps: list[float] = []

        def do_call() -> str:
            calls["n"] += 1
            if calls["n"] <= 2:
                raise TransientAcpDisconnect("stream closed before response.completed")
            return "ok"

        out = run_acp_with_retry(
            do_call,
            reset=lambda: resets.append(1),
            max_retries=3,
            sleep=sleeps.append,
            backoff_base=1.0,
        )
        assert out == "ok"
        assert calls["n"] == 3
        assert len(resets) == 2  # reset before each retry
        assert sleeps == [1.0, 2.0]  # exponential backoff

    def test_exhausts_and_raises(self) -> None:
        calls = {"n": 0}

        def do_call() -> str:
            calls["n"] += 1
            raise TransientAcpDisconnect("stream disconnected before completion")

        with pytest.raises(TransientAcpDisconnect):
            run_acp_with_retry(
                do_call,
                reset=lambda: None,
                max_retries=3,
                sleep=lambda _s: None,
            )
        assert calls["n"] == 4  # 1 initial + 3 retries

    def test_no_retry_on_non_transient(self) -> None:
        calls = {"n": 0}
        resets: list[int] = []

        def do_call() -> str:
            calls["n"] += 1
            raise RuntimeError("real failure: ZeroDivisionError")

        with pytest.raises(RuntimeError, match="real failure"):
            run_acp_with_retry(
                do_call,
                reset=lambda: resets.append(1),
                max_retries=3,
                sleep=lambda _s: None,
            )
        assert calls["n"] == 1
        assert resets == []

    def test_backoff_capped(self) -> None:
        sleeps: list[float] = []

        def do_call() -> str:
            raise TransientAcpDisconnect("stream closed before response.completed")

        with pytest.raises(TransientAcpDisconnect):
            run_acp_with_retry(
                do_call,
                reset=lambda: None,
                max_retries=5,
                sleep=sleeps.append,
                backoff_base=10.0,
                backoff_cap=25.0,
            )
        # 10, 20, then capped at 25
        assert sleeps == [10.0, 20.0, 25.0, 25.0, 25.0]

    def test_zero_retries_means_single_attempt(self) -> None:
        calls = {"n": 0}

        def do_call() -> str:
            calls["n"] += 1
            raise TransientAcpDisconnect("stream closed before response.completed")

        with pytest.raises(TransientAcpDisconnect):
            run_acp_with_retry(
                do_call, reset=lambda: None, max_retries=0, sleep=lambda _s: None
            )
        assert calls["n"] == 1
