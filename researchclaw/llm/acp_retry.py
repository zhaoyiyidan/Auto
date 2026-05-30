"""Shared bounded-retry helper for transient ACP (acpx) stream disconnects.

Both ACP call sites — the LLM client (``acp_client.ACPClient``) and the
workspace code-agent session (``acp_workspace_session.AcpWorkspaceSession``) —
can hit a transient upstream SSE disconnect mid-turn. acpx may even exit 0 after
exhausting its own internal reconnects, leaving a disconnect banner in stdout.
Without a retry, a single transient drop surfaces as a hard failure (e.g. a
stage-fatal ``E10_CODE_AGENT_FAIL``) and can abort an otherwise-healthy run.

This module centralizes (a) detection of the transient-disconnect signature and
(b) a small bounded retry loop with exponential backoff, so every ACP path gets
uniform resilience. Genuine failures (non-transient errors, an agent that ran
but produced no commit) are NOT retried.
"""

from __future__ import annotations

import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")

# Substrings that indicate a transient acpx/gateway stream disconnect. Matched
# case-insensitively against both stdout and stderr. These also include the
# reconnect markers historically used by ACPClient so the two stacks share one
# definition.
_TRANSIENT_SUBSTRINGS = (
    "stream disconnected before completion",
    "stream closed before response.completed",
    "agent needs reconnect",
    "session not found",
    "query closed",
)

# acpx prints its own internal reconnect attempts as "Reconnecting... N/M".
_RECONNECTING_RE = re.compile(r"reconnecting\.\.\.\s*\d+\s*/\s*\d+", re.IGNORECASE)


class TransientAcpDisconnect(RuntimeError):
    """Raised when an ACP call fails due to a transient stream disconnect.

    Callers raise this (instead of a plain ``RuntimeError``) when they detect a
    transient signature — including the case where the subprocess exited 0 but
    the disconnect banner is present in its stdout — so ``run_acp_with_retry``
    knows the failure is retryable.
    """


def is_transient_text(text: str | None) -> bool:
    """Return True if *text* carries a transient ACP-disconnect signature."""
    if not text:
        return False
    lowered = text.lower()
    if any(sub in lowered for sub in _TRANSIENT_SUBSTRINGS):
        return True
    return bool(_RECONNECTING_RE.search(text))


def run_acp_with_retry(
    do_call: Callable[[], T],
    *,
    reset: Callable[[], None],
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    backoff_base: float = 1.0,
    backoff_cap: float = 30.0,
) -> T:
    """Call ``do_call`` with bounded retry on transient ACP disconnects.

    Parameters
    ----------
    do_call:
        The ACP invocation. Should raise :class:`TransientAcpDisconnect` (or a
        subclass) when it detects a transient disconnect, and any other
        exception for a genuine failure.
    reset:
        Called before each retry to recover the session (e.g. close + re-ensure
        the named session). Not called after the final attempt.
    max_retries:
        Maximum number of retries AFTER the initial attempt (so total attempts
        are ``1 + max_retries``). ``0`` means a single attempt, no retry.
    sleep:
        Injectable sleep (defaults to :func:`time.sleep`); tests pass a no-op.
    backoff_base, backoff_cap:
        Exponential backoff: the delay before retry ``i`` (0-indexed) is
        ``min(backoff_cap, backoff_base * 2**i)``.

    Returns the result of ``do_call``. Re-raises the last
    :class:`TransientAcpDisconnect` if all attempts are exhausted, and
    immediately re-raises any non-transient exception.
    """
    attempts = max(0, int(max_retries)) + 1
    last_exc: TransientAcpDisconnect | None = None
    for attempt in range(attempts):
        try:
            return do_call()
        except TransientAcpDisconnect as exc:
            last_exc = exc
            if attempt >= attempts - 1:
                break
            delay = min(backoff_cap, backoff_base * (2 ** attempt))
            sleep(delay)
            reset()
    assert last_exc is not None
    raise last_exc
