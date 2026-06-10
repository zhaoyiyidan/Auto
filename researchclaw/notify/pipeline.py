"""Pipeline-level notification helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from os import PathLike

from researchclaw.notify.lark import LarkNotifier, LarkNotifyResult

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_failure_message(
    *,
    run_id: str,
    stage_name: str,
    stage_num: int,
    error: str | None,
    run_dir: str | PathLike[str] | None = None,
) -> tuple[str, str]:
    """Return the Lark title/body for a terminal pipeline stage failure."""
    run_dir_text = str(run_dir) if run_dir is not None else "unknown"
    error_text = error or "unknown error"
    title = f"ResearchClaw pipeline FAILED: {run_id}"
    body = "\n".join(
        [
            f"Run: {run_id}",
            f"Stage {stage_num:02d}: {stage_name} - FAILED",
            f"Error: {error_text}",
            f"Run dir: {run_dir_text}",
            "",
            "Log in to the server to inspect & fix:",
            f"  researchclaw status {run_dir_text}",
            "  researchclaw run --resume",
            f"Time (UTC): {_utcnow_iso()}",
        ]
    )
    return title, body


def notify_terminal_failure(
    *,
    config: object,
    run_id: str,
    stage_name: str,
    stage_num: int,
    error: str | None,
    run_dir: str | PathLike[str] | None = None,
) -> LarkNotifyResult | None:
    """Send a best-effort Lark alert for a failure that stops the pipeline."""
    notifications = getattr(config, "notifications", None)
    if not getattr(notifications, "on_stage_fail", False):
        return None

    try:
        notifier = LarkNotifier.from_rc_config(config)
        title, body = build_failure_message(
            run_id=run_id,
            stage_name=stage_name,
            stage_num=stage_num,
            error=error,
            run_dir=run_dir,
        )
        return notifier.send(title, body)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Terminal-failure Lark notification failed (non-blocking)",
            exc_info=True,
        )
        return None
