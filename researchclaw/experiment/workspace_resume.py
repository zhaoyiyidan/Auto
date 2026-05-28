"""Resume helpers for persistent workspace ACP sessions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, TypeVar

from researchclaw.experiment.workspace_agent_ledger import WorkspaceAgentLedger

logger = logging.getLogger(__name__)

SessionT = TypeVar("SessionT", bound="ResumableWorkspaceSession")


class ResumableWorkspaceSession(Protocol):
    """Session methods needed by the resume manager."""

    session_name: str

    def ensure_session(self) -> None:
        """Reconnect to an existing named session or create it."""
        ...

    def fork_from_archive(self: SessionT, archive_path: Path, fork_name: str) -> SessionT:
        """Import a ledger export under *fork_name* and return that session."""
        ...


class WorkspaceResumeManager:
    """Reconnect or restore a workspace ACP session from the ledger."""

    def __init__(
        self,
        ledger: WorkspaceAgentLedger,
        session: ResumableWorkspaceSession,
    ) -> None:
        self.ledger = ledger
        self.session = session

    def resume(self) -> ResumableWorkspaceSession | None:
        """Return a usable session, or None when the caller must rebuild context."""
        try:
            self.session.ensure_session()
            return self.session
        except Exception as exc:  # noqa: BLE001
            logger.debug("Workspace ACP reconnect failed: %s", exc)

        archive = self.ledger.latest_export()
        if archive is None:
            return None
        try:
            return self.session.fork_from_archive(archive, self.session.session_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Workspace ACP import failed: %s", exc)
            return None
