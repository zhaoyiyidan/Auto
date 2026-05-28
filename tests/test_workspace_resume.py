from __future__ import annotations

from pathlib import Path

from researchclaw.experiment.workspace_agent_ledger import WorkspaceAgentLedger
from researchclaw.experiment.workspace_resume import WorkspaceResumeManager


class FakeSession:
    def __init__(self, *, ensure_error: Exception | None = None) -> None:
        self.session_name = "researchclaw-code-run-1"
        self.ensure_error = ensure_error
        self.ensure_calls = 0
        self.imports: list[tuple[Path, str]] = []
        self.imported: FakeSession | None = None

    def ensure_session(self) -> None:
        self.ensure_calls += 1
        if self.ensure_error:
            raise self.ensure_error

    def fork_from_archive(self, archive_path: Path, fork_name: str) -> FakeSession:
        self.imports.append((archive_path, fork_name))
        if self.imported is None:
            raise RuntimeError("import failed")
        return self.imported


def test_resume_reconnects_existing_named_session(tmp_path: Path) -> None:
    ledger = WorkspaceAgentLedger(tmp_path)
    session = FakeSession()

    resumed = WorkspaceResumeManager(ledger, session).resume()

    assert resumed is session
    assert session.ensure_calls == 1


def test_resume_returns_none_without_session_or_export(tmp_path: Path) -> None:
    ledger = WorkspaceAgentLedger(tmp_path)
    session = FakeSession(ensure_error=RuntimeError("missing session"))

    resumed = WorkspaceResumeManager(ledger, session).resume()

    assert resumed is None
    assert session.ensure_calls == 1
    assert session.imports == []


def test_resume_imports_latest_export_when_reconnect_fails(tmp_path: Path) -> None:
    ledger = WorkspaceAgentLedger(tmp_path)
    stage_dir = ledger.stage_dir(13)
    archive = stage_dir / "session_export.tar.gz"
    archive.write_bytes(b"export")
    session = FakeSession(ensure_error=RuntimeError("missing session"))
    imported = FakeSession()
    session.imported = imported

    resumed = WorkspaceResumeManager(ledger, session).resume()

    assert resumed is imported
    assert session.imports == [(archive, "researchclaw-code-run-1")]


def test_resume_returns_none_when_import_fails(tmp_path: Path) -> None:
    ledger = WorkspaceAgentLedger(tmp_path)
    archive = ledger.stage_dir(10) / "session_export.tar.gz"
    archive.write_bytes(b"export")
    session = FakeSession(ensure_error=RuntimeError("missing session"))

    resumed = WorkspaceResumeManager(ledger, session).resume()

    assert resumed is None
    assert session.imports == [(archive, "researchclaw-code-run-1")]
