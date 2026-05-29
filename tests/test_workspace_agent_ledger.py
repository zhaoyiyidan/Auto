from __future__ import annotations

import json
from pathlib import Path

from researchclaw.experiment.workspace import WorkspaceAgentResult
from researchclaw.experiment.workspace_agent_ledger import WorkspaceAgentLedger


class DummySession:
    def __init__(self) -> None:
        self.exports: list[Path] = []

    def export_session(self, output_path: Path) -> None:
        self.exports.append(output_path)
        output_path.write_bytes(b"session")


def test_ledger_root_is_researchclaw_workspace_agent_dir(tmp_path: Path) -> None:
    ledger = WorkspaceAgentLedger(tmp_path)

    assert ledger.root == tmp_path / ".researchclaw" / "workspace-agent"
    assert ledger.root.exists()


def test_stage_dir_names(tmp_path: Path) -> None:
    ledger = WorkspaceAgentLedger(tmp_path)

    assert ledger.stage_dir(10).name == "stage-10"
    assert ledger.stage_dir(13).name == "stage-13"
    assert ledger.stage_dir(14, iteration=1).name == "repair-001"


def test_write_prompt_and_base_sha(tmp_path: Path) -> None:
    ledger = WorkspaceAgentLedger(tmp_path)
    stage_dir = ledger.stage_dir(10)

    ledger.write_prompt(stage_dir, "modify the workspace")
    ledger.write_base_sha(stage_dir, "abc123")

    assert (stage_dir / "prompt.md").read_text(encoding="utf-8") == "modify the workspace"
    assert (stage_dir / "base_sha.txt").read_text(encoding="utf-8") == "abc123\n"


def test_write_agent_result(tmp_path: Path) -> None:
    ledger = WorkspaceAgentLedger(tmp_path)
    stage_dir = ledger.stage_dir(13)
    result = WorkspaceAgentResult(
        base_sha="base",
        agent_commit_sha="head",
        manifest_path="run_manifest.json",
        diff_stat=" train.py | 1 +",
        raw_log="done",
        provider_name="acp",
        elapsed_sec=1.5,
    )

    ledger.write_agent_result(stage_dir, result)

    payload = json.loads((stage_dir / "agent_result.json").read_text(encoding="utf-8"))
    assert payload["base_sha"] == "base"
    assert payload["agent_commit_sha"] == "head"
    assert (stage_dir / "agent_commit_sha.txt").read_text(encoding="utf-8") == "head\n"
    assert (stage_dir / "diff_stat.txt").read_text(encoding="utf-8") == " train.py | 1 +\n"


def test_write_session_export_and_latest_export(tmp_path: Path) -> None:
    ledger = WorkspaceAgentLedger(tmp_path)
    session = DummySession()
    first = ledger.stage_dir(10)
    second = ledger.stage_dir(13)

    first_export = ledger.write_session_export(first, session)
    second_export = ledger.write_session_export(second, session)

    assert first_export == first / "session_export.tar.gz"
    assert second_export == second / "session_export.tar.gz"
    assert session.exports == [first_export, second_export]
    assert ledger.latest_export() == second_export


def test_session_meta_roundtrip(tmp_path: Path) -> None:
    ledger = WorkspaceAgentLedger(tmp_path)
    meta = {
        "session_name": "researchclaw-code-run-1",
        "agent": "claude",
        "cwd": "/workspace",
    }

    ledger.save_session_meta(meta)

    assert ledger.load_session_meta() == meta
