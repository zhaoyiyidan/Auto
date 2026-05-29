"""Append-only ledger files for workspace-native ACP agent runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from researchclaw.experiment.workspace import WorkspaceAgentResult


class WorkspaceAgentLedger:
    """Manage ResearchClaw-owned workspace agent provenance files."""

    def __init__(self, base_dir: Path) -> None:
        base = Path(base_dir)
        if base.name == "workspace-agent" and base.parent.name == ".researchclaw":
            self.root = base
        else:
            self.root = base / ".researchclaw" / "workspace-agent"
        self.root.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: int, iteration: int | None = None) -> Path:
        """Return the ledger directory for a stage or repair iteration."""
        if stage == 14 and iteration is not None:
            name = f"repair-{iteration:03d}"
        elif iteration is not None:
            name = f"stage-{stage}-iter-{iteration:03d}"
        else:
            name = f"stage-{stage}"
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_prompt(self, stage_dir: Path, prompt: str) -> Path:
        return self._write_text(stage_dir / "prompt.md", prompt)

    def write_base_sha(self, stage_dir: Path, base_sha: str) -> Path:
        return self._write_text(stage_dir / "base_sha.txt", base_sha.rstrip() + "\n")

    def write_agent_result(
        self,
        stage_dir: Path,
        result: WorkspaceAgentResult,
    ) -> Path:
        payload = asdict(result)
        path = self._write_json(stage_dir / "agent_result.json", payload)
        if result.agent_commit_sha:
            self._write_text(
                stage_dir / "agent_commit_sha.txt",
                result.agent_commit_sha.rstrip() + "\n",
            )
        self._write_text(stage_dir / "diff_stat.txt", result.diff_stat.rstrip() + "\n")
        return path

    def write_session_export(self, stage_dir: Path, session: Any) -> Path:
        output_path = stage_dir / "session_export.tar.gz"
        session.export_session(output_path)
        return output_path

    def copy_manifest(self, stage_dir: Path, manifest_path: Path) -> Path:
        destination = stage_dir / "run_manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, destination)
        return destination

    def write_submit_result(self, stage_dir: Path, submit_result: Any) -> Path:
        return self._write_json(stage_dir / "submit_result.json", _to_payload(submit_result))

    def write_registry_record(self, stage_dir: Path, record: Any) -> Path:
        return self._write_json(stage_dir / "registry_record.json", _to_payload(record))

    def latest_export(self) -> Path | None:
        exports = list(self.root.glob("**/session_export.tar.gz"))
        if not exports:
            return None
        return max(exports, key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))

    def save_session_meta(self, meta: dict[str, Any]) -> Path:
        return self._write_json(self.root / "session.json", meta)

    def load_session_meta(self) -> dict[str, Any]:
        path = self.root / "session.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_text(self, path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path


def _to_payload(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unsupported ledger payload: {type(value).__name__}")
