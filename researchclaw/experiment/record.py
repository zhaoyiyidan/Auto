"""Append-only records for workspace-native experiment provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from researchclaw.experiment.workspace import ExperimentRecord


def compute_result_hashes(
    result_paths: list[str],
    workspace_path: Path,
) -> dict[str, str]:
    """Return SHA256 hashes for result paths relative to *workspace_path*."""
    hashes: dict[str, str] = {}
    root = workspace_path.resolve()
    for rel in result_paths:
        path = (root / rel).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[rel] = digest.hexdigest()
    return hashes


class ExperimentRegistry:
    """Append-only JSONL registry for workspace experiment records."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path

    def append(self, record: ExperimentRecord) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            handle.flush()

    def read_all(self) -> list[ExperimentRecord]:
        if not self.registry_path.exists():
            return []
        records: list[ExperimentRecord] = []
        for line in self.registry_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(ExperimentRecord.from_dict(payload))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return records

    def read_by_stage(self, stage: int) -> list[ExperimentRecord]:
        return [record for record in self.read_all() if record.stage == stage]
