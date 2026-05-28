from __future__ import annotations

from pathlib import Path

import pytest

from researchclaw.experiment.record import ExperimentRegistry, compute_result_hashes
from researchclaw.experiment.workspace import ExperimentRecord


class TestComputeResultHashes:
    def test_sha256_of_known_content(self, tmp_path: Path) -> None:
        result = tmp_path / "results" / "metrics.json"
        result.parent.mkdir()
        result.write_text("hello\n", encoding="utf-8")

        hashes = compute_result_hashes(["results/metrics.json"], tmp_path)

        assert hashes["results/metrics.json"] == (
            "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
        )

    def test_empty_paths(self, tmp_path: Path) -> None:
        assert compute_result_hashes([], tmp_path) == {}

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_result_hashes(["missing.json"], tmp_path)


class TestExperimentRegistry:
    def test_append(self, tmp_path: Path) -> None:
        registry = ExperimentRegistry(tmp_path / "registry.jsonl")
        record = _record(stage=10)

        registry.append(record)

        assert registry.read_all() == [record]

    def test_read_all(self, tmp_path: Path) -> None:
        registry = ExperimentRegistry(tmp_path / "registry.jsonl")
        records = [_record(stage=10), _record(stage=13)]
        for record in records:
            registry.append(record)

        assert registry.read_all() == records

    def test_read_by_stage(self, tmp_path: Path) -> None:
        registry = ExperimentRegistry(tmp_path / "registry.jsonl")
        stage_10 = _record(stage=10)
        stage_13 = _record(stage=13)
        registry.append(stage_10)
        registry.append(stage_13)

        assert registry.read_by_stage(13) == [stage_13]

    def test_creates_file_on_first_append(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "registry.jsonl"
        registry = ExperimentRegistry(path)

        registry.append(_record(stage=10))

        assert path.exists()

    def test_handles_corrupt_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.jsonl"
        path.write_text('{"stage": 10}\nnot-json\n', encoding="utf-8")
        registry = ExperimentRegistry(path)

        assert registry.read_all() == []


def _record(stage: int) -> ExperimentRecord:
    return ExperimentRecord(
        workspace="/tmp/workspace",
        stage=stage,
        base_sha=f"base-{stage}",
        agent_commit_sha=f"agent-{stage}",
        provider="codex",
        agent_manifest=".researchclaw/run_manifest.json",
        submitter="local",
        job_id=f"local_{stage}",
        result_paths=["outputs/metrics.json"],
        result_hashes={"outputs/metrics.json": f"hash-{stage}"},
        recorded_at="2026-05-28T00:00:00Z",
    )
