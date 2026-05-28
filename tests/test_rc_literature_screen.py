from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.stage_impls import _literature as literature
from researchclaw.pipeline.stages import StageStatus


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> object:
        _ = kwargs
        self.calls.append(messages)
        from researchclaw.llm.client import LLMResponse

        return LLMResponse(content=self.response, model="fake-model")


@pytest.fixture()
def adapters() -> AdapterBundle:
    return AdapterBundle()


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    (path / "stage-04").mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def stage_dir(run_dir: Path) -> Path:
    path = run_dir / "stage-05"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _config(tmp_path: Path) -> RCConfig:
    data: dict[str, Any] = {
        "project": {"name": "rc-screen-test", "mode": "docs-first"},
        "research": {
            "topic": "retrieval augmented generation for code agents",
            "domains": ["ml", "agents"],
            "quality_threshold": 0.7,
        },
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "",
        },
        "security": {"hitl_required_stages": []},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def _candidate(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cite_key": "smith2024ragagents",
        "title": "Retrieval Augmented Generation for Code Agents",
        "authors": [{"name": "Smith, J."}, {"name": "Jones, K."}],
        "year": 2024,
        "venue": "ICSE",
        "doi": "10.1145/example",
        "arxiv_id": "2401.12345",
        "url": "https://example.test/paper",
        "pdf_url": "https://example.test/paper.pdf",
        "source": "manual_search_agent",
        "paper_type": "method",
        "abstract": "Retrieval augmented generation improves code agents on repository tasks.",
        "bibtex": "@inproceedings{smith2024ragagents,title={RAG for Code Agents}}",
    }
    row.update(overrides)
    return row


def _evidence(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cite_key": "smith2024ragagents",
        "full_text_available": True,
        "full_text_summary": "Full text evaluates RAG code agents on SWE-bench.",
        "key_evidence": ["Reports pass@1 improvement over no-retrieval baseline."],
        "datasets": ["SWE-bench"],
        "metrics": {"pass@1": 0.42},
        "limitations": "Limited ablations.",
        "relevance_reason": "Directly studies code-agent RAG.",
        "quality_notes": "Peer-reviewed with baselines.",
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_candidates(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(run_dir / "stage-04" / "candidates.jsonl", rows)


def _write_evidence(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(run_dir / "stage-04" / "paper_evidence.jsonl", rows)


def _read_shortlist(stage_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (stage_dir / "shortlist.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_screen_preserves_all_original_candidate_fields(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_candidates(run_dir, [_candidate()])

    result = literature._execute_literature_screen(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    row = _read_shortlist(stage_dir)[0]
    for key in ("cite_key", "doi", "arxiv_id", "url", "pdf_url", "bibtex", "authors"):
        assert row[key] == _candidate()[key]


def test_screen_reads_paper_evidence_when_available(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_candidates(run_dir, [_candidate()])
    _write_evidence(run_dir, [_evidence()])
    llm = FakeLLMClient(
        json.dumps(
            {
                "shortlist": [
                    {
                        "cite_key": "smith2024ragagents",
                        "title": "Retrieval Augmented Generation for Code Agents",
                        "relevance_score": 0.95,
                        "quality_score": 0.9,
                        "keep_reason": "Direct evidence",
                    }
                ]
            }
        )
    )

    result = literature._execute_literature_screen(
        stage_dir, run_dir, cfg, adapters, llm=llm
    )

    assert result.status is StageStatus.DONE
    sent_text = "\n".join(message["content"] for call in llm.calls for message in call)
    assert "Full text evaluates RAG code agents on SWE-bench" in sent_text
    assert "Reports pass@1 improvement" in sent_text


def test_screen_gracefully_degrades_without_paper_evidence(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_candidates(run_dir, [_candidate()])

    result = literature._execute_literature_screen(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    assert _read_shortlist(stage_dir)


def test_screen_preserves_authors_field_in_shortlist(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_candidates(run_dir, [_candidate()])

    result = literature._execute_literature_screen(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    assert _read_shortlist(stage_dir)[0]["authors"] == _candidate()["authors"]


def test_screen_injects_evidence_into_shortlist_rows(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_candidates(run_dir, [_candidate()])
    _write_evidence(run_dir, [_evidence()])

    result = literature._execute_literature_screen(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    row = _read_shortlist(stage_dir)[0]
    assert row["key_evidence"] == [
        "Reports pass@1 improvement over no-retrieval baseline."
    ]
    assert row["datasets"] == ["SWE-bench"]
    assert row["metrics"] == {"pass@1": 0.42}
    assert row["limitations"] == "Limited ablations."


def test_screen_evidence_no_match_does_not_add_empty_fields(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_candidates(run_dir, [_candidate()])
    _write_evidence(run_dir, [_evidence(cite_key="other2024paper")])

    result = literature._execute_literature_screen(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    row = _read_shortlist(stage_dir)[0]
    assert "key_evidence" not in row
    assert "datasets" not in row
    assert "metrics" not in row
    assert "limitations" not in row


def test_screen_fallback_shortlist_preserves_fields(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_candidates(run_dir, [_candidate(cite_key="fallback2024rag")])
    _write_evidence(run_dir, [_evidence(cite_key="fallback2024rag")])

    result = literature._execute_literature_screen(
        stage_dir, run_dir, cfg, adapters, llm=None
    )

    assert result.status is StageStatus.DONE
    row = _read_shortlist(stage_dir)[0]
    assert row["cite_key"] == "fallback2024rag"
    assert row["authors"] == _candidate()["authors"]
    assert row["key_evidence"] == [
        "Reports pass@1 improvement over no-retrieval baseline."
    ]
