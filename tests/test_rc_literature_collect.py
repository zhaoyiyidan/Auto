from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.stage_impls import _literature as literature
from researchclaw.pipeline.stages import StageStatus


MANUAL_SEARCH_FIELDS = (
    "cite_key",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "arxiv_id",
    "url",
    "pdf_url",
    "source",
    "matched_query",
    "paper_type",
    "abstract",
    "full_text_available",
    "full_text_summary",
    "key_evidence",
    "datasets",
    "metrics",
    "limitations",
    "relevance_reason",
    "quality_notes",
    "bibtex",
)


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> object:
        _ = kwargs
        self.calls.append(messages)
        raise AssertionError("Stage 4 manual handoff must not call the LLM")


class FakePaper:
    source = "unit_test"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cite_key": "legacy2024paper",
            "title": "Legacy API Search Paper",
            "authors": [{"name": "Legacy Author"}],
            "year": 2024,
            "venue": "Unit Tests",
            "url": "https://example.test/legacy",
            "abstract": "A legacy API result for the literature collect stage.",
        }

    def to_bibtex(self) -> str:
        return "@article{legacy2024paper,\n  title={Legacy API Search Paper},\n  year={2024}\n}"


@pytest.fixture()
def adapters() -> AdapterBundle:
    return AdapterBundle()


@pytest.fixture(autouse=True)
def no_external_literature_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "researchclaw.literature.search.search_papers_multi_query",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr("researchclaw.data.load_seminal_papers", lambda topic: [])


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    _write_stage3_artifacts(path)
    return path


@pytest.fixture()
def stage_dir(run_dir: Path) -> Path:
    path = run_dir / "stage-04"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _config(tmp_path: Path, *, manual_search: bool = True) -> RCConfig:
    data: dict[str, Any] = {
        "project": {"name": "rc-literature-test", "mode": "docs-first"},
        "research": {
            "topic": "retrieval augmented generation for code agents",
            "domains": ["ml", "agents"],
            "daily_paper_count": 2,
            "quality_threshold": 0.7,
            "manual_search": manual_search,
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
        "web_search": {"enabled": False},
        "security": {"hitl_required_stages": []},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def _write_stage3_artifacts(run_dir: Path) -> None:
    stage3 = run_dir / "stage-03"
    stage3.mkdir(parents=True, exist_ok=True)
    (stage3 / "search_plan.yaml").write_text(
        """
topic: retrieval augmented generation for code agents
search_strategies:
  - name: core
    queries:
      - retrieval augmented generation code agents
      - agentic software engineering literature review
filters:
  min_year: 2021
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (stage3 / "queries.json").write_text(
        json.dumps(
            {
                "queries": [
                    "retrieval augmented generation code agents",
                    "agentic software engineering literature review",
                ],
                "year_min": 2021,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "stage-01").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage-01" / "goal.md").write_text(
        "Build evidence-backed research around RAG code agents.",
        encoding="utf-8",
    )
    (run_dir / "stage-02").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage-02" / "problem_tree.md").write_text(
        "1. What retrieval signals matter?\n2. Which baselines are credible?",
        encoding="utf-8",
    )


def _manual_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cite_key": "smith2024ragagents",
        "title": "Retrieval Augmented Generation for Code Agents",
        "authors": "Smith, J. and Jones, K.",
        "year": 2024,
        "venue": "ICSE",
        "doi": "10.1145/example",
        "arxiv_id": "2401.12345",
        "url": "https://example.test/paper",
        "pdf_url": "https://example.test/paper.pdf",
        "source": "search_agent",
        "matched_query": "retrieval augmented generation code agents",
        "paper_type": "method",
        "abstract": "RAG methods for code agents improve repository-level tasks.",
        "full_text_available": True,
        "full_text_summary": "The paper evaluates a RAG code agent on repository tasks.",
        "key_evidence": [
            "Full text reports stronger pass@1 than non-retrieval baselines."
        ],
        "datasets": ["SWE-bench"],
        "metrics": {"pass@1": 0.42},
        "limitations": "Single benchmark family and limited ablations.",
        "relevance_reason": "Directly studies RAG for code agents.",
        "quality_notes": "Peer-reviewed and includes baseline comparisons.",
        "bibtex": (
            "@inproceedings{smith2024ragagents,\n"
            "  title={Retrieval Augmented Generation for Code Agents},\n"
            "  author={Smith, J. and Jones, K.},\n"
            "  year={2024}\n"
            "}"
        ),
    }
    row.update(overrides)
    return row


def _write_manual_results(stage_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (stage_dir / "manual_search_results.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_first_run_generates_template_files_and_returns_paused(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.PAUSED
    assert result.decision == "awaiting_manual_search"
    assert (stage_dir / "search_agent_prompt.md").is_file()
    assert (stage_dir / "search_agent_output_template.jsonl").is_file()
    assert (stage_dir / "manual_literature_instructions.md").is_file()
    assert (stage_dir / "search_meta.json").is_file()


def test_first_run_does_not_call_search_apis(
    tmp_path: Path,
    run_dir: Path,
    stage_dir: Path,
    adapters: AdapterBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path)

    def forbidden_search(*args: object, **kwargs: object) -> list[object]:
        _ = args, kwargs
        raise AssertionError("Stage 4 manual handoff must not call search APIs")

    monkeypatch.setattr(
        "researchclaw.literature.search.search_papers_multi_query",
        forbidden_search,
    )

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.PAUSED


def test_first_run_does_not_generate_placeholder_or_candidates(
    tmp_path: Path,
    run_dir: Path,
    stage_dir: Path,
    adapters: AdapterBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr(
        "researchclaw.literature.search.search_papers_multi_query",
        lambda *args, **kwargs: [],
    )

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.PAUSED
    assert not (stage_dir / "candidates.jsonl").exists()
    assert "[Placeholder]" not in "\n".join(
        path.read_text(encoding="utf-8") for path in stage_dir.glob("*")
    )


def test_first_run_does_not_call_llm_for_papers(
    tmp_path: Path,
    run_dir: Path,
    stage_dir: Path,
    adapters: AdapterBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr(
        "researchclaw.literature.search.search_papers_multi_query",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr("researchclaw.data.load_seminal_papers", lambda topic: [])
    llm = FakeLLMClient()

    result = literature._execute_literature_collect(
        stage_dir, run_dir, cfg, adapters, llm=llm
    )

    assert result.status is StageStatus.PAUSED
    assert llm.calls == []


def test_resume_detects_manual_results_and_converts_to_candidates(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_manual_results(stage_dir, [_manual_row()])

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    assert (stage_dir / "candidates.jsonl").is_file()
    assert (stage_dir / "references.bib").is_file()
    assert (stage_dir / "paper_evidence.jsonl").is_file()


def test_resume_writes_correct_search_meta(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_manual_results(stage_dir, [_manual_row()])

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    meta = json.loads((stage_dir / "search_meta.json").read_text(encoding="utf-8"))
    assert meta["source"] == "manual_search_agent"
    assert meta["real_search"] is False
    assert meta["external_manual"] is True
    assert meta["total_candidates"] == 1
    assert meta["status"] == "completed"


def test_resume_maps_all_22_fields_correctly(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    row = _manual_row()
    _write_manual_results(stage_dir, [row])

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    candidate = _read_jsonl(stage_dir / "candidates.jsonl")[0]
    evidence = _read_jsonl(stage_dir / "paper_evidence.jsonl")[0]
    assert candidate["cite_key"] == row["cite_key"]
    assert candidate["title"] == row["title"]
    assert candidate["doi"] == row["doi"]
    assert candidate["arxiv_id"] == row["arxiv_id"]
    assert candidate["pdf_url"] == row["pdf_url"]
    assert candidate["matched_query"] == row["matched_query"]
    assert candidate["paper_type"] == row["paper_type"]
    assert candidate["abstract"] == row["abstract"]
    assert evidence["cite_key"] == row["cite_key"]
    assert evidence["full_text_available"] is True
    assert evidence["full_text_summary"] == row["full_text_summary"]
    assert evidence["key_evidence"] == row["key_evidence"]
    assert evidence["datasets"] == row["datasets"]
    assert evidence["metrics"] == row["metrics"]
    assert evidence["limitations"] == row["limitations"]
    assert evidence["relevance_reason"] == row["relevance_reason"]
    assert evidence["quality_notes"] == row["quality_notes"]


def test_resume_preserves_bibtex_in_references_bib(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    row = _manual_row()
    _write_manual_results(stage_dir, [row])

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    assert row["bibtex"] in (stage_dir / "references.bib").read_text(
        encoding="utf-8"
    )


def test_resume_authors_string_to_list_conversion(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_manual_results(stage_dir, [_manual_row(authors="Smith, J. and Jones, K.")])

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    candidate = _read_jsonl(stage_dir / "candidates.jsonl")[0]
    assert candidate["authors"] == [{"name": "Smith, J."}, {"name": "Jones, K."}]


def test_manual_results_empty_file_returns_failed(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    (stage_dir / "manual_search_results.jsonl").write_text("", encoding="utf-8")

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.FAILED
    assert "empty" in (result.error or "").lower()


def test_manual_results_invalid_jsonl_returns_failed(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    (stage_dir / "manual_search_results.jsonl").write_text("{not-json}\n", encoding="utf-8")

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.FAILED
    assert "line 1" in (result.error or "").lower()


def test_manual_results_missing_required_fields_returns_failed(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_manual_results(stage_dir, [{"title": "Missing cite key"}])

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.FAILED
    assert "cite_key" in (result.error or "")


def test_resume_idempotent_when_candidates_already_exist(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    existing = _manual_row(title="Already Converted")
    (stage_dir / "candidates.jsonl").write_text(
        json.dumps(existing, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    assert json.loads((stage_dir / "candidates.jsonl").read_text(encoding="utf-8"))[
        "title"
    ] == "Already Converted"


def test_search_agent_prompt_contains_topic_queries_and_format(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.PAUSED
    prompt = (stage_dir / "search_agent_prompt.md").read_text(encoding="utf-8")
    assert "retrieval augmented generation for code agents" in prompt
    assert "retrieval augmented generation code agents" in prompt
    assert "JSONL" in prompt
    assert "full_text_summary" in prompt
    assert "key_evidence" in prompt


def test_template_jsonl_has_all_22_fields(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.PAUSED
    template = _read_jsonl(stage_dir / "search_agent_output_template.jsonl")[0]
    assert tuple(template.keys()) == MANUAL_SEARCH_FIELDS


def test_legacy_mode_still_works_when_manual_search_false(
    tmp_path: Path,
    run_dir: Path,
    stage_dir: Path,
    adapters: AdapterBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path, manual_search=False)
    calls: list[list[str]] = []

    def fake_search(queries: list[str], **kwargs: object) -> list[FakePaper]:
        _ = kwargs
        calls.append(queries)
        return [FakePaper()]

    monkeypatch.setattr(
        "researchclaw.literature.search.search_papers_multi_query",
        fake_search,
    )
    monkeypatch.setattr("researchclaw.data.load_seminal_papers", lambda topic: [])

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    assert calls
    assert (stage_dir / "candidates.jsonl").is_file()
    assert "legacy2024paper" in (stage_dir / "references.bib").read_text(
        encoding="utf-8"
    )


def test_paper_evidence_jsonl_maps_by_cite_key(
    tmp_path: Path, run_dir: Path, stage_dir: Path, adapters: AdapterBundle
) -> None:
    cfg = _config(tmp_path)
    _write_manual_results(
        stage_dir,
        [
            _manual_row(cite_key="smith2024ragagents"),
            _manual_row(
                cite_key="lee2023baselines",
                title="Baselines for Code Agents",
                key_evidence=["Full text includes baseline ablations."],
            ),
        ],
    )

    result = literature._execute_literature_collect(stage_dir, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    evidence_by_key = {
        row["cite_key"]: row for row in _read_jsonl(stage_dir / "paper_evidence.jsonl")
    }
    assert evidence_by_key["smith2024ragagents"]["key_evidence"] == [
        "Full text reports stronger pass@1 than non-retrieval baselines."
    ]
    assert evidence_by_key["lee2023baselines"]["key_evidence"] == [
        "Full text includes baseline ablations."
    ]
