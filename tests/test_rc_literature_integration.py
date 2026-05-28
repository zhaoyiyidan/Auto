from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline import runner as rc_runner
from researchclaw.pipeline.contracts import CONTRACTS
from researchclaw.pipeline.executor import StageResult
from researchclaw.pipeline.stage_impls import _literature as literature
from researchclaw.pipeline.stages import Stage, StageStatus


def _disable_external_literature_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        "researchclaw.literature.search.search_papers_multi_query",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr("researchclaw.data.load_seminal_papers", lambda topic: [])


def _config(tmp_path: Path) -> RCConfig:
    data: dict[str, Any] = {
        "project": {"name": "rc-literature-integration", "mode": "docs-first"},
        "research": {
            "topic": "retrieval augmented generation for code agents",
            "domains": ["ml", "agents"],
            "manual_search": True,
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


def _write_stage3(run_dir: Path) -> None:
    stage3 = run_dir / "stage-03"
    stage3.mkdir(parents=True, exist_ok=True)
    (stage3 / "search_plan.yaml").write_text(
        "search_strategies:\n"
        "  - name: core\n"
        "    queries: [retrieval augmented generation code agents]\n"
        "filters:\n"
        "  min_year: 2021\n",
        encoding="utf-8",
    )
    (stage3 / "queries.json").write_text(
        json.dumps(
            {
                "queries": ["retrieval augmented generation code agents"],
                "year_min": 2021,
            }
        ),
        encoding="utf-8",
    )


def _manual_row() -> dict[str, Any]:
    return {
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
        "abstract": "Retrieval augmented generation improves code agents.",
        "full_text_available": True,
        "full_text_summary": "Full text evaluates RAG code agents.",
        "key_evidence": ["Reports pass@1 improvement."],
        "datasets": ["SWE-bench"],
        "metrics": {"pass@1": 0.42},
        "limitations": "Limited ablations.",
        "relevance_reason": "Directly studies the topic.",
        "quality_notes": "Peer-reviewed with baselines.",
        "bibtex": "@inproceedings{smith2024ragagents,title={RAG for Code Agents}}",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_full_pause_resume_cycle(tmp_path: Path, monkeypatch) -> None:
    _disable_external_literature_sources(monkeypatch)
    cfg = _config(tmp_path)
    adapters = AdapterBundle()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_stage3(run_dir)
    stage4 = run_dir / "stage-04"
    stage4.mkdir()

    first = literature._execute_literature_collect(stage4, run_dir, cfg, adapters)

    assert first.status is StageStatus.PAUSED
    (stage4 / "manual_search_results.jsonl").write_text(
        json.dumps(_manual_row(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    resumed = literature._execute_literature_collect(stage4, run_dir, cfg, adapters)

    assert resumed.status is StageStatus.DONE
    assert (stage4 / "candidates.jsonl").is_file()
    assert (stage4 / "paper_evidence.jsonl").is_file()


def test_stage4_to_stage5_flow_with_evidence(tmp_path: Path, monkeypatch) -> None:
    _disable_external_literature_sources(monkeypatch)
    cfg = _config(tmp_path)
    adapters = AdapterBundle()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_stage3(run_dir)
    stage4 = run_dir / "stage-04"
    stage4.mkdir()
    (stage4 / "manual_search_results.jsonl").write_text(
        json.dumps(_manual_row(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    collect = literature._execute_literature_collect(stage4, run_dir, cfg, adapters)
    stage5 = run_dir / "stage-05"
    stage5.mkdir()
    screen = literature._execute_literature_screen(stage5, run_dir, cfg, adapters)

    assert collect.status is StageStatus.DONE
    assert screen.status is StageStatus.DONE
    shortlist = _read_jsonl(stage5 / "shortlist.jsonl")
    assert shortlist[0]["cite_key"] == "smith2024ragagents"
    assert shortlist[0]["key_evidence"] == ["Reports pass@1 improvement."]


def test_stage6_reads_enriched_shortlist(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    adapters = AdapterBundle()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stage5 = run_dir / "stage-05"
    stage5.mkdir(parents=True)
    (stage5 / "shortlist.jsonl").write_text(
        json.dumps(
            {
                "cite_key": "smith2024ragagents",
                "title": "Retrieval Augmented Generation for Code Agents",
                "key_evidence": ["Reports pass@1 improvement."],
                "datasets": ["SWE-bench"],
                "metrics": {"pass@1": 0.42},
                "limitations": "Limited ablations.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    stage6 = run_dir / "stage-06"
    stage6.mkdir()

    result = literature._execute_knowledge_extract(stage6, run_dir, cfg, adapters)

    assert result.status is StageStatus.DONE
    cards = list((stage6 / "cards").glob("*.md"))
    assert cards
    assert "smith2024ragagents" in cards[0].read_text(encoding="utf-8")


def test_runner_summary_reports_paused_correctly(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _config(tmp_path)
    adapters = AdapterBundle()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_execute_stage(stage: Stage, **kwargs: object) -> StageResult:
        _ = kwargs
        if stage is Stage.LITERATURE_COLLECT:
            return StageResult(
                stage=stage,
                status=StageStatus.PAUSED,
                artifacts=(
                    "search_agent_prompt.md",
                    "search_agent_output_template.jsonl",
                    "manual_literature_instructions.md",
                    "search_meta.json",
                ),
                decision="awaiting_manual_search",
                error="Manual literature search required",
            )
        return StageResult(stage=stage, status=StageStatus.DONE, artifacts=("ok.md",))

    monkeypatch.setattr(rc_runner, "execute_stage", fake_execute_stage)

    results = rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="run-paused",
        config=cfg,
        adapters=adapters,
    )

    assert results[-1].stage is Stage.LITERATURE_COLLECT
    assert results[-1].status is StageStatus.PAUSED
    summary = json.loads((run_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["final_status"] == "paused"
    assert summary["final_stage"] == 4


def test_resume_from_checkpoint_after_stage3_returns_stage4(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint.json").write_text(
        json.dumps({"last_completed_stage": 3, "run_id": "run-resume"}),
        encoding="utf-8",
    )

    assert rc_runner.resume_from_checkpoint(run_dir) is Stage.LITERATURE_COLLECT


def test_contracts_updated() -> None:
    collect = CONTRACTS[Stage.LITERATURE_COLLECT]
    screen = CONTRACTS[Stage.LITERATURE_SCREEN]

    assert collect.output_files == ("candidates.jsonl", "search_meta.json")
    assert collect.max_retries == 1
    assert "Manual search agent handoff" in collect.dod
    assert screen.input_files == ("candidates.jsonl",)
    assert screen.output_files == ("shortlist.jsonl",)
    assert "evidence enrichment" in screen.dod
