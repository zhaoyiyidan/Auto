"""Stages 3-6: Search strategy, literature collection, screening, and knowledge extraction."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import (
    StageResult,
    _build_fallback_queries,
    _chat_with_prompt,
    _extract_topic_keywords,
    _extract_yaml_block,
    _get_evolution_overlay,
    _parse_jsonl_rows,
    _read_prior_artifact,
    _safe_filename,
    _safe_json_loads,
    _utcnow_iso,
    _write_jsonl,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)

MANUAL_SEARCH_FIELDS: tuple[str, ...] = (
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
MANUAL_REQUIRED_FIELDS: tuple[str, ...] = ("cite_key", "title")
MANUAL_EVIDENCE_FIELDS: tuple[str, ...] = (
    "full_text_available",
    "full_text_summary",
    "key_evidence",
    "datasets",
    "metrics",
    "limitations",
    "relevance_reason",
    "quality_notes",
)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _expand_search_queries(queries: list[str], topic: str) -> list[str]:
    """Expand search queries for broader literature coverage.

    Generates additional queries by extracting key phrases from the topic
    and creating focused sub-queries. This ensures we find papers even when
    the original queries are too narrow or specific for arXiv.
    """
    expanded = list(queries)  # keep originals
    seen = {q.lower().strip() for q in queries}

    # Extract key phrases from topic by splitting on common delimiters
    # e.g. "Comparing A, B, and C on X with Y" → ["A", "B", "C", "X", "Y"]
    topic_words = topic.split()

    # Generate shorter, broader queries from the topic
    if len(topic_words) > 5:
        # First 5 words as a broader query
        broad = " ".join(topic_words[:5])
        if broad.lower().strip() not in seen:
            expanded.append(broad)
            seen.add(broad.lower().strip())

        # Last 5 words as another perspective
        tail = " ".join(topic_words[-5:])
        if tail.lower().strip() not in seen:
            expanded.append(tail)
            seen.add(tail.lower().strip())

    # Add "survey" and "benchmark" variants of the topic
    for suffix in ("survey", "benchmark", "comparison"):
        # Take first 4 content words + suffix
        short_topic = " ".join(topic_words[:4])
        variant = f"{short_topic} {suffix}"
        if variant.lower().strip() not in seen:
            expanded.append(variant)
            seen.add(variant.lower().strip())

    return expanded


def _load_manual_search_context(
    run_dir: Path, config: RCConfig
) -> tuple[list[str], int, str, str, str]:
    """Read Stage 3 and prior context for manual search handoff files."""
    topic = config.research.topic
    queries_text = _read_prior_artifact(run_dir, "queries.json")
    queries_data = _safe_json_loads(queries_text or "{}", {})
    raw_queries = queries_data.get("queries", []) if isinstance(queries_data, dict) else []
    queries = [str(q).strip() for q in raw_queries if str(q).strip()]
    if not queries:
        queries = _build_fallback_queries(topic)[:5]
    year_min = 2020
    if isinstance(queries_data, dict):
        try:
            year_min = int(queries_data.get("year_min", 2020))
        except (TypeError, ValueError):
            year_min = 2020
    plan_text = _read_prior_artifact(run_dir, "search_plan.yaml") or ""
    goal_text = _read_prior_artifact(run_dir, "goal.md") or ""
    problem_tree = _read_prior_artifact(run_dir, "problem_tree.md") or ""
    return queries, year_min, plan_text, goal_text, problem_tree


def _manual_search_template_row() -> dict[str, Any]:
    return {
        "cite_key": "smith2024ragagents",
        "title": "Retrieval Augmented Generation for Code Agents",
        "authors": "Smith, J. and Jones, K.",
        "year": 2024,
        "venue": "ICSE",
        "doi": "10.1145/example",
        "arxiv_id": "2401.12345",
        "url": "https://example.org/paper",
        "pdf_url": "https://example.org/paper.pdf",
        "source": "manual_search_agent",
        "matched_query": "retrieval augmented generation code agents",
        "paper_type": "method",
        "abstract": "One-paragraph abstract or concise abstract summary.",
        "full_text_available": True,
        "full_text_summary": "Short summary grounded in the paper body.",
        "key_evidence": [
            "Concrete finding, experiment, theorem, or comparison from the full text."
        ],
        "datasets": ["SWE-bench"],
        "metrics": {"pass@1": 0.42},
        "limitations": "Known limitations or threats to validity.",
        "relevance_reason": "Why this paper matters for the research topic.",
        "quality_notes": "Venue, reproducibility, citations, or methodological notes.",
        "bibtex": "@inproceedings{smith2024ragagents, title={...}, year={2024}}",
    }


def _render_search_agent_prompt(
    *,
    topic: str,
    queries: list[str],
    year_min: int,
    plan_text: str,
    goal_text: str,
    problem_tree: str,
) -> str:
    fields = ", ".join(MANUAL_SEARCH_FIELDS)
    query_lines = "\n".join(f"- {q}" for q in queries)
    template = json.dumps(_manual_search_template_row(), ensure_ascii=False)
    return (
        "# Manual Literature Search Agent Prompt\n\n"
        "You are a literature search agent. Find real, relevant papers for the "
        "research topic below. Prefer peer-reviewed papers, strong preprints, "
        "surveys, baselines, benchmarks, seminal work, and credible negative "
        "or limitation evidence. Inspect the abstract and paper body whenever "
        "a full text or PDF is available.\n\n"
        f"## Topic\n{topic}\n\n"
        f"## Research Goal\n{goal_text.strip() or '(not provided)'}\n\n"
        f"## Problem Tree\n{problem_tree.strip() or '(not provided)'}\n\n"
        f"## Search Queries\n{query_lines}\n\n"
        f"Minimum publication year: {year_min}\n\n"
        f"## Stage 3 Search Plan\n```yaml\n{plan_text.strip()}\n```\n\n"
        "## Output Requirements\n"
        "Return only JSONL. Each line must be one JSON object for one paper. "
        "Do not wrap the output in Markdown. Use these fields exactly:\n\n"
        f"{fields}\n\n"
        "paper_type must be one of: seminal, survey, method, benchmark, "
        "baseline, negative_result, related.\n\n"
        "If full text is available, fill full_text_summary and key_evidence "
        "with claims grounded in the paper body. If it is not available, set "
        "full_text_available to false and explain the evidence limit in "
        "quality_notes.\n\n"
        "## Example JSONL Line\n"
        f"{template}\n"
    )


def _render_manual_instructions(stage_dir: Path) -> str:
    return (
        "# Manual Literature Search Instructions\n\n"
        "Stage 4 is waiting for external manual search results.\n\n"
        "1. Send `stage-04/search_agent_prompt.md` to your search agent.\n"
        "2. Ask the search agent to return only JSONL in the requested schema.\n"
        "3. Save the returned JSONL to `stage-04/manual_search_results.jsonl`.\n"
        "4. Resume the pipeline from Stage 4, for example with `--resume` or "
        "`--from-stage 4`.\n\n"
        f"Current Stage 4 directory: `{stage_dir}`\n"
    )


def _generate_search_agent_handoff_files(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
) -> StageResult:
    stage_dir.mkdir(parents=True, exist_ok=True)
    queries, year_min, plan_text, goal_text, problem_tree = _load_manual_search_context(
        run_dir, config
    )
    prompt = _render_search_agent_prompt(
        topic=config.research.topic,
        queries=queries,
        year_min=year_min,
        plan_text=plan_text,
        goal_text=goal_text,
        problem_tree=problem_tree,
    )
    template = json.dumps(
        _manual_search_template_row(), ensure_ascii=False
    ) + "\n"
    instructions = _render_manual_instructions(stage_dir)
    (stage_dir / "search_agent_prompt.md").write_text(prompt, encoding="utf-8")
    (stage_dir / "search_agent_output_template.jsonl").write_text(
        template, encoding="utf-8"
    )
    (stage_dir / "manual_literature_instructions.md").write_text(
        instructions, encoding="utf-8"
    )
    (stage_dir / "search_meta.json").write_text(
        json.dumps(
            {
                "source": "manual_search_agent",
                "status": "awaiting_input",
                "real_search": False,
                "external_manual": True,
                "queries_used": queries,
                "year_min": year_min,
                "manual_results_file": "manual_search_results.jsonl",
                "ts": _utcnow_iso(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts = (
        "search_agent_prompt.md",
        "search_agent_output_template.jsonl",
        "manual_literature_instructions.md",
        "search_meta.json",
    )
    return StageResult(
        stage=Stage.LITERATURE_COLLECT,
        status=StageStatus.PAUSED,
        artifacts=artifacts,
        error=(
            "Manual literature search required: save search-agent JSONL to "
            "stage-04/manual_search_results.jsonl and resume."
        ),
        decision="awaiting_manual_search",
        evidence_refs=tuple(f"stage-04/{a}" for a in artifacts),
    )


def _validate_manual_results_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, [f"Missing manual search results: {path.name}"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return rows, [f"Cannot read {path.name}: {exc}"]
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSONL ({exc.msg})")
            continue
        if not isinstance(parsed, dict):
            errors.append(f"line {line_no}: expected a JSON object")
            continue
        missing = [
            field
            for field in MANUAL_REQUIRED_FIELDS
            if not str(parsed.get(field, "")).strip()
        ]
        if missing:
            errors.append(
                f"line {line_no}: missing required fields: {', '.join(missing)}"
            )
            continue
        rows.append(parsed)
    if not rows and not errors:
        errors.append(f"{path.name} is empty")
    return rows, errors


def _coerce_author_list(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        names = [part.strip() for part in re.split(r"\s+and\s+", value) if part.strip()]
        return [{"name": name} for name in names]
    if isinstance(value, list):
        authors: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                authors.append({"name": item.strip()})
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    authors.append({"name": name})
                elif item:
                    authors.append({str(k): str(v) for k, v in item.items()})
        return authors
    return []


def _coerce_year(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _manual_candidate_from_row(row: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "cite_key": str(row.get("cite_key", "")).strip(),
        "title": str(row.get("title", "")).strip(),
        "authors": _coerce_author_list(row.get("authors", [])),
        "year": _coerce_year(row.get("year", 0)),
        "venue": str(row.get("venue", "") or ""),
        "doi": str(row.get("doi", "") or ""),
        "arxiv_id": str(row.get("arxiv_id", "") or ""),
        "url": str(row.get("url", "") or ""),
        "pdf_url": str(row.get("pdf_url", "") or ""),
        "source": str(row.get("source", "") or "manual_search_agent"),
        "matched_query": str(row.get("matched_query", "") or ""),
        "paper_type": str(row.get("paper_type", "") or "related"),
        "abstract": str(row.get("abstract", "") or ""),
        "collected_at": _utcnow_iso(),
    }
    if str(row.get("bibtex", "")).strip():
        candidate["bibtex"] = str(row["bibtex"])
    return candidate


def _manual_evidence_from_row(row: dict[str, Any]) -> dict[str, Any]:
    evidence = {"cite_key": str(row.get("cite_key", "")).strip()}
    for field in MANUAL_EVIDENCE_FIELDS:
        if field in row:
            evidence[field] = row[field]
    return evidence


def _author_string(authors: Any) -> str:
    names: list[str] = []
    if isinstance(authors, list):
        for item in authors:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                names.append(str(item.get("name", "")).strip())
    elif isinstance(authors, str):
        names = [part.strip() for part in re.split(r"\s+and\s+", authors) if part.strip()]
    return " and ".join(name for name in names if name)


def _fallback_bibtex(candidate: dict[str, Any]) -> str:
    cite_key = str(candidate.get("cite_key", "") or "manualpaper")
    title = str(candidate.get("title", "") or "Untitled")
    year = candidate.get("year", "")
    authors = _author_string(candidate.get("authors", [])) or "Unknown"
    venue = str(candidate.get("venue", "") or "")
    doi = str(candidate.get("doi", "") or "")
    url = str(candidate.get("url", "") or "")
    lines = [
        f"@article{{{cite_key},",
        f"  title={{{title}}},",
        f"  author={{{authors}}},",
    ]
    if year:
        lines.append(f"  year={{{year}}},")
    if venue:
        lines.append(f"  journal={{{venue}}},")
    if doi:
        lines.append(f"  doi={{{doi}}},")
    if url:
        lines.append(f"  url={{{url}}},")
    lines.append("}")
    return "\n".join(lines)


def _write_manual_search_meta(
    stage_dir: Path,
    *,
    status: str,
    queries: list[str],
    year_min: int,
    total_candidates: int = 0,
    bibtex_entries: int = 0,
    errors: list[str] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "source": "manual_search_agent",
        "status": status,
        "real_search": False,
        "external_manual": True,
        "queries_used": queries,
        "year_min": year_min,
        "total_candidates": total_candidates,
        "bibtex_entries": bibtex_entries,
        "manual_results_file": "manual_search_results.jsonl",
        "ts": _utcnow_iso(),
    }
    if errors:
        payload["errors"] = errors
    (stage_dir / "search_meta.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _manual_done_result(stage_dir: Path) -> StageResult:
    artifacts = ["candidates.jsonl"]
    for optional in ("references.bib", "paper_evidence.jsonl"):
        if (stage_dir / optional).is_file():
            artifacts.append(optional)
    artifacts.append("search_meta.json")
    return StageResult(
        stage=Stage.LITERATURE_COLLECT,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-04/{a}" for a in artifacts),
    )


def _convert_manual_results_to_pipeline_format(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
) -> StageResult:
    stage_dir.mkdir(parents=True, exist_ok=True)
    queries, year_min, _plan_text, _goal_text, _problem_tree = _load_manual_search_context(
        run_dir, config
    )
    candidates_path = stage_dir / "candidates.jsonl"
    if candidates_path.is_file() and candidates_path.stat().st_size > 0:
        rows = _parse_jsonl_rows(candidates_path.read_text(encoding="utf-8"))
        if not (stage_dir / "search_meta.json").is_file():
            _write_manual_search_meta(
                stage_dir,
                status="completed",
                queries=queries,
                year_min=year_min,
                total_candidates=len(rows),
                bibtex_entries=0,
            )
        return _manual_done_result(stage_dir)

    manual_path = stage_dir / "manual_search_results.jsonl"
    if not manual_path.exists():
        return _generate_search_agent_handoff_files(stage_dir, run_dir, config)

    rows, errors = _validate_manual_results_jsonl(manual_path)
    if errors:
        _write_manual_search_meta(
            stage_dir,
            status="failed",
            queries=queries,
            year_min=year_min,
            errors=errors,
        )
        return StageResult(
            stage=Stage.LITERATURE_COLLECT,
            status=StageStatus.FAILED,
            artifacts=("search_meta.json",),
            error="; ".join(errors),
            decision="retry",
            evidence_refs=("stage-04/search_meta.json",),
        )

    candidates = [_manual_candidate_from_row(row) for row in rows]
    evidence_rows = [_manual_evidence_from_row(row) for row in rows]
    bibtex_entries = [
        str(row.get("bibtex", "")).strip() or _fallback_bibtex(candidate)
        for row, candidate in zip(rows, candidates, strict=False)
    ]
    _write_jsonl(candidates_path, candidates)
    (stage_dir / "references.bib").write_text(
        "\n\n".join(entry for entry in bibtex_entries if entry) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(stage_dir / "paper_evidence.jsonl", evidence_rows)
    _write_manual_search_meta(
        stage_dir,
        status="completed",
        queries=queries,
        year_min=year_min,
        total_candidates=len(candidates),
        bibtex_entries=len([entry for entry in bibtex_entries if entry]),
    )
    return _manual_done_result(stage_dir)


# ---------------------------------------------------------------------------
# Stage executors
# ---------------------------------------------------------------------------


def _execute_search_strategy(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    problem_tree = _read_prior_artifact(run_dir, "problem_tree.md") or ""
    topic = config.research.topic
    plan: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "search_strategy")
        sp = _pm.for_stage("search_strategy", evolution_overlay=_overlay, topic=topic, problem_tree=problem_tree)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict):
            yaml_text = str(payload.get("search_plan_yaml", "")).strip()
            if yaml_text:
                try:
                    parsed = yaml.safe_load(_extract_yaml_block(yaml_text))
                except yaml.YAMLError:
                    parsed = None
                if isinstance(parsed, dict):
                    plan = parsed
            src = payload.get("sources", [])
            if isinstance(src, list):
                sources = [item for item in src if isinstance(item, dict)]
    if plan is None:
        # Build smart fallback queries by extracting key terms from topic
        # instead of using the raw (often very long) topic string.
        _fallback_queries = _build_fallback_queries(topic)
        plan = {
            "topic": topic,
            "generated": _utcnow_iso(),
            "search_strategies": [
                {
                    "name": "keyword_core",
                    "queries": _fallback_queries[:5],
                    "sources": ["arxiv", "semantic_scholar", "openreview"],
                    "max_results_per_query": 60,
                },
                {
                    "name": "backward_forward_citation",
                    "queries": _fallback_queries[5:10] or _fallback_queries[:3],
                    "sources": ["semantic_scholar", "google_scholar"],
                    "depth": 1,
                },
            ],
            "filters": {
                "min_year": 2020,
                "language": ["en"],
                "peer_review_preferred": True,
            },
            "deduplication": {"method": "title_doi_hash", "fuzzy_threshold": 0.9},
        }
    if not sources:
        sources = [
            {
                "id": "arxiv",
                "name": "arXiv",
                "type": "api",
                "url": "https://export.arxiv.org/api/query",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
            {
                "id": "semantic_scholar",
                "name": "Semantic Scholar",
                "type": "api",
                "url": "https://api.semanticscholar.org/graph/v1/paper/search",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
        ]
    if config.openclaw_bridge.use_web_fetch:
        for src in sources:
            try:
                response = adapters.web_fetch.fetch(str(src.get("url", "")))
                src["status"] = (
                    "verified"
                    if response.status_code in (200, 301, 302, 405)
                    else "unreachable"
                )
                src["http_status"] = response.status_code
            except Exception:  # noqa: BLE001
                src["status"] = "unknown"
    (stage_dir / "search_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    (stage_dir / "sources.json").write_text(
        json.dumps(
            {"sources": sources, "count": len(sources), "generated": _utcnow_iso()},
            indent=2,
        ),
        encoding="utf-8",
    )

    # F1.5: Extract queries from plan for Stage 4 real literature search
    queries_list: list[str] = []
    year_min = 2020
    if isinstance(plan, dict):
        strategies = plan.get("search_strategies", [])
        if isinstance(strategies, list):
            for strat in strategies:
                if isinstance(strat, dict):
                    qs = strat.get("queries", [])
                    if isinstance(qs, list):
                        queries_list.extend(str(q) for q in qs if q)
        filters = plan.get("filters", {})
        if isinstance(filters, dict) and filters.get("min_year"):
            try:
                year_min = int(filters["min_year"])
            except (ValueError, TypeError):
                pass

    # --- Sanitize queries: shorten overly long queries ---
    # LLMs often produce the full topic title as a query, which is too long for
    # arXiv and Semantic Scholar (they work best with 3-8 keyword queries).
    _stop = {
        "a", "an", "the", "of", "for", "in", "on", "and", "or", "with",
        "to", "by", "from", "its", "is", "are", "was", "be", "as", "at",
        "via", "using", "based", "study", "analysis", "empirical",
        "towards", "toward", "into", "exploring", "comparison", "tasks",
        "effectiveness", "investigation", "comprehensive", "novel",
        "challenge", "challenges", "gaps", "gap", "critical", "survey", "review",
    }

    def _extract_search_terms(text: str) -> list[str]:
        """Extract meaningful search terms from text, removing stop words."""
        return [
            w for w in re.split(r"[^a-zA-Z0-9]+", text)
            if w.lower() not in _stop and len(w) > 1
        ]

    _MAX_QUERY_LEN = 60  # characters — beyond this, shorten to keywords
    _SEARCH_SUFFIXES = ["benchmark", "survey", "seminal", "state of the art"]

    def _shorten_query(q: str, max_kw: int = 6) -> str:
        """Shorten a query to *max_kw* keywords, preserving any trailing suffix."""
        q_stripped = q.strip()
        # Check if query ends with a known search suffix
        suffix = ""
        q_core = q_stripped
        for sfx in _SEARCH_SUFFIXES:
            if q_stripped.lower().endswith(sfx):
                suffix = sfx
                q_core = q_stripped[: -len(sfx)].strip()
                break
        # Extract keywords from the core part
        kws = _extract_search_terms(q_core)
        shortened = " ".join(kws[:max_kw])
        if suffix:
            shortened = f"{shortened} {suffix}"
        return shortened

    if queries_list:
        sanitized: list[str] = []
        for q in queries_list:
            if len(q) > _MAX_QUERY_LEN:
                shortened = _shorten_query(q)
                if shortened.strip():
                    sanitized.append(shortened)
            else:
                sanitized.append(q)
        queries_list = sanitized

    def _build_default_search_queries(topic_text: str) -> list[str]:
        """Generate concept-style search queries from the topic instead of copying the title."""
        _words = _extract_search_terms(topic_text)
        if not _words:
            return [topic_text[:60]]
        kw_primary = " ".join(_words[:6])
        kw_short = " ".join(_words[:4])
        kw_alt = " ".join(_words[1:5]) if len(_words) > 4 else kw_short
        return [
            kw_primary,
            f"{kw_short} benchmark",
            f"{kw_short} survey",
            kw_alt,
            f"{kw_short} recent advances",
        ]

    if not queries_list:
        queries_list = _build_default_search_queries(topic)

    # Ensure minimum query diversity — if dedup leaves too few, add variants
    _all_kw = _extract_search_terms(topic)
    _seen_q: set[str] = set()
    unique_queries: list[str] = []
    for q in queries_list:
        q_lower = q.strip().lower()
        if q_lower and q_lower not in _seen_q:
            _seen_q.add(q_lower)
            unique_queries.append(q.strip())
    # If we have fewer than 5 unique queries, generate supplemental keyword variants
    if len(unique_queries) < 5 and len(_all_kw) >= 3:
        supplements = [
            " ".join(_all_kw[:4]) + " survey",
            " ".join(_all_kw[:4]) + " benchmark",
            " ".join(_all_kw[1:5]),  # shifted window for diversity
            " ".join(_all_kw[:3]) + " comparison",
            " ".join(_all_kw[:3]) + " deep learning",
            " ".join(_all_kw[2:6]),  # another shifted window
        ]
        for s in supplements:
            s_lower = s.strip().lower()
            if s_lower not in _seen_q:
                _seen_q.add(s_lower)
                unique_queries.append(s.strip())
            if len(unique_queries) >= 8:
                break
    queries_list = unique_queries
    (stage_dir / "queries.json").write_text(
        json.dumps({"queries": queries_list, "year_min": year_min}, indent=2),
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.SEARCH_STRATEGY,
        status=StageStatus.DONE,
        artifacts=("search_plan.yaml", "sources.json", "queries.json"),
        evidence_refs=(
            "stage-03/search_plan.yaml",
            "stage-03/sources.json",
            "stage-03/queries.json",
        ),
    )


def _execute_literature_collect(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """Stage 4: collect literature via manual handoff or legacy APIs."""
    if getattr(config.research, "manual_search", True):
        return _convert_manual_results_to_pipeline_format(stage_dir, run_dir, config)
    return _execute_api_search_legacy(
        stage_dir, run_dir, config, adapters, llm=llm, prompts=prompts
    )


def _execute_api_search_legacy(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """Stage 4: Collect literature — prefer real APIs, fallback to LLM."""
    topic = config.research.topic

    # Read queries.json from Stage 3 (F1.5 output)
    queries_text = _read_prior_artifact(run_dir, "queries.json")
    queries_data = _safe_json_loads(queries_text or "{}", {})
    queries: list[str] = queries_data.get("queries", [topic])
    year_min: int = queries_data.get("year_min", 2020)

    # --- Try real API search first ---
    candidates: list[dict[str, Any]] = []
    bibtex_entries: list[str] = []
    real_search_succeeded = False

    try:
        from researchclaw.literature.search import (
            search_papers_multi_query,
            papers_to_bibtex,
        )

        # Expand queries for broader coverage
        expanded_queries = _expand_search_queries(queries, config.research.topic)
        logger.info(
            "[literature] Searching %d queries (expanded from %d) "
            "across OpenAlex → S2 → arXiv…",
            len(expanded_queries),
            len(queries),
        )
        papers = search_papers_multi_query(
            expanded_queries,
            limit_per_query=40,
            year_min=year_min,
            s2_api_key=config.llm.s2_api_key,
        )
        if papers:
            real_search_succeeded = True
            # Count by source
            src_counts: dict[str, int] = {}
            for p in papers:
                src_counts[p.source] = src_counts.get(p.source, 0) + 1
                d = p.to_dict()
                d["collected_at"] = _utcnow_iso()
                candidates.append(d)
                bibtex_entries.append(p.to_bibtex())
            src_str = ", ".join(f"{s}: {n}" for s, n in src_counts.items())
            logger.info(
                "[literature] Found %d papers (%s)", len(papers), src_str
            )
    except Exception:  # noqa: BLE001
        logger.warning(
            "[rate-limit] Literature search failed — falling back to LLM",
            exc_info=True,
        )

    # --- Inject foundational/seminal papers ---
    try:
        from researchclaw.data import load_seminal_papers
        seminal = load_seminal_papers(topic)
        if seminal:
            _existing_titles = {c.get("title", "").lower() for c in candidates}
            _injected = 0
            for sp in seminal:
                if sp.get("title", "").lower() not in _existing_titles:
                    candidates.append({
                        "id": f"seminal-{sp.get('cite_key', '')}",
                        "title": sp.get("title", ""),
                        "source": "seminal_library",
                        "url": "",
                        "year": sp.get("year", 2020),
                        "abstract": f"Foundational paper on {', '.join(sp.get('keywords', [])[:3])}.",
                        "authors": [{"name": sp.get("authors", "")}],
                        "cite_key": sp.get("cite_key", ""),
                        "venue": sp.get("venue", ""),
                        "collected_at": _utcnow_iso(),
                    })
                    _injected += 1
            if _injected:
                logger.info("Stage 4: Injected %d seminal papers from seed library", _injected)
    except Exception:  # noqa: BLE001
        logger.debug("Seminal paper injection skipped", exc_info=True)

    # --- Fallback: LLM-generated candidates ---
    if not candidates and llm is not None:
        plan_text = _read_prior_artifact(run_dir, "search_plan.yaml") or ""
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "literature_collect")
        sp = _pm.for_stage("literature_collect", evolution_overlay=_overlay, topic=topic, plan_text=plan_text)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
            candidates = [row for row in payload["candidates"] if isinstance(row, dict)]

    # --- Web search augmentation (Tavily/DDG + Google Scholar + Crawl4AI) ---
    web_context_parts: list[str] = []
    if config.web_search.enabled:
        try:
            from researchclaw.web.agent import WebSearchAgent
            import os

            tavily_key = config.web_search.tavily_api_key or os.environ.get(
                config.web_search.tavily_api_key_env, ""
            )
            web_agent = WebSearchAgent(
                tavily_api_key=tavily_key,
                enable_scholar=config.web_search.enable_scholar,
                enable_crawling=config.web_search.enable_crawling,
                enable_pdf=config.web_search.enable_pdf_extraction,
                max_web_results=config.web_search.max_web_results,
                max_scholar_results=config.web_search.max_scholar_results,
                max_crawl_urls=config.web_search.max_crawl_urls,
            )
            web_result = web_agent.search_and_extract(
                topic, search_queries=queries,
            )

            # Convert Google Scholar papers into candidates
            for sp in web_result.scholar_papers:
                _existing_titles = {
                    str(c.get("title", "")).lower().strip() for c in candidates
                }
                if sp.title.lower().strip() not in _existing_titles:
                    lit_paper = sp.to_literature_paper()
                    d = lit_paper.to_dict()
                    d["collected_at"] = _utcnow_iso()
                    candidates.append(d)
                    bibtex_entries.append(lit_paper.to_bibtex())

            # Save web search context for downstream stages
            web_context = web_result.to_context_string(max_length=20_000)
            if web_context.strip():
                (stage_dir / "web_context.md").write_text(
                    web_context, encoding="utf-8"
                )
                web_context_parts.append(web_context)

            # Save full web search metadata
            (stage_dir / "web_search_result.json").write_text(
                json.dumps(web_result.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )

            logger.info(
                "[web-search] Added %d scholar papers, %d web results, %d crawled pages",
                len(web_result.scholar_papers),
                len(web_result.web_results),
                len(web_result.crawled_pages),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[web-search] Web search augmentation failed — continuing with academic APIs only",
                exc_info=True,
            )

    # --- Ultimate fallback: placeholder data ---
    # BUG-L2: Do NOT overwrite real_search_succeeded here — it was already
    # set correctly in the search block above. Overwriting would mislabel
    # LLM-hallucinated or seminal papers as "real search" results.
    if not candidates:
        logger.warning("Stage 4: All literature searches failed — using placeholder papers")
        candidates = [
            {
                "id": f"candidate-{idx + 1}",
                "title": f"[Placeholder] Study {idx + 1} on {topic}",
                "source": "arxiv" if idx % 2 == 0 else "semantic_scholar",
                "url": f"https://example.org/{_safe_filename(topic.lower())}/{idx + 1}",
                "year": 2024,
                "abstract": f"This candidate investigates {topic} and reports preliminary findings.",
                "collected_at": _utcnow_iso(),
                "is_placeholder": True,
            }
            for idx in range(max(20, config.research.daily_paper_count or 20))
        ]

    # Write candidates
    out = stage_dir / "candidates.jsonl"
    _write_jsonl(out, candidates)

    # BUG-50 fix: Generate BibTeX from candidates when real search failed
    # (LLM/placeholder fallback paths don't populate bibtex_entries)
    if not bibtex_entries and candidates:
        for c in candidates:
            if c.get("is_placeholder"):
                continue
            _ck = c.get("cite_key", "")
            if not _ck:
                # Derive cite_key from first author surname + year
                _authors = c.get("authors", [])
                _surname = "unknown"
                if isinstance(_authors, list) and _authors:
                    _a0 = _authors[0] if isinstance(_authors[0], str) else (_authors[0].get("name", "") if isinstance(_authors[0], dict) else "")
                    _surname = _a0.split()[-1].lower() if _a0.strip() else "unknown"
                _yr = c.get("year", 2024)
                _title_word = "".join(
                    w[0] for w in str(c.get("title", "study")).split()[:3]
                ).lower()
                _ck = f"{_surname}{_yr}{_title_word}"
            _title = c.get("title", "Untitled")
            _year = c.get("year", 2024)
            _author_str = ""
            _raw_authors = c.get("authors", [])
            if isinstance(_raw_authors, list):
                _names = []
                for _a in _raw_authors:
                    if isinstance(_a, str):
                        _names.append(_a)
                    elif isinstance(_a, dict):
                        _names.append(_a.get("name", ""))
                _author_str = " and ".join(n for n in _names if n)
            bibtex_entries.append(
                f"@article{{{_ck},\n"
                f"  title={{{_title}}},\n"
                f"  author={{{_author_str or 'Unknown'}}},\n"
                f"  year={{{_year}}},\n"
                f"  url={{{c.get('url', '')}}},\n"
                f"}}"
            )
        logger.info(
            "Stage 4: Generated %d BibTeX entries from candidates (fallback)",
            len(bibtex_entries),
        )

    # Write references.bib (F2.4)
    artifacts = ["candidates.jsonl"]
    if web_context_parts:
        artifacts.append("web_context.md")
    if (stage_dir / "web_search_result.json").exists():
        artifacts.append("web_search_result.json")
    if bibtex_entries:
        bib_content = "\n\n".join(bibtex_entries) + "\n"
        (stage_dir / "references.bib").write_text(bib_content, encoding="utf-8")
        artifacts.append("references.bib")
        logger.info(
            "Stage 4: Wrote %d BibTeX entries to references.bib", len(bibtex_entries)
        )

    # Write search metadata
    (stage_dir / "search_meta.json").write_text(
        json.dumps(
            {
                "real_search": real_search_succeeded,
                "queries_used": queries,
                "year_min": year_min,
                "total_candidates": len(candidates),
                "bibtex_entries": len(bibtex_entries),
                "ts": _utcnow_iso(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts.append("search_meta.json")

    return StageResult(
        stage=Stage.LITERATURE_COLLECT,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-04/{a}" for a in artifacts),
    )


_MAX_ABSTRACT_LEN = 800  # Truncate long abstracts to reduce token usage
_MAX_CANDIDATES_CHARS = 30_000  # Cap total candidates text sent to LLM
_EVIDENCE_PROMPT_LIMITS: dict[str, int] = {
    "full_text_summary": 500,
    "key_evidence": 500,
    "datasets": 200,
    "metrics": 200,
    "limitations": 200,
    "relevance_reason": 200,
    "quality_notes": 200,
}


def _truncate_for_prompt(value: Any, limit: int) -> Any:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    if len(text) <= limit:
        return value
    return text[:limit] + "..."


def _load_paper_evidence(run_dir: Path) -> dict[str, dict[str, Any]]:
    evidence_text = _read_prior_artifact(run_dir, "paper_evidence.jsonl") or ""
    evidence_by_key: dict[str, dict[str, Any]] = {}
    for row in _parse_jsonl_rows(evidence_text):
        cite_key = str(row.get("cite_key", "")).strip()
        if cite_key:
            evidence_by_key[cite_key] = row
    return evidence_by_key


def _prompt_evidence_fields(evidence: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, limit in _EVIDENCE_PROMPT_LIMITS.items():
        if key in evidence and evidence[key] not in ("", None, [], {}):
            compact[key] = _truncate_for_prompt(evidence[key], limit)
    return compact


def _attach_evidence(row: dict[str, Any], evidence: dict[str, Any]) -> None:
    for key in MANUAL_EVIDENCE_FIELDS:
        if key in evidence and evidence[key] not in ("", None, [], {}):
            row[key] = evidence[key]


def _merge_shortlist_with_candidates(
    shortlist: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
    evidence_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        str(row.get("cite_key", "")).strip(): row
        for row in filtered_rows
        if str(row.get("cite_key", "")).strip()
    }
    by_title = {
        str(row.get("title", "")).strip().lower(): row
        for row in filtered_rows
        if str(row.get("title", "")).strip()
    }
    merged: list[dict[str, Any]] = []
    for item in shortlist:
        cite_key = str(item.get("cite_key", "")).strip()
        title = str(item.get("title", "")).strip().lower()
        base = by_key.get(cite_key) or by_title.get(title) or {}
        row = dict(base)
        row.update(item)
        evidence = evidence_by_key.get(str(row.get("cite_key", "")).strip())
        if evidence:
            _attach_evidence(row, evidence)
        merged.append(row)
    return merged


def _execute_literature_screen(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    candidates_text = _read_prior_artifact(run_dir, "candidates.jsonl") or ""
    evidence_by_key = _load_paper_evidence(run_dir)

    # --- P1-1: keyword relevance pre-filter ---
    # Before LLM screening, drop papers whose title+abstract share no keywords
    # with the research topic.  This catches cross-domain noise cheaply.
    topic_keywords = _extract_topic_keywords(
        config.research.topic, config.research.domains
    )
    filtered_rows: list[dict[str, Any]] = []
    dropped_count = 0
    for raw_line in candidates_text.strip().splitlines():
        row = _safe_json_loads(raw_line, {})
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).lower()
        abstract = str(row.get("abstract", "")).lower()
        text_blob = f"{title} {abstract}"
        overlap = sum(1 for kw in topic_keywords if kw in text_blob)
        # T2.2: Relaxed from ≥2 to ≥1 keyword hit — previous threshold was
        # too aggressive (94% rejection rate).  Single-keyword matches are
        # still screened by the LLM in the next step.
        if overlap >= 1:
            row["keyword_overlap"] = overlap
            filtered_rows.append(row)
        else:
            dropped_count += 1
    # If pre-filter dropped everything, fall back to original (safety valve)
    if not filtered_rows:
        filtered_rows = _parse_jsonl_rows(candidates_text)
    # Truncate abstracts and attach full-text evidence when available.
    for row in filtered_rows:
        abstract = row.get("abstract", "")
        if isinstance(abstract, str) and len(abstract) > _MAX_ABSTRACT_LEN:
            row["abstract"] = abstract[:_MAX_ABSTRACT_LEN] + "..."
        evidence = evidence_by_key.get(str(row.get("cite_key", "")).strip())
        if evidence:
            _attach_evidence(row, evidence)

    # Rebuild candidates_text from filtered rows with compact evidence fields.
    prompt_rows: list[dict[str, Any]] = []
    for row in filtered_rows:
        prompt_row = dict(row)
        evidence = evidence_by_key.get(str(row.get("cite_key", "")).strip())
        if evidence:
            for key in MANUAL_EVIDENCE_FIELDS:
                prompt_row.pop(key, None)
            prompt_row.update(_prompt_evidence_fields(evidence))
        prompt_rows.append(prompt_row)
    candidates_text = "\n".join(
        json.dumps(r, ensure_ascii=False) for r in prompt_rows
    )
    # Cap total candidates text size to avoid blowing token budget
    if len(candidates_text) > _MAX_CANDIDATES_CHARS:
        # Truncate at newline boundary to avoid cutting mid-JSON-line
        candidates_text = candidates_text[:_MAX_CANDIDATES_CHARS].rsplit("\n", 1)[0]
        logger.info(
            "Candidates text truncated to %d chars for screening",
            len(candidates_text),
        )
    logger.info(
        "Domain pre-filter: kept %d, dropped %d (keywords: %s)",
        len(filtered_rows),
        dropped_count,
        topic_keywords[:8],
    )

    shortlist: list[dict[str, Any]] = []
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "literature_screen")
        sp = _pm.for_stage(
            "literature_screen",
            evolution_overlay=_overlay,
            topic=config.research.topic,
            domains=", ".join(config.research.domains)
            if config.research.domains
            else "general",
            quality_threshold=config.research.quality_threshold,
            candidates_text=candidates_text,
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict) and isinstance(payload.get("shortlist"), list):
            shortlist = [row for row in payload["shortlist"] if isinstance(row, dict)]
            shortlist = _merge_shortlist_with_candidates(
                shortlist, filtered_rows, evidence_by_key
            )
    # T2.2: Ensure minimum shortlist size of 15 for adequate related work
    _MIN_SHORTLIST = 15
    if not shortlist:
        rows = (
            filtered_rows[:_MIN_SHORTLIST]
            if filtered_rows
            else _parse_jsonl_rows(candidates_text)[:_MIN_SHORTLIST]
        )
        for idx, item in enumerate(rows):
            item["relevance_score"] = round(0.75 - idx * 0.02, 3)
            item["quality_score"] = round(0.72 - idx * 0.015, 3)
            item["keep_reason"] = "Template screened entry"
            shortlist.append(item)
    elif len(shortlist) < _MIN_SHORTLIST:
        # T2.2: LLM returned too few — supplement from filtered candidates
        existing_titles = {
            str(s.get("title", "")).lower().strip() for s in shortlist
        }
        for row in filtered_rows:
            if len(shortlist) >= _MIN_SHORTLIST:
                break
            title_lower = str(row.get("title", "")).lower().strip()
            if title_lower and title_lower not in existing_titles:
                row.setdefault("relevance_score", 0.5)
                row.setdefault("quality_score", 0.5)
                row.setdefault("keep_reason", "Supplemented to meet minimum shortlist")
                shortlist.append(row)
                existing_titles.add(title_lower)
        logger.info(
            "Stage 5: Supplemented shortlist to %d papers (minimum: %d)",
            len(shortlist), _MIN_SHORTLIST,
        )
    out = stage_dir / "shortlist.jsonl"
    _write_jsonl(out, shortlist)
    return StageResult(
        stage=Stage.LITERATURE_SCREEN,
        status=StageStatus.DONE,
        artifacts=("shortlist.jsonl",),
        evidence_refs=("stage-05/shortlist.jsonl",),
    )


def _execute_knowledge_extract(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    shortlist = _read_prior_artifact(run_dir, "shortlist.jsonl") or ""

    # Inject web context from Stage 4 if available
    web_context = _read_prior_artifact(run_dir, "web_context.md") or ""
    if web_context:
        shortlist = shortlist + "\n\n--- Web Search Context ---\n" + web_context[:10_000]

    cards_dir = stage_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "knowledge_extract")
        sp = _pm.for_stage("knowledge_extract", evolution_overlay=_overlay, shortlist=shortlist)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict) and isinstance(payload.get("cards"), list):
            cards = [item for item in payload["cards"] if isinstance(item, dict)]
    if not cards:
        rows = _parse_jsonl_rows(shortlist)
        for idx, paper in enumerate(rows[:6]):
            title = str(paper.get("title", f"Paper {idx + 1}"))
            cards.append(
                {
                    "card_id": f"card-{idx + 1}",
                    "title": title,
                    "problem": f"How to improve {config.research.topic}",
                    "method": "Template method summary",
                    "data": "Template dataset",
                    "metrics": "Template metric",
                    "findings": "Template key finding",
                    "limitations": "Template limitation",
                    "citation": str(paper.get("url", "")),
                    "cite_key": str(paper.get("cite_key", "")),
                }
            )
    for idx, card in enumerate(cards):
        card_id = _safe_filename(str(card.get("card_id", f"card-{idx + 1}")))
        parts = [f"# {card.get('title', card_id)}", ""]
        for key in (
            "cite_key",
            "problem",
            "method",
            "data",
            "metrics",
            "findings",
            "limitations",
            "citation",
        ):
            parts.append(f"## {key.title()}")
            parts.append(str(card.get(key, "")))
            parts.append("")
        (cards_dir / f"{card_id}.md").write_text("\n".join(parts), encoding="utf-8")
    return StageResult(
        stage=Stage.KNOWLEDGE_EXTRACT,
        status=StageStatus.DONE,
        artifacts=("cards/",),
        evidence_refs=("stage-06/cards/",),
    )
