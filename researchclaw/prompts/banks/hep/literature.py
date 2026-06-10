"""Generated prompt bank segment.

This module was split from the legacy monolithic prompt bank without
changing rendered prompt content.
"""

from __future__ import annotations

from typing import Any


STAGES: dict[str, dict[str, Any]] = {
    'search_strategy': {
        "system": (
            "You design literature retrieval strategies for an HEP-ph study. "
            "Primary sources are arXiv (hep-ph, hep-ex, astro-ph.CO), INSPIRE-HEP, "
            "the PDG, and the published collaboration pages "
            "(ATLAS/CMS/LHCb/LZ/XENONnT/PandaX/Fermi-LAT/AMS-02)."
        ),
        "user": (
            "Create a merged search strategy package for an HEP-ph topic.\n"
            "Return a JSON object with keys: search_plan_yaml, sources.\n"
            "search_plan_yaml must be valid YAML text. It should enumerate "
            "arXiv categories (hep-ph, hep-ex, astro-ph.CO), INSPIRE record "
            "queries, and keyword combinations involving BSM model names, "
            "experimental collaborations, and observables.\n"
            "sources must include id, name, type (arxiv | inspire | pdg | "
            "collaboration_page | journal), url, status, query, verified_at.\n"
            "Topic: {topic}\n"
            "Problem tree:\n{problem_tree}"
        ),
        "json_mode": True,
    },
    'literature_collect': {
        "system": (
            "You are a literature mining assistant for HEP-ph. You prefer "
            "peer-reviewed papers from JHEP, PRD, PRL, EPJC, Phys.Lett.B, "
            "and well-cited arXiv preprints (hep-ph, hep-ex, astro-ph.CO)."
        ),
        "user": (
            "Generate candidate papers from the search plan.\n"
            "Return JSON: {candidates:[...]} with >=8 rows.\n"
            "Each candidate must include id, title, source (arxiv / JHEP / "
            "PRD / PRL / EPJC / …), url, year, abstract, collected_at. "
            "Prefer arXiv numbers and DOIs. Include at least one experimental "
            "collaboration paper (ATLAS/CMS/LZ/XENONnT/Fermi-LAT) for every "
            "experimental bound that will be quoted.\n"
            "Topic: {topic}\n"
            "Search plan:\n{plan_text}"
        ),
        "json_mode": True,
    },
    'literature_screen': {
        "system": (
            "You are a strict HEP-ph reviewer with zero tolerance for "
            "cross-domain false positives. A paper about 'dark matter' in "
            "financial engineering is NOT relevant to particle dark matter. "
            "A paper about 'Higgs boson' in condensed-matter analogues is "
            "NOT relevant to electroweak phenomenology. Reject papers whose "
            "actual physics domain differs from the topic, even if the title "
            "keywords overlap."
        ),
        "user": (
            "Perform merged relevance + quality screening and return shortlist.\n"
            "Return JSON: {shortlist:[...]} each with title, cite_key "
            "(if present), relevance_score (0-1), quality_score (0-1), "
            "keep_reason.\n"
            "Preserve all original fields (paper_id, doi, arxiv_id, cite_key, "
            "etc.) from the input.\n"
            "Topic: {topic}\n"
            "Domains: {domains}\n"
            "Threshold: {quality_threshold}\n\n"
            "SCREENING RULES (apply strictly):\n"
            "1. DOMAIN MATCH: The paper's research must be in hep-ph, hep-ex, "
            "or astro-ph.CO (for DM/cosmology). Condensed-matter, "
            "mathematical, or AI-ML papers that use 'dark matter' or "
            "'gauge theory' metaphorically do NOT count.\n"
            "2. METHOD RELEVANCE: The paper must discuss the specific "
            "BSM model, EFT operator, experimental channel, or observable "
            "directly applicable to the topic.\n"
            "3. EXPERIMENTAL PROVENANCE: For any paper invoked as an "
            "experimental bound, prefer the original collaboration paper "
            "(ATLAS/CMS/LZ/XENONnT/Fermi-LAT) over a secondary recast.\n"
            "4. RECENCY PREFERENCE: Prefer papers from 2020+ for experimental "
            "limits; foundational theory papers (pre-2020) are welcome if "
            "they introduced the model/operator framework still in use.\n"
            "5. SEMINAL PAPERS: Papers flagged source='seminal_library' are "
            "pre-vetted (PDG reviews, classical BSM reviews, standard "
            "cross-section references) — keep them if keywords match "
            "(relevance_score >= 0.7).\n"
            "6. QUALITY FLOOR: Reject papers without an abstract, arXiv "
            "number, or citation count (likely not real hep-ph papers).\n"
            "Candidates JSONL:\n{candidates_text}"
        ),
        "json_mode": True,
    },
    'knowledge_extract': {
        "system": (
            "You extract high-signal physics evidence cards from HEP-ph papers. "
            "Each card captures the BSM framework, the observable, the "
            "experimental input, and the quantitative result."
        ),
        "user": (
            "Extract structured knowledge cards from shortlist.\n"
            "Return JSON: {cards:[{card_id, title, cite_key, problem, method, "
            "data, metrics, findings, limitations, citation}]}.\n"
            "For HEP-ph papers, interpret the fields as:\n"
            "- problem: the physics question (e.g. 'constraints on scalar "
            "mediator couplings from monojet').\n"
            "- method: the BSM model / EFT operator set + calculation "
            "pipeline (tree-level, NLO, recast, global fit).\n"
            "- data: the experimental dataset reinterpreted (LHC Run-2 139/fb, "
            "LZ 2022 WS, Fermi-LAT dSph 15yr, …).\n"
            "- metrics: the quantitative output (95% CL limits in natural "
            "units, relic-density-compatible coupling ranges, s-channel "
            "resonance peak σ·BR).\n"
            "- findings: 1-2 sentence physics result with numbers in natural "
            "units.\n"
            "- limitations: dominant theoretical/experimental systematics.\n"
            "IMPORTANT: If the input contains cite_key fields, preserve them "
            "exactly in the output.\n"
            "Shortlist:\n{shortlist}"
        ),
        "json_mode": True,
    },
}
