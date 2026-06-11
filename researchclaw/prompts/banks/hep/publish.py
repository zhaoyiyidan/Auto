"""Generated prompt bank segment.

This module was split from the legacy monolithic prompt bank without
changing rendered prompt content.
"""

from __future__ import annotations

from typing import Any


STAGES: dict[str, dict[str, Any]] = {
    'quality_gate': {
        "system": (
            "You are the final HEP-ph quality gate evaluator for a "
            "JHEP/PRD submission candidate."
        ),
        "user": (
            "Evaluate the revised HEP-ph paper quality and return JSON.\n"
            "Schema: {score_1_to_10:number, verdict:string, strengths:[...], "
            "weaknesses:[...], required_actions:[...]}.\n"
            "Threshold: {quality_threshold}\n"
            "Physics checklist used for scoring:\n"
            "- Lagrangian gauge-invariant, parameters enumerated.\n"
            "- Every experimental bound cites the ORIGINAL collaboration "
            "paper at stated CL.\n"
            "- Results in natural units; exclusion contours with log-log "
            "axes; theoretical uncertainties acknowledged.\n"
            "- No ML-venue artefacts (Broader Impact, Reproducibility "
            "Checklist, seed variance).\n"
            "- Section order follows HEP convention.\n"
            "- 30-60 hep-ph / hep-ex / astro-ph citations.\n"
            "Paper:\n{revised}"
        ),
        "json_mode": True,
    },
    'knowledge_archive': {
        "system": (
            "You produce reproducibility-focused HEP-ph research "
            "retrospectives."
        ),
        "user": (
            "{preamble}\n\n"
            "Write a retrospective archive in markdown for an HEP-ph study. "
            "Include:\n"
            "- Physics lessons learned (which approximations held, which "
            "broke).\n"
            "- Reproducibility notes: Lagrangian, parameter ranges, PDF "
            "set, form factor, CL procedure, scan grid.\n"
            "- Comparison with previously-published results, noting where "
            "the current analysis agrees or disagrees.\n"
            "- Future work: higher-order corrections, additional channels, "
            "planned-experiment projections.\n"
            "Decision:\n{decision}\n\nAnalysis:\n{analysis}\n\n"
            "Revised paper:\n{revised}"
        ),
        "max_tokens": 8192,
    },
    'export_publish': {
        "system": (
            "You are a publication formatting editor for an HEP-ph "
            "manuscript. Preserve natural units, Feynman-diagram language, "
            "and citations to original experimental papers. Do NOT insert "
            "NeurIPS checklists or broader-impact paragraphs during final "
            "formatting."
        ),
        "user": (
            "Format the revised HEP-ph paper into clean final markdown for "
            "publication export. Preserve content quality and readability.\n"
            "CITATION FORMAT (CRITICAL): All citations MUST remain in "
            "[cite_key] bracket format, e.g. [buchmueller2014]. Do NOT "
            "convert to author-year format like [Buchmueller et al., 2014]. "
            "The [cite_key] format is required for downstream LaTeX "
            "\\cite{{}} generation.\n"
            "HEP FORMATTING CHECKS:\n"
            "- Natural units preserved (GeV, pb, fb, cm², Ω h²).\n"
            "- Exclusion-contour figures present in Results.\n"
            "- Section order: Introduction → Model → Phenomenology → Setup "
            "→ Results → Discussion → Conclusions → References.\n"
            "- No 'Broader Impact' or 'Reproducibility Checklist' sections.\n"
            "Input paper:\n{revised}"
        ),
        "max_tokens": 16384,
    },
}
