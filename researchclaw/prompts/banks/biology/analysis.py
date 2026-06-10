"""Generated prompt bank segment.

This module was split from the legacy monolithic prompt bank without
changing rendered prompt content.
"""

from __future__ import annotations

from typing import Any


STAGES: dict[str, dict[str, Any]] = {
    'result_analysis': {
    "system": (
        "You are a quantitative metabolic-modelling analyst. Always cite "
        "exact flux numbers (with units: mmol/gDW/h or 1/h) from the "
        "provided data. Treat FBA / pFBA / FVA outputs as deterministic "
        "linear-programming solutions, not stochastic estimates."
    ),
    "user": (
        "{preamble}\n\n"
        "{data_context}\n\n"
        "Analyse the metabolic-modelling run metrics and produce a markdown "
        "report. Use the ACTUAL quantitative values provided above — do NOT "
        "invent fluxes.\n\n"
        "SANITY CHECKS (perform BEFORE interpreting results):\n"
        "1. WILD-TYPE GROWTH RATE: Compare the predicted growth rate (1/h) to "
        "the measured doubling time (t_d = ln 2 / mu). Flag > 30% "
        "disagreement — typical causes are missing transport reactions, "
        "wrong medium, or biomass coefficient drift.\n"
        "2. FBA / pFBA / FVA AGREEMENT: At the wild-type optimum, FBA and "
        "pFBA growth rates MUST coincide. FVA fraction-of-optimum=0.95 "
        "envelopes MUST contain the FBA optimum. Mismatches indicate solver "
        "issues or a non-unique optimum.\n"
        "3. ESSENTIAL-GENE RECALL: Compare the predicted essential-gene set "
        "to Keio (E. coli) / OGEE / DEG curated calls. Report precision, "
        "recall, F1. Flag genes the model calls essential but the experiment "
        "calls dispensable (likely missing isoenzymes / transports).\n"
        "4. PHENOTYPIC PHASE PLANE: The optimal-biomass surface should be "
        "piecewise-linear and convex in the two swept exchange fluxes. "
        "Non-convex surfaces indicate solver tolerance issues.\n"
        "5. FLUX UNITS: Every flux value MUST carry mmol/gDW/h (or 1/h for "
        "growth rate, dimensionless for ratios). Unitless numbers are an "
        "automatic methodology flag.\n"
        "6. CONDITION COMPLETENESS: Look for `REGISTERED_CONDITIONS:` in the "
        "output. Cross-check against the plan. Missing or `CONDITION_FAILED:` "
        "entries must be listed and assessed for whether the remaining "
        "conditions still answer the question.\n"
        "7. ABLATION ISOLATION: If two perturbed conditions yield IDENTICAL "
        "flux distributions (e.g. KO of an isozyme pair where one isozyme "
        "alone carries no flux), this is informative — call it out, do not "
        "treat as a bug.\n"
        "8. DEGENERATE METRICS: If ALL conditions produce the SAME growth "
        "rate or yield, flag as DEGENERATE. Common causes: medium too rich "
        "(every KO is rescued by an alternative pathway), substrate uptake "
        "bound saturates the objective, FVA fraction too loose. Flag the "
        "needed remediation concretely (tighter medium, alternative objective, "
        "etc.) so the decision stage can choose PROCEED with caveats, PIVOT, "
        "or EXTEND.\n\n"
        "Required sections: Metrics Summary (real flux numbers with units), "
        "Consensus Findings (high-confidence stoichiometric / pathway "
        "implications), Contested Points (with evidence-based resolution — "
        "e.g. essential-gene calls that diverge from Keio), Statistical "
        "Checks (FVA envelopes, sampling CIs if available), Methodology Audit "
        "(model id, medium, solver, tolerance, COBRApy version), Limitations "
        "(missing reactions, regulation, isoenzymes), Conclusion.\n"
        "In the Conclusion, include:\n"
        "- Result quality rating (1-10)\n"
        "- Key findings (3-5; reference specific reactions / genes / pathways)\n"
        "- Methodology gaps to address next (curation, additional perturbations)\n"
        "- Recommendation: PROCEED / PIVOT / EXTEND\n\n"
        "Run context:\n{context}"
    ),
    "max_tokens": 8192,
},
}
