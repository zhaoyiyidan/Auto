"""Generated prompt bank segment.

This module was split from the legacy monolithic prompt bank without
changing rendered prompt content.
"""

from __future__ import annotations

from typing import Any


STAGES: dict[str, dict[str, Any]] = {
    'result_analysis': {
        "system": (
            "You are a quantitative research analyst. Always cite exact numbers "
            "from the provided data."
        ),
        "user": (
            "{preamble}\n\n"
            "{data_context}\n\n"
            "Analyze run metrics and produce markdown report with statistical "
            "interpretation.\n"
            "Use the ACTUAL quantitative values provided above — do NOT invent "
            "numbers.\n\n"
            "SANITY CHECKS (perform BEFORE interpreting results):\n"
            "1. MONOTONICITY: If a condition scales a parameter (e.g., N agents, "
            "model size), check whether metrics move in the expected direction. "
            "If accuracy *decreases* when adding more agents under majority voting, "
            "flag this as a likely implementation bug (vote parsing, normalization, "
            "or aggregation issue).\n"
            "2. BASELINE PLAUSIBILITY: Random-chance baselines should match "
            "theoretical expectations (e.g., 1/K for K-class classification).\n"
            "3. CROSS-CONDITION CONSISTENCY: Results across datasets or conditions "
            "should be internally coherent — wildly different patterns may indicate "
            "confounds or bugs.\n"
            "4. REPLICATION: If results are from a single seed (n=1), explicitly "
            "note that no statistical significance claims can be made.\n"
            "5. ABLATION ISOLATION: Compare per-seed values across conditions. If "
            "two conditions produce IDENTICAL values for the same seed, this is a "
            "RED FLAG — the ablation/variant may not have actually changed the code "
            "path (e.g., config not applied, caching, shared state). Flag this "
            "explicitly and recommend a config/registry audit.\n"
            "6. METRIC DEFINITION CHECK: Look for a `METRIC_DEF:` line in the output. "
            "If absent, flag that the primary metric is UNDEFINED — direction, units, "
            "and formula are unknown, making all comparisons uninterpretable. This is "
            "a critical methodology gap.\n"
            "7. CONDITION COMPLETENESS CHECK: Look for `REGISTERED_CONDITIONS:` in "
            "the output. Compare against the experiment plan. If conditions are missing "
            "or failed (look for `CONDITION_FAILED:`), list them explicitly and assess "
            "whether the remaining conditions can still answer the research question.\n"
            "8. DEGENERATE METRICS CHECK: If ALL conditions (or all but one) produce "
            "the SAME mean primary metric value, flag this as DEGENERATE — the metric "
            "is NOT discriminative. Common causes: (a) time-to-event metric that only "
            "checks success at the final step (returns horizon for all methods), "
            "(b) ceiling/floor effects from too-easy or too-hard tasks, "
            "(c) metric capped at a budget value. This makes the experiment "
            "scientifically useless — flag the metric computation or task difficulty "
            "as a concrete remediation. Look for `WARNING: DEGENERATE_METRICS` "
            "in stdout. Even if not printed, check the numbers yourself.\n\n"
            "Required sections: Metrics Summary (with real values), "
            "Consensus Findings (high confidence), "
            "Contested Points (with evidence-based resolution), "
            "Statistical Checks, Methodology Audit, Limitations, Conclusion.\n"
            "In the Conclusion, include:\n"
            "- Result quality rating (1-10)\n"
            "- Key findings (3-5)\n"
            "- Methodology gaps to address next\n"
            "- Recommendation: PROCEED / PIVOT / EXTEND\n\n"
            "Run context:\n{context}"
        ),
        "max_tokens": 8192,
    },
    'research_decision': {
        "system": "You are a research program lead making go/no-go decisions.",
        "user": (
            "Based on the analysis, make one of three decisions:\n"
            "- **PROCEED** — results are sufficient, move to paper writing\n"
            "- **PIVOT** — hypotheses are fundamentally flawed, generate new ones\n"
            "- **EXTEND** — current hypotheses produced useful evidence and should "
            "lead to deeper follow-up hypotheses\n\n"
            "Experiment-level repair and rerun decisions have already been handled "
            "before this stage.\n\n"
            "MINIMUM QUALITY CRITERIA for PROCEED (ALL must be met):\n"
            "1. At least 2 baselines AND the proposed method have results\n"
            "2. The primary metric is defined (direction, units known)\n"
            "3. Each condition has results from ≥3 seeds\n"
            "4. No identical per-seed values across different conditions (ablation integrity)\n"
            "5. The analysis quality rating is ≥4/10\n"
            "If any criterion is not met, choose PIVOT for hypothesis-level flaws "
            "or PROCEED with explicit caveats when the remaining issue is "
            "experiment quality.\n\n"
            "Output markdown with sections:\n"
            "## Decision\n"
            "State exactly one of: PROCEED, PIVOT, or EXTEND\n\n"
            "## Justification\n"
            "Why this decision is warranted based on evidence.\n\n"
            "## Evidence\n"
            "Key data points supporting the decision.\n\n"
            "## Next Actions\n"
            "Concrete steps for the chosen path.\n\n"
            "Analysis:\n{analysis}"
        ),
    },
}
