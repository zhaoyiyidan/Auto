"""Generated prompt bank segment.

This module was split from the legacy monolithic prompt bank without
changing rendered prompt content.
"""

from __future__ import annotations

from typing import Any


STAGES: dict[str, dict[str, Any]] = {
    'synthesis': {
        "system": (
            "You are a synthesis specialist for HEP-ph literature reviews. "
            "You cluster papers by BSM framework, by observable, and by "
            "experimental channel, then identify physics gaps."
        ),
        "user": (
            "Produce merged synthesis output (topic clusters + research gaps) "
            "for an HEP phenomenology study.\n"
            "Output markdown with sections: Cluster Overview, Cluster 1..N, "
            "Gap 1..N, Prioritized Opportunities.\n"
            "For each cluster, note the dominant BSM framework (simplified "
            "model, EFT, UV-complete), the main observable, and the current "
            "best experimental bound in natural units.\n"
            "For each gap, identify:\n"
            "1. Model-building gaps — which BSM frameworks (simplified "
            "models, EFT operators, UV completions) remain underexplored.\n"
            "2. Experimental coverage gaps — which regions of (m_DM, m_med, "
            "coupling, spin) are not yet constrained.\n"
            "3. Complementarity gaps — which combinations of direct + "
            "indirect + collider channels could provide stronger joint limits.\n"
            "4. Theoretical precision gaps — where current calculations have "
            "large uncertainties (QCD scale, PDF, nuclear form factors, "
            "astrophysical J-factors).\n"
            "The 'Prioritized Opportunities' section MUST propose concrete, "
            "physically-motivated research directions — not generic statistical "
            "or ML methods. Each opportunity should be expressible as: "
            "'Study X using observable Y in experiment Z, closing the gap "
            "at parameter region (…)'.\n"
            "Topic: {topic}\n"
            "{domain_context}"
            "Cards context:\n{cards_context}"
        ),
        "max_tokens": 8192,
    },
    'hypothesis_gen': {
        "system": (
            "You formulate testable HEP-phenomenology hypotheses that address "
            "gaps NOT covered by existing experimental results. Every "
            "hypothesis must be:\n"
            "1. NOVEL: Not replicating a published recast or an existing "
            "collaboration exclusion.\n"
            "2. GAP-FILLING: Targets a specific weakness or blind spot "
            "identified in the literature synthesis.\n"
            "3. PHYSICALLY TESTABLE: Evaluable with analytical cross-section "
            "formulas, a parameter-space scan in Python, or a recast of "
            "published experimental data.\n"
            "{feasibility_constraint}"
            "4. FALSIFIABLE: Has a specific numerical threshold, in natural "
            "units, whose violation rejects the hypothesis (e.g. 'σ_SI > "
            "2·10⁻⁴⁶ cm² for m_DM = 40 GeV is excluded by LZ').\n"
            "5. PHYSICALLY INTERESTING: At least one hypothesis should "
            "challenge a common assumption or test a counter-intuitive "
            "prediction (interference effect, accidental cancellation, "
            "non-thermal production, velocity-dependent cross section)."
        ),
        "user": (
            "Generate at least 2 falsifiable HEP-ph hypotheses from the "
            "synthesis below.\n"
            "For each hypothesis provide:\n"
            "- **Hypothesis statement**: A clear claim in physics language, "
            "naming the BSM model / operator and the observable.\n"
            "- **Novelty argument**: Why this has NOT been tested before, "
            "citing the specific gaps in the synthesis (experimental, "
            "theoretical, complementarity, …).\n"
            "- **Rationale**: Theoretical basis (symmetry, UV completion, "
            "matching) or empirical basis (tension in data, anomaly) for "
            "expecting this result.\n"
            "- **Measurable prediction**: A number in natural units — cross "
            "section in pb/fb/cm², branching ratio, relic Ω_DM h², mass "
            "window in GeV/TeV, coupling upper bound.\n"
            "- **Failure condition**: Which experimental bound, at what CL, "
            "would reject the hypothesis.\n"
            "- **Required baselines**: The Standard Model prediction and any "
            "already-excluded BSM scenarios that must be compared against to "
            "make the finding meaningful.\n\n"
            "AVOID:\n"
            "- Hypotheses that restate a known PDG-level fact.\n"
            "- Hypotheses that replicate a published collaboration exclusion.\n"
            "- Hypotheses requiring a full MadGraph+PYTHIA+Delphes chain "
            "inside the sandbox — those are not feasible here.\n"
            "- ML-style hypotheses ('a neural network will improve signal "
            "efficiency') unless tied to a specific physics observable.\n\n"
            "{domain_context}"
            "{extension_context}\n"
            "Synthesis:\n{synthesis}"
        ),
    },
}
