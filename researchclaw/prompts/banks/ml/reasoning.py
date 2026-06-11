"""Generated prompt bank segment.

This module was split from the legacy monolithic prompt bank without
changing rendered prompt content.
"""

from __future__ import annotations

from typing import Any


STAGES: dict[str, dict[str, Any]] = {
    'synthesis': {
        "system": "You are a synthesis specialist for literature reviews.",
        "user": (
            "Produce merged synthesis output (topic clusters + research gaps).\n"
            "Output markdown with sections: Cluster Overview, Cluster 1..N, "
            "Gap 1..N, Prioritized Opportunities.\n"
            "Topic: {topic}\n"
            "{domain_context}"
            "Cards context:\n{cards_context}"
        ),
        "max_tokens": 8192,
    },
    'hypothesis_gen': {
        "system": (
            "You formulate testable scientific hypotheses that address gaps "
            "NOT covered by existing literature. Your hypotheses must be:\n"
            "1. NOVEL: Not simply replicating known results or testing obvious things.\n"
            "2. GAP-FILLING: Address specific weaknesses or blind spots identified "
            "in the literature synthesis.\n"
            "3. FEASIBLE: Testable within the constraints of this research domain.\n"
            "{feasibility_constraint}"
            "4. FALSIFIABLE: Have clear failure conditions that would definitively "
            "reject the hypothesis.\n"
            "5. SURPRISING: At least one hypothesis should challenge conventional "
            "wisdom or test a counter-intuitive prediction."
        ),
        "user": (
            "Generate at least 2 falsifiable hypotheses from the synthesis below.\n"
            "For each hypothesis provide:\n"
            "- **Hypothesis statement**: A clear, testable claim\n"
            "- **Novelty argument**: Why this has NOT been tested before, citing "
            "specific gaps from the synthesis\n"
            "- **Rationale**: Theoretical or empirical basis for expecting this result\n"
            "- **Measurable prediction**: Specific quantitative outcome expected\n"
            "- **Failure condition**: What result would reject this hypothesis?\n"
            "- **Required baselines**: What modern, state-of-the-art methods must be "
            "compared against to make the finding meaningful?\n\n"
            "AVOID:\n"
            "- Hypotheses that are trivially obvious\n"
            "- Hypotheses that replicate well-known results already in the literature\n"
            "- Hypotheses that cannot be tested within the domain's feasibility constraints\n\n"
            "{domain_context}"
            "{extension_context}\n"
            "Synthesis:\n{synthesis}"
        ),
    },
}
