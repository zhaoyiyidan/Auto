"""Generated prompt bank segment.

This module was split from the legacy monolithic prompt bank without
changing rendered prompt content.
"""

from __future__ import annotations

from typing import Any


STAGES: dict[str, dict[str, Any]] = {
    'hypothesis_gen': {
    "system": (
        "You formulate testable hypotheses about metabolic phenotypes (gene "
        "essentiality, growth rate, flux distribution, by-product yield, "
        "phenotypic phase planes, knockout viability). Your hypotheses must be:\n"
        "1. NOVEL: Not simply re-deriving textbook FBA predictions for the "
        "wild-type strain — target a perturbation, condition, or mechanism that "
        "the cited literature has not pinned down.\n"
        "2. ORGANISM-SPECIFIC: Reference an exact genome-scale metabolic model "
        "(GEM) — e.g. *E. coli* iJO1366 / iML1515, *S. cerevisiae* iMM904 / "
        "Yeast8, *human* Recon3D — and state why that model is appropriate.\n"
        "3. FEASIBLE in-silico: Solvable as an LP / MILP within COBRApy "
        "(FBA / pFBA / FVA / single- or double-knockout / phase plane).\n"
        "{feasibility_constraint}"
        "4. FALSIFIABLE with a numeric threshold: e.g. \"growth rate ratio "
        "KO/WT < 0.05\", \"succinate yield > 0.6 mol/mol glucose\", "
        "\"FVA flux range > 5 mmol/gDW/h\".\n"
        "5. CONTRASTIVE: At least one hypothesis should challenge a canonical "
        "FBA prediction — e.g. predict a synthetically lethal pair the GEM "
        "currently misses, or a medium under which an essential gene becomes "
        "dispensable."
    ),
    "user": (
        "Generate at least 2 falsifiable metabolic-modelling hypotheses from "
        "the synthesis below. For EACH hypothesis provide:\n"
        "- **Hypothesis statement**: A clear, testable claim referencing a "
        "specific organism / GEM, a measurable observable (growth rate, flux, "
        "yield, essentiality), and a perturbation (gene KO, medium swap, "
        "exchange-bound change, double-KO).\n"
        "- **Novelty argument**: Why this prediction is not already in the "
        "synthesis literature — cite the specific gap.\n"
        "- **Rationale**: Stoichiometric / pathway / regulatory reasoning that "
        "motivates the prediction. Reference the relevant subsystem "
        "(e.g. central carbon metabolism, oxidative phosphorylation).\n"
        "- **Measurable prediction**: Numeric threshold with units "
        "(mmol/gDW/h for fluxes, 1/h for growth, dimensionless ratios).\n"
        "- **Failure condition**: What in-silico result rejects the hypothesis "
        "(e.g. KO ratio > 0.5, FVA range < 0.1).\n"
        "- **Required references / baselines**: Published GEM, experimental "
        "essentiality set (Keio / OGEE), 13C-MFA fluxes, Edwards-Palsson "
        "phase-plane data — whichever is needed to make the test meaningful.\n\n"
        "AVOID:\n"
        "- Hypotheses that simply restate the GEM's textbook biomass coefficient.\n"
        "- Hypotheses that depend on dynamic / kinetic modelling (FBA is "
        "steady-state by construction).\n"
        "- ML-style 'predict X with a neural network' — FBA is a linear "
        "program; learning is not the right tool here.\n\n"
        "{domain_context}"
        "{extension_context}\n"
        "Synthesis:\n{synthesis}"
    ),
},
}
