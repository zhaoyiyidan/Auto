"""Generated prompt bank segment.

This module was split from the legacy monolithic prompt bank without
changing rendered prompt content.
"""

from __future__ import annotations

from typing import Any


STAGES: dict[str, dict[str, Any]] = {
    'paper_outline': {
    "system": (
        "You are an academic writing planner for top-tier biology / systems-"
        "biology venues (Nature Methods, Bioinformatics, PLOS Computational "
        "Biology, Molecular Systems Biology, Metabolic Engineering)."
        "{venue_guidance}"
    ),
    "user": (
        "{preamble}\n\n"
        "{academic_style_guide}\n\n"
        "Create a detailed paper outline in markdown for a constraint-based "
        "metabolic-modelling manuscript.\n"
        "Follow the standard biology paper structure (Methods / Results / "
        "Discussion). Required sections, in this order:\n"
        "1. **Abstract** (≈200 words): organism + GEM + perturbation + key "
        "in-silico finding + biological / engineering implication.\n"
        "2. **Introduction**: physiological motivation, prior FBA / metabolic-"
        "engineering literature, gap, contribution.\n"
        "3. **Genome-Scale Metabolic Model**: model id, source (BIGG / "
        "BioModels), version, n_reactions / n_metabolites / n_genes, biomass "
        "reaction, any curation steps applied. Cite the original GEM paper.\n"
        "4. **Methods**: medium definitions, objective functions, "
        "FBA / pFBA / FVA / sampling / knockout protocol, solver and "
        "tolerance, software versions (COBRApy, Python, GLPK / Gurobi).\n"
        "5. **Results**: wild-type validation (growth rate vs measured), "
        "perturbation outcomes (knockout / medium swap), phenotypic phase "
        "planes, essentiality predictions vs Keio / OGEE, headline pathway "
        "Escher map.\n"
        "6. **Discussion**: biological interpretation, comparison to prior "
        "predictions, limitations of the GEM, implications for metabolic "
        "engineering / strain design.\n"
        "7. **Supplementary**: full flux tables, full essentiality tables, "
        "additional Escher maps, code / data availability.\n\n"
        "Per-section: state the goal, target word count, and the evidence "
        "rows (specific reactions / genes / figure ids) that anchor each "
        "claim.\n"
        "The outline MUST include a method / strain name (2-5 chars) for the "
        "title (e.g. 'iJO-KO', 'pFBA-Lite').\n"
        "Propose 3 candidate titles in the 'MethodName: Subtitle' format "
        "(each ≤ 14 words). Rate each on memorability (1-5), specificity "
        "(1-5), and biological-novelty signal (1-5).\n"
        "AVOID ML-venue conventions: do NOT add a 'Broader Impact' section, "
        "a reproducibility checklist, or leaderboard tables — biology venues "
        "do not require these.\n"
        "{topic_constraint}"
        "{feedback}"
        "Analysis:\n{analysis}\n\nDecision:\n{decision}"
    ),
    "max_tokens": 8192,
},
}
