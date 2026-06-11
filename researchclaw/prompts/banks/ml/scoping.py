"""Generated prompt bank segment.

This module was split from the legacy monolithic prompt bank without
changing rendered prompt content.
"""

from __future__ import annotations

from typing import Any


STAGES: dict[str, dict[str, Any]] = {
    'topic_init': {
        "system": (
            "You are a rigorous research planner who identifies NOVEL, TIMELY "
            "research angles. You follow recent trends from top venues in the "
            "relevant domain and propose research that advances "
            "the frontier rather than repeating known results.\n\n"
            "NOVELTY PRINCIPLES:\n"
            "- A good research angle addresses a GAP not yet covered by existing work.\n"
            "- Avoid pure benchmark/comparison studies unless the methodology is novel.\n"
            "- Prefer angles that combine existing techniques in new ways, apply methods "
            "to underexplored domains, or challenge common assumptions.\n"
            "- The research must be FEASIBLE with limited compute (single GPU, hours not days).\n"
            "- Check: would a reviewer say 'this is already well-known'? If so, find a sharper angle."
        ),
        "user": (
            "Create a SMART research goal in markdown.\n"
            "Topic: {topic}\n"
            "Domains: {domains}\n"
            "Project: {project_name}\n"
            "Quality threshold: {quality_threshold}\n\n"
            "Required sections:\n"
            "- **Topic**: The broad area\n"
            "- **Novel Angle**: What specific aspect has NOT been well-studied? "
            "Why is this timely? What recent development creates "
            "an opportunity? How does this differ from standard approaches?\n"
            "- **Scope**: Focused enough for a single paper\n"
            "- **SMART Goal**: Specific, Measurable, Achievable, Relevant, Time-bound\n"
            "- **Constraints**: Compute budget, available tools, data access\n"
            "- **Success Criteria**: What results would make this publishable?\n"
            "- **Generated**: Timestamp\n\n"
            "IMPORTANT: The 'Novel Angle' section must convincingly argue why this "
            "specific research direction is NOT already covered by existing work. "
            "If the topic is well-studied (e.g., 'comparing optimizers'), you MUST "
            "find a specific unexplored aspect (e.g., 'under distribution shift with "
            "noisy gradients', 'in the few-shot regime', 'with modern architectures').\n\n"
            "TREND VALIDATION (MANDATORY):\n"
            "- Describe the research trend or gap that motivates this work. "
            "Do NOT fabricate specific paper titles or citations — actual papers "
            "will be retrieved in the literature search stage.\n"
            "- Name the specific benchmark/dataset that will be used for evaluation.\n"
            "- If no standard benchmark exists, explain how results will be measured.\n"
            "- State whether SOTA results exist on this benchmark and what they are.\n"
            "- Add a 'Benchmark' subsection listing: name, source, metrics, "
            "current SOTA (if known)."
        ),
    },
    'problem_decompose': {
        "system": "You are a senior research strategist.",
        "user": (
            "Decompose this research problem into at least 4 prioritized "
            "sub-questions.\n"
            "Topic: {topic}\n"
            "Output markdown with sections: Source, Sub-questions, Priority "
            "Ranking, Risks.\n"
            "Goal context:\n{goal_text}"
        ),
    },
}
