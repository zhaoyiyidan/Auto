"""Prompt metadata models and default catalog annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_APPLICABLE_DOMAINS = ("ml", "hep_ph", "biology_metabolic")


@dataclass(frozen=True)
class PromptMetadata:
    """Read-only metadata describing a stage or sub-prompt."""

    prompt_id: str
    version: str = "1.0.0"
    purpose: str = ""
    required_variables: tuple[str, ...] = ()
    optional_variables: tuple[str, ...] = ()
    output_schema: str = ""
    json_mode: bool = False
    token_budget: int | None = None
    applicable_domains: tuple[str, ...] = DEFAULT_APPLICABLE_DOMAINS
    overridable: bool = True

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        prompt_id: str = "",
    ) -> "PromptMetadata":
        """Build metadata from YAML/catalog dictionaries."""
        raw = dict(data or {})
        resolved_prompt_id = str(raw.get("prompt_id") or prompt_id)
        token_budget = raw.get("token_budget")
        if token_budget is not None:
            token_budget = int(token_budget)

        return cls(
            prompt_id=resolved_prompt_id,
            version=str(raw.get("version") or "1.0.0"),
            purpose=str(raw.get("purpose") or ""),
            required_variables=tuple(raw.get("required_variables") or ()),
            optional_variables=tuple(raw.get("optional_variables") or ()),
            output_schema=str(raw.get("output_schema") or ""),
            json_mode=bool(raw.get("json_mode", False)),
            token_budget=token_budget,
            applicable_domains=tuple(
                raw.get("applicable_domains") or DEFAULT_APPLICABLE_DOMAINS
            ),
            overridable=bool(raw.get("overridable", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a complete metadata dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "version": self.version,
            "purpose": self.purpose,
            "required_variables": self.required_variables,
            "optional_variables": self.optional_variables,
            "output_schema": self.output_schema,
            "json_mode": self.json_mode,
            "token_budget": self.token_budget,
            "applicable_domains": self.applicable_domains,
            "overridable": self.overridable,
        }


CANONICAL_STAGE_METADATA: dict[str, dict[str, Any]] = {
    "topic_init": {
        "purpose": "Create the initial scoped research goal.",
        "required_variables": [
            "topic",
            "domains",
            "project_name",
            "quality_threshold",
        ],
    },
    "problem_decompose": {
        "purpose": "Break the research goal into prioritized sub-questions.",
        "required_variables": ["topic", "goal_text"],
    },
    "search_strategy": {
        "purpose": "Build the literature retrieval strategy and source plan.",
        "required_variables": ["topic", "problem_tree"],
        "output_schema": "JSON object with search_plan_yaml and sources.",
    },
    "literature_collect": {
        "purpose": "Collect candidate literature from the search plan.",
        "required_variables": ["topic", "plan_text"],
        "output_schema": "JSON object with candidate literature records.",
    },
    "literature_screen": {
        "purpose": "Filter candidate literature for relevance and quality.",
        "required_variables": [
            "topic",
            "domains",
            "quality_threshold",
            "candidates_text",
        ],
        "output_schema": "JSON object with screened papers and rejection reasons.",
    },
    "knowledge_extract": {
        "purpose": "Extract reusable evidence cards from screened literature.",
        "required_variables": ["shortlist"],
        "output_schema": "JSON object with extracted knowledge cards.",
    },
    "synthesis": {
        "purpose": "Synthesize extracted literature into research gaps and themes.",
        "required_variables": ["topic", "domain_context", "cards_context"],
    },
    "hypothesis_gen": {
        "purpose": "Generate falsifiable hypotheses from the synthesis.",
        "required_variables": [
            "feasibility_constraint",
            "domain_context",
            "synthesis",
        ],
        "optional_variables": ["extension_context"],
    },
    "experiment_design": {
        "purpose": "Design a feasible experiment plan for testing hypotheses.",
        "required_variables": [
            "preamble",
            "domain_design_context",
            "metric_key",
            "metric_direction",
            "dataset_guidance",
            "hardware_profile",
            "time_budget_sec",
            "per_condition_budget_sec",
            "available_tier1_datasets",
            "hypotheses",
        ],
    },
    "code_generation": {
        "purpose": "Generate runnable experiment code from the experiment plan.",
        "required_variables": [
            "topic",
            "metric",
            "pkg_hint",
            "metric_direction_hint",
            "time_budget",
            "exp_plan",
        ],
    },
    "resource_planning": {
        "purpose": "Plan compute resources and execution constraints.",
        "required_variables": ["exp_plan"],
        "output_schema": "JSON object with resource plan and risks.",
    },
    "result_analysis": {
        "purpose": "Analyze experiment results and extract evidence.",
        "required_variables": ["preamble", "data_context", "context"],
    },
    "research_decision": {
        "purpose": "Decide whether to proceed, extend, or revise based on analysis.",
        "required_variables": ["analysis"],
    },
    "paper_outline": {
        "purpose": "Create the paper outline from decision and evidence.",
        "required_variables": [
            "preamble",
            "venue_guidance",
            "academic_style_guide",
            "topic_constraint",
            "analysis",
            "decision",
            "feedback",
        ],
    },
    "paper_draft": {
        "purpose": "Draft the research paper in markdown.",
        "required_variables": [
            "preamble",
            "venue_guidance",
            "academic_style_guide",
            "narrative_writing_rules",
            "anti_hedging_rules",
            "anti_repetition_rules",
            "writing_structure",
            "citation_instruction",
            "exp_metrics_instruction",
            "topic_constraint",
            "outline",
            "lcc",
            "tabular",
        ],
    },
    "peer_review": {
        "purpose": "Review the draft and identify required revisions.",
        "required_variables": ["topic", "experiment_evidence", "draft"],
    },
    "paper_revision": {
        "purpose": "Revise the paper based on reviews and writing rules.",
        "required_variables": [
            "academic_style_guide",
            "narrative_writing_rules",
            "anti_hedging_rules",
            "anti_repetition_rules",
            "writing_structure",
            "topic_constraint",
            "draft",
            "reviews",
        ],
    },
    "quality_gate": {
        "purpose": "Evaluate final paper quality against the configured threshold.",
        "required_variables": ["quality_threshold", "revised"],
        "output_schema": "JSON object with score, verdict, strengths, weaknesses, actions.",
    },
    "knowledge_archive": {
        "purpose": "Write a reproducibility-focused retrospective archive.",
        "required_variables": ["preamble", "decision", "analysis", "revised"],
    },
    "export_publish": {
        "purpose": "Format the revised paper for publication export.",
        "required_variables": ["revised"],
    },
}


SUB_PROMPT_METADATA: dict[str, dict[str, Any]] = {
    "workspace_codegen": {
        "purpose": "Instruct a workspace-native code agent to implement an experiment.",
        "required_variables": [
            "topic",
            "exp_plan",
            "metric",
            "pkg_hint",
            "compute_budget",
            "extra_guidance",
            "manifest_filename",
            "manifest_schema_example",
            "stage10_validation_boundary",
        ],
    },
    "workspace_repair": {
        "purpose": "Instruct a workspace-native code agent to repair or refine an experiment.",
        "required_variables": [
            "request_section",
            "topic",
            "metric_direction",
            "metric_key",
            "exp_plan",
            "project_files",
            "run_summaries",
            "results_section",
            "manifest_filename",
            "manifest_schema_example",
            "stage10_validation_boundary",
        ],
    },
    "hypothesis_synthesize": {
        "purpose": "Merge multiple hypothesis debate perspectives.",
        "required_variables": ["perspectives"],
    },
    "analysis_synthesize": {
        "purpose": "Merge multiple result-analysis perspectives.",
        "required_variables": ["perspectives"],
    },
    "code_repair": {
        "purpose": "Repair a generated code file from validation issues.",
        "required_variables": ["fname", "issues_text", "all_files_ctx"],
    },
    "iterative_improve": {
        "purpose": "Improve experiment code using previous run summaries.",
        "required_variables": [
            "topic",
            "metric_key",
            "metric_direction",
            "files_context",
            "run_summaries",
            "condition_coverage_hint",
            "exp_plan_anchor",
        ],
    },
    "iterative_repair": {
        "purpose": "Repair an experiment workspace from a high-level issue.",
        "required_variables": ["issue_text", "all_files_ctx"],
    },
    "architecture_planning": {
        "purpose": "Plan the experiment codebase structure.",
        "required_variables": ["topic", "metric", "exp_plan"],
    },
    "generate_single_file": {
        "purpose": "Generate one code file from the architecture plan.",
        "required_variables": [
            "topic",
            "exp_plan",
            "file_name",
            "file_spec",
            "blueprint",
            "dependency_summaries",
            "dependency_code",
            "pkg_hint",
        ],
    },
    "code_exec_fix": {
        "purpose": "Fix runtime errors in generated experiment code.",
        "required_variables": [
            "stderr",
            "stdout_tail",
            "returncode",
            "files_context",
        ],
    },
    "code_reviewer": {
        "purpose": "Review generated experiment code for scientific correctness.",
        "required_variables": ["topic", "metric", "exp_plan", "files_context"],
        "output_schema": "JSON object with verdict, score, critical_issues, suggestions.",
    },
}


def _catalog_meta_dict(
    prompt_id: str,
    base: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    token_budget = entry.get("max_tokens")
    meta = {
        "prompt_id": prompt_id,
        "version": str(base.get("version") or "1.0.0"),
        "purpose": str(base.get("purpose") or ""),
        "required_variables": list(base.get("required_variables") or []),
        "optional_variables": list(base.get("optional_variables") or []),
        "output_schema": str(base.get("output_schema") or ""),
        "json_mode": bool(entry.get("json_mode", False)),
        "token_budget": int(token_budget) if token_budget is not None else None,
        "applicable_domains": list(
            base.get("applicable_domains") or DEFAULT_APPLICABLE_DOMAINS
        ),
        "overridable": bool(base.get("overridable", True)),
    }
    return meta


def apply_stage_metadata(stages: dict[str, dict[str, Any]]) -> None:
    """Attach default metadata to known stage entries in-place."""
    for stage, base in CANONICAL_STAGE_METADATA.items():
        if stage in stages:
            stages[stage].setdefault(
                "meta",
                _catalog_meta_dict(stage, base, stages[stage]),
            )


def apply_sub_prompt_metadata(sub_prompts: dict[str, dict[str, Any]]) -> None:
    """Attach default metadata to known sub-prompt entries in-place."""
    for name, base in SUB_PROMPT_METADATA.items():
        if name in sub_prompts:
            sub_prompts[name].setdefault(
                "meta",
                _catalog_meta_dict(name, base, sub_prompts[name]),
            )
