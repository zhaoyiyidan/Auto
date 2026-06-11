"""Cross-domain prompt catalog consistency checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from researchclaw.prompts.manager import SUPPORTED_DOMAINS, _load_bank
from researchclaw.prompts.metadata import PromptMetadata


CANONICAL_STAGES = (
    "topic_init",
    "problem_decompose",
    "search_strategy",
    "literature_collect",
    "literature_screen",
    "knowledge_extract",
    "synthesis",
    "hypothesis_gen",
    "experiment_design",
    "code_generation",
    "resource_planning",
    "result_analysis",
    "research_decision",
    "paper_outline",
    "paper_draft",
    "peer_review",
    "paper_revision",
    "quality_gate",
    "knowledge_archive",
    "export_publish",
)


@dataclass(frozen=True)
class ConsistencyReport:
    """Result of checking domain prompt banks against the canonical contract."""

    missing_stages: dict[str, tuple[str, ...]]
    json_mode_mismatches: dict[str, dict[str, bool]]
    required_var_mismatches: dict[str, dict[str, tuple[str, ...]]]
    meta_gaps: dict[str, tuple[str, ...]]

    @property
    def ok(self) -> bool:
        return (
            all(not missing for missing in self.missing_stages.values())
            and not self.json_mode_mismatches
            and not self.required_var_mismatches
            and not self.meta_gaps
        )


def _stage_meta(stage: str, entry: dict[str, Any]) -> PromptMetadata:
    return PromptMetadata.from_dict(entry.get("meta"), prompt_id=stage)


def check_domain_consistency(
    domains: tuple[str, ...] = SUPPORTED_DOMAINS,
) -> ConsistencyReport:
    """Check stage and metadata consistency across prompt domains.

    Banks are inspected through ``_load_bank`` so partially inherited banks,
    such as biology, are evaluated after their inheritance/override layer has
    produced the effective runtime ``STAGES`` mapping.
    """
    baseline_stages, _, _ = _load_bank("ml")
    baseline_json = {
        stage: bool(baseline_stages[stage].get("json_mode", False))
        for stage in CANONICAL_STAGES
        if stage in baseline_stages
    }
    baseline_required = {
        stage: _stage_meta(stage, baseline_stages[stage]).required_variables
        for stage in CANONICAL_STAGES
        if stage in baseline_stages
    }

    missing_stages: dict[str, tuple[str, ...]] = {}
    json_mode_mismatches: dict[str, dict[str, bool]] = {}
    required_var_mismatches: dict[str, dict[str, tuple[str, ...]]] = {}
    meta_gaps: dict[str, tuple[str, ...]] = {}

    for domain in domains:
        stages, _, _ = _load_bank(domain)
        missing = tuple(stage for stage in CANONICAL_STAGES if stage not in stages)
        missing_stages[domain] = missing

        domain_meta_gaps: list[str] = []
        for stage in CANONICAL_STAGES:
            if stage not in stages:
                continue

            entry = stages[stage]
            meta = _stage_meta(stage, entry)
            if not entry.get("meta") or not meta.purpose.strip():
                domain_meta_gaps.append(stage)
            elif not meta.required_variables:
                domain_meta_gaps.append(stage)

            actual_json = bool(entry.get("json_mode", False))
            expected_json = baseline_json.get(stage)
            if expected_json is not None and actual_json != expected_json:
                json_mode_mismatches.setdefault(stage, {})[domain] = actual_json

            actual_required = meta.required_variables
            expected_required = baseline_required.get(stage)
            if expected_required is not None and actual_required != expected_required:
                required_var_mismatches.setdefault(stage, {})[domain] = actual_required

        if domain_meta_gaps:
            meta_gaps[domain] = tuple(domain_meta_gaps)

    return ConsistencyReport(
        missing_stages=missing_stages,
        json_mode_mismatches=json_mode_mismatches,
        required_var_mismatches=required_var_mismatches,
        meta_gaps=meta_gaps,
    )
