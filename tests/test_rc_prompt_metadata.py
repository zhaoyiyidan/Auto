"""Tests for prompt metadata introspection."""

from __future__ import annotations

from researchclaw.prompts import PromptManager
from researchclaw.prompts.manager import SUPPORTED_DOMAINS
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


def test_prompt_metadata_from_dict_roundtrip() -> None:
    meta = PromptMetadata.from_dict(
        {
            "prompt_id": "search_strategy",
            "version": "1.2.3",
            "purpose": "Build literature retrieval strategy.",
            "required_variables": ["topic", "problem_tree"],
            "optional_variables": ["extension_context"],
            "output_schema": '{"search_plan_yaml": "...", "sources": []}',
            "json_mode": True,
            "token_budget": 1200,
            "applicable_domains": ["ml", "hep_ph"],
            "overridable": False,
        }
    )

    assert meta.prompt_id == "search_strategy"
    assert meta.required_variables == ("topic", "problem_tree")
    assert meta.optional_variables == ("extension_context",)
    assert meta.applicable_domains == ("ml", "hep_ph")
    assert meta.to_dict() == {
        "prompt_id": "search_strategy",
        "version": "1.2.3",
        "purpose": "Build literature retrieval strategy.",
        "required_variables": ("topic", "problem_tree"),
        "optional_variables": ("extension_context",),
        "output_schema": '{"search_plan_yaml": "...", "sources": []}',
        "json_mode": True,
        "token_budget": 1200,
        "applicable_domains": ("ml", "hep_ph"),
        "overridable": False,
    }


def test_pm_meta_returns_empty_for_unannotated_stage() -> None:
    pm = PromptManager()
    pm._stages["scratch_stage"] = {"system": "sys", "user": "usr"}  # type: ignore[attr-defined]

    meta = pm.meta("scratch_stage")

    assert meta == PromptMetadata(prompt_id="scratch_stage")
    assert meta.applicable_domains == SUPPORTED_DOMAINS


def test_pm_required_variables_reads_meta() -> None:
    pm = PromptManager()

    assert pm.required_variables("topic_init") == (
        "topic",
        "domains",
        "project_name",
        "quality_threshold",
    )
    assert pm.required_variables("search_strategy") == ("topic", "problem_tree")


def test_meta_json_mode_matches_entry() -> None:
    pm = PromptManager()

    for stage in CANONICAL_STAGES:
        assert pm.meta(stage).json_mode is pm.json_mode(stage), stage


def test_meta_complete_for_all_canonical_stages() -> None:
    pm = PromptManager()

    for stage in CANONICAL_STAGES:
        meta = pm.meta(stage)
        assert meta.prompt_id == stage
        assert meta.purpose.strip(), stage
        assert meta.required_variables, stage
        assert meta.json_mode is pm.json_mode(stage)
