"""Migration tests for Stage 10 code-generation prompts."""

from __future__ import annotations

import json

from researchclaw.prompts import PromptManager


def _workspace_prompt_kwargs() -> dict[str, object]:
    return {
        "topic": "topic",
        "plan_md": "plan",
        "expected_outputs": ["outputs/metrics.json"],
        "manifest_filename": "run_manifest.json",
    }


def _repair_prompt_kwargs() -> dict[str, object]:
    return {
        "topic": "topic",
        "plan_md": "plan",
        "expected_outputs": ["outputs/metrics.json"],
        "project_files": ["train.py"],
        "run_summaries": ["previous run"],
        "manifest_filename": "run_manifest.json",
        "repair_request": {"reason": "code_defect", "errors": ["accuracy stuck at 0"]},
    }


def _extract_run_manifest_example(prompt: str) -> dict[str, object]:
    marker = "Required run_manifest.json format example"
    start = prompt.index("{", prompt.index(marker))
    end = prompt.index("\n\nStage 10 validation boundary", start)
    payload = json.loads(prompt[start:end])
    assert isinstance(payload, dict)
    return payload


def test_codegen_catalog_entries_exist() -> None:
    pm = PromptManager()

    assert pm.sub_prompt_meta("workspace_codegen").prompt_id == "workspace_codegen"
    assert pm.sub_prompt_meta("workspace_repair").prompt_id == "workspace_repair"


def test_workspace_codegen_prompt_preserves_constraints() -> None:
    from researchclaw.pipeline.stage_impls._code_generation import (
        _workspace_codegen_prompt,
    )

    prompt = _workspace_codegen_prompt(**_workspace_prompt_kwargs())

    assert "Completion contract (MUST)" in prompt
    assert "Do not submit" in prompt
    assert "Stage 10 validation boundary" in prompt
    assert "MUST NOT run the formal experiment" in prompt
    assert "Required run_manifest.json format example" in prompt
    assert "manifest.code_commit to the final HEAD commit" in prompt
    assert "schema_version, code_commit, launch.command, launch.cwd" in prompt
    manifest_example = _extract_run_manifest_example(prompt)
    assert manifest_example["result_paths"] == ["outputs/metrics.json"]
    assert "metrics" not in manifest_example
    assert "primary_metric" not in manifest_example
    assert "metric_direction" not in manifest_example


def test_repair_prompt_preserves_constraints() -> None:
    from researchclaw.pipeline.stage_impls._code_generation import (
        _repair_or_refine_prompt,
    )

    prompt = _repair_or_refine_prompt(**_repair_prompt_kwargs())

    assert "REPAIR REQUEST" in prompt
    assert "Completion contract (MUST)" in prompt
    assert "Do not submit" in prompt
    assert "Stage 10 validation boundary" in prompt
    assert "MUST NOT run the formal experiment" in prompt
    assert "Required run_manifest.json format example" in prompt
    assert "schema_version, code_commit, launch.command, launch.cwd" in prompt
    manifest_example = _extract_run_manifest_example(prompt)
    assert manifest_example["result_paths"] == ["outputs/metrics.json"]
    assert "metrics" not in manifest_example


def test_workspace_codegen_helper_sources_from_catalog(monkeypatch) -> None:
    from researchclaw.pipeline.stage_impls import _code_generation

    pm = PromptManager()
    pm._sub_prompts["workspace_codegen"]["system"] = "SENTINEL CODEGEN SYSTEM"  # type: ignore[attr-defined]
    monkeypatch.setattr(_code_generation, "PromptManager", lambda: pm)

    prompt = _code_generation._workspace_codegen_prompt(**_workspace_prompt_kwargs())

    assert "SENTINEL CODEGEN SYSTEM" in prompt


def test_repair_helper_sources_from_catalog(monkeypatch) -> None:
    from researchclaw.pipeline.stage_impls import _code_generation

    pm = PromptManager()
    pm._sub_prompts["workspace_repair"]["system"] = "SENTINEL REPAIR SYSTEM"  # type: ignore[attr-defined]
    monkeypatch.setattr(_code_generation, "PromptManager", lambda: pm)

    prompt = _code_generation._repair_or_refine_prompt(**_repair_prompt_kwargs())

    assert "SENTINEL REPAIR SYSTEM" in prompt
