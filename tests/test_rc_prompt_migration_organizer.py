"""Migration tests for the Stage 14 evidence organizer prompt."""

from __future__ import annotations

from researchclaw.pipeline.evidence_organizer import build_organizer_prompt
from researchclaw.prompts import PromptManager


def _bundle() -> dict[str, object]:
    return {
        "run_dir": "/tmp/run",
        "stage_dir": "/tmp/run/stage-14",
        "workspace_path": "/tmp/workspace",
        "default_inputs": [
            {"label": "task_spec", "path": "/tmp/run/stage-09/task_spec.yaml", "exists": True}
        ],
        "optional_inputs": [],
        "result_files": [
            {"path": "/tmp/workspace/outputs/metrics.json", "exists": True}
        ],
        "reproducibility": {"launch_command": "python train.py"},
    }


def test_organizer_catalog_entry_exists() -> None:
    pm = PromptManager()

    assert pm.sub_prompt_meta("evidence_organizer").prompt_id == "evidence_organizer"


def test_organizer_prompt_preserves_constraints() -> None:
    prompt = build_organizer_prompt(_bundle())

    assert "Stage 14 Evidence Organizer Agent" in prompt
    assert "analysis.md must use exactly these sections" in prompt
    assert "## Experiment Objective" in prompt
    assert "## Reproducibility" in prompt
    assert "DO NOT" in prompt
    assert "PROCEED" in prompt
    assert "Evidence bundle" in prompt


def test_organizer_helper_sources_from_catalog(monkeypatch) -> None:
    import researchclaw.pipeline.evidence_organizer as organizer_mod

    pm = PromptManager()
    pm._sub_prompts["evidence_organizer"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL ORGANIZER SYSTEM",
        "user": "{sections}\n{retry_block}\n{bundle_json}",
    }
    monkeypatch.setattr(organizer_mod, "PromptManager", lambda: pm, raising=False)

    prompt = organizer_mod.build_organizer_prompt(_bundle(), ["bad heading"])

    assert "SENTINEL ORGANIZER SYSTEM" in prompt
    assert "bad heading" in prompt
    assert '"run_dir": "/tmp/run"' in prompt
