"""Tests for the split prompt bank catalog layout."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from researchclaw.prompts import (
    DEBATE_ROLES_ANALYSIS,
    DEBATE_ROLES_HYPOTHESIS,
    PromptManager,
    RenderedPrompt,
    SECTION_WORD_TARGETS,
    _DEFAULT_STAGES,
)


ML_STAGES_SNAPSHOT_SHA256 = "8867b5a50949145d28e0d306af11659503587313654751649b7fbcdc4cc21308"


def _stable_stage_bytes(stages: dict[str, object]) -> bytes:
    return json.dumps(
        stages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_assembled_ml_stages_match_snapshot() -> None:
    bank = importlib.import_module("researchclaw.prompts.banks.ml")

    digest = hashlib.sha256(_stable_stage_bytes(bank.STAGES)).hexdigest()

    assert digest == ML_STAGES_SNAPSHOT_SHA256


def test_legacy_imports_still_resolve() -> None:
    from researchclaw.prompts.ml import STAGES as ml_stages
    from researchclaw.prompts.ml import _DEFAULT_STAGES as legacy_ml_stages
    from researchclaw.prompts.hep import STAGES as hep_stages
    from researchclaw.prompts.biology import STAGES as biology_stages

    assert PromptManager
    assert RenderedPrompt
    assert SECTION_WORD_TARGETS
    assert DEBATE_ROLES_HYPOTHESIS
    assert DEBATE_ROLES_ANALYSIS
    assert _DEFAULT_STAGES is ml_stages
    assert legacy_ml_stages is ml_stages
    assert hep_stages
    assert biology_stages


def test_no_bank_file_exceeds_line_budget() -> None:
    banks_root = Path("researchclaw/prompts/banks")
    bank_files = sorted(
        path
        for path in banks_root.glob("**/*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    )

    assert bank_files, "expected split bank modules under researchclaw/prompts/banks"
    for path in bank_files:
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        assert line_count < 600, f"{path} has {line_count} lines"
