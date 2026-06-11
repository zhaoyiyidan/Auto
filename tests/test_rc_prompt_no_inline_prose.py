"""Acceptance tests for prompt/code separation in pipeline modules."""

from __future__ import annotations

from pathlib import Path


PROMPT_PROSE_PATTERNS = (
    "You are a",
    "Return only JSON",
    "MUST NOT",
    "Do not mention agent",
)


def test_no_instruction_prose_in_pipeline() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_dir = repo_root / "researchclaw" / "pipeline"
    files = sorted(pipeline_dir.glob("*.py"))
    files.extend(sorted((pipeline_dir / "stage_impls").glob("*.py")))

    matches: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in PROMPT_PROSE_PATTERNS:
                if pattern in line:
                    rel = path.relative_to(repo_root)
                    matches.append(f"{rel}:{lineno}: {pattern}")

    assert matches == []
