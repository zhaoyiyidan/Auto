from __future__ import annotations

from pathlib import Path


LEGACY_STAGE_9_14_TOKENS = (
    "experiment_final",
    "refinement_log",
    "_extract_multi_file_blocks",
    "create_sandbox",
    "collider_agent",
    "EXPERIMENT_RUN",
    "ITERATIVE_REFINE",
    "exp_plan.yaml",
)


def test_runtime_package_has_no_legacy_stage_9_14_tokens() -> None:
    root = Path(__file__).resolve().parents[1] / "researchclaw"
    offenders: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in LEGACY_STAGE_9_14_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(root)}: {token}")

    assert not offenders, "Legacy Stage 9-14 tokens remain:\n" + "\n".join(offenders)
