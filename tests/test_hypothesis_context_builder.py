from __future__ import annotations

import json
from pathlib import Path

from researchclaw.pipeline.stage_impls._hypothesis_context import (
    USER_PRIOR_HEADER,
    build_hypothesis_context,
)


def _stage_dir(run_dir: Path, stage_num: int) -> Path:
    path = run_dir / f"stage-{stage_num:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_build_synthesis_only_context(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage7 = _stage_dir(run_dir, 7)
    stage8 = _stage_dir(run_dir, 8)
    synthesis = "# Synthesis\nGap found."
    (stage7 / "synthesis.md").write_text(synthesis, encoding="utf-8")

    ctx = build_hypothesis_context(run_dir, stage8)

    assert ctx.synthesis == synthesis
    assert ctx.research_context == synthesis
    assert ctx.user_prior == ""
    assert ctx.extension_context == ""
    assert ctx.extension_block == ""
    assert USER_PRIOR_HEADER not in ctx.research_context
    user_source = next(source for source in ctx.sources if source.role == "user_prior")
    assert user_source.present is False
    assert user_source.chars == 0


def test_build_context_with_user_prior(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage7 = _stage_dir(run_dir, 7)
    stage8 = _stage_dir(run_dir, 8)
    (stage7 / "synthesis.md").write_text("# Synthesis\nGap found.", encoding="utf-8")
    user_text = "Prior expert note: focus on calibration drift."
    (stage8 / "user_context.md").write_text(f"\n{user_text}\n", encoding="utf-8")

    ctx = build_hypothesis_context(run_dir, stage8)

    assert ctx.user_prior == user_text
    assert USER_PRIOR_HEADER in ctx.research_context
    assert "treat as authoritative context to inform" in ctx.research_context
    assert user_text in ctx.research_context
    user_source = next(source for source in ctx.sources if source.role == "user_prior")
    assert user_source.present is True
    assert user_source.chars == len(user_text)
    assert user_source.loaded_from == "stage-08/user_context.md"


def test_build_context_with_extension_keeps_extension_separate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stage7 = _stage_dir(run_dir, 7)
    stage8 = _stage_dir(run_dir, 8)
    (stage7 / "synthesis.md").write_text("# Synthesis\nGap found.", encoding="utf-8")
    extension = "Prior H1 opened a follow-up mechanism."
    (run_dir / "hypothesis_extension_context.md").write_text(
        extension, encoding="utf-8"
    )

    ctx = build_hypothesis_context(run_dir, stage8)

    assert ctx.extension_context == extension
    assert "## Hypothesis Extension Context" in ctx.extension_block
    assert "Generate deeper follow-up hypotheses" in ctx.extension_block
    assert extension in ctx.extension_block
    assert extension not in ctx.research_context
    extension_source = next(source for source in ctx.sources if source.role == "extension")
    assert extension_source.present is True
    assert extension_source.chars == len(extension)


def test_user_prior_and_extension_stay_in_distinct_partitions(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stage7 = _stage_dir(run_dir, 7)
    stage8 = _stage_dir(run_dir, 8)
    (stage7 / "synthesis.md").write_text("# Synthesis\nGap found.", encoding="utf-8")
    user_text = "USER PRIOR: instrument latency matters."
    extension = "EXTEND SEED: prior result suggested a retry policy."
    (stage8 / "user_context.md").write_text(user_text, encoding="utf-8")
    (run_dir / "hypothesis_extension_context.md").write_text(
        extension, encoding="utf-8"
    )

    ctx = build_hypothesis_context(run_dir, stage8)

    assert user_text in ctx.research_context
    assert extension not in ctx.research_context
    assert extension in ctx.extension_block
    assert user_text not in ctx.extension_block


def test_user_context_falls_back_to_latest_stage8_version(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stage7 = _stage_dir(run_dir, 7)
    stage8 = _stage_dir(run_dir, 8)
    (stage7 / "synthesis.md").write_text("# Synthesis\nGap found.", encoding="utf-8")
    (run_dir / "stage-08_v1").mkdir(parents=True)
    (run_dir / "stage-08_v1" / "user_context.md").write_text(
        "older user prior", encoding="utf-8"
    )
    (run_dir / "stage-08_v2").mkdir(parents=True)
    (run_dir / "stage-08_v2" / "user_context.md").write_text(
        "latest user prior", encoding="utf-8"
    )

    ctx = build_hypothesis_context(run_dir, stage8)

    assert ctx.user_prior == "latest user prior"
    assert "latest user prior" in ctx.research_context
    user_source = next(source for source in ctx.sources if source.role == "user_prior")
    assert user_source.loaded_from == "stage-08_v2/user_context.md"


def test_whitespace_only_user_context_is_absent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage7 = _stage_dir(run_dir, 7)
    stage8 = _stage_dir(run_dir, 8)
    synthesis = "# Synthesis\nGap found."
    (stage7 / "synthesis.md").write_text(synthesis, encoding="utf-8")
    (stage8 / "user_context.md").write_text("\n\t  \n", encoding="utf-8")

    ctx = build_hypothesis_context(run_dir, stage8)

    assert ctx.user_prior == ""
    assert ctx.research_context == synthesis
    assert USER_PRIOR_HEADER not in ctx.research_context
    user_source = next(source for source in ctx.sources if source.role == "user_prior")
    assert user_source.present is False
    assert user_source.chars == 0


def test_audit_snapshot_and_manifest_are_written(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage7 = _stage_dir(run_dir, 7)
    stage8 = _stage_dir(run_dir, 8)
    (stage7 / "synthesis.md").write_text("# Synthesis\nGap found.", encoding="utf-8")
    (stage8 / "user_context.md").write_text("USER PRIOR", encoding="utf-8")
    (run_dir / "hypothesis_extension_context.md").write_text(
        "EXTENSION SEED", encoding="utf-8"
    )

    ctx = build_hypothesis_context(run_dir, stage8)

    snapshot = (stage8 / "context_snapshot.md").read_text(encoding="utf-8")
    manifest = json.loads(
        (stage8 / "context_manifest.json").read_text(encoding="utf-8")
    )
    assert ctx.research_context in snapshot
    assert ctx.extension_block in snapshot
    assert manifest["built_at"]
    assert {source["role"] for source in manifest["sources"]} == {
        "synthesis",
        "user_prior",
        "extension",
    }
    assert any(
        source["role"] == "user_prior" and source["present"] is True
        for source in manifest["sources"]
    )
