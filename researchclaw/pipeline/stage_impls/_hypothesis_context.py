"""Context assembly for Stage 8 hypothesis generation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from researchclaw.pipeline._helpers import (
    _find_prior_file,
    _read_prior_artifact,
    _utcnow_iso,
)

USER_CONTEXT_FILENAME = "user_context.md"
USER_PRIOR_HEADER = "## User Prior Knowledge"


@dataclass(frozen=True)
class ContextSource:
    role: str
    path: str
    present: bool
    chars: int
    loaded_from: str


@dataclass(frozen=True)
class HypothesisContext:
    synthesis: str
    user_prior: str
    extension_context: str
    research_context: str
    extension_block: str
    sources: list[ContextSource]


def build_hypothesis_context(
    run_dir: Path,
    stage_dir: Path,
    *,
    write_audit: bool = True,
) -> HypothesisContext:
    """Build the full pre-generation context for Stage 8."""
    run_dir = Path(run_dir)
    stage_dir = Path(stage_dir)

    synthesis = _read_prior_artifact(run_dir, "synthesis.md") or ""
    synthesis_path = _find_prior_file(run_dir, "synthesis.md")
    user_prior, user_loaded_from = _read_user_context(run_dir, stage_dir)
    extension_context, extension_loaded_from = _read_extension_context(run_dir)

    research_context = _build_research_context(synthesis, user_prior)
    extension_block = _build_extension_block(extension_context)

    sources = [
        ContextSource(
            role="synthesis",
            path="stage-07/synthesis.md",
            present=bool(synthesis),
            chars=len(synthesis),
            loaded_from=_relative_to_run(run_dir, synthesis_path),
        ),
        ContextSource(
            role="user_prior",
            path="stage-08/user_context.md",
            present=bool(user_prior),
            chars=len(user_prior),
            loaded_from=_relative_to_run(run_dir, user_loaded_from),
        ),
        ContextSource(
            role="extension",
            path="hypothesis_extension_context.md",
            present=bool(extension_context),
            chars=len(extension_context),
            loaded_from=_relative_to_run(run_dir, extension_loaded_from),
        ),
    ]
    ctx = HypothesisContext(
        synthesis=synthesis,
        user_prior=user_prior,
        extension_context=extension_context,
        research_context=research_context,
        extension_block=extension_block,
        sources=sources,
    )

    if write_audit:
        _write_audit(stage_dir, ctx)

    return ctx


def _build_research_context(synthesis: str, user_prior: str) -> str:
    if not user_prior:
        return synthesis
    separator = "\n\n" if synthesis and not synthesis.endswith("\n\n") else ""
    return (
        f"{synthesis}{separator}"
        f"{USER_PRIOR_HEADER}\n"
        "Please treat as authoritative context to inform - not replace - "
        "the synthesis-driven gap analysis.\n\n"
        f"{user_prior}"
    )


def _build_extension_block(extension_context: str) -> str:
    if not extension_context:
        return ""
    return (
        "\n\n## Hypothesis Extension Context\n"
        "Generate deeper follow-up hypotheses from the prior hypothesis "
        "and experiment evidence below. Do not treat this as a blank-slate "
        "pivot.\n\n"
        f"{extension_context}"
    )


def _read_user_context(run_dir: Path, stage_dir: Path) -> tuple[str, Path | None]:
    current = stage_dir / USER_CONTEXT_FILENAME
    text = _read_stripped(current)
    if text:
        return text, current

    candidates = [
        path
        for path in run_dir.glob(f"stage-08_v*/{USER_CONTEXT_FILENAME}")
        if _stage8_version(path.parent.name) >= 0
    ]
    for candidate in sorted(
        candidates,
        key=lambda path: _stage8_version(path.parent.name),
        reverse=True,
    ):
        text = _read_stripped(candidate)
        if text:
            return text, candidate
    return "", None


def _read_extension_context(run_dir: Path) -> tuple[str, Path | None]:
    path = run_dir / "hypothesis_extension_context.md"
    text = _read_stripped(path)
    if text:
        return text, path
    return "", None


def _read_stripped(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _stage8_version(dirname: str) -> int:
    match = re.fullmatch(r"stage-08_v(\d+)", dirname)
    if not match:
        return -1
    return int(match.group(1))


def _relative_to_run(run_dir: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _write_audit(stage_dir: Path, ctx: HypothesisContext) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    snapshot = ctx.research_context
    if ctx.extension_block:
        snapshot = f"{snapshot}{ctx.extension_block}"
    (stage_dir / "context_snapshot.md").write_text(snapshot, encoding="utf-8")

    manifest = {
        "built_at": _utcnow_iso(),
        "sources": [asdict(source) for source in ctx.sources],
    }
    (stage_dir / "context_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
