"""Pluggable code-generation backend for experiment Stages 10 & 13.

The legacy non-workspace path still uses the LLM chat API and parses code
blocks. Workspace-native code editing is handled separately by the ACP
workspace agent path.

Usage::

    from researchclaw.experiment.code_agent import create_code_agent

    agent = create_code_agent(config, llm=llm_client, prompts=pm)
    result = agent.generate(exp_plan=plan, topic=topic, ...)
    if result.ok:
        files = result.files  # dict[str, str]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from researchclaw.config import RCConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodeAgentResult:
    """Output from a code agent invocation."""

    files: dict[str, str]  # filename -> code content
    provider_name: str
    elapsed_sec: float
    raw_output: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.files)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class CodeAgentProvider(Protocol):
    """Protocol for code generation backends."""

    @property
    def name(self) -> str: ...

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        """Generate experiment code from scratch (Stage 10)."""
        ...

    def refine(
        self,
        *,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        """Refine existing experiment code based on run results (Stage 13)."""
        ...

    def repair(
        self,
        *,
        files: dict[str, str],
        issues: str,
        workdir: Path,
        timeout_sec: int = 300,
    ) -> CodeAgentResult:
        """Fix validation or runtime issues in code."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _collect_py_files(workdir: Path) -> dict[str, str]:
    """Read all .py files from a directory (flat, no subdirs)."""
    files: dict[str, str] = {}
    for pyfile in sorted(workdir.glob("*.py")):
        if pyfile.name.startswith("_codex_") or pyfile.name.startswith("_agent_"):
            continue
        files[pyfile.name] = pyfile.read_text(encoding="utf-8")
    return files


def _seed_workdir(workdir: Path, files: dict[str, str]) -> None:
    """Pre-populate workdir with files for refinement/repair."""
    workdir.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        (workdir / fname).write_text(content, encoding="utf-8")


def format_feedback_for_agent(
    sandbox_result: Any,
    metric_key: str,
    metric_direction: str,
    best_metric: float | None,
) -> str:
    """Format sandbox run results as structured feedback for CLI agents."""
    parts = ["## Previous Run Results"]
    parts.append(f"Return code: {sandbox_result.returncode}")
    parts.append(f"Elapsed: {sandbox_result.elapsed_sec:.1f}s")
    parts.append(f"Timed out: {sandbox_result.timed_out}")
    if sandbox_result.metrics:
        parts.append("Metrics:")
        for k, v in sandbox_result.metrics.items():
            parts.append(f"  {k}: {v}")
    if sandbox_result.stderr:
        parts.append(f"Stderr (last 1000 chars):\n{sandbox_result.stderr[-1000:]}")
    parts.append(f"\nTarget: {metric_direction} '{metric_key}'")
    if best_metric is not None:
        parts.append(f"Best so far: {best_metric}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LlmCodeAgent — wraps existing LLM chat API (backward-compatible)
# ---------------------------------------------------------------------------

class LlmCodeAgent:
    """Code agent backed by the existing OpenAI-compatible LLM chat API.

    This implementation extracts the LLM call + response parsing logic that was
    previously inline in ``_execute_code_generation`` and
    ``_execute_iterative_refine``, preserving exact behavior.
    """

    def __init__(
        self,
        llm: Any,
        prompts: Any,
        config: RCConfig,
    ) -> None:
        self._llm = llm
        self._pm = prompts
        self._config = config

    @property
    def name(self) -> str:
        return "llm"

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        from researchclaw.pipeline.executor import (
            _chat_with_prompt,
            _extract_multi_file_blocks,
        )

        start = time.monotonic()
        sp = self._pm.for_stage(
            "code_generation",
            topic=topic,
            metric=metric_key,
            pkg_hint=pkg_hint + "\n" + compute_budget + "\n" + extra_guidance,
            exp_plan=exp_plan,
        )
        # Higher max_tokens for reasoning models
        _code_max_tokens = sp.max_tokens or 8192
        if any(
            self._config.llm.primary_model.startswith(p)
            for p in ("gpt-5", "o3", "o4")
        ):
            _code_max_tokens = max(_code_max_tokens, 16384)

        try:
            resp = _chat_with_prompt(
                self._llm,
                sp.system,
                sp.user,
                json_mode=sp.json_mode,
                max_tokens=_code_max_tokens,
            )
            files = _extract_multi_file_blocks(resp.content)
            # Retry on empty response with higher token limit
            if not files and not resp.content.strip():
                logger.warning(
                    "LlmCodeAgent: empty response (len=%d, finish=%s). "
                    "Retrying with 32768 tokens.",
                    len(resp.content),
                    resp.finish_reason,
                )
                resp = _chat_with_prompt(
                    self._llm,
                    sp.system,
                    sp.user,
                    json_mode=sp.json_mode,
                    max_tokens=32768,
                )
                files = _extract_multi_file_blocks(resp.content)

            elapsed = time.monotonic() - start
            if not files:
                logger.warning(
                    "LlmCodeAgent: no files extracted (resp len=%d)",
                    len(resp.content),
                )
            return CodeAgentResult(
                files=files,
                provider_name="llm",
                elapsed_sec=elapsed,
                raw_output=resp.content[:2000],
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("LlmCodeAgent.generate failed: %s", exc)
            return CodeAgentResult(
                files={},
                provider_name="llm",
                elapsed_sec=elapsed,
                error=str(exc),
            )

    def refine(
        self,
        *,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        from researchclaw.pipeline.executor import (
            _chat_with_prompt,
            _extract_code_block,
            _extract_multi_file_blocks,
        )

        start = time.monotonic()

        def _files_to_context(project_files: dict[str, str]) -> str:
            parts = []
            for fname, code in sorted(project_files.items()):
                parts.append(f"```filename:{fname}\n{code}\n```")
            return "\n\n".join(parts)

        try:
            ip = self._pm.sub_prompt(
                "iterative_improve",
                metric_key=metric_key,
                metric_direction=metric_direction,
                files_context=_files_to_context(current_files),
                run_summaries=chr(10).join(run_summaries[:20]),
                condition_coverage_hint="",
                topic=topic,
            )
            user_prompt = ip.user + extra_hints

            response = _chat_with_prompt(
                self._llm,
                ip.system,
                user_prompt,
                max_tokens=ip.max_tokens or 8192,
            )
            extracted_files = _extract_multi_file_blocks(response.content)
            if not extracted_files:
                single_code = _extract_code_block(response.content)
                if single_code.strip():
                    extracted_files = {"main.py": single_code}

            elapsed = time.monotonic() - start
            return CodeAgentResult(
                files=extracted_files,
                provider_name="llm",
                elapsed_sec=elapsed,
                raw_output=response.content[:2000],
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("LlmCodeAgent.refine failed: %s", exc)
            return CodeAgentResult(
                files={},
                provider_name="llm",
                elapsed_sec=elapsed,
                error=str(exc),
            )

    def repair(
        self,
        *,
        files: dict[str, str],
        issues: str,
        workdir: Path,
        timeout_sec: int = 300,
    ) -> CodeAgentResult:
        from researchclaw.pipeline.executor import (
            _chat_with_prompt,
            _extract_code_block,
            _extract_multi_file_blocks,
        )

        start = time.monotonic()
        all_files_ctx = "\n\n".join(
            f"```filename:{f}\n{c}\n```" for f, c in files.items()
        )
        try:
            rp = self._pm.sub_prompt(
                "code_repair",
                fname="main.py",
                issues_text=issues,
                all_files_ctx=all_files_ctx,
            )
            resp = _chat_with_prompt(self._llm, rp.system, rp.user)
            # Try multi-file extraction first, then single-block
            repaired = _extract_multi_file_blocks(resp.content)
            if not repaired:
                code = _extract_code_block(resp.content)
                if code.strip():
                    repaired = {"main.py": code}

            elapsed = time.monotonic() - start
            return CodeAgentResult(
                files=repaired,
                provider_name="llm",
                elapsed_sec=elapsed,
                raw_output=resp.content[:2000],
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return CodeAgentResult(
                files={},
                provider_name="llm",
                elapsed_sec=elapsed,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_code_agent(
    config: RCConfig,
    llm: Any | None = None,
    prompts: Any | None = None,
) -> CodeAgentProvider:
    """Create the legacy non-workspace LLM code agent."""
    agent_cfg = config.experiment.cli_agent
    provider = agent_cfg.provider

    if provider == "llm":
        if llm is None:
            raise RuntimeError("LLM code agent requires an LLM client")
        from researchclaw.prompts import PromptManager

        return LlmCodeAgent(llm, prompts or PromptManager(), config)  # type: ignore[arg-type]

    if provider in {"claude_code", "codex"}:
        raise ValueError(
            "One-shot CLI code agents were removed. Use "
            "experiment.workspace_agent.enabled=true with "
            "workspace_agent.transport=acp and workspace_agent.agent="
            f"{provider!r} instead."
        )

    raise ValueError(f"Unknown code agent provider: {provider}")
