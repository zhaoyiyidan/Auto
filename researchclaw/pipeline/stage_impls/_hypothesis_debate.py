"""ACP-forked multi-agent debate for Stage 8 hypothesis generation."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.llm.acp_client import ACPClient, ACPConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import _read_prior_artifact
from researchclaw.prompts import PromptManager, _render

logger = logging.getLogger(__name__)


def run_acp_debate(
    run_dir: Path,
    stage_dir: Path,
    config: RCConfig,
    llm: LLMClient,
    prompts: PromptManager,
) -> str:
    """Run ACP-forked claim/judge/revise debate and return clean hypotheses."""
    if not isinstance(llm, ACPClient):
        raise RuntimeError("ACP debate requires ACPClient")

    synthesis = _read_prior_artifact(run_dir, "synthesis.md") or ""
    # EXTEND context is a ONE-TIME round-1 seed, not part of synthesis. Keeping
    # it separate means it is injected into the round-1 claim's
    # {extension_context} slot only; later rounds rely on prior_claim + judge
    # criticisms, and the judge / final synthesizer see the clean synthesis.
    extension_context = _read_extension_context(run_dir)
    roles = prompts.debate_roles_hypothesis()
    if not roles:
        raise RuntimeError("No hypothesis debate roles configured")

    acp_config = getattr(config.llm, "acp", None)
    max_rounds = max(1, int(getattr(acp_config, "debate_max_rounds", 2)))
    confidence_min = float(getattr(acp_config, "debate_confidence_min", 0.6))

    archive_dir = stage_dir / "fork_debate"
    archive_path = _export_ancestor(llm, archive_dir)

    try:
        accepted_claims = _run_role_debates_parallel(
            roles=roles,
            archive_path=archive_path,
            base_config=llm.config,
            synthesis=synthesis,
            extension_context=extension_context,
            topic=config.research.topic,
            max_rounds=max_rounds,
            confidence_min=confidence_min,
        )

        if not accepted_claims:
            raise RuntimeError("All ACP debate claim agents failed")

        final_fork_name = "debate-final-synth"
        final_client = _open_fork_or_fresh(
            archive_path, final_fork_name, llm.config
        )
        try:
            hypotheses_md = _run_final_synthesizer(
                final_client, accepted_claims, synthesis, config.research.topic
            )
        finally:
            _close_fork(final_client, final_fork_name)

        hypotheses_md = _strip_debate_metadata(
            hypotheses_md,
            role_names=accepted_claims.keys(),
        ).strip()
        if not hypotheses_md:
            raise RuntimeError("ACP final synthesizer returned empty hypotheses")

        _commit_summary_to_main(llm, hypotheses_md)
        return hypotheses_md
    finally:
        _cleanup_archives(archive_dir)


def _run_role_debates_parallel(
    *,
    roles: dict[str, dict[str, str]],
    archive_path: Path,
    base_config: ACPConfig,
    synthesis: str,
    extension_context: str,
    topic: str,
    max_rounds: int,
    confidence_min: float,
) -> dict[str, str]:
    """Run each perspective independently while preserving final role order."""
    role_items = list(roles.items())
    accepted_by_role: dict[str, str] = {}

    with ThreadPoolExecutor(
        max_workers=max(1, len(role_items)),
        thread_name_prefix="stage8-debate",
    ) as executor:
        future_to_role = {
            executor.submit(
                _run_role_debate,
                role_name=role_name,
                role_prompts=role_prompts,
                archive_path=archive_path,
                base_config=base_config,
                synthesis=synthesis,
                extension_context=extension_context,
                topic=topic,
                max_rounds=max_rounds,
                confidence_min=confidence_min,
            ): role_name
            for role_name, role_prompts in role_items
        }

        for future in as_completed(future_to_role):
            role_name = future_to_role[future]
            try:
                claim = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("ACP debate role '%s' failed: %s", role_name, exc)
                continue
            if claim:
                accepted_by_role[role_name] = claim

    return {
        role_name: accepted_by_role[role_name]
        for role_name, _role_prompts in role_items
        if role_name in accepted_by_role
    }


def _run_role_debate(
    *,
    role_name: str,
    role_prompts: dict[str, str],
    archive_path: Path,
    base_config: ACPConfig,
    synthesis: str,
    extension_context: str,
    topic: str,
    max_rounds: int,
    confidence_min: float,
) -> str | None:
    prior_claim: str | None = None
    judge_criticisms: list[str] | None = None

    for round_index in range(1, max_rounds + 1):
        claim_fork_name = _fork_name(role_name, "claim", round_index)
        try:
            claim_client = _open_fork_or_fresh(
                archive_path, claim_fork_name, base_config
            )
            try:
                candidate_claim = _run_claim_agent(
                    claim_client,
                    role_name,
                    role_prompts,
                    synthesis,
                    extension_context,
                    topic,
                    prior_claim,
                    judge_criticisms,
                )
            finally:
                _close_fork(claim_client, claim_fork_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ACP claim agent failed for role '%s' round %d: %s",
                role_name,
                round_index,
                exc,
            )
            return None

        if not candidate_claim.strip():
            logger.warning(
                "ACP claim agent returned empty output for role '%s' round %d",
                role_name,
                round_index,
            )
            return None

        judge_fork_name = _fork_name(role_name, "judge", round_index)
        try:
            judge_client = _open_fork_or_fresh(
                archive_path, judge_fork_name, base_config
            )
            try:
                verdict = _run_judge_agent(
                    judge_client,
                    candidate_claim,
                    synthesis,
                    topic,
                )
            finally:
                _close_fork(judge_client, judge_fork_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ACP judge agent failed for role '%s' round %d; accepting claim: %s",
                role_name,
                round_index,
                exc,
            )
            return candidate_claim

        if (
            verdict.get("verdict") == "pass"
            and float(verdict.get("confidence", 0.0)) >= confidence_min
        ):
            return candidate_claim

        prior_claim = candidate_claim
        judge_criticisms = _verdict_criticisms(verdict)
        if round_index == max_rounds:
            return candidate_claim

    return prior_claim


def _read_extension_context(run_dir: Path) -> str:
    """Read the EXTEND rollback context, returning '' when absent."""
    path = run_dir / "hypothesis_extension_context.md"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _export_ancestor(acp_client: ACPClient, archive_dir: Path) -> Path:
    """Export the main ACP session to an archive and return its path."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "ancestor.archive"
    if archive_path.exists():
        archive_path.unlink()
    acp_client.export_session(archive_path)
    return archive_path


def _open_fork_or_fresh(
    archive_path: Path,
    fork_name: str,
    base_config: ACPConfig,
) -> ACPClient:
    """Open an ACP fork when supported, otherwise a fresh isolated session.

    Current acpx export/import archives keep the provider session id. That
    makes them portable restores, not true forks, and import refuses a second
    local record with the same provider id. In that case we still preserve
    agent isolation by using a fresh named ACP session and relying on explicit
    Stage 8 prompt context.
    """
    try:
        return ACPClient.fork_from_archive(archive_path, fork_name, base_config)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "provider session id" not in msg and "session-provider-exists" not in msg:
            raise
        logger.warning(
            "ACP archive import is a restore, not a fork; using fresh isolated "
            "session '%s' with explicit context.",
            fork_name,
        )
        fork_config = _build_fork_config(base_config, fork_name)
        client = ACPClient(fork_config)
        client.close_session(fork_name)
        return client


def _import_fork(
    agent: str,
    cwd: str,
    archive_path: Path,
    fork_name: str,
    acpx_binary: str,
) -> None:
    """Import an ACP archive as a named fork session."""
    result = subprocess.run(
        [
            acpx_binary,
            "--ttl",
            "0",
            "--cwd",
            str(Path(cwd).resolve()),
            agent,
            "sessions",
            "import",
            str(archive_path),
            "--name",
            fork_name,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"Failed to import ACP fork '{fork_name}': {stderr}"
        )


def _build_fork_config(acp_config: Any, fork_name: str) -> ACPConfig:
    """Build an ACPConfig for a fork session from either config dataclass."""
    return ACPConfig(
        agent=getattr(acp_config, "agent", "claude"),
        cwd=getattr(acp_config, "cwd", "."),
        acpx_command=getattr(acp_config, "acpx_command", ""),
        session_name=fork_name,
        timeout_sec=int(getattr(acp_config, "timeout_sec", 1800)),
        base_url=getattr(acp_config, "base_url", ""),
        api_key_env=getattr(acp_config, "api_key_env", ""),
        model=getattr(acp_config, "model", ""),
        debate_max_rounds=int(getattr(acp_config, "debate_max_rounds", 2)),
        debate_confidence_min=float(
            getattr(acp_config, "debate_confidence_min", 0.6)
        ),
    )


def _run_claim_agent(
    fork_client: ACPClient,
    role_name: str,
    role_prompts: dict[str, str],
    synthesis: str,
    extension_context: str,
    topic: str,
    prior_claim: str | None,
    judge_criticisms: list[str] | None,
) -> str:
    """Generate a candidate claim in an isolated ACP fork."""
    messages, system = _build_claim_prompt(
        role_name,
        role_prompts,
        topic,
        synthesis,
        extension_context,
        prior_claim,
        judge_criticisms,
    )
    resp = fork_client.chat(
        messages,
        system=system,
        max_tokens=4096,
        strip_thinking=True,
    )
    return resp.content.strip()


def _run_judge_agent(
    fork_client: ACPClient,
    candidate_claim: str,
    synthesis: str,
    topic: str,
) -> dict[str, Any]:
    """Judge a candidate claim in an isolated ACP fork."""
    messages, system = _build_judge_prompt(candidate_claim, synthesis, topic)
    resp = fork_client.chat(
        messages,
        system=system,
        max_tokens=2048,
        json_mode=True,
        strip_thinking=True,
    )
    return _parse_judge_verdict(resp.content)


def _run_final_synthesizer(
    fork_client: ACPClient,
    accepted_claims: dict[str, str],
    synthesis: str,
    topic: str,
) -> str:
    """Synthesize accepted claims into final Stage 8 hypotheses."""
    messages, system = _build_synthesizer_prompt(accepted_claims, synthesis, topic)
    resp = fork_client.chat(
        messages,
        system=system,
        max_tokens=4096,
        strip_thinking=True,
    )
    return resp.content


def _build_claim_prompt(
    role_name: str,
    role_prompts: dict[str, str],
    topic: str,
    synthesis: str,
    extension_context: str,
    prior_claim: str | None,
    judge_criticisms: list[str] | None,
) -> tuple[list[dict[str, str]], str]:
    """Construct the claim-generation prompt for a role fork.

    The EXTEND context is a one-time round-1 seed: it is rendered into the
    template's ``{extension_context}`` slot only when there is no prior claim
    (i.e. the first round). Revision rounds drop it and instead carry the prior
    candidate + judge criticisms, so the two "prior" signals never overlap.
    """
    extension_block = ""
    if prior_claim is None and extension_context:
        extension_block = (
            "## Hypothesis Extension Context\n"
            "Generate deeper follow-up hypotheses from this prior hypothesis "
            "and experiment evidence. Do not treat this as a blank-slate "
            "pivot.\n\n"
            f"{extension_context}\n"
        )
    variables = {
        "topic": topic,
        "synthesis": synthesis,
        "extension_context": extension_block,
    }
    system = _render(role_prompts.get("system", ""), variables)
    user = _render(role_prompts.get("user", ""), variables)

    if prior_claim is None:
        content = (
            f"{user}\n\n"
            "Generate the strongest candidate hypothesis set for this perspective. "
            "Make every hypothesis falsifiable, measurable, and tied to the synthesis."
        )
    else:
        criticisms = "\n".join(
            f"- {item}" for item in (judge_criticisms or ["No specific criticism."])
        )
        content = (
            f"{user}\n\n"
            "You are revising a prior candidate claim. You do not inherit the prior "
            "claim agent's hidden session; use only the explicit text below.\n\n"
            "## Prior Candidate Claim\n"
            f"{prior_claim}\n\n"
            "## Judge Criticisms\n"
            f"{criticisms}\n\n"
            "Produce a stronger revised candidate. Do not discuss the revision "
            "process, debate rounds, or the judge."
        )

    return [{"role": "user", "content": content}], system


def _build_judge_prompt(
    candidate_claim: str,
    synthesis: str,
    topic: str,
) -> tuple[list[dict[str, str]], str]:
    """Construct a strict judge prompt that returns machine-readable JSON."""
    pm = PromptManager()
    prompt = pm.sub_prompt(
        "hypothesis_judge",
        topic=topic,
        synthesis=synthesis,
        candidate_claim=candidate_claim,
    )
    return [{"role": "user", "content": prompt.user}], prompt.system


def _build_synthesizer_prompt(
    accepted_claims: dict[str, str],
    synthesis: str,
    topic: str,
) -> tuple[list[dict[str, str]], str]:
    """Construct final synthesis prompt with explicit metadata exclusion."""
    claim_blocks = []
    for idx, claim in enumerate(accepted_claims.values(), start=1):
        claim_blocks.append(f"## Candidate Set {idx}\n{claim}")
    claims_text = "\n\n---\n\n".join(claim_blocks)
    pm = PromptManager()
    prompt = pm.sub_prompt(
        "hypothesis_synthesizer",
        topic=topic,
        synthesis=synthesis,
        claims_text=claims_text,
    )
    return [{"role": "user", "content": prompt.user}], prompt.system


def _parse_judge_verdict(raw_text: str) -> dict[str, Any]:
    """Parse a judge verdict with JSON-first logic and heuristic fallback."""
    text = (raw_text or "").strip()
    if not text:
        return _default_judge_verdict(parse_error=True)

    payload = _extract_json_object(text)
    if payload:
        try:
            parsed = json.loads(payload)
            return _normalize_judge_verdict(parsed, parse_error=False)
        except json.JSONDecodeError:
            logger.debug("Judge JSON parsing failed; using heuristic fallback")

    lower = text.lower()
    pass_hit = bool(re.search(r"\b(pass|approve|approved|accept|accepted)\b", lower))
    fail_hit = bool(re.search(r"\b(fail|reject|rejected|fatal|flaw|problem)\b", lower))
    verdict = "pass" if pass_hit and not fail_hit else "fail"
    criticisms = _extract_criticisms_from_text(text)
    confidence = _extract_confidence(text)
    return {
        "verdict": verdict,
        "criticisms": criticisms,
        "fatal_flaws": criticisms if verdict == "fail" else [],
        "confidence": confidence,
        "parse_error": True,
    }


def _close_fork(client: ACPClient, fork_name: str) -> None:
    """Close a fork session, ignoring cleanup errors."""
    try:
        client.close_session(fork_name)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to close ACP fork '%s'", fork_name, exc_info=True)


def _fork_name(role_name: str, kind: str, round_index: int) -> str:
    role_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", role_name).strip("-").lower()
    return f"debate-{role_slug}-{kind}-r{round_index}"


def _verdict_criticisms(verdict: dict[str, Any]) -> list[str]:
    criticisms = verdict.get("criticisms") or []
    fatal = verdict.get("fatal_flaws") or []
    result = [str(item) for item in criticisms if str(item).strip()]
    result.extend(str(item) for item in fatal if str(item).strip())
    if not result:
        result.append("The judge rejected the claim without specific criticism.")
    return result


def _extract_json_object(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return None


def _normalize_judge_verdict(
    parsed: Any,
    *,
    parse_error: bool,
) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return _default_judge_verdict(parse_error=True)

    verdict_raw = str(parsed.get("verdict", "fail")).strip().lower()
    verdict = "pass" if verdict_raw in {"pass", "approve", "approved", "accept"} else "fail"
    confidence = parsed.get("confidence", 0.0)
    try:
        confidence_float = float(confidence)
    except (TypeError, ValueError):
        confidence_float = 0.0
    confidence_float = max(0.0, min(1.0, confidence_float))

    return {
        "verdict": verdict,
        "criticisms": _as_string_list(parsed.get("criticisms")),
        "fatal_flaws": _as_string_list(parsed.get("fatal_flaws")),
        "confidence": confidence_float,
        "parse_error": parse_error,
    }


def _default_judge_verdict(*, parse_error: bool) -> dict[str, Any]:
    return {
        "verdict": "fail",
        "criticisms": ["Judge output was empty or unparseable."],
        "fatal_flaws": [],
        "confidence": 0.0,
        "parse_error": parse_error,
    }


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _extract_criticisms_from_text(text: str) -> list[str]:
    criticisms: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "*")):
            criticisms.append(stripped.lstrip("-* ").strip())
    if not criticisms:
        criticisms.append("Judge output was not valid JSON.")
    return criticisms


def _extract_confidence(text: str) -> float:
    match = re.search(r"\b(?:confidence|score)\D+([01](?:\.\d+)?)", text, re.I)
    if not match:
        return 0.0
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return 0.0


def _strip_debate_metadata(text: str, role_names: Any) -> str:
    metadata_labels = (
        "agent",
        "candidate set",
        "criticism",
        "criticisms",
        "critique",
        "debate",
        "fork",
        "judge",
        "perspective",
        "round",
        "session",
        "verdict",
    )
    label_re = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:"
        + "|".join(re.escape(label) for label in metadata_labels)
        + r")\b\s*(?::|\d+\b)",
        re.I,
    )
    role_terms = [re.escape(str(name)) for name in role_names]
    provenance_re = None
    if role_terms:
        provenance_re = re.compile(
            r"\b(?:from|by|as)\s+(?:the\s+)?(?:"
            + "|".join(role_terms)
            + r")\s+(?:agent|perspective|role)\b",
            re.I,
        )
    cleaned: list[str] = []
    for line in text.splitlines():
        if label_re.search(line):
            continue
        if provenance_re and provenance_re.search(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _commit_summary_to_main(llm: ACPClient, hypotheses_md: str) -> None:
    titles = re.findall(r"^#{1,3}\s+(.+)$", hypotheses_md, flags=re.MULTILINE)
    title_text = "; ".join(titles[:4]) if titles else "untitled hypotheses"
    summary = f"Stage 8 complete. Generated {max(1, len(titles))} hypotheses: {title_text}."
    try:
        llm.chat(
            [{"role": "user", "content": summary}],
            max_tokens=128,
            strip_thinking=True,
        )
    except Exception:  # noqa: BLE001
        logger.debug("ACP main-session Stage 8 summary commit failed", exc_info=True)


def _cleanup_archives(archive_dir: Path) -> None:
    try:
        for path in archive_dir.glob("*.archive"):
            path.unlink()
    except Exception:  # noqa: BLE001
        logger.debug("Failed to clean ACP debate archives", exc_info=True)
