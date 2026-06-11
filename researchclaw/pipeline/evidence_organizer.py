"""Stage 14 evidence bundle and organizer-agent helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.experiment.acp_workspace_session import AcpWorkspaceSession
from researchclaw.experiment.workspace import RunManifest
from researchclaw.pipeline._helpers import _find_prior_file, _read_prior_artifact
from researchclaw.prompts import PromptManager


@dataclass(frozen=True)
class PostCheckResult:
    ok: bool
    violations: tuple[str, ...] = ()


_DEFAULT_INPUTS: tuple[tuple[str, str], ...] = (
    ("task_spec", "task_spec.yaml"),
    ("run_manifest", "run_manifest.json"),
    ("execution_record", "execution_record.json"),
    ("result_artifacts", "result_artifacts.json"),
    ("contract_evidence", "contract_evidence.json"),
    ("experiment_decision", "experiment_decision.json"),
)

_OPTIONAL_INPUTS: tuple[tuple[str, str], ...] = (
    ("experiment_protocol", "experiment_protocol.json"),
    ("stage_12_local_log", "stage-12-local.log"),
    ("workspace_agent_result", "stage-10-workspace-agent-result.json"),
    ("manifest_validation", "manifest_validation.json"),
)

_FIXED_ANALYSIS_SECTIONS: tuple[str, ...] = (
    "## Experiment Objective",
    "## Experiment Plan",
    "## Executed Experiments",
    "## Results Summary",
    "## Artifact Locations",
    "## Reproducibility",
)


def _resolve_organizer_agent(config: RCConfig) -> str:
    agent_cfg = config.experiment.result_analysis_agent
    if agent_cfg.agent.strip():
        return agent_cfg.agent.strip()

    acp_cfg = getattr(config.llm, "acp", None)
    acp_agent = str(getattr(acp_cfg, "agent", "") or "").strip()
    return acp_agent or "claude"


def build_evidence_bundle(run_dir: Path, config: RCConfig) -> dict[str, Any]:
    """Build the path-only evidence bundle for the Stage 14 organizer agent."""
    run_dir = run_dir.resolve()
    stage_dir = (run_dir / "stage-14").resolve()
    workspace = _workspace_path(config)
    default_inputs = [
        _prior_path_entry(run_dir, label, filename)
        for label, filename in _DEFAULT_INPUTS
    ]
    optional_inputs = [
        _prior_path_entry(run_dir, label, filename)
        for label, filename in _OPTIONAL_INPUTS
    ]
    manifest, manifest_error = _load_manifest(run_dir)
    result_files = _manifest_result_files(workspace, manifest)
    execution_record = _load_json_from_prior(run_dir, "execution_record.json")

    bundle: dict[str, Any] = {
        "run_dir": str(run_dir),
        "stage_dir": str(stage_dir),
        "workspace_path": str(workspace),
        "default_inputs": default_inputs,
        "optional_inputs": optional_inputs,
        "result_files": result_files,
        "reproducibility": _reproducibility_fields(manifest, execution_record),
    }
    if manifest_error:
        bundle["manifest_error"] = manifest_error
    return bundle


def build_organizer_prompt(
    bundle: dict[str, Any],
    retry_violations: tuple[str, ...] | list[str] = (),
) -> str:
    """Create the organizer prompt. The bundle contains paths, not file bodies."""
    retry_block = ""
    if retry_violations:
        retry_block = (
            "\nSTRICT RETRY: the previous analysis.md crossed organizer boundaries. "
            "Rewrite it from scratch and avoid these violations: "
            f"{', '.join(str(v) for v in retry_violations)}.\n"
        )

    bundle_json = json.dumps(bundle, indent=2, sort_keys=True)
    sections = "\n".join(_FIXED_ANALYSIS_SECTIONS)
    pm = PromptManager()
    prompt = pm.sub_prompt(
        "evidence_organizer",
        stage_dir=bundle.get("stage_dir", "stage-14"),
        sections=sections,
        retry_block=retry_block,
        bundle_json=bundle_json,
    )
    return f"{prompt.system}\n\n{prompt.user}" if prompt.system else prompt.user


def create_evidence_organizer_agent(
    config: RCConfig,
    run_dir: Path,
) -> AcpWorkspaceSession | None:
    """Create the independent Stage 14 organizer ACP session."""
    agent_cfg = config.experiment.result_analysis_agent
    acp_cfg = getattr(config.llm, "acp", None)
    base_url = getattr(acp_cfg, "base_url", "") or getattr(config.llm, "base_url", "")
    api_key_env = (
        getattr(acp_cfg, "api_key_env", "") or getattr(config.llm, "api_key_env", "")
    )
    acpx_command = (
        agent_cfg.acpx_command
        or getattr(acp_cfg, "acpx_command", "")
        or ""
    )
    session = AcpWorkspaceSession(
        agent=_resolve_organizer_agent(config),
        cwd=run_dir,
        acpx_command=acpx_command,
        session_name=agent_cfg.session_name,
        timeout_sec=agent_cfg.timeout_sec,
        max_turns=agent_cfg.max_turns,
        base_url=base_url,
        api_key_env=api_key_env,
        model=getattr(config.llm, "primary_model", ""),
    )
    if session._resolve_acpx() is None:
        return None
    return session


def run_evidence_organizer(
    session: Any,
    prompt: str,
    timeout_sec: int,
) -> str:
    """Run the organizer prompt in the existing session primitive."""
    _ = timeout_sec
    return str(session.run_task(prompt))


def postcheck_analysis(text: str, run_dir: Path) -> PostCheckResult:
    """Enforce that Stage 14 output remains organization, not judgment."""
    _ = run_dir
    violations: list[str] = []
    if not text.strip():
        violations.append("empty")
        return PostCheckResult(ok=False, violations=tuple(violations))

    violations.extend(_forbidden_heading_violations(text))
    violations.extend(_standalone_decision_token_violations(text))
    violations.extend(_forbidden_reference_violations(text))
    return PostCheckResult(ok=not violations, violations=tuple(violations))


def _workspace_path(config: RCConfig) -> Path:
    raw = getattr(config.experiment.workspace_agent, "workspace_path", ".") or "."
    return Path(str(raw)).expanduser().resolve()


def _prior_path_entry(run_dir: Path, label: str, filename: str) -> dict[str, Any]:
    path = _find_prior_file(run_dir, filename)
    return {
        "label": label,
        "filename": filename,
        "path": str(path.resolve()) if path is not None else "",
        "exists": bool(path is not None and path.is_file()),
    }


def _load_manifest(run_dir: Path) -> tuple[RunManifest | None, str]:
    raw = _read_prior_artifact(run_dir, "run_manifest.json")
    if raw is None:
        return None, "run_manifest.json not found"
    try:
        return RunManifest.from_json(raw), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"failed to parse run_manifest.json: {exc}"


def _manifest_result_files(
    workspace: Path,
    manifest: RunManifest | None,
) -> list[dict[str, Any]]:
    if manifest is None:
        return []
    entries: list[dict[str, Any]] = []
    for declared_path in manifest.result_paths:
        path = Path(declared_path).expanduser()
        if not path.is_absolute():
            path = workspace / path
        path = path.resolve()
        entries.append(
            {
                "declared_path": declared_path,
                "path": str(path),
                "exists": path.is_file(),
            }
        )
    return entries


def _load_json_from_prior(run_dir: Path, filename: str) -> dict[str, Any]:
    raw = _read_prior_artifact(run_dir, filename)
    if raw is None:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _reproducibility_fields(
    manifest: RunManifest | None,
    execution_record: dict[str, Any],
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "code_commit": "",
        "launch_command": "",
        "launch_cwd": "",
        "launch_env": {},
        "resources": {},
        "result_paths": [],
        "metric_primary": "",
        "metric_direction": "",
        "job_id": execution_record.get("job_id", ""),
        "elapsed_sec": execution_record.get("elapsed_sec", ""),
        "submitter": execution_record.get("submitter", ""),
    }
    if manifest is None:
        return fields
    fields.update(
        {
            "code_commit": manifest.code_commit,
            "launch_command": manifest.launch.command,
            "launch_cwd": manifest.launch.cwd,
            "launch_env": dict(manifest.launch.env),
            "resources": asdict(manifest.launch.resources),
            "result_paths": list(manifest.result_paths),
            "metric_primary": manifest.metrics.primary,
            "metric_direction": manifest.metrics.direction,
        }
    )
    return fields


def _forbidden_heading_violations(text: str) -> list[str]:
    patterns: tuple[tuple[str, str], ...] = (
        ("decision", r"\bdecision\b"),
        ("recommendation", r"\brecommendation(s)?\b"),
        ("next actions", r"\bnext action(s)?\b"),
        ("quality assessment", r"\bquality assessment\b"),
        (
            "missing or ambiguous evidence",
            r"\bmissing\s+(or\s+)?ambiguous\s+evidence\b",
        ),
        ("proceed/pivot/extend", r"\b(proceed|pivot|extend)\b"),
    )
    violations: list[str] = []
    for match in re.finditer(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", text, re.MULTILINE):
        heading = re.sub(r"\s+", " ", match.group(1).strip().lower())
        for label, pattern in patterns:
            if re.search(pattern, heading):
                violations.append(f"forbidden_heading:{label}")
    return violations


def _standalone_decision_token_violations(text: str) -> list[str]:
    violations: list[str] = []
    for line in text.splitlines():
        token = line.strip().strip("*").strip("#").strip().rstrip(".:;,-").lower()
        if token in {"proceed", "pivot", "extend"}:
            violations.append(f"standalone_decision_token:{token}")
    return violations


def _forbidden_reference_violations(text: str) -> list[str]:
    lower = text.lower()
    violations: list[str] = []
    if "stage-15" in lower or "stage_15" in lower:
        violations.append("forbidden_reference:stage-15")
    if re.search(r"\bdecision\.md\b", lower):
        violations.append("forbidden_reference:decision.md")
    return violations
