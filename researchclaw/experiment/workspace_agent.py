"""Workspace-native factory and wrappers for ACP code agents."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Protocol, runtime_checkable

from researchclaw.experiment.workspace import (
    LaunchCommand,
    RunManifest,
    WorkspaceAgentResult,
)


@runtime_checkable
class WorkspaceAgentProvider(Protocol):
    """Protocol for agents that operate directly in an existing workspace."""

    name: str

    def generate_in_workspace(
        self,
        workspace_path: Path,
        prompt: str,
        workdir: Path | None = None,
        timeout_sec: int = 600,
    ) -> WorkspaceAgentResult:
        """Modify a git workspace and leave an agent run manifest."""
        ...


class GitWorkspaceAgent:
    """Thin wrapper around a workspace-native agent provider."""

    def __init__(
        self,
        inner: WorkspaceAgentProvider,
        workspace_path: Path,
        *,
        manifest_filename: str = "run_manifest.json",
    ) -> None:
        if not hasattr(inner, "generate_in_workspace"):
            raise TypeError("Workspace agent inner must define generate_in_workspace")
        self.inner = inner
        self.workspace_path = workspace_path
        self.manifest_filename = manifest_filename

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def session_name(self) -> str:
        return getattr(self.inner, "session_name", "")

    def generate_in_workspace(
        self,
        workspace_path: Path,
        prompt: str,
        workdir: Path | None = None,
        timeout_sec: int = 600,
    ) -> WorkspaceAgentResult:
        workspace = workspace_path.resolve()
        return self.inner.generate_in_workspace(
            workspace,
            prompt,
            workdir or workspace,
            timeout_sec,
        )


class SmokeWorkspaceAgent:
    """Deterministic workspace agent for opt-in E2E smoke validation."""

    name = "smoke"
    session_name = "researchclaw-smoke-code"

    def __init__(self, *, manifest_filename: str = "run_manifest.json") -> None:
        self.manifest_filename = manifest_filename

    def generate_in_workspace(
        self,
        workspace_path: Path,
        prompt: str,
        workdir: Path | None = None,
        timeout_sec: int = 600,
    ) -> WorkspaceAgentResult:
        _ = workdir, timeout_sec
        workspace = Path(workspace_path).resolve()
        start = time.monotonic()
        base_sha = _git(workspace, "rev-parse", "HEAD")
        outputs = _expected_outputs_from_prompt(prompt)
        command = _smoke_launch_command(workspace, outputs)
        marker = workspace / ".researchclaw_smoke_agent.txt"
        marker.write_text(
            f"Smoke workspace agent touched this repo at {time.time():.6f}\n",
            encoding="utf-8",
        )
        code_paths = [marker.name]
        fallback_script = workspace / "scripts" / "researchclaw_smoke_outputs.py"
        if fallback_script.is_file():
            code_paths.append(fallback_script.relative_to(workspace).as_posix())
        subprocess.run(
            ["git", "add", *code_paths],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = subprocess.run(
            ["git", "commit", "-m", "researchclaw smoke workspace code"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode != 0:
            error = commit.stderr.strip() or commit.stdout.strip() or "git commit failed"
            manifest_path = workspace / self.manifest_filename
            return WorkspaceAgentResult(
                base_sha=base_sha,
                agent_commit_sha=None,
                manifest_path=self.manifest_filename if manifest_path.is_file() else None,
                diff_stat=_git(workspace, "diff", "--stat"),
                raw_log=error,
                provider_name=self.name,
                elapsed_sec=time.monotonic() - start,
                error=error,
            )
        code_commit = _git(workspace, "rev-parse", "HEAD")
        manifest_path = workspace / self.manifest_filename
        manifest = RunManifest(
            code_commit=code_commit,
            launch=LaunchCommand(
                command=command,
                cwd=str(workspace),
                env={"PYTHONPATH": f"{workspace}:${{PYTHONPATH:-}}"},
            ),
            result_paths=outputs,
        )
        manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", self.manifest_filename],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = subprocess.run(
            ["git", "commit", "-m", "researchclaw smoke workspace manifest"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode != 0:
            error = commit.stderr.strip() or commit.stdout.strip() or "git commit failed"
            return WorkspaceAgentResult(
                base_sha=base_sha,
                agent_commit_sha=None,
                manifest_path=self.manifest_filename if manifest_path.is_file() else None,
                diff_stat=_git(workspace, "diff", "--stat", base_sha, "HEAD"),
                raw_log=error,
                provider_name=self.name,
                elapsed_sec=time.monotonic() - start,
                error=error,
            )
        head_sha = _git(workspace, "rev-parse", "HEAD")
        return WorkspaceAgentResult(
            base_sha=base_sha,
            agent_commit_sha=head_sha,
            manifest_path=self.manifest_filename,
            diff_stat=_git(workspace, "diff", "--stat", base_sha, "HEAD"),
            raw_log="smoke workspace manifest committed",
            provider_name=self.name,
            elapsed_sec=time.monotonic() - start,
            error=None,
        )


def _resolve_workspace_agent_command(config: Any) -> str:
    workspace_cfg = config.experiment.workspace_agent
    if str(getattr(workspace_cfg, "agent", "") or "").strip():
        return str(workspace_cfg.agent).strip()

    acp_cfg = getattr(getattr(config, "llm", None), "acp", None)
    acp_agent = str(getattr(acp_cfg, "agent", "") or "").strip()
    return acp_agent or "claude"


def create_workspace_agent(
    config: Any,
    llm: Any | None = None,
    prompts: Any | None = None,
) -> WorkspaceAgentProvider:
    """Create a configured workspace-native agent."""
    from researchclaw.experiment.acp_workspace_agent import AcpWorkspaceAgent
    from researchclaw.experiment.acp_workspace_session import AcpWorkspaceSession

    workspace_cfg = config.experiment.workspace_agent
    transport = getattr(workspace_cfg, "transport", "acp")
    if transport == "smoke":
        return GitWorkspaceAgent(
            SmokeWorkspaceAgent(
                manifest_filename=workspace_cfg.manifest_filename,
            ),
            Path(workspace_cfg.workspace_path),
            manifest_filename=workspace_cfg.manifest_filename,
        )
    if transport != "acp":
        raise ValueError(f"Unsupported workspace agent transport: {transport}")
    session_name = workspace_cfg.session_name or "researchclaw-code"
    acp_cfg = getattr(getattr(config, "llm", None), "acp", None)
    base_url = getattr(acp_cfg, "base_url", "") or getattr(config.llm, "base_url", "")
    api_key_env = getattr(acp_cfg, "api_key_env", "") or getattr(config.llm, "api_key_env", "")
    acpx_command = workspace_cfg.acpx_command or getattr(acp_cfg, "acpx_command", "")
    session = AcpWorkspaceSession(
        agent=_resolve_workspace_agent_command(config),
        cwd=Path(workspace_cfg.workspace_path),
        acpx_command=acpx_command,
        session_name=session_name,
        timeout_sec=workspace_cfg.timeout_sec,
        max_turns=workspace_cfg.max_turns,
        base_url=base_url,
        api_key_env=api_key_env,
        model=getattr(config.llm, "primary_model", ""),
        max_retries=getattr(workspace_cfg, "max_reruns", 3),
        reconnect_timeout_sec=getattr(workspace_cfg, "reconnect_timeout_sec", 300),
        reconnect_poll_interval_sec=getattr(
            workspace_cfg, "reconnect_poll_interval_sec", 5
        ),
    )
    agent = AcpWorkspaceAgent(
        session,
        manifest_filename=workspace_cfg.manifest_filename,
    )
    return GitWorkspaceAgent(
        agent,
        Path(workspace_cfg.workspace_path),
        manifest_filename=workspace_cfg.manifest_filename,
    )


def _expected_outputs_from_prompt(prompt: str) -> list[str]:
    match = re.search(
        r"EXPECTED OUTPUTS \(expected_outputs\.json\):\s*(\{.*?\})\s*Completion contract",
        prompt,
        flags=re.DOTALL,
    )
    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            payload = {}
        outputs = payload.get("outputs") if isinstance(payload, dict) else None
        if isinstance(outputs, list):
            parsed = [
                str(item).strip()
                for item in outputs
                if isinstance(item, str) and str(item).strip()
            ]
            if parsed:
                return parsed
    return ["outputs/metrics.json", "outputs/run_summary.json"]


def _smoke_launch_command(workspace: Path, outputs: list[str]) -> str:
    label_smoothing_script = workspace / "scripts" / "run_cifar10_label_smoothing_regimes.py"
    if label_smoothing_script.is_file() and {
        "outputs/metrics.json",
        "outputs/run_summary.json",
    }.issubset(set(outputs)):
        return (
            "python scripts/run_cifar10_label_smoothing_regimes.py "
            "--output-dir outputs --train-size 80 --val-size 40 --test-size 40 "
            "--seeds 0 --epsilons 0.0,0.1 --regimes weak --epochs 1 "
            "--batch-size 20 --eval-batch-size 40 --channels 3"
        )

    script = workspace / "scripts" / "researchclaw_smoke_outputs.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "from pathlib import Path\n"
        f"outputs = {outputs!r}\n"
        "for item in outputs:\n"
        "    path = Path(item)\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text(json.dumps({'status': 'smoke', 'path': item}, indent=2), encoding='utf-8')\n",
        encoding="utf-8",
    )
    return "python scripts/researchclaw_smoke_outputs.py"


def _git(workspace: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()
