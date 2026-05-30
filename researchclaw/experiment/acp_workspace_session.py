"""ACP persistent session runner for workspace-native code agents."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from researchclaw.llm.acp_client import _find_acpx
from researchclaw.llm.acp_retry import (
    TransientAcpDisconnect,
    is_transient_text,
    run_acp_with_retry,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcpWorkspaceSessionConfig:
    """Configuration for a persistent ACP session in a user workspace."""

    agent: str = "claude"
    cwd: Path = Path(".")
    acpx_command: str = ""
    session_name: str = "researchclaw-code"
    timeout_sec: int = 1800
    max_turns: int = 50
    base_url: str = ""
    api_key_env: str = ""
    model: str = ""
    max_retries: int = 3


class AcpWorkspaceSession:
    """Manage a named acpx session whose cwd is the user workspace repo."""

    _MAX_CLI_PROMPT_BYTES = 20_000 if sys.platform == "win32" else 100_000
    _MAX_CMD_WRAPPER_PROMPT_BYTES = 6_000 if sys.platform == "win32" else 100_000
    _CMD_TOO_LONG_HINTS = (
        "too long",
        "trop long",
        "zu lang",
        "demasiado larg",
        "e2big",
    )

    def __init__(
        self,
        *,
        agent: str = "claude",
        cwd: str | Path = ".",
        acpx_command: str = "",
        session_name: str = "researchclaw-code",
        timeout_sec: int = 1800,
        max_turns: int = 50,
        base_url: str = "",
        api_key_env: str = "",
        model: str = "",
        max_retries: int = 3,
    ) -> None:
        self.config = AcpWorkspaceSessionConfig(
            agent=agent,
            cwd=Path(cwd),
            acpx_command=acpx_command,
            session_name=session_name,
            timeout_sec=timeout_sec,
            max_turns=max_turns,
            base_url=base_url,
            api_key_env=api_key_env,
            model=model,
            max_retries=max_retries,
        )
        self._acpx: str | None = acpx_command or None
        self._session_ready = False
        # Backoff sleep for transient-disconnect retries; injectable in tests.
        self._retry_sleep = time.sleep

    @property
    def session_name(self) -> str:
        return self.config.session_name

    @property
    def agent(self) -> str:
        return self.config.agent

    @property
    def cwd(self) -> Path:
        return self.config.cwd

    def ensure_session(self) -> None:
        """Find or create the named acpx session without text-only warmup."""
        if self._session_ready:
            return
        acpx = self._resolve_acpx()
        if not acpx:
            raise RuntimeError("acpx not found")
        result = subprocess.run(
            [
                *self._acpx_base_args(acpx),
                "sessions",
                "ensure",
                "--name",
                self.config.session_name,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=self._build_env(),
        )
        if result.returncode != 0:
            result = subprocess.run(
                [
                    *self._acpx_base_args(acpx),
                    "sessions",
                    "new",
                    "--name",
                    self.config.session_name,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=self._build_env(),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to create ACP workspace session: {result.stderr.strip()}"
                )
        self._session_ready = True

    def run_task(self, prompt: str) -> str:
        """Send *prompt* to the persistent code session and return raw stdout.

        Transient ACP stream disconnects (including the case where acpx exits 0
        but leaves a disconnect banner in stdout) are retried up to
        ``config.max_retries`` times. Each retry closes and re-ensures the SAME
        named session so the agent resumes its own partial workspace state, and
        the resent prompt is prefixed with a continuation instruction telling the
        agent to inspect existing uncommitted/committed changes and continue
        rather than start over — so a resend does not duplicate work. If a retry
        leaves more than one new commit on top of the pre-task HEAD, a warning is
        logged so stacked commits are visible. Genuine failures (non-zero exit
        without a transient signature) are not retried.
        """
        prompt = prompt.replace("\x00", "")
        acpx = self._resolve_acpx()
        if not acpx:
            raise RuntimeError("acpx not found")

        base_head = self._git_head()
        attempt = {"n": 0}

        _CONTINUATION_PREFACE = (
            "NOTE: a previous attempt at this task was interrupted by a transient "
            "connection drop and has been resumed. FIRST inspect the current "
            "workspace state (`git status`, `git log`, and any uncommitted "
            "changes) and CONTINUE from there. Do NOT restart from scratch and do "
            "NOT create duplicate commits for work that is already committed.\n\n"
        )

        def _do_call() -> str:
            send_prompt = prompt if attempt["n"] == 0 else _CONTINUATION_PREFACE + prompt
            attempt["n"] += 1
            self.ensure_session()
            prompt_bytes = len(send_prompt.encode("utf-8"))
            use_stdin = prompt_bytes > self._cli_prompt_limit(acpx) or (
                sys.platform == "win32" and "\n" in send_prompt
            )
            try:
                if use_stdin:
                    result = self._send_prompt_via_stdin(acpx, send_prompt)
                else:
                    result = self._send_prompt_cli(acpx, send_prompt)
            except RuntimeError as exc:
                exc_lower = str(exc).lower()
                if use_stdin or not any(
                    hint in exc_lower for hint in self._CMD_TOO_LONG_HINTS
                ):
                    raise
                result = self._send_prompt_via_stdin(acpx, send_prompt)
            if result.returncode != 0:
                detail = (result.stderr or "").strip()
                # A non-zero exit that carries a transient-disconnect signature
                # is retryable; anything else is a genuine failure.
                if is_transient_text(result.stdout) or is_transient_text(result.stderr):
                    raise TransientAcpDisconnect(
                        f"ACP workspace stream disconnected (exit {result.returncode}): "
                        f"{detail[:200]}"
                    )
                raise RuntimeError(
                    f"ACP workspace task failed (exit {result.returncode}): {detail}"
                )
            # acpx can exit 0 after exhausting its own internal reconnects,
            # leaving a disconnect banner in stdout — treat as retryable.
            if is_transient_text(result.stdout) or is_transient_text(result.stderr):
                raise TransientAcpDisconnect(
                    "ACP workspace stream disconnected (exit 0): "
                    f"{(result.stdout or '').strip()[:200]}"
                )
            return result.stdout or ""

        def _reset() -> None:
            self.close()
            self._session_ready = False
            self.ensure_session()

        output = run_acp_with_retry(
            _do_call,
            reset=_reset,
            max_retries=self.config.max_retries,
            sleep=self._retry_sleep,
        )
        if attempt["n"] > 1:
            self._warn_if_stacked_commits(base_head)
        return output

    def _git_head(self) -> str | None:
        """Return the current git HEAD sha of the workspace, or None."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.config.cwd), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:  # noqa: BLE001
            return None
        if result.returncode != 0:
            return None
        return (result.stdout or "").strip() or None

    def _warn_if_stacked_commits(self, base_head: str | None) -> None:
        """Log a warning if more than one new commit landed since *base_head*.

        After a transient-disconnect retry the agent resumes the same session; if
        it created a fresh commit on a retry in addition to one from the
        interrupted attempt, the ``base..HEAD`` range will contain >1 commit.
        This surfaces that for review (it does not modify history).
        """
        if not base_head:
            return
        try:
            result = subprocess.run(
                ["git", "-C", str(self.config.cwd),
                 "rev-list", "--count", f"{base_head}..HEAD"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:  # noqa: BLE001
            return
        if result.returncode != 0:
            return
        try:
            count = int((result.stdout or "0").strip())
        except ValueError:
            return
        if count > 1:
            logger.warning(
                "ACP workspace task added %d commits after a transient-disconnect "
                "retry (base %s); review for duplicated/stacked commits.",
                count, base_head[:12],
            )

    def export_session(self, output_path: Path) -> None:
        """Export the current named ACP session to *output_path*."""
        acpx = self._resolve_acpx()
        if not acpx:
            raise RuntimeError("acpx not found")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _run_export() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [
                    *self._acpx_base_args(acpx),
                    "sessions",
                    "export",
                    self.config.session_name,
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                env=self._build_env(),
            )

        result = _run_export()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0 and "currently locked" in stderr.lower():
            self.close_session(self.config.session_name)
            self._session_ready = False
            result = _run_export()
            stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to export ACP workspace session "
                f"'{self.config.session_name}': {stderr}"
            )

    def fork_from_archive(self, archive_path: Path, fork_name: str) -> AcpWorkspaceSession:
        """Import *archive_path* as *fork_name* and return a session for it."""
        acpx = self._resolve_acpx()
        if not acpx:
            raise RuntimeError("acpx not found")
        result = subprocess.run(
            [
                *self._acpx_base_args(acpx),
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
            env=self._build_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to import ACP workspace session '{fork_name}': "
                f"{(result.stderr or '').strip()}"
            )
        fork_config = replace(self.config, session_name=fork_name)
        return self.__class__(
            agent=fork_config.agent,
            cwd=fork_config.cwd,
            acpx_command=self._resolve_acpx() or fork_config.acpx_command,
            session_name=fork_config.session_name,
            timeout_sec=fork_config.timeout_sec,
            max_turns=fork_config.max_turns,
            base_url=fork_config.base_url,
            api_key_env=fork_config.api_key_env,
            model=fork_config.model,
            max_retries=fork_config.max_retries,
        )

    def close_session(self, name: str) -> None:
        """Close a named ACP workspace session, ignoring cleanup failures."""
        acpx = self._resolve_acpx()
        if not acpx:
            return
        try:
            subprocess.run(
                [*self._acpx_base_args(acpx), "sessions", "close", name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                env=self._build_env(),
            )
        except Exception:  # noqa: BLE001
            logger.debug("ACP workspace session close failed", exc_info=True)

    def close(self) -> None:
        """Close this session if it was opened by this process."""
        if not self._session_ready:
            return
        self.close_session(self.config.session_name)
        self._session_ready = False

    def _resolve_acpx(self) -> str | None:
        if self._acpx:
            return self._acpx
        self._acpx = _find_acpx()
        return self._acpx

    def _abs_cwd(self) -> str:
        return str(self.config.cwd.resolve())

    def _acpx_base_args(self, acpx: str) -> list[str]:
        return [acpx, "--ttl", "0", "--cwd", self._abs_cwd(), self.config.agent]

    def _provider_env_names(self) -> tuple[str, str]:
        agent_name = os.path.basename(self.config.agent).lower()
        if "codex" in agent_name:
            return "OPENAI_BASE_URL", "OPENAI_API_KEY"
        if "claude" in agent_name:
            return "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"
        return "", ""

    def _codex_acp_config(self) -> str:
        provider_name = "custom-gateway"
        config: dict[str, Any] = {
            "model_provider": provider_name,
            "model_providers": {
                provider_name: {
                    "name": "Custom Gateway",
                    "base_url": self.config.base_url,
                    "env_key": self.config.api_key_env,
                    "requires_openai_auth": False,
                    "wire_api": "responses",
                }
            },
        }
        if self.config.model:
            config["model"] = self.config.model
        return json.dumps(config, separators=(",", ":"))

    def _build_env(self) -> dict[str, str]:
        env = {**os.environ}
        base_url_env, api_key_env = self._provider_env_names()
        if self.config.base_url and base_url_env:
            env[base_url_env] = self.config.base_url
        if self.config.api_key_env and api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if api_key:
                env[api_key_env] = api_key
        if (
            "codex" in os.path.basename(self.config.agent).lower()
            and self.config.base_url
            and self.config.api_key_env
        ):
            env["MODEL_PROVIDER"] = "custom-gateway"
            env["CODEX_CONFIG"] = self._codex_acp_config()
        return env

    def _cli_prompt_limit(self, acpx: str | None) -> int:
        limit = self._MAX_CLI_PROMPT_BYTES
        if sys.platform == "win32" and acpx:
            lower = acpx.lower()
            if lower.endswith((".cmd", ".bat")):
                return min(limit, self._MAX_CMD_WRAPPER_PROMPT_BYTES)
        return limit

    def _send_prompt_cli(
        self,
        acpx: str,
        prompt: str,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            acpx,
            "--approve-all",
            "--max-turns",
            str(self.config.max_turns),
            "--ttl",
            "0",
            "--cwd",
            self._abs_cwd(),
            self.config.agent,
            "-s",
            self.config.session_name,
            prompt,
        ]
        try:
            return self._run_acp_with_heartbeat(cmd, label="ACP workspace task")
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ACP workspace task timed out after {self.config.timeout_sec}s"
            ) from exc

    def _send_prompt_via_stdin(
        self,
        acpx: str,
        prompt: str,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            acpx,
            "--approve-all",
            "--max-turns",
            str(self.config.max_turns),
            "--ttl",
            "0",
            "--cwd",
            self._abs_cwd(),
            self.config.agent,
            "-s",
            self.config.session_name,
            "-f",
            "-",
        ]
        try:
            return self._run_acp_with_heartbeat(
                cmd,
                label="ACP workspace task",
                input_data=prompt,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ACP workspace task timed out after {self.config.timeout_sec}s"
            ) from exc

    def _run_acp_with_heartbeat(
        self,
        cmd: list[str],
        *,
        label: str = "ACP workspace task",
        input_data: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run acpx with heartbeat logging while the code agent works."""
        timeout = self.config.timeout_sec
        heartbeat_interval = 30
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_data else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            env=self._build_env(),
        )
        if input_data and proc.stdin:
            try:
                proc.stdin.write(input_data)
                proc.stdin.close()
            except OSError:
                pass

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def _reader(stream: Any, buf: list[str]) -> None:
            try:
                for line in stream:
                    buf.append(line)
            except Exception:  # noqa: BLE001
                pass

        t_out = threading.Thread(target=_reader, args=(proc.stdout, stdout_chunks), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, stderr_chunks), daemon=True)
        t_out.start()
        t_err.start()

        start = time.monotonic()
        while True:
            try:
                proc.wait(timeout=heartbeat_interval)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    proc.kill()
                    t_out.join(timeout=5)
                    t_err.join(timeout=5)
                    raise subprocess.TimeoutExpired(
                        cmd,
                        timeout,
                        output="".join(stdout_chunks),
                        stderr="".join(stderr_chunks),
                    )
                logger.info(
                    "%s still running... %.0fs elapsed (timeout: %ds)",
                    label,
                    elapsed,
                    timeout,
                )

        t_out.join(timeout=5)
        t_err.join(timeout=5)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode or 0,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )
