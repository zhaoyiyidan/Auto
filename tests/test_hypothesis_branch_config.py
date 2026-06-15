from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.executor import StageResult
from researchclaw.pipeline.stages import Stage, StageStatus


def _config(tmp_path: Path) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "branch-config-test", "mode": "docs-first"},
            "research": {"topic": "branch config"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {},
            "llm": {
                "provider": "acp",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "acp": {
                    "session_name": "main-session",
                },
            },
            "experiment": {
                "workspace_agent": {
                    "enabled": True,
                    "workspace_path": "/original/workspace",
                    "session_name": "base-session",
                    "agent": "codex",
                },
                "result_analysis_agent": {
                    "session_name": "analysis-session",
                    "agent": "codex",
                }
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def test_branch_config_replaces_workspace_agent_without_mutating_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline import runner as rc_runner
    from researchclaw.pipeline.hypothesis_branch import branch_config

    config = _config(tmp_path)
    branch_workspace = tmp_path / "workspaces" / "h-001-attempt-001"
    branch = branch_config(
        config,
        workspace_path=branch_workspace,
        session_name="base-session-h-001-attempt-001",
    )
    seen: dict[str, str] = {}

    def fake_execute_stage(stage: Stage, **kwargs: Any) -> StageResult:
        cfg = kwargs["config"]
        seen["workspace_path"] = cfg.experiment.workspace_agent.workspace_path
        seen["session_name"] = cfg.experiment.workspace_agent.session_name
        seen["llm_session_name"] = cfg.llm.acp.session_name
        seen["analysis_session_name"] = (
            cfg.experiment.result_analysis_agent.session_name
        )
        return StageResult(
            stage=stage,
            status=StageStatus.DONE,
            artifacts=("out.md",),
        )

    monkeypatch.setattr(rc_runner, "execute_stage", fake_execute_stage)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    rc_runner.execute_pipeline(
        run_dir=run_dir,
        run_id="branch-config-threading",
        config=branch,
        adapters=AdapterBundle(),
        from_stage=Stage.TOPIC_INIT,
        to_stage=Stage.TOPIC_INIT,
    )

    assert seen == {
        "workspace_path": str(branch_workspace),
        "session_name": "base-session-h-001-attempt-001",
        "llm_session_name": "base-session-h-001-attempt-001",
        "analysis_session_name": "base-session-h-001-attempt-001-analysis",
    }
    assert config.experiment.workspace_agent.workspace_path == "/original/workspace"
    assert config.experiment.workspace_agent.session_name == "base-session"
    assert config.experiment.result_analysis_agent.session_name == "analysis-session"
    assert config.llm.acp.session_name == "main-session"
