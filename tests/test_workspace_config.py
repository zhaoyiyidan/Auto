from __future__ import annotations

from researchclaw.config import (
    ExperimentConfig,
    RCConfig,
    SubmitterConfig,
    WorkspaceAgentConfig,
    _parse_experiment_config,
    _parse_submitter_config,
    _parse_workspace_agent_config,
)


def test_workspace_agent_config_defaults() -> None:
    cfg = WorkspaceAgentConfig()

    assert cfg.enabled is False
    assert cfg.transport == "acp"
    assert cfg.workspace_path == "."
    assert cfg.session_name == ""
    assert cfg.agent == "claude"
    assert cfg.acpx_command == ""
    assert cfg.manifest_filename == "run_manifest.json"
    assert cfg.timeout_sec == 1800
    assert cfg.max_turns == 50
    assert cfg.close_policy == "keep"
    assert not hasattr(cfg, "git_mode")


def test_workspace_agent_config_from_dict() -> None:
    cfg = _parse_workspace_agent_config(
        {
            "enabled": True,
            "transport": "acp",
            "workspace_path": "/tmp/workspace",
            "session_name": "researchclaw-code-test",
            "agent": "codex",
            "acpx_command": "/usr/local/bin/acpx",
            "manifest_filename": "manifest.json",
            "timeout_sec": 2400,
            "max_turns": 80,
            "close_policy": "close",
        }
    )

    assert cfg.enabled is True
    assert cfg.transport == "acp"
    assert cfg.workspace_path == "/tmp/workspace"
    assert cfg.session_name == "researchclaw-code-test"
    assert cfg.agent == "codex"
    assert cfg.acpx_command == "/usr/local/bin/acpx"
    assert cfg.manifest_filename == "manifest.json"
    assert cfg.timeout_sec == 2400
    assert cfg.max_turns == 80
    assert cfg.close_policy == "close"


def test_submitter_config_defaults() -> None:
    cfg = SubmitterConfig()

    assert cfg.type == "local"
    assert cfg.custom_callable == ""
    assert cfg.ssh_host == ""
    assert cfg.ssh_user == ""
    assert cfg.ssh_port == 22
    assert cfg.ssh_key_path == ""
    assert cfg.wait_for_completion is True
    assert cfg.poll_interval_sec == 15
    assert cfg.wait_timeout_sec == 0


def test_submitter_config_from_dict() -> None:
    cfg = _parse_submitter_config(
        {
            "type": "ssh_slurm",
            "custom_callable": "pkg.module:create",
            "ssh_host": "cluster.example.com",
            "ssh_user": "alice",
            "ssh_port": 2222,
            "ssh_key_path": "~/.ssh/id_rsa",
            "wait_for_completion": False,
            "poll_interval_sec": 5,
            "wait_timeout_sec": 600,
        }
    )

    assert cfg.type == "ssh_slurm"
    assert cfg.custom_callable == "pkg.module:create"
    assert cfg.ssh_host == "cluster.example.com"
    assert cfg.ssh_user == "alice"
    assert cfg.ssh_port == 2222
    assert cfg.ssh_key_path == "~/.ssh/id_rsa"
    assert cfg.wait_for_completion is False
    assert cfg.poll_interval_sec == 5
    assert cfg.wait_timeout_sec == 600


def test_legacy_experiment_config_symbols_are_removed() -> None:
    import researchclaw.config as cfg

    removed = [
        "SandboxConfig",
        "DockerSandboxConfig",
        "AgenticConfig",
        "ColliderAgentConfig",
        "BiologyAgentConfig",
        "StatAgentConfig",
        "SshRemoteConfig",
        "ColabDriveConfig",
        "OpenCodeConfig",
        "CodeAgentConfig",
        "CliAgentConfig",
        "EXPERIMENT_MODES",
        "CLI_AGENT_PROVIDERS",
    ]

    assert [name for name in removed if hasattr(cfg, name)] == []


def test_experiment_config_includes_workspace_agent() -> None:
    cfg = RCConfig(experiment=ExperimentConfig())

    assert isinstance(cfg.experiment.workspace_agent, WorkspaceAgentConfig)


def test_experiment_config_includes_submitter() -> None:
    cfg = RCConfig(experiment=ExperimentConfig())

    assert isinstance(cfg.experiment.submitter, SubmitterConfig)


def test_parse_experiment_config_includes_workspace_agent_and_submitter() -> None:
    cfg = _parse_experiment_config(
        {
            "workspace_agent": {"enabled": True, "workspace_path": "/tmp/workspace"},
            "submitter": {"type": "manual"},
        }
    )

    assert cfg.workspace_agent.enabled is True
    assert cfg.workspace_agent.workspace_path == "/tmp/workspace"
    assert cfg.workspace_agent.transport == "acp"
    assert cfg.submitter.type == "manual"


def test_workspace_agent_config_roundtrip_through_rc_config() -> None:
    payload = {
        "project": {"name": "test", "mode": "docs-first"},
        "research": {"topic": "test"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"root": "knowledge"},
        "llm": {
            "base_url": "https://example.invalid/v1",
            "api_key_env": "TEST_KEY",
        },
        "security": {"hitl_required_stages": [5, 9, 20]},
        "experiment": {
            "workspace_agent": {
                "enabled": True,
                "transport": "acp",
                "workspace_path": "/tmp/repo",
                "session_name": "researchclaw-code-run-1",
                "agent": "codex",
                "acpx_command": "/opt/bin/acpx",
                "manifest_filename": "agent_manifest.json",
                "timeout_sec": 3600,
                "max_turns": 120,
                "close_policy": "close",
            }
        },
    }
    loaded = RCConfig.from_dict(payload, check_paths=False)

    cfg = loaded.experiment.workspace_agent
    assert cfg.enabled is True
    assert cfg.transport == "acp"
    assert cfg.workspace_path == "/tmp/repo"
    assert cfg.session_name == "researchclaw-code-run-1"
    assert cfg.agent == "codex"
    assert cfg.acpx_command == "/opt/bin/acpx"
    assert cfg.manifest_filename == "agent_manifest.json"
    assert cfg.timeout_sec == 3600
    assert cfg.max_turns == 120
    assert cfg.close_policy == "close"
    assert "git_mode" not in loaded.to_dict()["experiment"]["workspace_agent"]
