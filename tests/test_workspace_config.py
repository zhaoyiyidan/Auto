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
    assert cfg.workspace_path == "."
    assert cfg.git_mode == "auto"
    assert cfg.manifest_filename == "run_manifest.json"
    assert cfg.timeout_sec == 1800


def test_workspace_agent_config_from_dict() -> None:
    cfg = _parse_workspace_agent_config(
        {
            "enabled": True,
            "workspace_path": "/tmp/workspace",
            "git_mode": "strict",
            "manifest_filename": "manifest.json",
            "timeout_sec": 2400,
        }
    )

    assert cfg.enabled is True
    assert cfg.workspace_path == "/tmp/workspace"
    assert cfg.git_mode == "strict"
    assert cfg.manifest_filename == "manifest.json"
    assert cfg.timeout_sec == 2400


def test_submitter_config_defaults() -> None:
    cfg = SubmitterConfig()

    assert cfg.type == "local"
    assert cfg.custom_callable == ""
    assert cfg.ssh_host == ""
    assert cfg.ssh_user == ""
    assert cfg.ssh_port == 22
    assert cfg.ssh_key_path == ""


def test_submitter_config_from_dict() -> None:
    cfg = _parse_submitter_config(
        {
            "type": "ssh_slurm",
            "custom_callable": "pkg.module:create",
            "ssh_host": "cluster.example.com",
            "ssh_user": "alice",
            "ssh_port": 2222,
            "ssh_key_path": "~/.ssh/id_rsa",
        }
    )

    assert cfg.type == "ssh_slurm"
    assert cfg.custom_callable == "pkg.module:create"
    assert cfg.ssh_host == "cluster.example.com"
    assert cfg.ssh_user == "alice"
    assert cfg.ssh_port == 2222
    assert cfg.ssh_key_path == "~/.ssh/id_rsa"


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
    assert cfg.submitter.type == "manual"
