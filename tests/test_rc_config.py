import json
from pathlib import Path
from typing import cast

import pytest

from researchclaw.config import (
    AcpConfig,
    ExperimentConfig,
    LarkNotifyConfig,
    LarkTargetConfig,
    RCConfig,
    ResultAnalysisAgentConfig,
    SecurityConfig,
    ValidationResult,
    load_config,
    validate_config,
)


def _write_valid_config(tmp_path: Path) -> Path:
    kb_root = tmp_path / "docs" / "kb"
    for name in (
        "questions",
        "literature",
        "experiments",
        "findings",
        "decisions",
        "reviews",
    ):
        (kb_root / name).mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "config.rc.yaml"
    _ = config_path.write_text(
        """
project:
  name: demo
  mode: docs-first
research:
  topic: Test topic
  domains: [ml, agents]
runtime:
  timezone: America/New_York
notifications:
  channel: discord
knowledge_base:
  backend: markdown
  root: docs/kb
openclaw_bridge:
  use_cron: true
  use_message: true
  use_memory: true
  use_sessions_spawn: true
  use_web_fetch: true
  use_browser: false
llm:
  provider: openai-compatible
  base_url: https://example.invalid/v1
  api_key_env: OPENAI_API_KEY
security:
  hitl_required_stages: [5, 9, 20]
experiment:
  mode: workspace
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _valid_config_data() -> dict[str, dict[str, object]]:
    return {
        "project": {"name": "demo", "mode": "docs-first"},
        "research": {"topic": "Test topic", "domains": ["ml", "agents"]},
        "runtime": {"timezone": "America/New_York"},
        "notifications": {"channel": "discord"},
        "knowledge_base": {"backend": "markdown", "root": "docs/kb"},
        "openclaw_bridge": {
            "use_cron": True,
            "use_message": True,
            "use_memory": True,
            "use_sessions_spawn": True,
            "use_web_fetch": True,
            "use_browser": False,
        },
        "llm": {
            "provider": "openai-compatible",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "OPENAI_API_KEY",
            "primary_model": "gpt-4.1",
            "fallback_models": ["gpt-4o-mini", "gpt-4o"],
        },
        "security": {"hitl_required_stages": [5, 9, 20]},
        "experiment": {"mode": "workspace"},
    }


def test_valid_config_data_helper_returns_expected_baseline_shape():
    data = _valid_config_data()
    assert data["project"]["name"] == "demo"
    assert data["knowledge_base"]["root"] == "docs/kb"
    assert data["security"]["hitl_required_stages"] == [5, 9, 20]


def test_validate_config_with_valid_data_returns_ok_true(tmp_path: Path):
    result = validate_config(
        _valid_config_data(), project_root=tmp_path, check_paths=False
    )

    assert isinstance(result, ValidationResult)
    assert result.ok is True
    assert result.errors == ()


def test_validate_config_missing_required_fields_returns_errors(tmp_path: Path):
    data = _valid_config_data()
    data["research"] = {}

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "Missing required field: research.topic" in result.errors


def test_validate_config_rejects_invalid_project_mode(tmp_path: Path):
    data = _valid_config_data()
    data["project"]["mode"] = "invalid-mode"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "Invalid project.mode: invalid-mode" in result.errors


def test_validate_config_rejects_invalid_knowledge_base_backend(tmp_path: Path):
    data = _valid_config_data()
    data["knowledge_base"]["backend"] = "sqlite"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "Invalid knowledge_base.backend: sqlite" in result.errors


def test_validate_config_accepts_llm_wire_api_responses(tmp_path: Path):
    data = _valid_config_data()
    data["llm"]["wire_api"] = "responses"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is True


def test_validate_config_rejects_invalid_llm_wire_api(tmp_path: Path):
    data = _valid_config_data()
    data["llm"]["wire_api"] = "responses_only"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "Invalid llm.wire_api: responses_only" in result.errors


@pytest.mark.parametrize("entry", [0, 24, "5", 9.1])
def test_validate_config_rejects_invalid_hitl_required_stages_entries(
    tmp_path: Path, entry: object
):
    data = _valid_config_data()
    data["security"]["hitl_required_stages"] = [5, entry, 20]

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert f"Invalid security.hitl_required_stages entry: {entry}" in result.errors


def test_validate_config_rejects_non_list_hitl_required_stages(tmp_path: Path):
    data = _valid_config_data()
    data["security"]["hitl_required_stages"] = "5,9,20"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is False
    assert "security.hitl_required_stages must be a list" in result.errors


def test_validate_config_ignores_legacy_metric_fields(tmp_path: Path):
    data = _valid_config_data()
    data["experiment"]["metric_key"] = "accuracy"
    data["experiment"]["metric_direction"] = "upward"

    result = validate_config(data, project_root=tmp_path, check_paths=False)
    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert result.ok is True
    assert result.errors == ()
    assert not hasattr(config.experiment, "metric_key")
    assert not hasattr(config.experiment, "metric_direction")


def test_rcconfig_from_dict_happy_path(tmp_path: Path):
    config = RCConfig.from_dict(
        _valid_config_data(),
        project_root=tmp_path,
        check_paths=False,
    )

    assert isinstance(config, RCConfig)
    assert config.project.name == "demo"
    assert config.research.domains == ("ml", "agents")
    assert config.llm.fallback_models == ("gpt-4o-mini", "gpt-4o")
    assert config.notifications.on_stage_fail is True


def test_rcconfig_from_dict_parses_llm_wire_api(tmp_path: Path):
    data = _valid_config_data()
    data["llm"]["wire_api"] = "responses"

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.llm.wire_api == "responses"


def test_rcconfig_from_dict_parses_experiment_repair_max_cycles(tmp_path: Path):
    data = _valid_config_data()
    data["experiment"]["repair"] = {"enabled": True, "max_cycles": 7}

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.experiment.repair.enabled is True
    assert config.experiment.repair.max_cycles == 7


def test_hypothesis_validation_config_defaults_off() -> None:
    try:
        from researchclaw.config import HypothesisValidationConfig
    except ImportError:
        pytest.fail("HypothesisValidationConfig is not implemented")

    defaults = HypothesisValidationConfig()

    assert defaults.enabled is False
    assert defaults.max_concurrent_branches == 1
    assert defaults.max_attempts_per_node == 1
    assert defaults.workspace_isolation == "shared"


def test_rcconfig_from_dict_parses_hypothesis_validation_config(
    tmp_path: Path,
) -> None:
    data = _valid_config_data()
    data["hypothesis_validation"] = {
        "enabled": True,
        "max_concurrent_branches": 3,
        "max_attempts_per_node": 2,
        "workspace_isolation": "worktree",
    }

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.hypothesis_validation.enabled is True
    assert config.hypothesis_validation.max_concurrent_branches == 3
    assert config.hypothesis_validation.max_attempts_per_node == 2
    assert config.hypothesis_validation.workspace_isolation == "worktree"


def test_result_analysis_agent_config_defaults() -> None:
    defaults = ResultAnalysisAgentConfig()

    assert defaults.session_name == "researchclaw-analysis"
    assert defaults.agent == ""
    assert defaults.acpx_command == ""
    assert defaults.timeout_sec == 1800
    assert defaults.max_turns == 50
    assert defaults.max_postcheck_retries == 1


def test_rcconfig_from_dict_parses_result_analysis_agent_config(tmp_path: Path) -> None:
    data = _valid_config_data()
    data["experiment"]["result_analysis_agent"] = {
        "session_name": "custom-analysis",
        "agent": "codex",
        "acpx_command": "/usr/local/bin/acpx",
        "timeout_sec": 99,
        "max_turns": 8,
        "max_postcheck_retries": 2,
    }

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.experiment.result_analysis_agent.session_name == "custom-analysis"
    assert config.experiment.result_analysis_agent.agent == "codex"
    assert config.experiment.result_analysis_agent.acpx_command == "/usr/local/bin/acpx"
    assert config.experiment.result_analysis_agent.timeout_sec == 99
    assert config.experiment.result_analysis_agent.max_turns == 8
    assert config.experiment.result_analysis_agent.max_postcheck_retries == 2


def test_acp_config_default_base_url_and_api_key_env():
    cfg = AcpConfig()

    assert cfg.base_url == ""
    assert cfg.api_key_env == ""
    assert cfg.debate_max_rounds == 2
    assert cfg.debate_confidence_min == 0.6


def test_acp_config_max_retries_default_is_three():
    assert AcpConfig().max_retries == 3


def test_acp_config_parses_max_retries(tmp_path: Path):
    data = _valid_config_data()
    data["llm"] = {
        "provider": "acp",
        "acp": {"agent": "codex", "max_retries": 5},
    }

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.llm.acp.max_retries == 5


def test_acp_config_max_retries_defaults_when_absent(tmp_path: Path):
    data = _valid_config_data()
    data["llm"] = {"provider": "acp", "acp": {"agent": "codex"}}

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.llm.acp.max_retries == 3


def test_acp_config_enable_debate_defaults_true():
    cfg = AcpConfig()

    assert cfg.enable_debate is True


def test_acp_config_enable_debate_override_false(tmp_path: Path):
    data = _valid_config_data()
    data["llm"] = {
        "provider": "acp",
        "acp": {
            "agent": "codex",
            "enable_debate": False,
        },
    }

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.llm.acp.enable_debate is False


def test_acp_config_enable_debate_defaults_true_when_absent(tmp_path: Path):
    data = _valid_config_data()
    data["llm"] = {
        "provider": "acp",
        "acp": {"agent": "codex"},
    }

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.llm.acp.enable_debate is True


def test_acp_config_roundtrip_custom_provider_fields(tmp_path: Path):
    data = _valid_config_data()
    data["llm"] = {
        "provider": "acp",
        "acp": {
            "agent": "codex",
            "base_url": "https://provider.example.com/v1",
            "api_key_env": "MY_ACP_KEY",
            "debate_max_rounds": 3,
            "debate_confidence_min": 0.75,
        },
    }

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.llm.acp.agent == "codex"
    assert config.llm.acp.base_url == "https://provider.example.com/v1"
    assert config.llm.acp.api_key_env == "MY_ACP_KEY"
    assert config.llm.acp.debate_max_rounds == 3
    assert config.llm.acp.debate_confidence_min == 0.75


def test_acp_client_from_rc_config_uses_llm_primary_model_and_provider_fields():
    from researchclaw.llm.acp_client import ACPClient

    rc_config = RCConfig.from_dict(
        {
            **_valid_config_data(),
            "llm": {
                "provider": "acp",
                "primary_model": "gpt-5.5",
                "acp": {
                    "agent": "codex",
                    "base_url": "https://provider.example.com/v1",
                    "api_key_env": "MY_ACP_KEY",
                    "debate_max_rounds": 4,
                    "debate_confidence_min": 0.8,
                },
            },
        },
        check_paths=False,
    )

    client = ACPClient.from_rc_config(rc_config)

    assert client.config.model == "gpt-5.5"
    assert client.config.base_url == "https://provider.example.com/v1"
    assert client.config.api_key_env == "MY_ACP_KEY"
    assert client.config.debate_max_rounds == 4
    assert client.config.debate_confidence_min == 0.8


def test_rcconfig_from_dict_missing_fields_raises_value_error(tmp_path: Path):
    data = _valid_config_data()
    del data["runtime"]

    with pytest.raises(ValueError, match="Missing required field: runtime.timezone"):
        _ = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def test_rcconfig_load_from_yaml_file(tmp_path: Path):
    config_path = _write_valid_config(tmp_path)
    config = RCConfig.load(config_path, project_root=tmp_path)

    assert isinstance(config, RCConfig)
    assert config.project.name == "demo"
    assert config.knowledge_base.root == "docs/kb"


def test_load_config_wrapper_returns_rcconfig(tmp_path: Path):
    config_path = _write_valid_config(tmp_path)
    config = load_config(config_path, project_root=tmp_path)

    assert isinstance(config, RCConfig)
    assert config.security.hitl_required_stages == (5, 9, 20)


def test_security_config_defaults_match_expected_values():
    defaults = SecurityConfig()

    assert defaults.hitl_required_stages == (5, 9, 20)
    assert defaults.allow_publish_without_approval is False
    assert defaults.redact_sensitive_logs is True


def test_experiment_config_defaults_mode_is_workspace_native():
    defaults = ExperimentConfig()

    assert defaults.mode == "workspace"
    assert not hasattr(defaults, "metric_key")
    assert not hasattr(defaults, "metric_direction")


def test_to_dict_roundtrip_rehydrates_equivalent_rcconfig(tmp_path: Path):
    original = RCConfig.from_dict(
        _valid_config_data(),
        project_root=tmp_path,
        check_paths=False,
    )

    normalized = cast(dict[str, object], json.loads(json.dumps(original.to_dict())))

    rehydrated = RCConfig.from_dict(
        normalized,
        project_root=tmp_path,
        check_paths=False,
    )

    assert rehydrated == original
    assert isinstance(original.to_dict()["security"]["hitl_required_stages"], tuple)


def test_lark_config_defaults_when_notifications_lark_absent(tmp_path: Path):
    config = RCConfig.from_dict(
        _valid_config_data(),
        project_root=tmp_path,
        check_paths=False,
    )

    assert isinstance(config.notifications.lark, LarkNotifyConfig)
    assert config.notifications.lark.enabled is False
    assert config.notifications.lark.targets == ()
    assert config.notifications.lark.command == "lark-cli"
    assert config.notifications.lark.app_id_env == "LARK_APP_ID"
    assert config.notifications.lark.app_secret_env == "LARK_APP_SECRET"


def test_lark_block_present_but_empty_uses_defaults(tmp_path: Path):
    data = _valid_config_data()
    data["notifications"]["lark"] = {}

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.notifications.lark == LarkNotifyConfig()


def test_lark_individual_field_defaults(tmp_path: Path):
    data = _valid_config_data()
    data["notifications"]["lark"] = {"enabled": True}

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.notifications.lark.enabled is True
    assert config.notifications.lark.backend == "cli"
    assert config.notifications.lark.command == "lark-cli"
    assert config.notifications.lark.targets == ()
    assert config.notifications.lark.timeout_sec == 15
    assert config.notifications.lark.dry_run is False


def test_lark_targets_mapping_parsed_to_tuple_of_targetconfig(tmp_path: Path):
    data = _valid_config_data()
    data["notifications"]["lark"] = {
        "targets": {
            "me": {
                "kind": "user",
                "receive_id_type": "open_id",
                "receive_id": "ou_xxx",
            },
            "lab_group": {
                "kind": "chat",
                "receive_id_type": "chat_id",
                "receive_id": "oc_xxx",
            },
        },
    }

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.notifications.lark.targets == (
        LarkTargetConfig(
            name="me",
            kind="user",
            receive_id_type="open_id",
            receive_id="ou_xxx",
        ),
        LarkTargetConfig(
            name="lab_group",
            kind="chat",
            receive_id_type="chat_id",
            receive_id="oc_xxx",
        ),
    )


def test_lark_target_defaults_for_missing_subfields(tmp_path: Path):
    data = _valid_config_data()
    data["notifications"]["lark"] = {
        "targets": {
            "me": {
                "receive_id": "ou_xxx",
            },
        },
    }

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.notifications.lark.targets == (
        LarkTargetConfig(name="me", receive_id="ou_xxx"),
    )


def test_lark_targets_empty_mapping_yields_empty_tuple(tmp_path: Path):
    data = _valid_config_data()
    data["notifications"]["lark"] = {"targets": {}}

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.notifications.lark.targets == ()


def test_lark_config_no_new_required_fields(tmp_path: Path):
    result = validate_config(
        _valid_config_data(),
        project_root=tmp_path,
        check_paths=False,
    )

    assert result.ok is True


def test_lark_full_block_roundtrips_via_from_dict(tmp_path: Path):
    data = _valid_config_data()
    data["notifications"]["lark"] = {
        "enabled": True,
        "backend": "cli",
        "command": "custom-lark-cli",
        "app_id": "cli_app_id",
        "app_secret": "cli_app_secret",
        "app_id_env": "CUSTOM_LARK_APP_ID",
        "app_secret_env": "CUSTOM_LARK_APP_SECRET",
        "timeout_sec": 7,
        "dry_run": True,
        "targets": {
            "app_notice": {
                "kind": "chat",
                "receive_id_type": "chat_id",
                "receive_id": "oc_xxx",
            },
        },
    }

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.notifications.lark == LarkNotifyConfig(
        enabled=True,
        backend="cli",
        command="custom-lark-cli",
        app_id="cli_app_id",
        app_secret="cli_app_secret",
        app_id_env="CUSTOM_LARK_APP_ID",
        app_secret_env="CUSTOM_LARK_APP_SECRET",
        timeout_sec=7,
        dry_run=True,
        targets=(
            LarkTargetConfig(
                name="app_notice",
                kind="chat",
                receive_id_type="chat_id",
                receive_id="oc_xxx",
            ),
        ),
    )


def test_to_dict_roundtrip_with_lark_targets(tmp_path: Path):
    data = _valid_config_data()
    data["notifications"]["lark"] = {
        "enabled": True,
        "targets": {
            "me": {
                "receive_id_type": "open_id",
                "receive_id": "ou_xxx",
            },
            "lab_group": {
                "kind": "chat",
                "receive_id_type": "chat_id",
                "receive_id": "oc_xxx",
            },
        },
    }
    original = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    normalized = cast(dict[str, object], json.loads(json.dumps(original.to_dict())))
    rehydrated = RCConfig.from_dict(
        normalized,
        project_root=tmp_path,
        check_paths=False,
    )

    assert rehydrated == original


def test_existing_notifications_fields_unchanged(tmp_path: Path):
    data = _valid_config_data()
    data["notifications"].update(
        {
            "target": "ops",
            "on_stage_start": True,
            "on_stage_fail": True,
            "on_gate_required": False,
        }
    )

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert config.notifications.channel == "discord"
    assert config.notifications.target == "ops"
    assert config.notifications.on_stage_start is True
    assert config.notifications.on_stage_fail is True
    assert config.notifications.on_gate_required is False


def test_check_paths_false_skips_missing_kb_root_validation(tmp_path: Path):
    data = _valid_config_data()
    data["knowledge_base"]["root"] = "docs/missing-kb"

    result = validate_config(data, project_root=tmp_path, check_paths=False)

    assert result.ok is True
    assert not any(error.startswith("Missing path:") for error in result.errors)


def test_path_validation_missing_kb_root_is_error(tmp_path: Path):
    result = validate_config(
        _valid_config_data(), project_root=tmp_path, check_paths=True
    )

    assert result.ok is False
    assert any(error.startswith("Missing path:") for error in result.errors)


def test_validate_config_missing_kb_subdirs_emits_warnings(tmp_path: Path):
    data = _valid_config_data()
    _ = (tmp_path / "docs" / "kb").mkdir(parents=True)

    result = validate_config(data, project_root=tmp_path, check_paths=True)

    assert result.ok is True
    assert len(result.warnings) == 6
    assert all(
        warning.startswith("Missing recommended kb subdir:")
        for warning in result.warnings
    )


def test_rcconfig_from_dict_uses_default_security_when_missing(tmp_path: Path):
    data = _valid_config_data()
    del data["security"]

    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
    assert config.security.hitl_required_stages == (5, 9, 20)


def test_load_uses_file_parent_as_default_project_root(tmp_path: Path):
    config_path = _write_valid_config(tmp_path)
    config = RCConfig.load(config_path)

    assert config.project.name == "demo"
    assert config.knowledge_base.root == "docs/kb"
