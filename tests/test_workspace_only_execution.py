from __future__ import annotations

import importlib.util
import inspect


def test_legacy_execution_backend_modules_are_removed() -> None:
    removed_modules = [
        "researchclaw.experiment.agentic_sandbox",
        "researchclaw.experiment.biology_agent_sandbox",
        "researchclaw.experiment.code_agent",
        "researchclaw.experiment.colab_sandbox",
        "researchclaw.experiment.collider_agent_sandbox",
        "researchclaw.experiment.docker_sandbox",
        "researchclaw.experiment.factory",
        "researchclaw.experiment.runner",
        "researchclaw.experiment.sandbox",
        "researchclaw.experiment.ssh_sandbox",
        "researchclaw.experiment.stat_agent_sandbox",
        "researchclaw.pipeline.code_agent",
        "researchclaw.pipeline.opencode_bridge",
    ]

    assert [
        name for name in removed_modules if importlib.util.find_spec(name) is not None
    ] == []


def test_workspace_stage_execution_has_no_legacy_backend_references() -> None:
    from researchclaw.pipeline.stage_impls import _execution

    source = inspect.getsource(_execution)

    forbidden = [
        "create_sandbox",
        "ExperimentRunner",
        "ColliderAgentSandbox",
        "agentic_sandbox",
        "biology_agent",
        "stat_agent",
        "docker_sandbox",
        "colab_sandbox",
        "ssh_sandbox",
    ]
    assert [token for token in forbidden if token in source] == []
