"""Optional per-hypothesis validation E2E tests.

Run smoke validation explicitly with:

    RUN_SMOKE_PER_HYPOTHESIS_E2E=1 pytest tests/e2e_per_hypothesis_real.py -q

Run live ACP validation explicitly with:

    RUN_REAL_PER_HYPOTHESIS_E2E=1 pytest tests/e2e_per_hypothesis_real.py -q

Both tests require a configured LLM/provider and a local config file. The smoke
test uses deterministic workspace/analysis transports for repeatability. The
live test exercises the configured ACP workspace code agent and Stage 14 agent.
This file is not collected by the default ``test_*.py`` pattern.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest


def test_smoke_per_hypothesis_run(tmp_path: Path) -> None:
    if os.environ.get("RUN_SMOKE_PER_HYPOTHESIS_E2E") != "1":
        pytest.skip(
            "smoke per-hypothesis E2E requires RUN_SMOKE_PER_HYPOTHESIS_E2E=1"
        )
    _run_per_hypothesis_e2e(
        tmp_path,
        workspace_transport="smoke",
        analysis_agent="smoke",
    )


def test_real_per_hypothesis_run(tmp_path: Path) -> None:
    if os.environ.get("RUN_REAL_PER_HYPOTHESIS_E2E") != "1":
        pytest.skip(
            "real per-hypothesis E2E requires RUN_REAL_PER_HYPOTHESIS_E2E=1"
        )
    _run_per_hypothesis_e2e(
        tmp_path,
        workspace_transport=os.environ.get(
            "RESEARCHCLAW_E2E_WORKSPACE_TRANSPORT",
            "acp",
        ),
        analysis_agent=os.environ.get("RESEARCHCLAW_E2E_ANALYSIS_AGENT", ""),
    )


def _run_per_hypothesis_e2e(
    tmp_path: Path,
    *,
    workspace_transport: str,
    analysis_agent: str,
) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.config import RCConfig
    from researchclaw.pipeline.hypothesis_coordinator import (
        HypothesisValidationCoordinator,
    )
    from researchclaw.pipeline.stage15_verdict import read_stage15_verdict

    config_path = Path(os.environ.get("RESEARCHCLAW_E2E_CONFIG", "config_test.yaml"))
    if not config_path.exists():
        pytest.skip(f"config not found: {config_path}")
    topic = os.environ.get(
        "RESEARCHCLAW_E2E_TOPIC",
        (
            "Compare label smoothing vs. standard cross-entropy on a small CNN "
            "for CIFAR-10-style image classification"
        ),
    )
    config = RCConfig.load(config_path, check_paths=False)
    config = replace(
        config,
        research=replace(
            config.research,
            topic=topic,
            manual_search=False,
        ),
        notifications=replace(
            config.notifications,
            lark=replace(
                config.notifications.lark,
                enabled=False,
                targets=(),
            ),
        ),
        knowledge_base=replace(
            config.knowledge_base,
            root=str(tmp_path / "kb"),
        ),
        hypothesis_validation=replace(
            config.hypothesis_validation,
            enabled=True,
            max_concurrent_branches=2,
            max_total_attempts=2,
        ),
        experiment=replace(
            config.experiment,
            workspace_agent=replace(
                config.experiment.workspace_agent,
                transport=workspace_transport,
                timeout_sec=int(os.environ.get("RESEARCHCLAW_E2E_AGENT_TIMEOUT", "90")),
                reconnect_timeout_sec=0,
                max_reruns=0,
            ),
            result_analysis_agent=replace(
                config.experiment.result_analysis_agent,
                agent=analysis_agent,
                timeout_sec=int(
                    os.environ.get("RESEARCHCLAW_E2E_ANALYSIS_TIMEOUT", "90")
                ),
                max_postcheck_retries=0,
            ),
        ),
    )
    run_dir = tmp_path / "real-per-hypothesis"
    run_dir.mkdir(parents=True, exist_ok=True)
    hypotheses_md = _e2e_hypotheses_md()
    stage8 = run_dir / "stage-08"
    stage8.mkdir(parents=True, exist_ok=True)
    (stage8 / "hypotheses.md").write_text(hypotheses_md, encoding="utf-8")

    coordinator = HypothesisValidationCoordinator(run_dir)
    coordinator.split_and_queue(hypotheses_md)
    coordinator.run_until_queue_empty(
        config=config,
        adapters=AdapterBundle(),
        max_concurrent=2,
        max_attempts_per_node=1,
    )

    aggregate_path = run_dir / "hypothesis_aggregate.json"
    assert aggregate_path.is_file()
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert len(aggregate["validation_summary"]) == 2
    for row in aggregate["validation_summary"][:2]:
        branch_run_dir = Path(row["branch_run_dir"])
        assert branch_run_dir.is_dir()
        verdict_path = branch_run_dir / "stage-15" / "verdict.json"
        assert verdict_path.is_file()
        verdict = read_stage15_verdict(verdict_path)
        assert verdict["decision"] in {"proceed", "extend", "pivot", "inconclusive"}
        assert (branch_run_dir / "attempt_result.json").is_file()
    assert not (run_dir / "current_node.txt").exists()
    assert not (run_dir / "pending_transition.json").exists()
    events_path = run_dir / "hypothesis_tree" / "events.jsonl"
    if events_path.exists():
        assert "pivot_rollback" not in events_path.read_text(encoding="utf-8")


def _e2e_hypotheses_md() -> str:
    return """
## H1
Statement: Mild label smoothing improves calibration on the existing synthetic CIFAR-10-style CNN without reducing validation accuracy by more than 1 percentage point.
Prediction: Epsilon 0.05 or 0.10 lowers expected calibration error versus standard cross-entropy while keeping validation accuracy within 1 percentage point.
Falsification: Reject if all smoothed settings either increase ECE or reduce validation accuracy by more than 1 percentage point.
Rationale: The smoke workspace already contains a label-smoothing experiment, so this branch can validate the per-hypothesis machinery without asking the code agent to invent a new project.
Baselines: standard cross-entropy, epsilon 0.05, epsilon 0.10

## H2
Statement: Strong label smoothing behaves like a capacity tax under weak regularization on the existing synthetic CIFAR-10-style CNN.
Prediction: Epsilon 0.20 reduces validation accuracy relative to mild smoothing or standard cross-entropy while not providing a commensurate NLL or calibration gain.
Falsification: Reject if epsilon 0.20 matches the best milder setting within 0.5 percentage points and improves NLL or ECE.
Rationale: The repository README describes strong-smoothing checks, so this branch is aligned with the existing workspace artifacts.
Baselines: standard cross-entropy, epsilon 0.10, epsilon 0.20
""".strip() + "\n"
