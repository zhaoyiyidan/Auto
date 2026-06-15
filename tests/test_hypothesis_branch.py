from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
from typing import Any

import pytest


def _hypothesis_node_cls() -> Any:
    try:
        from researchclaw.pipeline.hypothesis_store import HypothesisNode
    except ImportError:
        pytest.fail("HypothesisNode is not implemented")
    return HypothesisNode


def _write_stage14_candidate(
    run_dir: Path,
    dirname: str,
    score: float,
    analysis: str,
) -> None:
    stage_dir = run_dir / dirname
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "experiment_summary.json").write_text(
        json.dumps(
            {
                "metrics_summary": {
                    "primary_metric": {
                        "min": score,
                        "max": score,
                        "mean": score,
                        "count": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (stage_dir / "analysis.md").write_text(analysis, encoding="utf-8")


def _run_git(cwd: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("git is not available")
    return completed.stdout.strip()


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _run_git(path, "init", "-q")
    _run_git(path, "config", "user.email", "tests@example.com")
    _run_git(path, "config", "user.name", "Tests")
    (path / "README.md").write_text("base workspace\n", encoding="utf-8")
    _run_git(path, "add", "README.md")
    _run_git(path, "commit", "-q", "-m", "initial")


def _workspace_config(tmp_path: Path) -> Any:
    from researchclaw.config import RCConfig

    return RCConfig.from_dict(
        {
            "project": {"name": "branch-workspace-test", "mode": "docs-first"},
            "research": {"topic": "branch workspace"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
            },
            "experiment": {
                "workspace_agent": {
                    "enabled": True,
                    "workspace_path": "/base/workspace",
                    "session_name": "base-session",
                    "agent": "codex",
                }
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def test_branch_config_clears_hitl_required_stages(tmp_path: Path) -> None:
    from researchclaw.pipeline.hypothesis_branch import branch_config

    config = _workspace_config(tmp_path)
    branch = branch_config(config)

    assert config.security.hitl_required_stages
    assert branch.security.hitl_required_stages == ()


def test_seed_branch_dir_links_shared_context_and_writes_single_hypothesis(
    tmp_path: Path,
) -> None:
    try:
        from researchclaw.experiment.protocol import parse_hypotheses_md
        from researchclaw.pipeline._helpers import _read_prior_artifact
        from researchclaw.pipeline.hypothesis_branch import seed_branch_dir
    except ImportError:
        pytest.fail("seed_branch_dir dependencies are not implemented")

    run_dir = tmp_path / "run"
    branch_run_dir = run_dir / "hypothesis_branches" / "h-001" / "attempt-001"
    for stage_number in range(1, 8):
        stage_dir = run_dir / f"stage-{stage_number:02d}"
        stage_dir.mkdir(parents=True)
        (stage_dir / f"context_{stage_number}.txt").write_text(
            f"context {stage_number}",
            encoding="utf-8",
        )
    (run_dir / "stage-03" / "hardware_profile.json").write_text(
        '{"gpu": "A100"}',
        encoding="utf-8",
    )
    node = _hypothesis_node_cls()(
        id="h-001",
        statement="Treatment improves accuracy.",
        prediction="Accuracy increases by at least 5 points.",
        falsification="Accuracy does not improve.",
        rationale="Prior runs show useful signal.",
        baselines=("baseline-a",),
        source="stage8_batch",
        parent_id=None,
        created_at="2026-01-01T00:00:00+00:00",
    )

    seed_branch_dir(branch_run_dir, run_dir, node)

    for stage_number in range(1, 8):
        link = branch_run_dir / f"stage-{stage_number:02d}"
        assert link.is_symlink()
        assert os.readlink(link) == f"../../../stage-{stage_number:02d}"
    assert (
        _read_prior_artifact(branch_run_dir, "hardware_profile.json")
        == '{"gpu": "A100"}'
    )

    branch_stage8 = branch_run_dir / "stage-08"
    assert branch_stage8.is_dir()
    assert not branch_stage8.is_symlink()
    hypotheses_md = (branch_stage8 / "hypotheses.md").read_text(encoding="utf-8")
    parsed = parse_hypotheses_md(hypotheses_md)
    assert len(parsed) == 1
    assert parsed[0].statement == "Treatment improves accuracy."
    assert parsed[0].prediction == "Accuracy increases by at least 5 points."
    assert parsed[0].falsification == "Accuracy does not improve."


def test_validate_branch_runs_stage9_to_stage15_and_writes_attempt_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from researchclaw.pipeline import hypothesis_branch
        from researchclaw.pipeline.executor import StageResult
        from researchclaw.pipeline.hypothesis_branch import validate_branch
        from researchclaw.pipeline.hypothesis_store import ValidationAttempt
        from researchclaw.pipeline.stages import Stage, StageStatus
    except ImportError:
        pytest.fail("validate_branch dependencies are not implemented")

    branch_run_dir = (
        tmp_path / "run" / "hypothesis_branches" / "h-001" / "attempt-001"
    )
    branch_run_dir.mkdir(parents=True)
    node = _hypothesis_node_cls()(
        id="h-001",
        statement="Treatment improves accuracy.",
    )
    attempt = ValidationAttempt(
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        branch_run_dir=str(branch_run_dir),
    )
    recorded: dict[str, Any] = {}

    def fake_execute_pipeline(**kwargs: Any) -> list[Any]:
        recorded.update(kwargs)
        return [
            StageResult(
                stage=Stage.EXPERIMENT_TASK_SPEC,
                status=StageStatus.DONE,
                artifacts=("plan.md", "expected_outputs.json"),
            ),
            StageResult(
                stage=Stage.RESEARCH_DECISION,
                status=StageStatus.DONE,
                artifacts=("decision.md",),
                decision="pivot",
            ),
        ]

    monkeypatch.setattr(
        hypothesis_branch,
        "execute_pipeline",
        fake_execute_pipeline,
        raising=False,
    )

    result = validate_branch(
        branch_run_dir=branch_run_dir,
        node=node,
        attempt=attempt,
        config=object(),
        adapters=object(),
    )

    assert recorded["run_dir"] == branch_run_dir
    assert recorded["from_stage"] is Stage.EXPERIMENT_TASK_SPEC
    assert recorded["to_stage"] is Stage.RESEARCH_DECISION
    assert recorded["config"] is not None
    assert recorded["adapters"] is not None
    assert recorded["auto_approve_gates"] is True
    assert recorded["initialize_run_globals"] is False
    assert result["decision"] == "pivot"
    assert result["artifacts"] == ["decision.md"]

    payload = json.loads(
        (branch_run_dir / "attempt_result.json").read_text(encoding="utf-8")
    )
    assert payload == {
        "attempt_id": "h-001/attempt-001",
        "node_id": "h-001",
        "status": "succeeded",
        "decision": "pivot",
        "artifacts": ["decision.md"],
        "error": None,
    }


def test_validate_branch_resumes_from_branch_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline import hypothesis_branch
    from researchclaw.pipeline.branch_checkpoint import write_branch_stage_done
    from researchclaw.pipeline.executor import StageResult
    from researchclaw.pipeline.hypothesis_branch import validate_branch
    from researchclaw.pipeline.hypothesis_store import HypothesisNode, ValidationAttempt
    from researchclaw.pipeline.stages import Stage, StageStatus

    branch_run_dir = (
        tmp_path / "run" / "hypothesis_branches" / "h-001" / "attempt-001"
    )
    workspace_path = tmp_path / "worktrees" / "h-001-attempt-001"
    attempt = ValidationAttempt(
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        branch_run_dir=str(branch_run_dir),
        workspace_path=str(workspace_path),
    )
    write_branch_stage_done(
        branch_run_dir,
        Stage.MANIFEST_VALIDATE_AND_PREPARE,
        attempt_id=attempt.attempt_id,
        node_id=attempt.node_id,
        workspace_path=workspace_path,
    )
    recorded: dict[str, Any] = {}

    def fake_execute_pipeline(**kwargs: Any) -> list[Any]:
        recorded.update(kwargs)
        return [
            StageResult(
                stage=Stage.RESEARCH_DECISION,
                status=StageStatus.DONE,
                artifacts=("decision.md",),
                decision="proceed",
            )
        ]

    monkeypatch.setattr(hypothesis_branch, "execute_pipeline", fake_execute_pipeline)

    validate_branch(
        branch_run_dir=branch_run_dir,
        node=HypothesisNode(id="h-001", statement="Treatment improves accuracy."),
        attempt=attempt,
        config=_workspace_config(tmp_path),
        adapters=object(),
    )

    assert recorded["from_stage"] is Stage.HARNESS_SUBMIT_AND_COLLECT


def test_validate_branch_finalizes_stage15_done_without_rerunning_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline import hypothesis_branch
    from researchclaw.pipeline.branch_checkpoint import write_branch_stage_done
    from researchclaw.pipeline.hypothesis_branch import validate_branch
    from researchclaw.pipeline.hypothesis_store import HypothesisNode, ValidationAttempt
    from researchclaw.pipeline.stages import Stage

    branch_run_dir = (
        tmp_path / "run" / "hypothesis_branches" / "h-001" / "attempt-001"
    )
    stage15 = branch_run_dir / "stage-15"
    stage15.mkdir(parents=True)
    (stage15 / "verdict.json").write_text(
        json.dumps(
            {
                "decision": "proceed",
                "confidence": 0.91,
                "evidence_summary": "Stage 15 already completed.",
                "key_metrics": {"score": 0.87},
            }
        ),
        encoding="utf-8",
    )
    attempt = ValidationAttempt(
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        branch_run_dir=str(branch_run_dir),
        workspace_path=str(tmp_path / "worktrees" / "h-001-attempt-001"),
    )
    write_branch_stage_done(
        branch_run_dir,
        Stage.RESEARCH_DECISION,
        attempt_id=attempt.attempt_id,
        node_id=attempt.node_id,
        workspace_path=attempt.workspace_path or branch_run_dir,
    )

    def fake_execute_pipeline(**kwargs: Any) -> list[Any]:
        _ = kwargs
        raise AssertionError("completed branch must not rerun execute_pipeline")

    monkeypatch.setattr(hypothesis_branch, "execute_pipeline", fake_execute_pipeline)

    result = validate_branch(
        branch_run_dir=branch_run_dir,
        node=HypothesisNode(id="h-001", statement="Treatment improves accuracy."),
        attempt=attempt,
        config=_workspace_config(tmp_path),
        adapters=object(),
    )

    assert result == {
        "attempt_id": "h-001/attempt-001",
        "node_id": "h-001",
        "status": "succeeded",
        "decision": "proceed",
        "artifacts": ["verdict.json"],
        "error": None,
        "next_hypothesis": None,
        "confidence": 0.91,
        "evidence_summary": "Stage 15 already completed.",
        "metrics": {"score": 0.87},
    }
    assert json.loads(
        (branch_run_dir / "attempt_result.json").read_text(encoding="utf-8")
    ) == result


def test_validate_branch_stage_complete_callback_writes_branch_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline import hypothesis_branch
    from researchclaw.pipeline.branch_checkpoint import read_branch_state
    from researchclaw.pipeline.executor import StageResult
    from researchclaw.pipeline.hypothesis_branch import validate_branch
    from researchclaw.pipeline.hypothesis_store import HypothesisNode, ValidationAttempt
    from researchclaw.pipeline.stages import Stage, StageStatus

    branch_run_dir = (
        tmp_path / "run" / "hypothesis_branches" / "h-001" / "attempt-001"
    )
    branch_run_dir.mkdir(parents=True)
    workspace_path = tmp_path / "worktrees" / "h-001-attempt-001"
    attempt = ValidationAttempt(
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        branch_run_dir=str(branch_run_dir),
        workspace_path=str(workspace_path),
    )

    def fake_execute_pipeline(**kwargs: Any) -> list[Any]:
        on_stage_complete = kwargs["on_stage_complete"]
        on_stage_complete(Stage.EXPERIMENT_TASK_SPEC)
        on_stage_complete(Stage.HARNESS_SUBMIT_AND_COLLECT)
        on_stage_complete(Stage.RESEARCH_DECISION)
        return [
            StageResult(
                stage=Stage.RESEARCH_DECISION,
                status=StageStatus.DONE,
                artifacts=("decision.md",),
                decision="proceed",
            )
        ]

    monkeypatch.setattr(hypothesis_branch, "execute_pipeline", fake_execute_pipeline)

    validate_branch(
        branch_run_dir=branch_run_dir,
        node=HypothesisNode(id="h-001", statement="Treatment improves accuracy."),
        attempt=attempt,
        config=_workspace_config(tmp_path),
        adapters=object(),
    )

    state = read_branch_state(branch_run_dir)

    assert state is not None
    assert state["attempt_id"] == "h-001/attempt-001"
    assert state["node_id"] == "h-001"
    assert state["last_completed_stage"] == int(Stage.RESEARCH_DECISION)
    assert state["stage_status"] == {
        str(int(Stage.EXPERIMENT_TASK_SPEC)): "done",
        str(int(Stage.HARNESS_SUBMIT_AND_COLLECT)): "done",
        str(int(Stage.RESEARCH_DECISION)): "done",
    }
    assert state["workspace_path"] == str(workspace_path)


def test_promote_best_stage14_for_branch_is_scoped_to_attempt(
    tmp_path: Path,
) -> None:
    try:
        from researchclaw.pipeline.hypothesis_branch import (
            promote_best_stage14_for_branch,
        )
    except ImportError:
        pytest.fail("promote_best_stage14_for_branch is not implemented")

    shared_run_dir = tmp_path / "run"
    branch_run_dir = shared_run_dir / "hypothesis_branches" / "h-001" / "attempt-001"
    branch_run_dir.mkdir(parents=True)
    _write_stage14_candidate(
        branch_run_dir,
        "stage-14",
        0.10,
        "# Analysis\nWorse current branch result.",
    )
    _write_stage14_candidate(
        branch_run_dir,
        "stage-14_v1",
        0.91,
        "# Analysis\nBest branch result.",
    )

    promote_best_stage14_for_branch(branch_run_dir, config=object())

    branch_best = json.loads(
        (branch_run_dir / "experiment_summary_best.json").read_text(
            encoding="utf-8"
        )
    )
    branch_current = json.loads(
        (branch_run_dir / "stage-14" / "experiment_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert branch_best["metrics_summary"]["primary_metric"]["mean"] == 0.91
    assert branch_current["metrics_summary"]["primary_metric"]["mean"] == 0.91
    assert (branch_run_dir / "analysis_best.md").read_text(
        encoding="utf-8"
    ) == "# Analysis\nBest branch result."
    assert not (shared_run_dir / "experiment_summary_best.json").exists()
    assert not (shared_run_dir / "analysis_best.md").exists()


def test_coordinator_runs_mocked_branches_with_isolated_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.experiment.protocol import parse_hypotheses_md
    from researchclaw.pipeline import hypothesis_branch
    from researchclaw.pipeline.executor import StageResult
    from researchclaw.pipeline.hypothesis_coordinator import (
        HypothesisValidationCoordinator,
    )
    from researchclaw.pipeline.stages import Stage, StageStatus

    run_dir = tmp_path / "run"
    for stage_number in range(1, 8):
        stage_dir = run_dir / f"stage-{stage_number:02d}"
        stage_dir.mkdir(parents=True)
        (stage_dir / f"context_{stage_number}.txt").write_text(
            f"context {stage_number}",
            encoding="utf-8",
        )
    hypotheses_md = """
## H1: Accuracy hypothesis
Statement: Treatment improves accuracy.
Prediction: Accuracy improves.
Falsification: Accuracy does not improve.
Rationale: Signal one.

## H2: Robustness hypothesis
Statement: Treatment improves robustness.
Prediction: Robustness improves.
Falsification: Robustness does not improve.
Rationale: Signal two.
"""
    executed: list[Path] = []

    def fake_execute_pipeline(**kwargs: Any) -> list[Any]:
        branch_run_dir = Path(kwargs["run_dir"])
        executed.append(branch_run_dir)
        marker = branch_run_dir.parent.name
        stage9 = branch_run_dir / "stage-09"
        stage9.mkdir(parents=True)
        (stage9 / "branch_marker.txt").write_text(marker, encoding="utf-8")
        stage15 = branch_run_dir / "stage-15"
        stage15.mkdir(parents=True)
        (stage15 / "decision.md").write_text(
            "## Decision\nPROCEED",
            encoding="utf-8",
        )
        return [
            StageResult(
                stage=Stage.EXPERIMENT_TASK_SPEC,
                status=StageStatus.DONE,
                artifacts=("branch_marker.txt",),
            ),
            StageResult(
                stage=Stage.RESEARCH_DECISION,
                status=StageStatus.DONE,
                artifacts=("decision.md",),
                decision="proceed",
            ),
        ]

    monkeypatch.setattr(hypothesis_branch, "execute_pipeline", fake_execute_pipeline)

    attempts = HypothesisValidationCoordinator(run_dir).split_and_validate_sequential(
        hypotheses_md,
        config=object(),
        adapters=object(),
        created_at="2026-01-01T00:00:00+00:00",
    )

    expected_branches = [
        run_dir / "hypothesis_branches" / "h-001" / "attempt-001",
        run_dir / "hypothesis_branches" / "h-002" / "attempt-001",
    ]
    assert executed == expected_branches
    assert [attempt.status for attempt in attempts] == ["succeeded", "succeeded"]
    for branch_run_dir, expected_statement in zip(
        expected_branches,
        ["Treatment improves accuracy.", "Treatment improves robustness."],
        strict=True,
    ):
        parsed = parse_hypotheses_md(
            (branch_run_dir / "stage-08" / "hypotheses.md").read_text(
                encoding="utf-8"
            )
        )
        assert len(parsed) == 1
        assert parsed[0].statement == expected_statement
        assert (branch_run_dir / "stage-01").is_symlink()
        assert (branch_run_dir / "stage-09" / "branch_marker.txt").read_text(
            encoding="utf-8"
        ) == branch_run_dir.parent.name
        assert (branch_run_dir / "stage-15" / "decision.md").is_file()
        attempt_result = json.loads(
            (branch_run_dir / "attempt_result.json").read_text(encoding="utf-8")
        )
        assert attempt_result["status"] == "succeeded"
        assert attempt_result["decision"] == "proceed"


def test_provision_workspace_creates_distinct_git_worktrees(
    tmp_path: Path,
) -> None:
    try:
        from researchclaw.pipeline.hypothesis_branch import provision_workspace
        from researchclaw.pipeline.hypothesis_store import ValidationAttempt
    except ImportError:
        pytest.fail("provision_workspace dependencies are not implemented")

    repo = tmp_path / "repo"
    _init_git_repo(repo)
    workspace_root = tmp_path / "worktrees"
    attempt_one = ValidationAttempt(
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        branch_run_dir=str(tmp_path / "run" / "h-001" / "attempt-001"),
    )
    attempt_two = ValidationAttempt(
        attempt_id="h-002/attempt-001",
        node_id="h-002",
        branch_run_dir=str(tmp_path / "run" / "h-002" / "attempt-001"),
    )

    provisioned_one = provision_workspace(
        attempt_one,
        source_workspace=repo,
        workspace_root=workspace_root,
    )
    provisioned_two = provision_workspace(
        attempt_two,
        source_workspace=repo,
        workspace_root=workspace_root,
    )

    path_one = Path(provisioned_one.workspace_path or "")
    path_two = Path(provisioned_two.workspace_path or "")
    assert path_one == workspace_root / "h-001-attempt-001"
    assert path_two == workspace_root / "h-002-attempt-001"
    assert path_one != path_two
    assert attempt_one.workspace_path is None
    assert attempt_two.workspace_path is None
    assert _run_git(path_one, "rev-parse", "--is-inside-work-tree") == "true"
    assert _run_git(path_two, "rev-parse", "--is-inside-work-tree") == "true"
    assert (path_one / "README.md").read_text(encoding="utf-8") == "base workspace\n"
    assert (path_two / "README.md").read_text(encoding="utf-8") == "base workspace\n"


def test_provision_workspace_resolves_relative_workspace_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from researchclaw.pipeline.hypothesis_branch import (
            provision_workspace,
            release_workspace,
        )
        from researchclaw.pipeline.hypothesis_store import ValidationAttempt
    except ImportError:
        pytest.fail("provision_workspace dependencies are not implemented")

    repo = tmp_path / "repo"
    _init_git_repo(repo)
    monkeypatch.chdir(tmp_path)
    attempt = ValidationAttempt(
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        branch_run_dir=str(tmp_path / "branch"),
    )

    provisioned = provision_workspace(
        attempt,
        source_workspace=repo,
        workspace_root=Path("branch-worktrees"),
    )
    workspace_path = Path(provisioned.workspace_path or "")

    assert workspace_path.is_absolute()
    assert workspace_path == tmp_path / "branch-worktrees" / "h-001-attempt-001"
    assert workspace_path.exists()
    assert not (repo / "branch-worktrees").exists()

    release_workspace(
        replace(provisioned, status="succeeded"),
        source_workspace=repo,
    )


def test_release_workspace_removes_terminal_worktree_idempotently(
    tmp_path: Path,
) -> None:
    try:
        from researchclaw.pipeline.hypothesis_branch import (
            provision_workspace,
            release_workspace,
        )
        from researchclaw.pipeline.hypothesis_store import ValidationAttempt
    except ImportError:
        pytest.fail("release_workspace dependencies are not implemented")

    repo = tmp_path / "repo"
    _init_git_repo(repo)
    provisioned = provision_workspace(
        ValidationAttempt(
            attempt_id="h-001/attempt-001",
            node_id="h-001",
            branch_run_dir=str(tmp_path / "run" / "h-001" / "attempt-001"),
        ),
        source_workspace=repo,
        workspace_root=tmp_path / "worktrees",
    )
    workspace_path = Path(provisioned.workspace_path or "")
    running = replace(provisioned, status="running")
    succeeded = replace(provisioned, status="succeeded")

    release_workspace(running, source_workspace=repo)
    assert workspace_path.exists()

    release_workspace(succeeded, source_workspace=repo)
    assert not workspace_path.exists()
    release_workspace(succeeded, source_workspace=repo)
    assert not workspace_path.exists()


def test_validate_branch_threads_attempt_workspace_and_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.pipeline import hypothesis_branch
    from researchclaw.pipeline.executor import StageResult
    from researchclaw.pipeline.hypothesis_branch import validate_branch
    from researchclaw.pipeline.hypothesis_store import HypothesisNode, ValidationAttempt
    from researchclaw.pipeline.stages import Stage, StageStatus

    branch_run_dir = (
        tmp_path / "run" / "hypothesis_branches" / "h-001" / "attempt-001"
    )
    branch_run_dir.mkdir(parents=True)
    attempt = ValidationAttempt(
        attempt_id="h-001/attempt-001",
        node_id="h-001",
        branch_run_dir=str(branch_run_dir),
        workspace_path=str(tmp_path / "worktrees" / "h-001-attempt-001"),
        agent_session_name="base-session-h-001-attempt-001",
    )
    seen: dict[str, str] = {}

    def fake_execute_pipeline(**kwargs: Any) -> list[Any]:
        cfg = kwargs["config"]
        seen["workspace_path"] = cfg.experiment.workspace_agent.workspace_path
        seen["session_name"] = cfg.experiment.workspace_agent.session_name
        return [
            StageResult(
                stage=Stage.RESEARCH_DECISION,
                status=StageStatus.DONE,
                artifacts=("decision.md",),
                decision="proceed",
            )
        ]

    monkeypatch.setattr(hypothesis_branch, "execute_pipeline", fake_execute_pipeline)

    validate_branch(
        branch_run_dir=branch_run_dir,
        node=HypothesisNode(id="h-001", statement="Treatment improves accuracy."),
        attempt=attempt,
        config=_workspace_config(tmp_path),
        adapters=object(),
    )

    assert seen == {
        "workspace_path": str(tmp_path / "worktrees" / "h-001-attempt-001"),
        "session_name": "base-session-h-001-attempt-001",
    }


def test_validate_branch_disables_runner_global_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline import hypothesis_branch
    from researchclaw.pipeline.executor import StageResult
    from researchclaw.pipeline.hypothesis_branch import validate_branch
    from researchclaw.pipeline.hypothesis_store import HypothesisNode, ValidationAttempt
    from researchclaw.pipeline.stages import Stage, StageStatus

    branch_run_dir = (
        tmp_path / "run" / "hypothesis_branches" / "h-001" / "attempt-001"
    )
    branch_run_dir.mkdir(parents=True)
    recorded: dict[str, Any] = {}

    def fake_execute_pipeline(**kwargs: Any) -> list[Any]:
        recorded.update(kwargs)
        return [
            StageResult(
                stage=Stage.RESEARCH_DECISION,
                status=StageStatus.DONE,
                artifacts=("decision.md",),
                decision="proceed",
            )
        ]

    monkeypatch.setattr(hypothesis_branch, "execute_pipeline", fake_execute_pipeline)

    validate_branch(
        branch_run_dir=branch_run_dir,
        node=HypothesisNode(id="h-001", statement="Treatment improves accuracy."),
        attempt=ValidationAttempt(
            attempt_id="h-001/attempt-001",
            node_id="h-001",
            branch_run_dir=str(branch_run_dir),
        ),
        config=_workspace_config(tmp_path),
        adapters=AdapterBundle(hitl=object()),
    )

    assert recorded["initialize_run_globals"] is False
    assert recorded["adapters"].hitl is None
