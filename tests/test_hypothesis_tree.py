# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Any

import pytest

try:
    ht = importlib.import_module("researchclaw.pipeline.hypothesis_tree")
except ModuleNotFoundError:
    ht = None


@pytest.fixture(autouse=True)
def _require_hypothesis_tree_module() -> None:
    if ht is None:
        pytest.skip("researchclaw.pipeline.hypothesis_tree is not implemented yet")


def _events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "hypothesis_tree" / "events.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tree(run_dir: Path) -> dict[str, Any]:
    return json.loads(
        (run_dir / "hypothesis_tree" / "tree.json").read_text(encoding="utf-8")
    )


def _node_json(run_dir: Path, node_id: str) -> dict[str, Any]:
    return json.loads(
        (run_dir / "hypothesis_tree" / "nodes" / node_id / "node.json").read_text(
            encoding="utf-8"
        )
    )


def _hypothesis_text(number: int = 1) -> str:
    return f"# Hypotheses\nH{number}: hypothesis {number} explores mechanism {number}."


def _create_initial_node(run_dir: Path) -> Any:
    ht.init_tree_if_needed(run_dir)
    return ht.create_node(
        run_dir,
        "root",
        _hypothesis_text(1),
        status="active",
        pivoted_from=None,
        edge_type="initialize",
    )


def _seed_stage8(run_dir: Path, text: str | None = None) -> None:
    stage_dir = run_dir / "stage-08"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "hypotheses.md").write_text(
        text if text is not None else _hypothesis_text(1),
        encoding="utf-8",
    )


class TestHypothesisTreeUnit:
    def test_ensure_tree_dir_creates_directory(self, tmp_path: Path) -> None:
        ht.init_tree_if_needed(tmp_path)

        assert (tmp_path / "hypothesis_tree").is_dir()
        assert (tmp_path / "hypothesis_tree" / "nodes").is_dir()

    def test_init_tree_creates_root_node_and_skeleton(self, tmp_path: Path) -> None:
        root = ht.init_tree_if_needed(tmp_path)

        assert root.id == "root"
        assert (tmp_path / "hypothesis_tree" / "nodes" / "root" / "node.json").is_file()
        tree = _tree(tmp_path)
        assert tree["version"] == 1
        assert tree["nodes"]["root"]["parent_id"] is None

    def test_init_tree_idempotent(self, tmp_path: Path) -> None:
        ht.init_tree_if_needed(tmp_path)
        before = _events(tmp_path)
        root = ht.init_tree_if_needed(tmp_path)

        assert root.id == "root"
        assert _events(tmp_path) == before

    def test_create_node_writes_node_json_and_hypothesis_md(self, tmp_path: Path) -> None:
        ht.init_tree_if_needed(tmp_path)
        node = ht.create_node(
            tmp_path,
            "root",
            _hypothesis_text(1),
            status="active",
            pivoted_from=None,
            edge_type="initialize",
        )

        assert node.id == "h-1"
        assert _node_json(tmp_path, "h-1")["hypothesis_snippet"].startswith("# Hypotheses")
        assert (
            tmp_path / "hypothesis_tree" / "nodes" / "h-1" / "hypothesis.md"
        ).read_text(encoding="utf-8") == _hypothesis_text(1)

    def test_create_node_updates_tree_json(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)

        tree = _tree(tmp_path)
        assert "h-1" in tree["nodes"]
        assert tree["nodes"]["root"]["children"] == ["h-1"]
        assert tree["edges"][0]["edge_type"] == "initialize"

    def test_create_node_appends_event(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)

        events = _events(tmp_path)
        assert events[-1]["event_type"] == "node_created"
        assert events[-1]["node_id"] == "h-1"

    def test_create_node_id_sequential(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        node = ht.create_node(
            tmp_path,
            "root",
            _hypothesis_text(2),
            status="active",
            pivoted_from=None,
            edge_type="pivot",
        )

        assert node.id == "h-2"
        assert node.attempt_number == 2

    def test_get_current_node_id_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert ht.get_current_node_id(tmp_path) is None

    def test_set_and_get_current_node_roundtrip(self, tmp_path: Path) -> None:
        ht.init_tree_if_needed(tmp_path)
        ht.set_current_node(tmp_path, "root")

        assert ht.get_current_node_id(tmp_path) == "root"

    def test_write_and_read_pending_transition_roundtrip(self, tmp_path: Path) -> None:
        ht.init_tree_if_needed(tmp_path)
        pending = ht.PendingTransition(
            transition_type="extend",
            source_node_id="h-1",
            created_at="2026-01-01T00:00:00+00:00",
            decision_text_excerpt="extend this line",
            human_edited=True,
        )

        ht.write_pending_transition(tmp_path, pending)

        loaded = ht.read_pending_transition(tmp_path)
        assert loaded == pending

    def test_clear_pending_transition_removes_file(self, tmp_path: Path) -> None:
        ht.init_tree_if_needed(tmp_path)
        ht.write_pending_transition(
            tmp_path,
            ht.PendingTransition("pivot", "h-1", "2026-01-01T00:00:00+00:00", "", False),
        )

        ht.clear_pending_transition(tmp_path)

        assert ht.read_pending_transition(tmp_path) is None

    def test_read_pending_transition_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert ht.read_pending_transition(tmp_path) is None

    def test_read_tree_returns_empty_skeleton_when_missing(self, tmp_path: Path) -> None:
        tree = ht.read_tree(tmp_path)

        assert tree["version"] == 1
        assert tree["nodes"] == {}
        assert tree["edges"] == []

    def test_update_node_status_changes_all_artifacts(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)

        ht.update_node_status(tmp_path, "h-1", "completed")

        assert _node_json(tmp_path, "h-1")["status"] == "completed"
        assert _tree(tmp_path)["nodes"]["h-1"]["status"] == "completed"
        assert _events(tmp_path)[-1]["event_type"] == "node_status_changed"

    def test_events_jsonl_always_appends(self, tmp_path: Path) -> None:
        ht.init_tree_if_needed(tmp_path)
        first_count = len(_events(tmp_path))

        ht.append_event(
            tmp_path,
            ht.TreeEvent("custom_event", "root", {"n": 1}, "2026-01-01T00:00:00+00:00"),
        )
        ht.append_event(
            tmp_path,
            ht.TreeEvent("custom_event", "root", {"n": 2}, "2026-01-01T00:00:01+00:00"),
        )

        events = _events(tmp_path)
        assert len(events) == first_count + 2
        assert [event["data"]["n"] for event in events[-2:]] == [1, 2]

    def test_record_stage15_decision_proceed_marks_completed(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        ht.set_current_node(tmp_path, "h-1")

        ht.record_stage15_decision(tmp_path, "proceed", "## Decision\nPROCEED", human_edited=False)

        assert _node_json(tmp_path, "h-1")["status"] == "completed"
        assert ht.read_pending_transition(tmp_path) is None

    def test_record_stage15_decision_extend_writes_pending(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        ht.set_current_node(tmp_path, "h-1")

        ht.record_stage15_decision(tmp_path, "extend", "## Decision\nEXTEND", human_edited=False)

        pending = ht.read_pending_transition(tmp_path)
        assert pending is not None
        assert pending.transition_type == "extend"
        assert pending.source_node_id == "h-1"

    def test_record_stage15_decision_pivot_writes_pending(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        ht.set_current_node(tmp_path, "h-1")

        ht.record_stage15_decision(tmp_path, "pivot", "## Decision\nPIVOT", human_edited=False)

        pending = ht.read_pending_transition(tmp_path)
        assert pending is not None
        assert pending.transition_type == "pivot"

    def test_record_stage15_decision_no_tree_auto_inits(self, tmp_path: Path) -> None:
        _seed_stage8(tmp_path)

        ht.record_stage15_decision(tmp_path, "extend", "## Decision\nEXTEND", human_edited=False)

        assert (tmp_path / "hypothesis_tree" / "tree.json").is_file()
        assert ht.get_current_node_id(tmp_path) == "h-1"

    def test_record_stage15_decision_blocked_node_rejected(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        ht.set_current_node(tmp_path, "h-1")
        ht.update_node_status(tmp_path, "h-1", "blocked")

        with pytest.raises(ValueError, match="blocked"):
            ht.record_stage15_decision(tmp_path, "extend", "## Decision\nEXTEND", human_edited=False)

    def test_record_stage15_decision_human_edited_flag_preserved(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        ht.set_current_node(tmp_path, "h-1")

        ht.record_stage15_decision(tmp_path, "pivot", "## Decision\nPIVOT", human_edited=True)

        pending = ht.read_pending_transition(tmp_path)
        assert pending is not None
        assert pending.human_edited is True
        assert _events(tmp_path)[-1]["data"]["human_edited"] is True


class TestHypothesisTreeFinalize:
    def test_finalize_first_run_creates_h1_under_root(self, tmp_path: Path) -> None:
        node_id = ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))

        assert node_id == "h-1"
        assert ht.get_current_node_id(tmp_path) == "h-1"
        assert _tree(tmp_path)["edges"][0]["edge_type"] == "initialize"

    def test_finalize_extend_creates_child_and_inactivates_parent(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        ht.set_current_node(tmp_path, "h-1")
        ht.record_stage15_decision(tmp_path, "extend", "## Decision\nEXTEND", human_edited=False)

        node_id = ht.finalize_after_stage8(tmp_path, _hypothesis_text(2))

        assert node_id == "h-2"
        assert _node_json(tmp_path, "h-1")["status"] == "inactive"
        assert _node_json(tmp_path, "h-2")["parent_id"] == "h-1"
        assert ht.get_current_node_id(tmp_path) == "h-2"
        assert ht.read_pending_transition(tmp_path) is None

    def test_finalize_pivot_creates_sibling_and_blocks_source(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        ht.set_current_node(tmp_path, "h-1")
        ht.record_stage15_decision(tmp_path, "pivot", "## Decision\nPIVOT", human_edited=False)

        node_id = ht.finalize_after_stage8(tmp_path, _hypothesis_text(2))

        assert node_id == "h-2"
        assert _node_json(tmp_path, "h-1")["status"] == "blocked"
        assert _node_json(tmp_path, "h-2")["parent_id"] == "root"
        assert set(_tree(tmp_path)["nodes"]["root"]["children"]) == {"h-1", "h-2"}

    def test_finalize_pivot_records_pivoted_from_field(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        ht.set_current_node(tmp_path, "h-1")
        ht.record_stage15_decision(tmp_path, "pivot", "## Decision\nPIVOT", human_edited=False)

        ht.finalize_after_stage8(tmp_path, _hypothesis_text(2))

        assert _node_json(tmp_path, "h-2")["pivoted_from"] == "h-1"

    def test_finalize_empty_hypotheses_clears_pending_no_node(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        ht.set_current_node(tmp_path, "h-1")
        ht.record_stage15_decision(tmp_path, "extend", "## Decision\nEXTEND", human_edited=False)

        node_id = ht.finalize_after_stage8(tmp_path, " \n ")

        assert node_id is None
        assert ht.read_pending_transition(tmp_path) is None
        assert "h-2" not in _tree(tmp_path)["nodes"]

    def test_finalize_no_pending_transition_no_op(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        ht.set_current_node(tmp_path, "h-1")

        node_id = ht.finalize_after_stage8(tmp_path, _hypothesis_text(2))

        assert node_id is None
        assert set(_tree(tmp_path)["nodes"]) == {"root", "h-1"}

    def test_finalize_blocked_parent_rejected(self, tmp_path: Path) -> None:
        _create_initial_node(tmp_path)
        node = ht.create_node(
            tmp_path,
            "h-1",
            _hypothesis_text(2),
            status="active",
            pivoted_from=None,
            edge_type="extend",
        )
        ht.set_current_node(tmp_path, node.id)
        ht.update_node_status(tmp_path, "h-1", "blocked")
        ht.record_stage15_decision(tmp_path, "pivot", "## Decision\nPIVOT", human_edited=False)

        with pytest.raises(ValueError, match="blocked parent"):
            ht.finalize_after_stage8(tmp_path, _hypothesis_text(3))

    def test_finalize_two_extends_creates_linear_chain(self, tmp_path: Path) -> None:
        ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))
        ht.record_stage15_decision(tmp_path, "extend", "## Decision\nEXTEND", human_edited=False)
        ht.finalize_after_stage8(tmp_path, _hypothesis_text(2))
        ht.record_stage15_decision(tmp_path, "extend", "## Decision\nEXTEND", human_edited=False)
        ht.finalize_after_stage8(tmp_path, _hypothesis_text(3))

        assert _node_json(tmp_path, "h-2")["parent_id"] == "h-1"
        assert _node_json(tmp_path, "h-3")["parent_id"] == "h-2"
        assert ht.get_current_node_id(tmp_path) == "h-3"

    def test_finalize_pivot_from_root_child_creates_root_sibling(self, tmp_path: Path) -> None:
        ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))
        ht.record_stage15_decision(tmp_path, "pivot", "## Decision\nPIVOT", human_edited=False)

        ht.finalize_after_stage8(tmp_path, _hypothesis_text(2))

        assert _node_json(tmp_path, "h-2")["parent_id"] == "root"
        assert _tree(tmp_path)["nodes"]["root"]["children"] == ["h-1", "h-2"]

    def test_finalize_idempotent_same_hypothesis(self, tmp_path: Path) -> None:
        first = ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))
        second = ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))

        assert first == "h-1"
        assert second is None
        assert set(_tree(tmp_path)["nodes"]) == {"root", "h-1"}


def _pipeline_config(tmp_path: Path) -> Any:
    from researchclaw.config import RCConfig

    data = {
        "project": {"name": "hypothesis-tree-test", "mode": "docs-first"},
        "research": {"topic": "lineage testing"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline",
        },
        "experiment": {},
        "hypothesis_validation": {"enabled": False},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def _pipeline_adapters() -> Any:
    from researchclaw.adapters import AdapterBundle

    return AdapterBundle()


def _done_result(stage: Any, *, decision: str = "proceed") -> Any:
    from researchclaw.pipeline.executor import StageResult
    from researchclaw.pipeline.stages import StageStatus

    return StageResult(
        stage=stage,
        status=StageStatus.DONE,
        artifacts=("out.md",),
        decision=decision,
    )


def _write_pipeline_stage_artifacts(run_dir: Path, stage: Any, hypothesis_idx: int) -> None:
    from researchclaw.pipeline.stages import Stage

    stage_dir = run_dir / f"stage-{int(stage):02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    if stage == Stage.HYPOTHESIS_GEN:
        (stage_dir / "hypotheses.md").write_text(
            _hypothesis_text(hypothesis_idx), encoding="utf-8"
        )
    if stage == Stage.RESULT_ANALYSIS:
        (stage_dir / "analysis.md").write_text(
            f"# Analysis\nEvidence for hypothesis {hypothesis_idx}.",
            encoding="utf-8",
        )
        (stage_dir / "experiment_summary.json").write_text(
            json.dumps({"metrics_summary": {"accuracy": {"mean": 0.8}}}),
            encoding="utf-8",
        )


def _write_pipeline_decision_artifacts(
    run_dir: Path, decision: str, decision_md: str
) -> None:
    stage_dir = run_dir / "stage-15"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "decision.md").write_text(decision_md, encoding="utf-8")
    (stage_dir / "decision_structured.json").write_text(
        json.dumps({"decision": decision}),
        encoding="utf-8",
    )


def _cycle_dir(run_dir: Path, *ids: str) -> Path:
    return (
        run_dir
        / "hypothesis_tree"
        / "node_tree"
        / Path(*ids)
        / "_artifacts"
    )


class TestHypothesisTreePipeline:
    def test_pipeline_proceed_updates_tree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.pipeline import runner as rc_runner
        from researchclaw.pipeline.stages import Stage

        def mock_execute_stage(stage: Any, **kwargs: Any) -> Any:
            _ = kwargs
            _write_pipeline_stage_artifacts(tmp_path, stage, 1)
            if stage == Stage.RESEARCH_DECISION:
                stage_dir = tmp_path / "stage-15"
                stage_dir.mkdir(parents=True, exist_ok=True)
                decision_md = "## Decision\nPROCEED"
                (stage_dir / "decision.md").write_text(decision_md, encoding="utf-8")
                ht.record_stage15_decision(tmp_path, "proceed", decision_md, human_edited=False)
                return _done_result(stage, decision="proceed")
            return _done_result(stage)

        monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

        rc_runner.execute_pipeline(
            run_dir=tmp_path,
            run_id="run-proceed-tree",
            config=_pipeline_config(tmp_path),
            adapters=_pipeline_adapters(),
        )

        assert _node_json(tmp_path, "h-1")["status"] == "completed"

    def test_pipeline_extend_creates_child_node(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.pipeline import runner as rc_runner
        from researchclaw.pipeline.stages import Stage

        hypothesis_idx = 0
        decision_count = 0

        def mock_execute_stage(stage: Any, **kwargs: Any) -> Any:
            _ = kwargs
            nonlocal hypothesis_idx, decision_count
            if stage == Stage.HYPOTHESIS_GEN:
                hypothesis_idx += 1
            _write_pipeline_stage_artifacts(tmp_path, stage, max(1, hypothesis_idx))
            if stage == Stage.RESEARCH_DECISION:
                decision = "extend" if decision_count == 0 else "proceed"
                decision_count += 1
                stage_dir = tmp_path / "stage-15"
                stage_dir.mkdir(parents=True, exist_ok=True)
                decision_md = f"## Decision\n{decision.upper()}"
                (stage_dir / "decision.md").write_text(decision_md, encoding="utf-8")
                ht.record_stage15_decision(tmp_path, decision, decision_md, human_edited=False)
                return _done_result(stage, decision=decision)
            return _done_result(stage)

        monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

        rc_runner.execute_pipeline(
            run_dir=tmp_path,
            run_id="run-extend-tree",
            config=_pipeline_config(tmp_path),
            adapters=_pipeline_adapters(),
        )

        assert _node_json(tmp_path, "h-2")["parent_id"] == "h-1"
        assert _tree(tmp_path)["edges"][-1]["edge_type"] == "extend"

    def test_pipeline_pivot_creates_sibling_node(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.pipeline import runner as rc_runner
        from researchclaw.pipeline.stages import Stage

        hypothesis_idx = 0
        decision_count = 0

        def mock_execute_stage(stage: Any, **kwargs: Any) -> Any:
            _ = kwargs
            nonlocal hypothesis_idx, decision_count
            if stage == Stage.HYPOTHESIS_GEN:
                hypothesis_idx += 1
            _write_pipeline_stage_artifacts(tmp_path, stage, max(1, hypothesis_idx))
            if stage == Stage.RESEARCH_DECISION:
                decision = "pivot" if decision_count == 0 else "proceed"
                decision_count += 1
                stage_dir = tmp_path / "stage-15"
                stage_dir.mkdir(parents=True, exist_ok=True)
                decision_md = f"## Decision\n{decision.upper()}"
                (stage_dir / "decision.md").write_text(decision_md, encoding="utf-8")
                ht.record_stage15_decision(tmp_path, decision, decision_md, human_edited=False)
                return _done_result(stage, decision=decision)
            return _done_result(stage)

        monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

        rc_runner.execute_pipeline(
            run_dir=tmp_path,
            run_id="run-pivot-tree",
            config=_pipeline_config(tmp_path),
            adapters=_pipeline_adapters(),
        )

        assert _node_json(tmp_path, "h-1")["status"] == "blocked"
        assert _node_json(tmp_path, "h-2")["parent_id"] == "root"
        assert _tree(tmp_path)["edges"][-1]["edge_type"] == "pivot"

    def test_pipeline_human_gate_extend_follows_edited_decision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.pipeline import executor as rc_executor
        from researchclaw.pipeline import runner as rc_runner
        from researchclaw.pipeline.stages import Stage

        ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))
        stage15 = tmp_path / "stage-15"
        stage15.mkdir(parents=True, exist_ok=True)
        (stage15 / "decision.md").write_text("## Decision\nEXTEND", encoding="utf-8")
        (stage15 / ".gate_proposal.json").write_text("{}", encoding="utf-8")
        stage15_calls = 0

        def mock_execute_stage(stage: Any, **kwargs: Any) -> Any:
            nonlocal stage15_calls
            _write_pipeline_stage_artifacts(tmp_path, stage, 2)
            if stage == Stage.RESEARCH_DECISION and stage15_calls == 0:
                stage15_calls += 1
                return rc_executor.execute_stage(stage, **kwargs)
            if stage == Stage.RESEARCH_DECISION:
                decision_md = "## Decision\nPROCEED"
                (tmp_path / "stage-15" / "decision.md").write_text(
                    decision_md, encoding="utf-8"
                )
                ht.record_stage15_decision(tmp_path, "proceed", decision_md, human_edited=False)
                return _done_result(stage, decision="proceed")
            return _done_result(stage)

        monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

        rc_runner.execute_pipeline(
            run_dir=tmp_path,
            run_id="run-human-extend-tree",
            config=_pipeline_config(tmp_path),
            adapters=_pipeline_adapters(),
            from_stage=Stage.RESEARCH_DECISION,
        )

        assert _node_json(tmp_path, "h-2")["parent_id"] == "h-1"
        assert any(
            event["event_type"] == "transition_finalized"
            and event["data"]["transition_type"] == "extend"
            and event["data"]["human_edited"] is True
            for event in _events(tmp_path)
        )

    def test_pipeline_human_gate_pivot_follows_edited_decision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.pipeline import executor as rc_executor
        from researchclaw.pipeline import runner as rc_runner
        from researchclaw.pipeline.stages import Stage

        ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))
        stage15 = tmp_path / "stage-15"
        stage15.mkdir(parents=True, exist_ok=True)
        (stage15 / "decision.md").write_text("## Decision\nPIVOT", encoding="utf-8")
        (stage15 / ".gate_proposal.json").write_text("{}", encoding="utf-8")
        stage15_calls = 0

        def mock_execute_stage(stage: Any, **kwargs: Any) -> Any:
            nonlocal stage15_calls
            _write_pipeline_stage_artifacts(tmp_path, stage, 2)
            if stage == Stage.RESEARCH_DECISION and stage15_calls == 0:
                stage15_calls += 1
                return rc_executor.execute_stage(stage, **kwargs)
            if stage == Stage.RESEARCH_DECISION:
                decision_md = "## Decision\nPROCEED"
                (tmp_path / "stage-15" / "decision.md").write_text(
                    decision_md, encoding="utf-8"
                )
                ht.record_stage15_decision(tmp_path, "proceed", decision_md, human_edited=False)
                return _done_result(stage, decision="proceed")
            return _done_result(stage)

        monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

        rc_runner.execute_pipeline(
            run_dir=tmp_path,
            run_id="run-human-pivot-tree",
            config=_pipeline_config(tmp_path),
            adapters=_pipeline_adapters(),
            from_stage=Stage.RESEARCH_DECISION,
        )

        assert _node_json(tmp_path, "h-1")["status"] == "blocked"
        assert _node_json(tmp_path, "h-2")["pivoted_from"] == "h-1"
        assert any(
            event["event_type"] == "transition_finalized"
            and event["data"]["transition_type"] == "pivot"
            and event["data"]["human_edited"] is True
            for event in _events(tmp_path)
        )

    def test_pipeline_forced_proceed_clears_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.pipeline import runner as rc_runner
        from researchclaw.pipeline.stages import Stage

        ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))
        (tmp_path / "decision_history.json").write_text(
            json.dumps([{"decision": "pivot"}, {"decision": "extend"}]),
            encoding="utf-8",
        )
        for name in ("stage-14", "stage-14_v1"):
            stage14 = tmp_path / name
            stage14.mkdir(parents=True, exist_ok=True)
            (stage14 / "experiment_summary.json").write_text(
                json.dumps({"metrics_summary": {}}), encoding="utf-8"
            )

        def mock_execute_stage(stage: Any, **kwargs: Any) -> Any:
            _ = kwargs
            _write_pipeline_stage_artifacts(tmp_path, stage, 1)
            if stage == Stage.RESEARCH_DECISION:
                decision_md = "## Decision\nEXTEND"
                (tmp_path / "stage-15").mkdir(parents=True, exist_ok=True)
                (tmp_path / "stage-15" / "decision.md").write_text(
                    decision_md, encoding="utf-8"
                )
                ht.record_stage15_decision(tmp_path, "extend", decision_md, human_edited=False)
                return _done_result(stage, decision="extend")
            return _done_result(stage)

        monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

        rc_runner.execute_pipeline(
            run_dir=tmp_path,
            run_id="run-forced-proceed-tree",
            config=_pipeline_config(tmp_path),
            adapters=_pipeline_adapters(),
            from_stage=Stage.RESEARCH_DECISION,
        )

        assert ht.read_pending_transition(tmp_path) is None
        assert _node_json(tmp_path, "h-1")["status"] == "completed"
        assert _events(tmp_path)[-1]["data"]["forced"] is True

    def test_pipeline_first_run_initializes_tree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.pipeline import runner as rc_runner
        from researchclaw.pipeline.stages import Stage

        def mock_execute_stage(stage: Any, **kwargs: Any) -> Any:
            _ = kwargs
            _write_pipeline_stage_artifacts(tmp_path, stage, 1)
            return _done_result(stage)

        monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

        rc_runner.execute_pipeline(
            run_dir=tmp_path,
            run_id="run-first-tree",
            config=_pipeline_config(tmp_path),
            adapters=_pipeline_adapters(),
            to_stage=Stage.HYPOTHESIS_GEN,
        )

        assert (tmp_path / "hypothesis_tree" / "nodes" / "root").is_dir()
        assert ht.get_current_node_id(tmp_path) == "h-1"


class TestHypothesisCycleArchiveCompat:
    def test_pipeline_no_longer_writes_cycle_archive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.pipeline import runner as rc_runner
        from researchclaw.pipeline.stages import Stage

        def mock_execute_stage(stage: Any, **kwargs: Any) -> Any:
            _ = kwargs
            _write_pipeline_stage_artifacts(tmp_path, stage, 1)
            if stage == Stage.RESEARCH_DECISION:
                decision_md = "## Decision\nPROCEED"
                _write_pipeline_decision_artifacts(tmp_path, "proceed", decision_md)
                ht.record_stage15_decision(
                    tmp_path, "proceed", decision_md, human_edited=False
                )
                return _done_result(stage, decision="proceed")
            return _done_result(stage)

        monkeypatch.setattr(rc_runner, "execute_stage", mock_execute_stage)

        rc_runner.execute_pipeline(
            run_dir=tmp_path,
            run_id="run-no-cycle-archive",
            config=_pipeline_config(tmp_path),
            adapters=_pipeline_adapters(),
            from_stage=Stage.HYPOTHESIS_GEN,
        )

        assert not (
            tmp_path / "hypothesis_tree" / "node_tree" / "root" / "h-1" / "_artifacts"
        ).exists()
        assert _node_json(tmp_path, "h-1")["status"] == "completed"

    def test_existing_cycle_archive_manifest_still_readable(
        self, tmp_path: Path
    ) -> None:
        from researchclaw.pipeline import hypothesis_cycle_archive as ca
        from researchclaw.pipeline.stages import Stage

        _write_pipeline_stage_artifacts(tmp_path, Stage.HYPOTHESIS_GEN, 1)
        ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))
        _write_pipeline_stage_artifacts(tmp_path, Stage.RESULT_ANALYSIS, 1)
        decision_md = "## Decision\nPROCEED"
        _write_pipeline_decision_artifacts(tmp_path, "proceed", decision_md)
        ht.record_stage15_decision(
            tmp_path, "proceed", decision_md, human_edited=False
        )

        cycle = ca.archive_current_hypothesis_cycle(tmp_path, decision="proceed")

        assert cycle == _cycle_dir(tmp_path, "root", "h-1") / "cycle-001"
        manifest = json.loads((cycle / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["decision"] == "proceed"
        assert (cycle / "stage-15" / "decision.md").is_file()


class TestHypothesisTreeEdgeCases:
    def test_tree_survives_rollback_versioning(self, tmp_path: Path) -> None:
        from researchclaw.pipeline.runner import _version_rollback_stages
        from researchclaw.pipeline.stages import Stage

        ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))
        (tmp_path / "stage-08").mkdir()
        (tmp_path / "stage-08" / "hypotheses.md").write_text(
            _hypothesis_text(1), encoding="utf-8"
        )

        _version_rollback_stages(tmp_path, Stage.HYPOTHESIS_GEN, 1)

        assert (tmp_path / "hypothesis_tree" / "tree.json").is_file()
        assert (tmp_path / "stage-08_v1").is_dir()

    def test_empty_stage8_output_no_node_created(self, tmp_path: Path) -> None:
        node_id = ht.finalize_after_stage8(tmp_path, "")

        assert node_id is None
        assert set(_tree(tmp_path)["nodes"]) == {"root"}

    def test_rerun_without_pending_transition_no_op(self, tmp_path: Path) -> None:
        ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))
        before = _tree(tmp_path)

        ht.finalize_after_stage8(tmp_path, _hypothesis_text(2))

        assert _tree(tmp_path)["nodes"] == before["nodes"]

    def test_double_finalize_idempotent(self, tmp_path: Path) -> None:
        ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))
        ht.record_stage15_decision(tmp_path, "extend", "## Decision\nEXTEND", human_edited=False)
        first = ht.finalize_after_stage8(tmp_path, _hypothesis_text(2))
        second = ht.finalize_after_stage8(tmp_path, _hypothesis_text(2))

        assert first == "h-2"
        assert second is None
        assert set(_tree(tmp_path)["nodes"]) == {"root", "h-1", "h-2"}

    def test_node_counter_increments_correctly_across_runs(self, tmp_path: Path) -> None:
        ht.finalize_after_stage8(tmp_path, _hypothesis_text(1))
        ht.record_stage15_decision(tmp_path, "extend", "## Decision\nEXTEND", human_edited=False)
        ht.finalize_after_stage8(tmp_path, _hypothesis_text(2))
        ht.record_stage15_decision(tmp_path, "pivot", "## Decision\nPIVOT", human_edited=False)
        ht.finalize_after_stage8(tmp_path, _hypothesis_text(3))

        assert set(_tree(tmp_path)["nodes"]) == {"root", "h-1", "h-2", "h-3"}
        assert _node_json(tmp_path, "h-3")["attempt_number"] == 3
