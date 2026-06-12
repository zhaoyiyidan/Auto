"""Tests for experiment_repair — repair loop and prompt generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.pipeline.experiment_diagnosis import (
    DeficiencyType,
    Deficiency,
    ExperimentDiagnosis,
)
from researchclaw.pipeline.experiment_repair import (
    build_repair_prompt,
    select_best_results,
    _build_experiment_summary_from_run,
    _load_experiment_code,
    _load_experiment_summary,
    _summary_quality_score,
)


# ---------------------------------------------------------------------------
# build_repair_prompt tests
# ---------------------------------------------------------------------------


class TestBuildRepairPrompt:
    def test_basic_prompt(self):
        diag = ExperimentDiagnosis(
            deficiencies=[
                Deficiency(
                    type=DeficiencyType.MISSING_DEPENDENCY,
                    severity="critical",
                    description="Missing Python package: utils",
                    suggested_fix="Add 'utils' to requirements.txt",
                )
            ],
            conditions_completed=["CondA"],
            conditions_failed=["CondB"],
            total_planned=2,
            completion_rate=0.5,
            summary="1 deficiency. 1/2 conditions completed.",
        )
        prompt = build_repair_prompt(
            diagnosis=diag,
            original_code={"main.py": "import utils\nprint('hello')"},
            time_budget_sec=2400,
        )
        assert "EXPERIMENT REPAIR TASK" in prompt
        assert "utils" in prompt
        assert "main.py" in prompt
        assert "2400" in prompt

    def test_scope_reduction_included(self):
        diag = ExperimentDiagnosis(
            deficiencies=[
                Deficiency(
                    type=DeficiencyType.TIME_GUARD_DOMINANT,
                    severity="major",
                    description="Time guard killed 8/10 conditions",
                    affected_conditions=["C3", "C4", "C5"],
                    suggested_fix="Reduce conditions",
                )
            ],
            conditions_completed=["C1", "C2"],
            conditions_failed=["C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"],
            total_planned=10,
            completion_rate=0.2,
        )
        prompt = build_repair_prompt(diag, original_code={})
        assert "SCOPE REDUCTION" in prompt
        assert "BASELINE" in prompt

    def test_dep_fix_section(self):
        diag = ExperimentDiagnosis(
            deficiencies=[
                Deficiency(
                    type=DeficiencyType.MISSING_DEPENDENCY,
                    severity="critical",
                    description="Missing Python package: box2d-py",
                    suggested_fix="Add 'box2d-py' to requirements.txt",
                ),
            ],
        )
        prompt = build_repair_prompt(diag, original_code={})
        assert "DEPENDENCY FIXES" in prompt
        assert "box2d-py" in prompt

    def test_long_code_truncated(self):
        long_code = "x = 1\n" * 5000
        diag = ExperimentDiagnosis()
        prompt = build_repair_prompt(diag, original_code={"big.py": long_code})
        assert "truncated" in prompt

    def test_workspace_agent_instruction_section(self):
        diag = ExperimentDiagnosis()
        prompt = build_repair_prompt(diag, original_code={"main.py": "pass"})
        assert "WORKSPACE AGENT INSTRUCTIONS" in prompt
        assert "run_manifest.json" in prompt
        assert "final git commit after all task changes are ready" in prompt
        assert "code_commit to final HEAD" in prompt
        assert "git status --porcelain" in prompt
        assert "Do not submit" in prompt


class TestRepairLoopRemoval:
    def test_legacy_repair_helpers_are_removed(self) -> None:
        import researchclaw.pipeline._helpers as helpers
        import researchclaw.pipeline.experiment_repair as repair

        removed_from_helpers = [
            "_extract_code_block",
            "_extract_multi_file_blocks",
        ]
        removed_from_repair = [
            "_extract_code_blocks",
            "_repair_via_opencode",
            "_repair_via_llm",
            "_run_experiment_in_sandbox",
            "_get_repaired_code",
            "_repair_via_workspace_agent",
            "run_" + "repair_loop",
        ]

        assert [name for name in removed_from_helpers if hasattr(helpers, name)] == []
        assert [name for name in removed_from_repair if hasattr(repair, name)] == []

    def test_repair_loop_symbol_is_removed(self) -> None:
        import researchclaw.pipeline.experiment_repair as er

        assert not hasattr(er, "run_" + "repair_loop")

    def test_repair_config_drops_opencode_flag(self) -> None:
        from researchclaw.config import ExperimentRepairConfig

        assert not hasattr(ExperimentRepairConfig(), "use_opencode")


# ---------------------------------------------------------------------------
# Summary building tests
# ---------------------------------------------------------------------------


class TestBuildExperimentSummary:
    def test_basic_summary(self):
        run_result = {
            "stdout": "condition=Baseline metric=80.0\ncondition=Proposed metric=90.0",
            "stderr": "",
            "returncode": 0,
            "metrics": {
                "Baseline/0/accuracy": 80.0,
                "Proposed/0/accuracy": 90.0,
                "primary_metric": 90.0,
            },
            "elapsed_sec": 120.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {"main.py": "pass"})
        assert "condition_summaries" in summary
        assert "Baseline" in summary["condition_summaries"]
        assert "Proposed" in summary["condition_summaries"]
        assert summary["total_conditions"] == 2
        assert summary["best_run"]["status"] == "completed"

    def test_failed_run(self):
        run_result = {
            "stdout": "",
            "stderr": "Error: crash",
            "returncode": 1,
            "metrics": {},
            "elapsed_sec": 5.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        assert summary["best_run"]["status"] == "failed"
        assert summary["total_conditions"] == 0

    def test_multi_seed_grouping(self):
        run_result = {
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "metrics": {
                "Baseline/0/accuracy": 80.0,
                "Baseline/1/accuracy": 82.0,
                "Proposed/0/accuracy": 90.0,
                "Proposed/1/accuracy": 92.0,
            },
            "elapsed_sec": 300.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        assert len(summary["condition_summaries"]) == 2
        # Mean of 80.0 and 82.0
        bl = summary["condition_summaries"]["Baseline"]
        assert abs(bl["metrics"]["accuracy"] - 81.0) < 0.01
        assert bl["n_seeds"] == 2


# ---------------------------------------------------------------------------
# File loading tests
# ---------------------------------------------------------------------------


class TestLoadExperimentCode:
    def test_loads_stage10_manifest(self, tmp_path):
        stage_dir = tmp_path / "stage-10"
        stage_dir.mkdir(parents=True)
        (stage_dir / "run_manifest.json").write_text(
            json.dumps({"launch": {"command": "python train.py"}}),
            encoding="utf-8",
        )

        code = _load_experiment_code(tmp_path)
        assert "stage-10/run_manifest.json" in code
        assert "python train.py" in code["stage-10/run_manifest.json"]

    def test_loads_plan_when_manifest_missing(self, tmp_path):
        stage_dir = tmp_path / "stage-09"
        stage_dir.mkdir(parents=True)
        (stage_dir / "plan.md").write_text(
            "# Experiment Plan\n\nunlock files\n",
            encoding="utf-8",
        )
        (stage_dir / "expected_outputs.json").write_text(
            json.dumps(
                {
                    "schema_version": "researchclaw.expected_outputs.v1",
                    "outputs": ["outputs/results.json"],
                }
            ),
            encoding="utf-8",
        )

        code = _load_experiment_code(tmp_path)
        assert "stage-09/plan.md" in code
        assert "unlock files" in code["stage-09/plan.md"]

    def test_empty_when_no_code(self, tmp_path):
        code = _load_experiment_code(tmp_path)
        assert code == {}


class TestLoadExperimentSummary:
    def test_loads_summary(self, tmp_path):
        stage_dir = tmp_path / "stage-14"
        stage_dir.mkdir()
        summary = {"condition_summaries": {"A": {}}}
        (stage_dir / "experiment_summary.json").write_text(json.dumps(summary))

        result = _load_experiment_summary(tmp_path)
        assert result is not None
        assert "A" in result["condition_summaries"]


# ---------------------------------------------------------------------------
# select_best_results tests
# ---------------------------------------------------------------------------


class TestSelectBestResults:
    def test_picks_best_across_cycles(self, tmp_path):
        # Original (1 condition)
        s14 = tmp_path / "stage-14"
        s14.mkdir()
        (s14 / "experiment_summary.json").write_text(json.dumps({
            "condition_summaries": {"A": {}},
            "best_run": {"metrics": {}},
        }))

        # Repair v1 (3 conditions — better)
        r1 = tmp_path / "stage-14_repair_v1"
        r1.mkdir()
        (r1 / "experiment_summary.json").write_text(json.dumps({
            "condition_summaries": {"A": {}, "B": {}, "C": {}},
            "best_run": {"metrics": {"primary_metric": 90.0}},
        }))

        best = select_best_results(tmp_path, [])
        assert best is not None
        assert len(best["condition_summaries"]) == 3

    def test_returns_none_when_empty(self, tmp_path):
        result = select_best_results(tmp_path, [])
        assert result is None


# ---------------------------------------------------------------------------
# BUG-199: 2-part metric keys (condition/metric) in summary builder
# ---------------------------------------------------------------------------


class TestBuildExperimentSummaryTwoPartKeys:
    """BUG-199: Stage 13 repair output can produce 2-part keys (condition/metric)
    instead of 3-part keys (condition/seed/metric).  The parser must handle
    both formats.
    """

    def test_two_part_keys_parsed(self):
        """2-part keys like 'Baseline/accuracy' should create conditions."""
        run_result = {
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "metrics": {
                "Baseline/accuracy": 0.85,
                "Proposed/accuracy": 0.94,
                "Ablation/accuracy": 0.88,
            },
            "elapsed_sec": 120.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        assert summary["total_conditions"] == 3
        assert "Baseline" in summary["condition_summaries"]
        assert "Proposed" in summary["condition_summaries"]
        assert "Ablation" in summary["condition_summaries"]
        assert summary["condition_summaries"]["Proposed"]["metrics"]["accuracy"] == 0.94

    def test_two_part_keys_create_synthetic_seed(self):
        """2-part keys should create a synthetic seed '0' entry."""
        run_result = {
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "metrics": {
                "Baseline/accuracy": 0.80,
                "Baseline/loss": 0.45,
            },
            "elapsed_sec": 60.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        bl = summary["condition_summaries"]["Baseline"]
        assert bl["metrics"]["accuracy"] == 0.80
        assert bl["metrics"]["loss"] == 0.45
        assert bl["n_seeds"] == 1  # synthetic seed "0"

    def test_mixed_two_and_three_part_keys(self):
        """Mix of 2-part and 3-part keys for different conditions."""
        run_result = {
            "stdout": "",
            "stderr": "",
            "returncode": 0,
            "metrics": {
                # 3-part keys (with seed)
                "Baseline/0/accuracy": 0.80,
                "Baseline/1/accuracy": 0.82,
                # 2-part keys (Stage 13 repair output)
                "Proposed/accuracy": 0.94,
            },
            "elapsed_sec": 120.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        assert summary["total_conditions"] == 2
        # 3-part: mean of seeds
        bl = summary["condition_summaries"]["Baseline"]
        assert abs(bl["metrics"]["accuracy"] - 0.81) < 0.01
        assert bl["n_seeds"] == 2
        # 2-part: single value
        pr = summary["condition_summaries"]["Proposed"]
        assert pr["metrics"]["accuracy"] == 0.94
        assert pr["n_seeds"] == 1

    def test_empty_metrics_still_empty(self):
        """Empty metrics dict should still produce 0 conditions."""
        run_result = {
            "stdout": "",
            "stderr": "",
            "returncode": 1,
            "metrics": {},
            "elapsed_sec": 5.0,
            "timed_out": False,
        }
        summary = _build_experiment_summary_from_run(run_result, {})
        assert summary["total_conditions"] == 0


# ---------------------------------------------------------------------------
# BUG-198: Conditional promotion of repair summary in runner.py
# ---------------------------------------------------------------------------


class TestRepairSummaryPromotion:
    """BUG-198: runner.py should NOT overwrite a richer stage-14 summary
    with an empty/poorer repair result.
    """

    def test_empty_repair_does_not_overwrite_rich_summary(self, tmp_path):
        """Repair result with 0 conditions must NOT replace a summary
        that has real conditions and metrics.
        """
        # Create a rich existing stage-14 summary
        s14 = tmp_path / "stage-14"
        s14.mkdir()
        rich_summary = {
            "condition_summaries": {
                "Baseline": {"metrics": {"accuracy": 0.80}},
                "Proposed": {"metrics": {"accuracy": 0.94}},
                "Ablation": {"metrics": {"accuracy": 0.88}},
            },
            "best_run": {
                "metrics": {
                    "Baseline/0/accuracy": 0.80,
                    "Proposed/0/accuracy": 0.94,
                    "primary_metric": 0.94,
                },
            },
            "total_conditions": 3,
            "total_metric_keys": 3,
        }
        (s14 / "experiment_summary.json").write_text(json.dumps(rich_summary))

        # Compute scores to verify the logic
        rich_score = _summary_quality_score(rich_summary)

        empty_summary = {
            "condition_summaries": {},
            "best_run": {"metrics": {}},
            "total_conditions": 0,
            "total_metric_keys": 0,
        }
        empty_score = _summary_quality_score(empty_summary)

        # The rich summary must score higher
        assert rich_score > empty_score

        # Verify that the existing file is preserved (simulate what runner does)
        existing = json.loads(
            (s14 / "experiment_summary.json").read_text(encoding="utf-8")
        )
        existing_score = _summary_quality_score(existing)
        repair_score = _summary_quality_score(empty_summary)

        # runner.py should NOT overwrite because repair_score <= existing_score
        assert repair_score <= existing_score
        # The file should still contain the rich data
        after = json.loads(
            (s14 / "experiment_summary.json").read_text(encoding="utf-8")
        )
        assert len(after["condition_summaries"]) == 3

    def test_richer_repair_does_overwrite(self, tmp_path):
        """Repair result with MORE conditions should replace a poorer summary."""
        s14 = tmp_path / "stage-14"
        s14.mkdir()
        poor_summary = {
            "condition_summaries": {"A": {"metrics": {"m": 0.5}}},
            "best_run": {"metrics": {}},
            "total_conditions": 1,
            "total_metric_keys": 0,
        }
        (s14 / "experiment_summary.json").write_text(json.dumps(poor_summary))

        rich_repair = {
            "condition_summaries": {
                "A": {"metrics": {"m": 0.80}},
                "B": {"metrics": {"m": 0.85}},
                "C": {"metrics": {"m": 0.90}},
            },
            "best_run": {"metrics": {"primary_metric": 0.90}},
            "total_conditions": 3,
            "total_metric_keys": 4,
        }

        poor_score = _summary_quality_score(poor_summary)
        rich_score = _summary_quality_score(rich_repair)
        assert rich_score > poor_score
