"""Tests for experiment_diagnosis — failure analysis agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.pipeline.experiment_diagnosis import (
    DeficiencyType,
    ExperimentDiagnosis,
    ExperimentQualityAssessment,
    PaperMode,
    assess_experiment_quality,
    diagnose_experiment,
)

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


# ---------------------------------------------------------------------------
# Unit tests — individual checks
# ---------------------------------------------------------------------------


class TestMissingDependency:
    def test_detects_module_not_found(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stdout="",
            stderr="ModuleNotFoundError: No module named 'utils'",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.MISSING_DEPENDENCY in types

    def test_detects_box2d(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stdout="BOX2D_WARNING: Box2D/LunarLander-v3 not available; skipping",
            stderr="",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.MISSING_DEPENDENCY in types


class TestPermissionError:
    def test_detects_hf_permission(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stdout="",
            stderr="PermissionError: Cannot download huggingface model",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.PERMISSION_ERROR in types


class TestTimeGuard:
    def test_detects_dominant_time_guard(self):
        summary = {
            "condition_summaries": {"CondA": {"metrics": {"metric": 80.0}}},
            "best_run": {"metrics": {}},
        }
        plan = {"conditions": [{"name": "CondA"}, {"name": "CondB"}, {"name": "CondC"}, {"name": "CondD"}]}
        diag = diagnose_experiment(
            experiment_summary=summary,
            experiment_plan=plan,
            stdout="TIME_GUARD: skipping CondB\nTIME_GUARD: skipping CondC\nTIME_GUARD: skipping CondD",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.TIME_GUARD_DOMINANT in types

    def test_no_time_guard_if_most_complete(self):
        summary = {
            "condition_summaries": {
                "A": {"metrics": {"metric": 1.0}},
                "B": {"metrics": {"metric": 2.0}},
                "C": {"metrics": {"metric": 3.0}},
            },
            "best_run": {"metrics": {}},
        }
        plan = {"conditions": [{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}]}
        diag = diagnose_experiment(experiment_summary=summary, experiment_plan=plan)
        types = {d.type for d in diag.deficiencies}
        # 1/4 skipped = 25%, below 50% threshold
        assert DeficiencyType.TIME_GUARD_DOMINANT not in types


class TestSyntheticData:
    def test_detects_synthetic_fallback(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stdout="[data] WARNING: Alpaca load failed ... using synthetic data.",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.SYNTHETIC_DATA_FALLBACK in types


class TestGPUOOM:
    def test_detects_oom(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stderr="RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.GPU_OOM in types


class TestIdenticalConditions:
    def test_detects_from_ablation_warnings(self):
        summary = {
            "condition_summaries": {"A": {"metrics": {"m": 1}}, "B": {"metrics": {"m": 1}}},
            "best_run": {"metrics": {}},
            "ablation_warnings": [
                "ABLATION FAILURE: Conditions 'A' and 'B' produce identical outputs across all 1 metrics."
            ],
        }
        diag = diagnose_experiment(experiment_summary=summary)
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.IDENTICAL_CONDITIONS in types


class TestCodeCrash:
    def test_detects_traceback(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stderr=(
                "Traceback (most recent call last):\n"
                "  File 'main.py', line 42, in main\n"
                "    result = train(model)\n"
                "TypeError: train() missing argument 'data'\n"
            ),
        )
        types = {d.type for d in diag.deficiencies}
        assert DeficiencyType.CODE_CRASH in types


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------


class TestQualityAssessment:
    def test_full_paper_mode(self):
        summary = {
            "condition_summaries": {
                "A": {"metrics": {"metric": 80.0}},
                "B": {"metrics": {"metric": 85.0}},
                "C": {"metrics": {"metric": 90.0}},
            },
            "best_run": {
                "metrics": {
                    "A/0/m": 80.0, "A/1/m": 81.0,
                    "B/0/m": 85.0, "B/1/m": 86.0,
                    "C/0/m": 90.0, "C/1/m": 91.0,
                },
            },
        }
        qa = assess_experiment_quality(summary)
        assert qa.mode == PaperMode.FULL_PAPER
        assert qa.sufficient

    def test_preliminary_study_mode(self):
        summary = {
            "condition_summaries": {
                "A": {"metrics": {"metric": 80.0}},
                "B": {"metrics": {"metric": 85.0}},
            },
            "best_run": {"metrics": {"A/0/m": 80.0, "B/0/m": 85.0}},
        }
        qa = assess_experiment_quality(summary)
        assert qa.mode == PaperMode.PRELIMINARY_STUDY
        assert not qa.sufficient

    def test_technical_report_no_conditions(self):
        summary = {
            "condition_summaries": {},
            "best_run": {"metrics": {}},
        }
        qa = assess_experiment_quality(summary)
        assert qa.mode == PaperMode.TECHNICAL_REPORT
        assert not qa.sufficient

    def test_technical_report_synthetic_data(self):
        summary = {
            "condition_summaries": {"A": {"metrics": {"metric": 80.0}}},
            "best_run": {"metrics": {}, "stdout": "using synthetic data"},
        }
        qa = assess_experiment_quality(summary)
        assert qa.mode == PaperMode.TECHNICAL_REPORT


# ---------------------------------------------------------------------------
# Repair prompt generation
# ---------------------------------------------------------------------------


class TestRepairPrompt:
    def test_generates_prompt(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stderr="ModuleNotFoundError: No module named 'special_lib'",
        )
        prompt = diag.to_repair_prompt()
        assert "special_lib" in prompt
        assert "DIAGNOSIS" in prompt
        assert "CRITICAL" in prompt

    def test_serialization(self):
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {"A": {"metrics": {"m": 1}}}, "best_run": {"metrics": {}}},
        )
        d = diag.to_dict()
        assert isinstance(d, dict)
        assert "deficiencies" in d
        assert "conditions_completed" in d


# ---------------------------------------------------------------------------
# Integration — real artifacts
# ---------------------------------------------------------------------------


class TestRealArtifacts:
    def _load(self, run_id: str) -> dict:
        pattern = f"rc-*-{run_id}"
        matches = sorted(ARTIFACTS.glob(pattern))
        if not matches:
            pytest.skip(f"Artifact {run_id} not found")
        base = matches[0]
        summary_path = base / "stage-14" / "experiment_summary.json"
        if not summary_path.exists():
            pytest.skip(f"No experiment_summary for {run_id}")
        return json.loads(summary_path.read_text())

    def test_run_e57360_diagnosis(self):
        """Run 38 — 3/8 conditions completed, Box2D missing."""
        summary = self._load("e57360")
        qa = assess_experiment_quality(summary)
        # Should identify issues and NOT rate as full_paper
        assert qa.mode != PaperMode.FULL_PAPER or len(qa.deficiencies) > 0

    def test_run_8b4a1b_diagnosis(self):
        """Run 8b4a1b — all NaN, permission errors."""
        summary = self._load("8b4a1b")
        qa = assess_experiment_quality(summary)
        # Should be technical_report or preliminary_study at best
        assert qa.mode in (PaperMode.TECHNICAL_REPORT, PaperMode.PRELIMINARY_STUDY)

class TestDatasetNotFoundError:
    """BUG-203: HuggingFace DatasetNotFoundError should be caught."""

    def test_detects_hf_dataset_not_found(self):
        stderr = (
            "Traceback (most recent call last):\n"
            "  File \"/workspace/setup.py\", line 11, in main\n"
            "datasets.exceptions.DatasetNotFoundError: "
            "Dataset 'cifar10_corrupted' doesn't exist on the Hub or cannot be accessed.\n"
        )
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stderr=stderr,
        )
        ds_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.DATASET_UNAVAILABLE]
        assert len(ds_issues) >= 1
        assert "HuggingFace" in ds_issues[0].description
        # Should NOT also appear as a generic CODE_CRASH
        crashes = [d for d in diag.deficiencies if d.type == DeficiencyType.CODE_CRASH]
        assert not any("DatasetNotFoundError" in c.description for c in crashes)

    def test_suggested_fix_mentions_precached(self):
        stderr = (
            "DatasetNotFoundError: Dataset 'imagenet_v2' "
            "doesn't exist on the Hub or cannot be accessed.\n"
        )
        diag = diagnose_experiment(
            experiment_summary={"condition_summaries": {}, "best_run": {"metrics": {}}},
            stderr=stderr,
        )
        ds_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.DATASET_UNAVAILABLE]
        assert any("/opt/datasets" in d.suggested_fix for d in ds_issues)


class TestNearRandomAccuracy:
    """BUG-204: Detect near-random accuracy in experiment results."""

    def test_detects_near_random_cifar10(self):
        """8.91% accuracy on CIFAR-10 should be flagged."""
        diag = diagnose_experiment(
            experiment_summary={
                "condition_summaries": {"cond_a": {"metrics": {"top1_accuracy": 8.91}}},
                "metrics_summary": {"top1_accuracy": {"min": 8.42, "max": 8.91, "mean": 8.67}},
                "best_run": {"metrics": {}},
            },
        )
        hp_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.HYPERPARAMETER_ISSUE]
        assert any("random chance" in d.description for d in hp_issues)

    def test_normal_accuracy_not_flagged(self):
        """73% accuracy should NOT be flagged."""
        diag = diagnose_experiment(
            experiment_summary={
                "condition_summaries": {"baseline": {"metrics": {"accuracy": 73.07}}},
                "metrics_summary": {"accuracy": {"min": 68.0, "max": 73.07, "mean": 70.5}},
                "best_run": {"metrics": {}},
            },
        )
        hp_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.HYPERPARAMETER_ISSUE]
        assert not any("random chance" in d.description for d in hp_issues)

    def test_zero_accuracy_not_flagged(self):
        """0% accuracy (no data) should NOT be flagged by this check."""
        diag = diagnose_experiment(
            experiment_summary={
                "condition_summaries": {},
                "metrics_summary": {},
                "best_run": {"metrics": {}},
            },
        )
        hp_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.HYPERPARAMETER_ISSUE]
        assert not any("random chance" in d.description for d in hp_issues)

    def test_fraction_accuracy_not_flagged_when_high(self):
        """FIX#1: a perfect fractional accuracy (1.0 == 100%) must NOT be flagged.

        Reproduces the real arc-test bug: workspace agents store accuracy as a
        fraction, so 1.0 was misread as '1.0%' and flagged near-random.
        """
        diag = diagnose_experiment(
            experiment_summary={
                "condition_summaries": {"cond_a": {"metrics": {"accuracy": 0.999}}},
                "metrics_summary": {"accuracy": {"min": 0.99, "max": 1.0, "mean": 0.999}},
                "best_run": {"metrics": {}},
            },
        )
        hp_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.HYPERPARAMETER_ISSUE]
        assert not any("random chance" in d.description for d in hp_issues)

    def test_fraction_accuracy_flagged_when_near_random(self):
        """FIX#1: a genuinely low fractional accuracy (0.089 == 8.9%) IS flagged."""
        diag = diagnose_experiment(
            experiment_summary={
                "condition_summaries": {"cond_a": {"metrics": {"accuracy": 0.089}}},
                "metrics_summary": {"accuracy": {"min": 0.08, "max": 0.089, "mean": 0.085}},
                "best_run": {"metrics": {}},
            },
        )
        hp_issues = [d for d in diag.deficiencies if d.type == DeficiencyType.HYPERPARAMETER_ISSUE]
        assert any("random chance" in d.description for d in hp_issues)

    def test_percent_suffix_key_treated_as_percent(self):
        """FIX#1: a '*_percent' key is always interpreted as percent, never rescaled."""
        diag_high = diagnose_experiment(
            experiment_summary={
                "condition_summaries": {"cond_a": {"metrics": {"accuracy_percent": 99.0}}},
                "metrics_summary": {"accuracy_percent": {"min": 98.0, "max": 99.0, "mean": 98.5}},
                "best_run": {"metrics": {}},
            },
        )
        hp_high = [d for d in diag_high.deficiencies if d.type == DeficiencyType.HYPERPARAMETER_ISSUE]
        assert not any("random chance" in d.description for d in hp_high)

        diag_low = diagnose_experiment(
            experiment_summary={
                "condition_summaries": {"cond_a": {"metrics": {"accuracy_percent": 8.0}}},
                "metrics_summary": {"accuracy_percent": {"min": 7.0, "max": 8.0, "mean": 7.5}},
                "best_run": {"metrics": {}},
            },
        )
        hp_low = [d for d in diag_low.deficiencies if d.type == DeficiencyType.HYPERPARAMETER_ISSUE]
        assert any("random chance" in d.description for d in hp_low)


class TestRealArtifactsContinued(TestRealArtifacts):
    """Continuation of real artifact tests (after TestDatasetNotFoundError)."""

    def test_run_acbdfa_diagnosis(self):
        """Run acbdfa — 2 architectures, S4D nearly random."""
        summary = self._load("acbdfa")
        diag = diagnose_experiment(
            experiment_summary=summary,
            stdout=summary.get("best_run", {}).get("stdout", ""),
            stderr=summary.get("best_run", {}).get("stderr", ""),
        )
        assert diag.completion_rate > 0


class TestConditionCountingWorkspace:
    """FIX#3: workspace-mode condition counting from flat-float condition_summaries."""

    @staticmethod
    def _flat_condition(accuracy: float, n_seeds: int = 5) -> dict:
        return {
            "status": "completed",
            "n_runs": n_seeds,
            "n_seeds": n_seeds,
            "metrics": {
                "accuracy": accuracy,
                "accuracy_mean": accuracy,
                "primary_metric": accuracy,
                "primary_metric_mean": accuracy,
            },
        }

    def test_completed_conditions_from_flat_aggregate_summaries(self):
        summary = {
            "condition_summaries": {
                "cond_a": self._flat_condition(0.999),
                "cond_b": self._flat_condition(0.985),
            },
            "metrics_summary": {"accuracy": {"max": 1.0, "mean": 0.99}},
            "best_run": {"metrics": {}},
        }
        diag = diagnose_experiment(experiment_summary=summary)
        assert len(diag.conditions_completed) == 2
        assert diag.completion_rate == 1.0
        assert not any(
            d.type == DeficiencyType.NO_CONDITIONS_COMPLETED for d in diag.deficiencies
        )

    def test_no_conditions_guard_suppressed_when_top_level_numeric_metric(self):
        """Empty condition_summaries but a usable top-level metric => no phantom critical."""
        summary = {
            "condition_summaries": {},
            "metrics_summary": {"accuracy": {"max": 0.99, "mean": 0.98}},
            "best_run": {"metrics": {"accuracy": 0.99}},
        }
        diag = diagnose_experiment(experiment_summary=summary)
        assert not any(
            d.type == DeficiencyType.NO_CONDITIONS_COMPLETED for d in diag.deficiencies
        )

    def test_no_conditions_guard_fires_when_truly_empty(self):
        summary = {
            "condition_summaries": {},
            "metrics_summary": {},
            "best_run": {"metrics": {}},
        }
        diag = diagnose_experiment(experiment_summary=summary)
        assert any(
            d.type == DeficiencyType.NO_CONDITIONS_COMPLETED for d in diag.deficiencies
        )

    def test_insufficient_seeds_prefers_n_seeds(self):
        # n_seeds=5 => not flagged
        ok = {
            "condition_summaries": {"cond_a": self._flat_condition(0.99, n_seeds=5)},
            "metrics_summary": {"accuracy": {"max": 0.99}},
            "best_run": {"metrics": {}},
        }
        diag_ok = diagnose_experiment(experiment_summary=ok)
        assert not any(
            d.type == DeficiencyType.INSUFFICIENT_SEEDS for d in diag_ok.deficiencies
        )
        # n_seeds=1 => flagged
        bad = {
            "condition_summaries": {"cond_a": self._flat_condition(0.99, n_seeds=1)},
            "metrics_summary": {"accuracy": {"max": 0.99}},
            "best_run": {"metrics": {}},
        }
        diag_bad = diagnose_experiment(experiment_summary=bad)
        assert any(
            d.type == DeficiencyType.INSUFFICIENT_SEEDS for d in diag_bad.deficiencies
        )

    def test_identical_conditions_workspace_aware(self):
        # distinct primary metrics => not flagged
        distinct = {
            "condition_summaries": {
                "cond_a": self._flat_condition(0.99),
                "cond_b": self._flat_condition(0.95),
            },
            "metrics_summary": {"accuracy": {"max": 0.99}},
            "best_run": {"metrics": {}},
        }
        diag_distinct = diagnose_experiment(experiment_summary=distinct)
        assert not any(
            d.type == DeficiencyType.IDENTICAL_CONDITIONS for d in diag_distinct.deficiencies
        )
        # identical primary metrics across >=2 conditions => flagged
        identical = {
            "condition_summaries": {
                "cond_a": self._flat_condition(0.90),
                "cond_b": self._flat_condition(0.90),
            },
            "metrics_summary": {"accuracy": {"max": 0.90}},
            "best_run": {"metrics": {}},
        }
        diag_identical = diagnose_experiment(experiment_summary=identical)
        assert any(
            d.type == DeficiencyType.IDENTICAL_CONDITIONS for d in diag_identical.deficiencies
        )

    def test_full_paper_mode_uses_explicit_n_seeds(self):
        """FIX#3: a workspace run with >=3 conditions x >=2 seeds reaches FULL_PAPER
        (sufficient=True), even though there are no legacy per-seed metric keys."""
        cs = {
            f"cond_{i}": self._flat_condition(0.90 + i * 0.01, n_seeds=5)
            for i in range(3)
        }
        summary = {
            "condition_summaries": cs,
            "metrics_summary": {"accuracy": {"max": 0.92, "mean": 0.91}},
            "best_run": {"metrics": {}},
        }
        q = assess_experiment_quality(summary)
        assert q.mode == PaperMode.FULL_PAPER
        assert q.sufficient is True
