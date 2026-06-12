from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.contracts import CONTRACTS
from researchclaw.pipeline.stages import Stage


class FakeLLM:
    def __init__(self, responses: str | list[str]) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, "kwargs": kwargs})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        from researchclaw.llm.client import LLMResponse

        return LLMResponse(content=self.responses[index], model="fake")


def _config(tmp_path: Path) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "stage15-decision-review-test", "mode": "docs-first"},
            "research": {"topic": "decision review transparency"},
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
            "experiment": {},
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _write_analysis(run_dir: Path) -> None:
    stage14 = run_dir / "stage-14"
    stage14.mkdir(parents=True, exist_ok=True)
    (stage14 / "analysis.md").write_text(
        "# Analysis\nEvidence summary with baseline, seed runs, and metric deltas.",
        encoding="utf-8",
    )
    (stage14 / "experiment_summary.json").write_text(
        json.dumps({"primary_metric": "score", "best_metric": 0.8}),
        encoding="utf-8",
    )


def _write_plan(run_dir: Path) -> None:
    stage9 = run_dir / "stage-09"
    stage9.mkdir(parents=True, exist_ok=True)
    (stage9 / "plan.md").write_text(
        "# Experiment Plan\n\n"
        "## Hypotheses\nMetric should pass.\n\n"
        "## Baselines\nCompare against a baseline run.\n\n"
        "## Ablations\nRemove the main treatment.\n\n"
        "## Metrics\nUse score.\n\n"
        "## Decision Criteria\nProceed when score clears the planned threshold.\n\n"
        "## Expected Outputs\noutputs/results.json\n",
        encoding="utf-8",
    )
    (stage9 / "expected_outputs.json").write_text(
        json.dumps(
            {
                "schema_version": "researchclaw.expected_outputs.v1",
                "outputs": ["outputs/results.json"],
            }
        ),
        encoding="utf-8",
    )


def _run_decision(tmp_path: Path, run_dir: Path, llm=None):
    from researchclaw.pipeline.stage_impls._analysis import _execute_research_decision

    stage_dir = run_dir / "stage-15"
    stage_dir.mkdir(parents=True, exist_ok=True)
    return _execute_research_decision(
        stage_dir,
        run_dir,
        _config(tmp_path),
        AdapterBundle(),
        llm=llm,
    )


def _decision_md(decision: str) -> str:
    return (
        "# Research Decision\n\n"
        "## Decision\n"
        f"{decision}\n\n"
        "## Justification\n"
        "The baseline, seed runs, and primary metric all support this route.\n"
    )


def _review_text(decision: str, sentinel: str = "agent-authored-review") -> str:
    return (
        "## Decision Reviewed\n"
        f"{decision}\n\n"
        "## Short Rationale\n"
        f"{sentinel}: {decision}\n\n"
        "## Evidence Considered\n"
        "- Stage 14 analysis.\n\n"
        "## Criteria Assessment\n"
        "- Metrics and limitations were checked.\n\n"
        "## Why This Decision\n"
        "- It matches the finalized route.\n\n"
        "## Why Not The Alternatives\n"
        "- Alternatives were less consistent with the evidence.\n\n"
        "## Caveats For Reviewer\n"
        "- Check the supporting analysis.\n\n"
        "## Recommended Human Review Focus\n"
        "- Confirm decision/evidence consistency.\n"
    )


@pytest.mark.parametrize("decision", ["PROCEED", "PIVOT", "EXTEND"])
def test_standard_path_writes_agent_authored_decision_review_for_each_decision(
    tmp_path: Path,
    decision: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_analysis(run_dir)
    review_text = _review_text(decision, sentinel=f"sentinel-{decision.lower()}")
    llm = FakeLLM([_decision_md(decision), review_text])

    result = _run_decision(tmp_path, run_dir, llm=llm)

    path = run_dir / "stage-15" / "decision_review.md"
    assert path.is_file()
    assert path.read_text(encoding="utf-8").strip() == review_text.strip()
    assert decision in path.read_text(encoding="utf-8")
    assert result.decision == decision.lower()


def test_decision_review_prompt_is_anchored_to_authoritative_decision(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_analysis(run_dir)
    llm = FakeLLM([_decision_md("PIVOT"), _review_text("PIVOT")])

    _run_decision(tmp_path, run_dir, llm=llm)

    assert len(llm.calls) == 2
    review_call = llm.calls[1]
    messages = review_call["messages"]
    assert isinstance(messages, list)
    rendered_user = str(messages[0]["content"])
    rendered_system = str(review_call["kwargs"].get("system", ""))
    rendered_prompt = rendered_system + "\n" + rendered_user
    assert "PIVOT" in rendered_prompt
    assert _decision_md("PIVOT") in rendered_prompt


def test_standard_path_offline_writes_transparent_unavailable_review(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_analysis(run_dir)

    result = _run_decision(tmp_path, run_dir, llm=None)

    content = (run_dir / "stage-15" / "decision_review.md").read_text(encoding="utf-8")
    assert content.strip()
    assert result.decision.upper() in content
    assert "agent rationale unavailable" in content.lower()
    assert "## Short Rationale\n" not in content


def test_plan_path_offline_writes_decision_review(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_analysis(run_dir)
    _write_plan(run_dir)

    result = _run_decision(tmp_path, run_dir, llm=None)

    content = (run_dir / "stage-15" / "decision_review.md").read_text(encoding="utf-8")
    assert content.strip()
    assert result.decision.upper() in content
    assert "agent rationale unavailable" in content.lower()


def test_result_artifacts_and_contract_include_decision_review(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_analysis(run_dir)
    llm = FakeLLM([_decision_md("PROCEED"), _review_text("PROCEED")])

    result = _run_decision(tmp_path, run_dir, llm=llm)

    assert result.artifacts[0] == "decision.md"
    assert "decision_review.md" in result.artifacts
    output_files = CONTRACTS[Stage.RESEARCH_DECISION].output_files
    assert output_files[0] == "decision.md"
    assert "decision_review.md" in output_files


def test_agent_requirements_path_writes_agent_authored_decision_review(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_analysis(run_dir)
    (run_dir / "stage-09").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage-09" / "requirements.json").write_text(
        json.dumps([{"id": "req1", "description": "must pass", "must_pass": True}]),
        encoding="utf-8",
    )
    requirements_verdict = json.dumps(
        {
            "per_requirement": [
                {
                    "id": "req1",
                    "must_pass": True,
                    "met": True,
                    "evidence": "score present",
                    "missing": "",
                }
            ],
            "verdict": "proceed",
            "delta_feedback": "",
        }
    )
    review_text = _review_text("PROCEED", sentinel="requirements-review-sentinel")
    llm = FakeLLM([requirements_verdict, review_text])

    result = _run_decision(tmp_path, run_dir, llm=llm)

    assert result.decision == "proceed"
    assert result.artifacts[0] == "decision.md"
    assert "decision_review.md" in result.artifacts
    assert (
        run_dir / "stage-15" / "decision_review.md"
    ).read_text(encoding="utf-8").strip() == review_text.strip()


def test_write_decision_review_helper_with_llm_writes_agent_text(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.stage_impls._analysis import _write_decision_review

    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-15"
    _write_analysis(run_dir)
    review_text = _review_text("EXTEND", sentinel="direct-helper-sentinel")
    llm = FakeLLM(review_text)

    _write_decision_review(
        stage_dir=stage_dir,
        run_dir=run_dir,
        decision="extend",
        decision_md=_decision_md("EXTEND"),
        llm=llm,
    )

    assert (stage_dir / "decision_review.md").read_text(encoding="utf-8").strip() == (
        review_text.strip()
    )
    assert llm.calls


def test_write_decision_review_helper_without_llm_writes_unavailable_note(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.stage_impls._analysis import (
        _decision_review_unavailable,
        _write_decision_review,
    )

    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-15"
    _write_analysis(run_dir)

    text = _decision_review_unavailable("pivot")
    assert text.strip()
    assert "PIVOT" in text
    assert "agent rationale unavailable" in text.lower()

    _write_decision_review(
        stage_dir=stage_dir,
        run_dir=run_dir,
        decision="pivot",
        decision_md=_decision_md("PIVOT"),
        llm=None,
    )

    content = (stage_dir / "decision_review.md").read_text(encoding="utf-8")
    assert content.strip()
    assert "PIVOT" in content
    assert "agent rationale unavailable" in content.lower()
