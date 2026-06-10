"""Migration tests for requirements and hypothesis verdict judge prompts."""

from __future__ import annotations

import json

from researchclaw.experiment.protocol import DecisionRule
from researchclaw.prompts import PromptManager


class RecordingLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, **kwargs})
        from researchclaw.llm.client import LLMResponse

        return LLMResponse(content=json.dumps(self.payload), model="fake")


def test_judge_catalog_entries_exist() -> None:
    pm = PromptManager()

    assert pm.sub_prompt_meta("requirements_judge").prompt_id == "requirements_judge"
    assert (
        pm.sub_prompt_meta("hypothesis_verdict_fallback").prompt_id
        == "hypothesis_verdict_fallback"
    )


def test_requirements_judge_preserves_constraints() -> None:
    from researchclaw.pipeline.requirements_judge import _build_user_prompt

    prompt = _build_user_prompt(
        [{"id": "req1", "description": "accuracy > 0.8", "must_pass": True}],
        {"best_metric": 0.9},
        {"metrics": {"accuracy": 0.9}},
    )

    assert "REQUIREMENTS to verify" in prompt
    assert "Then produce a single overall verdict" in prompt
    assert "OUTPUT" in prompt
    assert "return ONLY a single JSON object" in prompt
    assert "delta_feedback" in prompt


def test_hypothesis_fallback_preserves_constraints() -> None:
    from researchclaw.pipeline.hypothesis_judge import evaluate_decision_rule

    llm = RecordingLLM({"verdict": "supported", "rationale": "summary supports H1"})
    verdict = evaluate_decision_rule(
        DecisionRule(hypothesis_id="H1", metric="missing"),
        {"analysis": "The observed trace supports H1."},
        llm=llm,
    )

    assert verdict == "supported"
    call = llm.calls[0]
    content = call["messages"][0]["content"]  # type: ignore[index]
    assert "Return only JSON" in content
    assert "`verdict`" in content
    assert "RULE" in content
    assert "SUMMARY" in content
    assert "strict scientific hypothesis judge" in str(call["system"])


def test_requirements_judge_sources_from_catalog(monkeypatch) -> None:
    import researchclaw.pipeline.requirements_judge as requirements_mod

    pm = PromptManager()
    pm._sub_prompts["requirements_judge"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL REQUIREMENTS SYSTEM",
        "user": "{requirements_json}\n{summary_excerpt}\n{results_excerpt}",
    }
    monkeypatch.setattr(requirements_mod, "PromptManager", lambda: pm, raising=False)

    llm = RecordingLLM(
        {
            "per_requirement": [
                {
                    "id": "req1",
                    "must_pass": True,
                    "met": True,
                    "evidence": "accuracy 0.9",
                    "missing": "",
                }
            ],
            "verdict": "proceed",
            "delta_feedback": "",
        }
    )
    verdict = requirements_mod.judge_requirements(
        [{"id": "req1", "description": "accuracy > 0.8", "must_pass": True}],
        {"best_metric": 0.9},
        {"metrics": {"accuracy": 0.9}},
        llm,
    )

    assert verdict["verdict"] == "proceed"
    assert llm.calls[0]["system"] == "SENTINEL REQUIREMENTS SYSTEM"


def test_hypothesis_fallback_sources_from_catalog(monkeypatch) -> None:
    import researchclaw.pipeline.hypothesis_judge as hypothesis_mod

    pm = PromptManager()
    pm._sub_prompts["hypothesis_verdict_fallback"] = {  # type: ignore[attr-defined]
        "system": "SENTINEL HYPOTHESIS SYSTEM",
        "user": "{rule_json}\n{summary_json}",
        "json_mode": True,
    }
    monkeypatch.setattr(hypothesis_mod, "PromptManager", lambda: pm, raising=False)

    llm = RecordingLLM({"verdict": "supported", "rationale": "summary supports H1"})
    verdict = hypothesis_mod.evaluate_decision_rule(
        DecisionRule(hypothesis_id="H1", metric="missing"),
        {"analysis": "The observed trace supports H1."},
        llm=llm,
    )

    assert verdict == "supported"
    assert llm.calls[0]["system"] == "SENTINEL HYPOTHESIS SYSTEM"
