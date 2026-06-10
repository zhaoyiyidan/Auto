"""Tests for prompt override precedence."""

from __future__ import annotations

import textwrap
from pathlib import Path

from researchclaw.prompts import PromptManager
from researchclaw.prompts.manager import PRECEDENCE_DOC


def _topic_vars() -> dict[str, str]:
    return {
        "topic": "RL",
        "domains": "ml",
        "project_name": "test",
        "quality_threshold": "4.0",
    }


def test_precedence_bank_lt_yaml(tmp_path: Path) -> None:
    override_file = tmp_path / "prompts.yaml"
    override_file.write_text(
        textwrap.dedent(
            """\
            stages:
              topic_init:
                system: YAML system
                user: YAML user for {topic}
            """
        ),
        encoding="utf-8",
    )

    sp = PromptManager(override_file).for_stage("topic_init", **_topic_vars())

    assert sp.system == "YAML system"
    assert sp.user == "YAML user for RL"


def test_precedence_yaml_lt_extra(tmp_path: Path) -> None:
    override_file = tmp_path / "prompts.yaml"
    override_file.write_text(
        textwrap.dedent(
            """\
            stages:
              topic_init:
                user: YAML user for {topic}
            """
        ),
        encoding="utf-8",
    )
    pm = PromptManager(override_file, extra_prompts={"topic_init": "EXTRA guidance"})

    sp = pm.for_stage("topic_init", **_topic_vars())

    assert sp.user.index("YAML user for RL") < sp.user.index("EXTRA guidance")


def test_overlay_and_extra_ordering_is_stable() -> None:
    pm = PromptManager(extra_prompts={"topic_init": "EXTRA guidance"})

    sp = pm.for_stage(
        "topic_init",
        evolution_overlay="EVOLUTION overlay",
        **_topic_vars(),
    )

    assert PRECEDENCE_DOC
    assert "domain bank < YAML custom_file < extra_prompts < evolution_overlay" in PRECEDENCE_DOC
    assert sp.user.index("EXTRA guidance") < sp.user.index("EVOLUTION overlay")
