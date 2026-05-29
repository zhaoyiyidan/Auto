from __future__ import annotations

import pytest

from researchclaw.hitl.intervention import HumanAction, HumanInput
from researchclaw.notify.lark_reply import ParsedReply, parse_reply


@pytest.mark.parametrize(
    "text",
    ["approve", "ok", "lgtm", "yes", "同意", "通过"],
)
def test_approve_synonyms(text: str):
    assert parse_reply(text) == ParsedReply(action=HumanAction.APPROVE)


def test_reject_with_reason():
    parsed = parse_reply("reject: needs stronger baselines")

    assert parsed == ParsedReply(
        action=HumanAction.REJECT,
        message="needs stronger baselines",
    )


def test_reject_without_reason():
    assert parse_reply("reject") == ParsedReply(action=HumanAction.REJECT)


@pytest.mark.parametrize("text", ["abort", "stop", "cancel", "终止", "取消"])
def test_abort_synonyms(text: str):
    assert parse_reply(text) == ParsedReply(action=HumanAction.ABORT)


def test_skip():
    assert parse_reply("跳过") == ParsedReply(action=HumanAction.SKIP)


@pytest.mark.parametrize("head", ["guidance", "guide", "inject", "指导", "建议"])
def test_guidance_maps_to_inject(head: str):
    parsed = parse_reply(f"{head}: compare against the ablation table")

    assert parsed == ParsedReply(
        action=HumanAction.INJECT,
        guidance="compare against the ablation table",
    )


@pytest.mark.parametrize("text", ["guidance", "inject:", "建议：  "])
def test_inject_empty_guidance_returns_none(text: str):
    assert parse_reply(text) is None


def test_rollback_numeric_sets_stage():
    assert parse_reply("rollback: 7") == ParsedReply(
        action=HumanAction.ROLLBACK,
        rollback_to_stage=7,
    )


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("edit: fix typos", HumanAction.EDIT),
        ("编辑：补充图注", HumanAction.EDIT),
        ("collaborate", HumanAction.COLLABORATE),
        ("collab", HumanAction.COLLABORATE),
        ("协作", HumanAction.COLLABORATE),
        ("resume", HumanAction.RESUME),
        ("继续", HumanAction.RESUME),
        ("take_over", HumanAction.TAKE_OVER),
        ("takeover", HumanAction.TAKE_OVER),
        ("接管", HumanAction.TAKE_OVER),
    ],
)
def test_edit_collaborate_resume_takeover_map(text: str, action: HumanAction):
    parsed = parse_reply(text)

    assert parsed is not None
    assert parsed.action is action


def test_case_insensitive():
    assert parse_reply("LGTM: Ship it") == ParsedReply(
        action=HumanAction.APPROVE,
        message="Ship it",
    )


def test_chinese_colon_separator():
    assert parse_reply("拒绝：还缺少消融实验") == ParsedReply(
        action=HumanAction.REJECT,
        message="还缺少消融实验",
    )


def test_unknown_returns_none():
    assert parse_reply("looks fine to me") is None


def test_whitespace_stripped():
    assert parse_reply("  approve :  Looks good  ") == ParsedReply(
        action=HumanAction.APPROVE,
        message="Looks good",
    )


def test_rollback_non_numeric_keeps_message():
    assert parse_reply("rollback: before analysis") == ParsedReply(
        action=HumanAction.ROLLBACK,
        message="before analysis",
    )


def test_to_human_input_maps_fields():
    parsed = ParsedReply(
        action=HumanAction.INJECT,
        message="reviewed",
        guidance="try a smaller model",
        rollback_to_stage=3,
    )

    human_input = parsed.to_human_input()

    assert isinstance(human_input, HumanInput)
    assert human_input.action is HumanAction.INJECT
    assert human_input.message == "reviewed"
    assert human_input.guidance == "try a smaller model"
    assert human_input.rollback_to_stage == 3
