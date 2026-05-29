"""Parse Feishu/Lark text replies into HITL actions."""

from __future__ import annotations

from dataclasses import dataclass

from researchclaw.hitl.intervention import HumanAction, HumanInput


@dataclass(frozen=True)
class ParsedReply:
    action: HumanAction
    message: str = ""
    guidance: str = ""
    rollback_to_stage: int | None = None

    def to_human_input(self) -> HumanInput:
        return HumanInput(
            action=self.action,
            message=self.message,
            guidance=self.guidance,
            rollback_to_stage=self.rollback_to_stage,
        )


_ACTION_HEADS: dict[str, HumanAction] = {
    "approve": HumanAction.APPROVE,
    "ok": HumanAction.APPROVE,
    "lgtm": HumanAction.APPROVE,
    "yes": HumanAction.APPROVE,
    "同意": HumanAction.APPROVE,
    "通过": HumanAction.APPROVE,
    "reject": HumanAction.REJECT,
    "no": HumanAction.REJECT,
    "拒绝": HumanAction.REJECT,
    "驳回": HumanAction.REJECT,
    "abort": HumanAction.ABORT,
    "stop": HumanAction.ABORT,
    "cancel": HumanAction.ABORT,
    "终止": HumanAction.ABORT,
    "取消": HumanAction.ABORT,
    "skip": HumanAction.SKIP,
    "跳过": HumanAction.SKIP,
    "guidance": HumanAction.INJECT,
    "guide": HumanAction.INJECT,
    "inject": HumanAction.INJECT,
    "指导": HumanAction.INJECT,
    "建议": HumanAction.INJECT,
    "edit": HumanAction.EDIT,
    "编辑": HumanAction.EDIT,
    "collaborate": HumanAction.COLLABORATE,
    "collab": HumanAction.COLLABORATE,
    "协作": HumanAction.COLLABORATE,
    "resume": HumanAction.RESUME,
    "继续": HumanAction.RESUME,
    "take_over": HumanAction.TAKE_OVER,
    "takeover": HumanAction.TAKE_OVER,
    "接管": HumanAction.TAKE_OVER,
    "rollback": HumanAction.ROLLBACK,
    "回滚": HumanAction.ROLLBACK,
}


def parse_reply(text: str) -> ParsedReply | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    head, rest = _split_head_rest(raw)
    action = _ACTION_HEADS.get(head.lower())
    if action is None:
        return None

    if action is HumanAction.INJECT:
        if not rest:
            return None
        return ParsedReply(action=action, guidance=rest)

    if action is HumanAction.ROLLBACK:
        if rest.isdigit():
            return ParsedReply(action=action, rollback_to_stage=int(rest))
        return ParsedReply(action=action, message=rest)

    return ParsedReply(action=action, message=rest)


def _split_head_rest(raw: str) -> tuple[str, str]:
    separators = [index for index in (raw.find(":"), raw.find("：")) if index >= 0]
    if not separators:
        return raw.strip(), ""

    index = min(separators)
    return raw[:index].strip(), raw[index + 1 :].strip()
