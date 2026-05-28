from __future__ import annotations

import ast
from pathlib import Path


def test_pipeline_llm_chat_calls_request_thinking_stripping() -> None:
    """Pipeline artifacts should not persist ACP/CoT reasoning blocks."""
    paths = [Path("researchclaw/pipeline/_helpers.py")]
    paths.extend(Path("researchclaw/pipeline/stage_impls").glob("*.py"))
    missing: list[str] = []

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "chat":
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id != "llm":
                continue
            if not any(kw.arg == "strip_thinking" for kw in node.keywords):
                missing.append(f"{path}:{node.lineno}")

    assert missing == []
