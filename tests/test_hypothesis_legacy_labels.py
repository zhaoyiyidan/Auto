from __future__ import annotations

from pathlib import Path


def test_legacy_hypothesis_test_modules_are_labeled() -> None:
    tests_dir = Path(__file__).resolve().parent
    legacy_modules = (
        "test_hypothesis_tree.py",
        "test_hypothesis_node_tree.py",
        "test_hypothesis_cycle_archive.py",
    )

    for module_name in legacy_modules:
        text = (tests_dir / module_name).read_text(encoding="utf-8")
        assert '"""Legacy hypothesis compatibility tests.' in text.splitlines()[:4]
