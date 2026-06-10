"""Feature-flag helpers for per-hypothesis validation mode."""

from __future__ import annotations

from typing import Any


def per_hypothesis_validation_enabled(config: Any) -> bool:
    validation = getattr(config, "hypothesis_validation", None)
    return bool(getattr(validation, "enabled", False))
