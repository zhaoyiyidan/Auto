"""ML-domain prompt bank compatibility shim."""

from __future__ import annotations

from researchclaw.prompts.banks.ml import (
    DEBATE_ROLES_ANALYSIS,
    DEBATE_ROLES_HYPOTHESIS,
    STAGES,
    _DEFAULT_STAGES,
)

__all__ = [
    "STAGES",
    "_DEFAULT_STAGES",
    "DEBATE_ROLES_HYPOTHESIS",
    "DEBATE_ROLES_ANALYSIS",
]
