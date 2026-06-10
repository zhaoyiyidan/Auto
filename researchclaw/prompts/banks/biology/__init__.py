"""Assembled split prompt bank."""

from __future__ import annotations

from researchclaw.prompts.banks.ml import STAGES as _ML_STAGES
from researchclaw.prompts.metadata import apply_stage_metadata

from researchclaw.prompts.banks.biology import reasoning
from researchclaw.prompts.banks.biology import experiment
from researchclaw.prompts.banks.biology import analysis
from researchclaw.prompts.banks.biology import writing
from researchclaw.prompts.banks.biology.debate import (
    DEBATE_ROLES_ANALYSIS,
    DEBATE_ROLES_HYPOTHESIS,
)


STAGES = {k: dict(v) for k, v in _ML_STAGES.items()}
STAGES.update(reasoning.STAGES)
STAGES.update(experiment.STAGES)
STAGES.update(analysis.STAGES)
STAGES.update(writing.STAGES)
apply_stage_metadata(STAGES)

_DEFAULT_STAGES = STAGES
