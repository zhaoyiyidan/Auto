"""Assembled split prompt bank."""

from __future__ import annotations

from researchclaw.prompts.metadata import apply_stage_metadata

from researchclaw.prompts.banks.hep import scoping
from researchclaw.prompts.banks.hep import literature
from researchclaw.prompts.banks.hep import reasoning
from researchclaw.prompts.banks.hep import experiment
from researchclaw.prompts.banks.hep import analysis
from researchclaw.prompts.banks.hep import writing
from researchclaw.prompts.banks.hep import publish
from researchclaw.prompts.banks.hep.debate import (
    DEBATE_ROLES_ANALYSIS,
    DEBATE_ROLES_HYPOTHESIS,
)


STAGES = {}
STAGES.update(scoping.STAGES)
STAGES.update(literature.STAGES)
STAGES.update(reasoning.STAGES)
STAGES.update(experiment.STAGES)
STAGES.update(analysis.STAGES)
STAGES.update(writing.STAGES)
STAGES.update(publish.STAGES)
apply_stage_metadata(STAGES)

_DEFAULT_STAGES = STAGES
