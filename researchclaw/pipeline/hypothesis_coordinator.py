"""Coordinator for per-hypothesis validation."""

from __future__ import annotations

from pathlib import Path

from researchclaw.experiment.protocol import parse_hypotheses_md
from researchclaw.pipeline.hypothesis_store import HypothesisNode, HypothesisStore


class HypothesisValidationCoordinator:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.store = HypothesisStore(self.run_dir)

    def split_stage8_hypotheses(
        self,
        hypotheses_md: str,
        *,
        created_at: str | None = None,
    ) -> list[HypothesisNode]:
        nodes: list[HypothesisNode] = []
        seen_hashes: set[str] = set()
        for spec in parse_hypotheses_md(hypotheses_md):
            candidate = HypothesisNode(
                id="candidate",
                statement=spec.statement,
                prediction=spec.prediction,
                falsification=spec.falsification,
                rationale=spec.rationale,
                baselines=spec.baselines,
                source="stage8_batch",
                parent_id=None,
                created_at=created_at or "",
            )
            if candidate.hypothesis_hash in seen_hashes:
                continue
            seen_hashes.add(candidate.hypothesis_hash)
            nodes.append(
                self.store.create_node(
                    statement=spec.statement,
                    prediction=spec.prediction,
                    falsification=spec.falsification,
                    rationale=spec.rationale,
                    baselines=spec.baselines,
                    source="stage8_batch",
                    parent_id=None,
                    created_at=created_at,
                )
            )
        return nodes
