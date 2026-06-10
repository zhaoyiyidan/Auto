"""Coordinator for per-hypothesis validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from researchclaw.experiment.protocol import parse_hypotheses_md
from researchclaw.pipeline.hypothesis_store import (
    HypothesisNode,
    HypothesisStore,
    ValidationAttempt,
)


def _result_field(result: Any, field: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(field, default)
    return getattr(result, field, default)


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

    def _branch_run_dir(self, node_id: str, attempt_name: str) -> Path:
        return self.run_dir / "hypothesis_branches" / node_id / attempt_name

    def validate_branch(
        self,
        node: HypothesisNode,
        attempt: ValidationAttempt,
        config: Any,
        adapters: Any,
    ) -> Any:
        raise NotImplementedError(
            "validate_branch is introduced by the branch-isolation phase"
        )

    def split_and_validate_sequential(
        self,
        hypotheses_md: str,
        *,
        config: Any,
        adapters: Any,
        created_at: str | None = None,
    ) -> list[ValidationAttempt]:
        nodes = self.split_stage8_hypotheses(
            hypotheses_md,
            created_at=created_at,
        )
        attempts: list[ValidationAttempt] = []
        for node in nodes:
            attempt = self.store.add_attempt(
                node_id=node.id,
                branch_run_dir=str(self._branch_run_dir(node.id, "attempt-001")),
                created_at=created_at,
            )
            self.store.set_node_status(
                node.id,
                "validating",
                created_at=created_at,
            )
            try:
                result = self.validate_branch(node, attempt, config, adapters)
            except Exception as exc:
                attempts.append(
                    self.store.update_attempt(
                        attempt.attempt_id,
                        status="failed",
                        error=str(exc),
                        finished_at=created_at,
                    )
                )
                continue

            decision = str(
                _result_field(result, "decision", "inconclusive")
                or "inconclusive"
            ).lower()
            updated = self.store.update_attempt(
                attempt.attempt_id,
                status="succeeded",
                metrics=dict(_result_field(result, "metrics", {}) or {}),
                artifacts=[
                    str(artifact)
                    for artifact in list(_result_field(result, "artifacts", []) or [])
                ],
                decision=decision,
                finished_at=created_at,
            )
            self.store.append_event(
                event_type="node_verdict",
                node_id=node.id,
                data={
                    "attempt_id": attempt.attempt_id,
                    "decision": decision,
                },
                timestamp=created_at,
            )
            attempts.append(updated)
        return attempts
