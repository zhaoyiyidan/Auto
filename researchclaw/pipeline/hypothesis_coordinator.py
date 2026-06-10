"""Coordinator for per-hypothesis validation."""

from __future__ import annotations

import json
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


def _followup_payload(result: Any) -> dict[str, Any] | None:
    payload = _result_field(result, "next_hypothesis")
    if payload is None:
        payload = _result_field(result, "hypothesis")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("Follow-up hypothesis must be a mapping")
    return payload


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
        from researchclaw.pipeline.hypothesis_branch import validate_branch

        return validate_branch(
            branch_run_dir=Path(attempt.branch_run_dir),
            node=node,
            attempt=attempt,
            config=config,
            adapters=adapters,
        )

    def _create_followup_node(
        self,
        *,
        source_node: HypothesisNode,
        decision: str,
        result: Any,
        created_at: str | None,
    ) -> HypothesisNode:
        payload = _followup_payload(result)
        if payload is None:
            raise ValueError(f"{decision.upper()} decision requires next_hypothesis")
        parent_id = source_node.id if decision == "extend" else source_node.parent_id
        return self.store.create_node(
            statement=str(payload.get("statement") or ""),
            prediction=str(payload.get("prediction") or ""),
            falsification=str(payload.get("falsification") or ""),
            rationale=str(payload.get("rationale") or ""),
            baselines=tuple(payload.get("baselines") or ()),
            source=decision,
            parent_id=parent_id,
            created_at=created_at,
        )

    def _apply_decision_to_tree(
        self,
        *,
        node: HypothesisNode,
        attempt: ValidationAttempt,
        decision: str,
        result: Any,
        created_at: str | None,
    ) -> None:
        status_by_decision = {
            "proceed": "supported",
            "inconclusive": "inconclusive",
            "extend": "superseded",
            "pivot": "superseded",
        }
        status = status_by_decision.get(decision, "inconclusive")
        self.store.set_node_status(
            node.id,
            status,
            created_at=created_at,
            event_data={
                "attempt_id": attempt.attempt_id,
                "decision": decision,
            },
        )
        if decision in {"extend", "pivot"}:
            followup = self._create_followup_node(
                source_node=node,
                decision=decision,
                result=result,
                created_at=created_at,
            )
            followup_attempt = self.store.add_attempt(
                node_id=followup.id,
                branch_run_dir=str(self._branch_run_dir(followup.id, "attempt-001")),
                created_at=created_at,
            )
            from researchclaw.pipeline.hypothesis_queue import (
                DurableWorkQueue,
                WorkItem,
            )

            DurableWorkQueue(self.run_dir).append(
                WorkItem(
                    node_id=followup.id,
                    attempt_id=followup_attempt.attempt_id,
                    branch_run_dir=followup_attempt.branch_run_dir,
                ),
                created_at=created_at,
            )

    def _finish_attempt_success(
        self,
        *,
        node: HypothesisNode,
        attempt: ValidationAttempt,
        result: Any,
        created_at: str | None,
    ) -> ValidationAttempt:
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
        self._apply_decision_to_tree(
            node=node,
            attempt=attempt,
            decision=decision,
            result=result,
            created_at=created_at,
        )
        return updated

    def _finish_attempt_failure(
        self,
        *,
        attempt: ValidationAttempt,
        error: BaseException,
        created_at: str | None,
    ) -> ValidationAttempt:
        return self.store.update_attempt(
            attempt.attempt_id,
            status="failed",
            error=str(error),
            finished_at=created_at,
        )

    def _attempt_result_already_reduced(self, attempt_id: str) -> bool:
        for event in self.store._read_events():
            event_type = event.get("event_type")
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            if (
                event_type in {"attempt_finished", "node_verdict"}
                and data.get("attempt_id") == attempt_id
            ):
                return True
        return False

    def reduce_attempt_result(
        self,
        result_path: Path,
        *,
        created_at: str | None = None,
    ) -> ValidationAttempt:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        attempt_id = str(payload.get("attempt_id") or "")
        node_id = str(payload.get("node_id") or "")
        if self._attempt_result_already_reduced(attempt_id):
            return self.store._read_attempt(attempt_id)

        node = self.store._read_node(node_id)
        attempt = self.store._read_attempt(attempt_id)
        if node.status == "proposed":
            self.store.set_node_status(
                node.id,
                "validating",
                created_at=created_at,
            )
        if str(payload.get("status") or "").lower() == "succeeded":
            return self._finish_attempt_success(
                node=node,
                attempt=attempt,
                result=payload,
                created_at=created_at,
            )
        return self._finish_attempt_failure(
            attempt=attempt,
            error=RuntimeError(str(payload.get("error") or "attempt failed")),
            created_at=created_at,
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
            from researchclaw.pipeline.hypothesis_branch import seed_branch_dir

            seed_branch_dir(Path(attempt.branch_run_dir), self.run_dir, node)
            self.store.set_node_status(
                node.id,
                "validating",
                created_at=created_at,
            )
            try:
                result = self.validate_branch(node, attempt, config, adapters)
            except Exception as exc:
                attempts.append(
                    self._finish_attempt_failure(
                        attempt=attempt,
                        error=exc,
                        created_at=created_at,
                    )
                )
                continue

            attempts.append(
                self._finish_attempt_success(
                    node=node,
                    attempt=attempt,
                    result=result,
                    created_at=created_at,
                )
            )
        return attempts

    def resume_pending_work(
        self,
        *,
        config: Any,
        adapters: Any,
        created_at: str | None = None,
    ) -> list[ValidationAttempt]:
        from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue

        resumed: list[ValidationAttempt] = []
        for item in DurableWorkQueue(self.run_dir).read_items():
            if (Path(item.branch_run_dir) / "attempt_result.json").exists():
                continue
            node = self.store._read_node(item.node_id)
            attempt = self.store._read_attempt(item.attempt_id)
            self.store.set_node_status(
                node.id,
                "validating",
                created_at=created_at,
            )
            try:
                result = self.validate_branch(node, attempt, config, adapters)
            except Exception as exc:
                resumed.append(
                    self._finish_attempt_failure(
                        attempt=attempt,
                        error=exc,
                        created_at=created_at,
                    )
                )
                continue
            resumed.append(
                self._finish_attempt_success(
                    node=node,
                    attempt=attempt,
                    result=result,
                    created_at=created_at,
                )
            )
        return resumed
