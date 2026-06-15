"""Coordinator for per-hypothesis validation."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import json
import logging
from pathlib import Path
from typing import Any

from researchclaw.experiment.protocol import parse_hypotheses_md
from researchclaw.pipeline.branch_checkpoint import (
    BRANCH_STAGE_MAX,
    BRANCH_STAGE_MIN,
    read_branch_state,
)
from researchclaw.pipeline.hypothesis_store import (
    HypothesisNode,
    HypothesisStore,
    ValidationAttempt,
    _hypothesis_hash,
)

logger = logging.getLogger(__name__)


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


def _attempt_name(attempt: ValidationAttempt) -> str:
    return attempt.attempt_id.split("/", 1)[-1]


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _attempt_file_class(branch_run_dir: Path) -> str:
    result_path = branch_run_dir / "attempt_result.json"
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return "pending"
        if isinstance(payload, dict):
            status = str(payload.get("status") or "").lower()
            if status == "failed":
                return "failed"
            return "completed"

    state = read_branch_state(branch_run_dir)
    if state is not None:
        last_completed = _coerce_int(state.get("last_completed_stage"))
        if (
            last_completed is not None
            and int(BRANCH_STAGE_MIN) <= last_completed <= int(BRANCH_STAGE_MAX)
        ):
            return "interrupted"
    return "pending"


def _session_name(
    run_dir: Path,
    node: HypothesisNode,
    attempt: ValidationAttempt,
) -> str:
    return f"{run_dir.name}-{node.id}-{_attempt_name(attempt)}"


def _workspace_isolation_enabled(
    config: Any,
    *,
    max_concurrent: int | None = None,
) -> bool:
    validation = getattr(config, "hypothesis_validation", None)
    mode = str(getattr(validation, "workspace_isolation", "shared")).lower()
    if mode == "worktree":
        return True
    if validation is not None and max_concurrent is not None:
        try:
            return int(max_concurrent) > 1
        except (TypeError, ValueError):
            return False
    return False


def _source_workspace(config: Any) -> Path:
    workspace_agent = getattr(
        getattr(config, "experiment", None),
        "workspace_agent",
        None,
    )
    configured = str(getattr(workspace_agent, "workspace_path", "") or "").strip()
    if configured:
        return Path(configured)
    return Path.cwd()


def _max_tree_depth(config: Any) -> int | None:
    validation = getattr(config, "hypothesis_validation", None)
    raw = getattr(validation, "max_tree_depth", None)
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return None


class WorkerAbandoned(RuntimeError):
    """Raised when a branch worker dies before producing a terminal result."""


class WorkspaceIsolationError(RuntimeError):
    """Raised when an isolated branch workspace cannot be provisioned."""


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

    def _queue_attempt(
        self,
        attempt: ValidationAttempt,
        *,
        created_at: str | None,
    ) -> None:
        from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue, WorkItem

        DurableWorkQueue(self.run_dir).append(
            WorkItem(
                node_id=attempt.node_id,
                attempt_id=attempt.attempt_id,
                branch_run_dir=attempt.branch_run_dir,
            ),
            created_at=created_at,
        )

    def _seed_branch_for_attempt(
        self,
        node: HypothesisNode,
        attempt: ValidationAttempt,
    ) -> None:
        from researchclaw.pipeline.hypothesis_branch import seed_branch_dir

        seed_branch_dir(Path(attempt.branch_run_dir), self.run_dir, node)

    def _complete_queue_item(
        self,
        attempt_id: str,
        *,
        created_at: str | None,
    ) -> None:
        from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue

        DurableWorkQueue(self.run_dir).complete_item(
            attempt_id,
            created_at=created_at,
        )

    def split_and_queue(
        self,
        hypotheses_md: str,
        *,
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
            self._seed_branch_for_attempt(node, attempt)
            self._queue_attempt(attempt, created_at=created_at)
            attempts.append(attempt)
        return attempts

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

    def _prepare_attempt_for_run(
        self,
        node: HypothesisNode,
        attempt: ValidationAttempt,
        config: Any,
        *,
        created_at: str | None,
        max_concurrent: int | None = None,
    ) -> ValidationAttempt:
        session_name = attempt.agent_session_name or _session_name(
            self.run_dir, node, attempt
        )
        workspace_path = attempt.workspace_path
        if (
            _workspace_isolation_enabled(config, max_concurrent=max_concurrent)
            and workspace_path is None
        ):
            from researchclaw.pipeline.hypothesis_branch import provision_workspace

            try:
                provisioned = provision_workspace(
                    attempt,
                    source_workspace=_source_workspace(config),
                    workspace_root=self.run_dir / ".worktrees",
                )
                workspace_path = provisioned.workspace_path
            except Exception as exc:  # noqa: BLE001
                raise WorkspaceIsolationError(
                    "Workspace isolation is enabled but provisioning failed for "
                    f"{attempt.attempt_id}: {exc}"
                ) from exc
        return self.store.update_attempt(
            attempt.attempt_id,
            status="running",
            workspace_path=workspace_path,
            agent_session_name=session_name,
            started_at=created_at,
        )

    def _sync_attempt_from_branch_state(
        self,
        attempt: ValidationAttempt,
    ) -> ValidationAttempt:
        state = read_branch_state(Path(attempt.branch_run_dir))
        if state is None:
            return attempt

        raw_stage_status = state.get("stage_status")
        stage_status: dict[int, str] = {}
        if isinstance(raw_stage_status, dict):
            for stage, status in raw_stage_status.items():
                stage_number = _coerce_int(stage)
                if stage_number is None:
                    continue
                stage_status[stage_number] = str(status)

        workspace_path = state.get("workspace_path")
        workspace = (
            str(workspace_path)
            if isinstance(workspace_path, str) and workspace_path.strip()
            else attempt.workspace_path
        )
        if stage_status == attempt.stage_status and workspace == attempt.workspace_path:
            return attempt
        return self.store.update_attempt(
            attempt.attempt_id,
            stage_status=stage_status,
            workspace_path=workspace,
        )

    def _release_attempt_workspace(
        self,
        attempt: ValidationAttempt,
        config: Any,
        *,
        max_concurrent: int | None = None,
    ) -> None:
        if not _workspace_isolation_enabled(config, max_concurrent=max_concurrent):
            return
        from researchclaw.pipeline.hypothesis_branch import release_workspace

        try:
            release_workspace(attempt, source_workspace=_source_workspace(config))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Workspace release failed for %s: %s",
                attempt.attempt_id,
                exc,
            )

    def _create_followup_node(
        self,
        *,
        source_node: HypothesisNode,
        decision: str,
        result: Any,
        created_at: str | None,
        max_tree_depth: int | None = None,
    ) -> tuple[HypothesisNode, bool] | None:
        payload = _followup_payload(result)
        if payload is None:
            raise ValueError(f"{decision.upper()} decision requires next_hypothesis")
        parent_id = source_node.id if decision == "extend" else source_node.parent_id
        if max_tree_depth is not None:
            candidate_depth = (
                self._node_depth(source_node) + 1
                if decision == "extend"
                else self._node_depth(source_node)
            )
            if candidate_depth > max_tree_depth:
                self.store.append_event(
                    event_type="followup_skipped",
                    node_id=source_node.id,
                    data={
                        "decision": decision,
                        "reason": "max_tree_depth",
                        "max_tree_depth": max_tree_depth,
                    },
                    timestamp=created_at,
                )
                return None

        existing = self._find_node_by_science_payload(payload)
        if existing is not None:
            return existing, False

        return self.store.create_node(
            statement=str(payload.get("statement") or ""),
            prediction=str(payload.get("prediction") or ""),
            falsification=str(payload.get("falsification") or ""),
            rationale=str(payload.get("rationale") or ""),
            baselines=tuple(payload.get("baselines") or ()),
            source=decision,
            parent_id=parent_id,
            created_at=created_at,
        ), True

    def _find_node_by_science_payload(
        self,
        payload: dict[str, Any],
    ) -> HypothesisNode | None:
        candidate_hash = _hypothesis_hash(
            str(payload.get("statement") or ""),
            str(payload.get("prediction") or ""),
            str(payload.get("falsification") or ""),
        )
        for node in self.store.list_nodes():
            if node.hypothesis_hash == candidate_hash:
                return node
        return None

    def _node_depth(self, node: HypothesisNode) -> int:
        depth = 0
        current = node
        seen = {node.id}
        while current.parent_id:
            if current.parent_id in seen:
                break
            seen.add(current.parent_id)
            try:
                current = self.store._read_node(current.parent_id)
            except ValueError:
                break
            depth += 1
        return depth

    def _apply_decision_to_tree(
        self,
        *,
        node: HypothesisNode,
        attempt: ValidationAttempt,
        decision: str,
        result: Any,
        created_at: str | None,
        max_tree_depth: int | None = None,
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
            followup_result = self._create_followup_node(
                source_node=node,
                decision=decision,
                result=result,
                created_at=created_at,
                max_tree_depth=max_tree_depth,
            )
            if followup_result is None:
                return
            followup, created = followup_result
            if not created:
                self.store.append_event(
                    event_type="followup_deduped",
                    node_id=node.id,
                    data={
                        "decision": decision,
                        "existing_node_id": followup.id,
                        "attempt_id": attempt.attempt_id,
                    },
                    timestamp=created_at,
                )
                return
            followup_attempt = self.store.add_attempt(
                node_id=followup.id,
                branch_run_dir=str(self._branch_run_dir(followup.id, "attempt-001")),
                created_at=created_at,
            )
            self._seed_branch_for_attempt(followup, followup_attempt)
            self._queue_attempt(followup_attempt, created_at=created_at)

    def _finish_attempt_success(
        self,
        *,
        node: HypothesisNode,
        attempt: ValidationAttempt,
        result: Any,
        created_at: str | None,
        max_tree_depth: int | None = None,
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
            max_tree_depth=max_tree_depth,
        )
        self._complete_queue_item(updated.attempt_id, created_at=created_at)
        return updated

    def _finish_attempt_failure(
        self,
        *,
        attempt: ValidationAttempt,
        error: BaseException,
        created_at: str | None,
        mark_node_terminal: bool = True,
    ) -> ValidationAttempt:
        updated = self.store.update_attempt(
            attempt.attempt_id,
            status="failed",
            error=str(error),
            finished_at=created_at,
        )
        if mark_node_terminal:
            self._mark_node_inconclusive(
                node_id=updated.node_id,
                attempt_id=updated.attempt_id,
                error=str(error),
                created_at=created_at,
            )
        self._complete_queue_item(updated.attempt_id, created_at=created_at)
        return updated

    def _finish_attempt_abandoned(
        self,
        *,
        attempt: ValidationAttempt,
        error: BaseException,
        created_at: str | None,
    ) -> ValidationAttempt:
        updated = self.store.update_attempt(
            attempt.attempt_id,
            status="abandoned",
            error=str(error),
            finished_at=created_at,
        )
        self._complete_queue_item(updated.attempt_id, created_at=created_at)
        return updated

    def _mark_node_inconclusive(
        self,
        *,
        node_id: str,
        attempt_id: str,
        error: str,
        created_at: str | None,
    ) -> None:
        node = self.store._read_node(node_id)
        if node.status in {"supported", "refuted", "inconclusive", "superseded"}:
            return
        if node.status == "proposed":
            node = self.store.set_node_status(
                node_id,
                "validating",
                created_at=created_at,
            )
        if node.status == "validating":
            self.store.set_node_status(
                node_id,
                "inconclusive",
                created_at=created_at,
                event_data={
                    "attempt_id": attempt_id,
                    "decision": "inconclusive",
                    "error": error,
                },
            )

    def _finish_attempt_from_result(
        self,
        *,
        node: HypothesisNode,
        attempt: ValidationAttempt,
        result: Any,
        created_at: str | None,
        max_tree_depth: int | None = None,
    ) -> ValidationAttempt:
        result_status = str(
            _result_field(result, "status", "succeeded") or ""
        ).lower()
        if result_status and result_status != "succeeded":
            return self._finish_attempt_failure(
                attempt=attempt,
                error=RuntimeError(
                    str(_result_field(result, "error", "attempt failed"))
                ),
                created_at=created_at,
            )
        return self._finish_attempt_success(
            node=node,
            attempt=attempt,
            result=result,
            created_at=created_at,
            max_tree_depth=max_tree_depth,
        )

    def _queue_retry_attempt(
        self,
        *,
        node: HypothesisNode,
        max_attempts_per_node: int,
        created_at: str | None,
    ) -> ValidationAttempt | None:
        next_attempt_id = self.store._next_attempt_id(node.id)
        attempt_name = next_attempt_id.split("/", 1)[1]
        try:
            attempt_number = int(attempt_name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            attempt_number = max_attempts_per_node + 1
        if attempt_number > max_attempts_per_node:
            return None
        retry = self.store.add_attempt(
            node_id=node.id,
            branch_run_dir=str(self._branch_run_dir(node.id, attempt_name)),
            created_at=created_at,
        )
        self._seed_branch_for_attempt(node, retry)
        self._queue_attempt(retry, created_at=created_at)
        return retry

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
            self._complete_queue_item(attempt_id, created_at=created_at)
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

    def reduce_existing_attempt_results(
        self,
        *,
        created_at: str | None = None,
    ) -> list[ValidationAttempt]:
        reduced: list[ValidationAttempt] = []
        branches_dir = self.run_dir / "hypothesis_branches"
        if not branches_dir.exists():
            return reduced
        for result_path in sorted(branches_dir.glob("*/attempt-*/attempt_result.json")):
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "Skipping unreadable attempt result during resume: %s",
                    result_path,
                )
                continue
            attempt_id = str(payload.get("attempt_id") or "")
            if not attempt_id or self._attempt_result_already_reduced(attempt_id):
                continue
            try:
                reduced.append(
                    self.reduce_attempt_result(
                        result_path,
                        created_at=created_at,
                    )
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Skipping invalid attempt result during resume: %s",
                    result_path,
                    exc_info=True,
                )
        return reduced

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
            self._seed_branch_for_attempt(node, attempt)
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
                self._finish_attempt_from_result(
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
            branch_run_dir = Path(item.branch_run_dir)
            item_class = _attempt_file_class(branch_run_dir)
            result_path = branch_run_dir / "attempt_result.json"
            if item_class in {"completed", "failed"}:
                try:
                    finished = self.reduce_attempt_result(
                        result_path,
                        created_at=created_at,
                    )
                except (json.JSONDecodeError, OSError, ValueError):
                    logger.warning(
                        "Ignoring unreadable attempt result and retrying: %s",
                        result_path,
                    )
                else:
                    self._release_attempt_workspace(finished, config)
                    resumed.append(finished)
                    continue
            node = self.store._read_node(item.node_id)
            attempt = self.store._read_attempt(item.attempt_id)
            if item_class == "interrupted":
                attempt = self._sync_attempt_from_branch_state(attempt)
            self._seed_branch_for_attempt(node, attempt)
            self.store.set_node_status(
                node.id,
                "validating",
                created_at=created_at,
            )
            attempt = self._prepare_attempt_for_run(
                node,
                attempt,
                config,
                created_at=created_at,
            )
            try:
                result = self.validate_branch(node, attempt, config, adapters)
            except Exception as exc:
                finished = self._finish_attempt_failure(
                    attempt=attempt,
                    error=exc,
                    created_at=created_at,
                )
                self._release_attempt_workspace(finished, config)
                resumed.append(finished)
                continue
            finished = self._finish_attempt_from_result(
                node=node,
                attempt=attempt,
                result=result,
                created_at=created_at,
                max_tree_depth=_max_tree_depth(config),
            )
            self._release_attempt_workspace(finished, config)
            resumed.append(finished)
        return resumed

    def run_pending_work_concurrent(
        self,
        *,
        config: Any,
        adapters: Any,
        max_concurrent: int,
        max_attempts_per_node: int = 1,
        created_at: str | None = None,
    ) -> list[ValidationAttempt]:
        from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue

        jobs: list[tuple[HypothesisNode, ValidationAttempt]] = []
        reduced: list[ValidationAttempt] = []
        for item in DurableWorkQueue(self.run_dir).read_items():
            branch_run_dir = Path(item.branch_run_dir)
            item_class = _attempt_file_class(branch_run_dir)
            result_path = branch_run_dir / "attempt_result.json"
            if item_class in {"completed", "failed"}:
                try:
                    finished = self.reduce_attempt_result(
                        result_path,
                        created_at=created_at,
                    )
                except (json.JSONDecodeError, OSError, ValueError):
                    logger.warning(
                        "Ignoring unreadable attempt result and retrying: %s",
                        result_path,
                    )
                else:
                    self._release_attempt_workspace(
                        finished,
                        config,
                        max_concurrent=max_concurrent,
                    )
                    reduced.append(finished)
                    continue
            node = self.store._read_node(item.node_id)
            attempt = self.store._read_attempt(item.attempt_id)
            if item_class == "interrupted":
                attempt = self._sync_attempt_from_branch_state(attempt)
            self._seed_branch_for_attempt(node, attempt)
            self.store.set_node_status(
                node.id,
                "validating",
                created_at=created_at,
            )
            jobs.append((node, attempt))

        if not jobs:
            return reduced

        max_workers = max(1, int(max_concurrent or 1))
        completed: list[ValidationAttempt | None] = [None] * len(jobs)

        def run_one(
            job: tuple[HypothesisNode, ValidationAttempt],
        ) -> tuple[ValidationAttempt, Any]:
            node, attempt = job
            prepared = self._prepare_attempt_for_run(
                node,
                attempt,
                config,
                created_at=created_at,
                max_concurrent=max_concurrent,
            )
            return prepared, self.validate_branch(node, prepared, config, adapters)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_indexes: dict[Any, int] = {}
            next_job_index = 0

            def submit_available() -> None:
                nonlocal next_job_index
                while (
                    next_job_index < len(jobs)
                    and len(future_indexes) < max_workers
                ):
                    future_indexes[pool.submit(run_one, jobs[next_job_index])] = (
                        next_job_index
                    )
                    next_job_index += 1

            submit_available()
            while future_indexes:
                done, _pending = wait(
                    future_indexes,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    index = future_indexes.pop(future)
                    node, attempt = jobs[index]
                    try:
                        attempt, result = future.result()
                    except WorkerAbandoned as exc:
                        attempt = self.store._read_attempt(attempt.attempt_id)
                        finished = self._finish_attempt_abandoned(
                            attempt=attempt,
                            error=exc,
                            created_at=created_at,
                        )
                        self._release_attempt_workspace(
                            finished,
                            config,
                            max_concurrent=max_concurrent,
                        )
                        completed[index] = finished
                        retry = self._queue_retry_attempt(
                            node=node,
                            max_attempts_per_node=max_attempts_per_node,
                            created_at=created_at,
                        )
                        if retry is None:
                            self._mark_node_inconclusive(
                                node_id=node.id,
                                attempt_id=finished.attempt_id,
                                error=str(exc),
                                created_at=created_at,
                            )
                        continue
                    except Exception as exc:
                        attempt = self.store._read_attempt(attempt.attempt_id)
                        finished = self._finish_attempt_failure(
                            attempt=attempt,
                            error=exc,
                            created_at=created_at,
                        )
                        self._release_attempt_workspace(
                            finished,
                            config,
                            max_concurrent=max_concurrent,
                        )
                        completed[index] = finished
                        continue
                    finished = self._finish_attempt_from_result(
                        node=node,
                        attempt=attempt,
                        result=result,
                        created_at=created_at,
                        max_tree_depth=_max_tree_depth(config),
                    )
                    self._release_attempt_workspace(
                        finished,
                        config,
                        max_concurrent=max_concurrent,
                    )
                    completed[index] = finished
                submit_available()
        return reduced + [attempt for attempt in completed if attempt is not None]

    def run_until_queue_empty(
        self,
        *,
        config: Any,
        adapters: Any,
        max_concurrent: int,
        max_attempts_per_node: int = 1,
        created_at: str | None = None,
        require_coordinator_gate: bool = False,
    ) -> list[HypothesisNode]:
        from researchclaw.pipeline.evidence_aggregator import EvidenceAggregator
        from researchclaw.pipeline.hypothesis_queue import DurableWorkQueue

        self.reduce_existing_attempt_results(created_at=created_at)
        max_total_attempts = getattr(
            getattr(config, "hypothesis_validation", None),
            "max_total_attempts",
            100,
        )
        try:
            max_total_attempts = int(max_total_attempts or 100)
        except (TypeError, ValueError):
            max_total_attempts = 100
        iterations = 0
        while True:
            pending = DurableWorkQueue(self.run_dir).read_items()
            if not pending:
                break
            iterations += len(pending)
            if iterations > max_total_attempts:
                logger.warning(
                    "Stopping per-hypothesis coordinator after %d queued attempts",
                    iterations,
                )
                break
            completed = self.run_pending_work_concurrent(
                config=config,
                adapters=adapters,
                max_concurrent=max_concurrent,
                max_attempts_per_node=max_attempts_per_node,
                created_at=created_at,
            )
            if not completed:
                break

        self.store.rebuild_tree(generated_at=created_at)
        try:
            aggregator = EvidenceAggregator(self.run_dir)
            aggregate = aggregator.aggregate(generated_at=created_at)
            aggregator.write_root_handoff(aggregate, generated_at=created_at)
        except Exception:  # noqa: BLE001
            logger.warning("Hypothesis evidence aggregation failed", exc_info=True)
        if require_coordinator_gate:
            gate_path = self.run_dir / "coordinator_gate.json"
            gate_path.write_text(
                json.dumps(
                    {
                        "status": "blocked_approval",
                        "reason": "coordinator_gate_required",
                        "generated": created_at,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return self.store.list_nodes()

    def reduce_attempt_results_concurrent(
        self,
        result_paths: list[Path],
        *,
        max_concurrent: int,
        created_at: str | None = None,
    ) -> list[ValidationAttempt]:
        if not result_paths:
            return []
        max_workers = max(1, int(max_concurrent or 1))
        completed: list[ValidationAttempt | None] = [None] * len(result_paths)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_indexes = {
                pool.submit(
                    self.reduce_attempt_result,
                    result_path,
                    created_at=created_at,
                ): index
                for index, result_path in enumerate(result_paths)
            }
            for future in as_completed(future_indexes):
                completed[future_indexes[future]] = future.result()
        return [attempt for attempt in completed if attempt is not None]
