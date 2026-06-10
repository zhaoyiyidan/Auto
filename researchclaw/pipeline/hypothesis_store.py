"""Per-hypothesis validation store models.

This module is additive to the legacy ``hypothesis_tree`` sidecars.  It keeps
immutable hypothesis science separate from runtime validation attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from researchclaw.experiment.protocol import HypothesisSpec
from researchclaw.pipeline.hypothesis_tree import _atomic_write_json, _atomic_write_text


TREE_DIRNAME = "hypothesis_tree"
_NODE_ID_RE = re.compile(r"^h-(\d+)$")
_ATTEMPT_ID_RE = re.compile(r"^attempt-(\d+)\.json$")


NODE_STATUSES = {
    "proposed",
    "validating",
    "supported",
    "refuted",
    "inconclusive",
    "superseded",
}

ATTEMPT_STATUSES = {"queued", "running", "succeeded", "failed", "abandoned"}


def _hypothesis_hash(statement: str, prediction: str, falsification: str) -> str:
    normalized = "\n".join(
        (
            str(statement or "").strip(),
            str(prediction or "").strip(),
            str(falsification or "").strip(),
        )
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _node_markdown(node: "HypothesisNode") -> str:
    lines = [
        f"# Hypothesis {node.id}",
        "",
        "## Statement",
        node.statement,
        "",
        "## Prediction",
        node.prediction,
        "",
        "## Falsification",
        node.falsification,
        "",
        "## Rationale",
        node.rationale,
        "",
        "## Baselines",
    ]
    if node.baselines:
        lines.extend(f"- {baseline}" for baseline in node.baselines)
    else:
        lines.append("_None specified._")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class HypothesisNode:
    id: str
    statement: str
    prediction: str = ""
    falsification: str = ""
    rationale: str = ""
    baselines: tuple[str, ...] = ()
    source: str = "stage8_batch"
    parent_id: str | None = None
    created_at: str = ""
    hypothesis_hash: str = field(default="", compare=True)
    status: str = "proposed"

    def __post_init__(self) -> None:
        spec = HypothesisSpec(
            id="H1",
            statement=self.statement,
            prediction=self.prediction,
            falsification=self.falsification,
            rationale=self.rationale,
            baselines=self.baselines,
        )
        status = str(self.status or "proposed").strip().lower()
        if status not in NODE_STATUSES:
            raise ValueError(f"Invalid hypothesis node status: {status}")
        object.__setattr__(self, "id", str(self.id or "").strip())
        object.__setattr__(self, "statement", spec.statement)
        object.__setattr__(self, "prediction", spec.prediction)
        object.__setattr__(self, "falsification", spec.falsification)
        object.__setattr__(self, "rationale", spec.rationale)
        object.__setattr__(self, "baselines", spec.baselines)
        object.__setattr__(self, "source", str(self.source or "").strip())
        object.__setattr__(self, "created_at", str(self.created_at or "").strip())
        object.__setattr__(self, "status", status)
        if not self.hypothesis_hash:
            object.__setattr__(
                self,
                "hypothesis_hash",
                _hypothesis_hash(spec.statement, spec.prediction, spec.falsification),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "prediction": self.prediction,
            "falsification": self.falsification,
            "rationale": self.rationale,
            "baselines": list(self.baselines),
            "source": self.source,
            "parent_id": self.parent_id,
            "hypothesis_hash": self.hypothesis_hash,
            "created_at": self.created_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Any) -> HypothesisNode:
        data = data if isinstance(data, dict) else {}
        return cls(
            id=data.get("id", ""),
            statement=data.get("statement") or data.get("hypothesis") or "",
            prediction=data.get("prediction") or "",
            falsification=data.get("falsification") or "",
            rationale=data.get("rationale") or "",
            baselines=data.get("baselines") or (),
            source=data.get("source") or "stage8_batch",
            parent_id=data.get("parent_id"),
            created_at=data.get("created_at") or "",
            hypothesis_hash=data.get("hypothesis_hash") or "",
            status=data.get("status") or "proposed",
        )


@dataclass(frozen=True)
class ValidationAttempt:
    attempt_id: str
    node_id: str
    status: str = "queued"
    branch_run_dir: str = ""
    workspace_path: str | None = None
    agent_session_name: str | None = None
    stage_status: dict[int, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    decision: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        status = str(self.status or "queued").strip().lower()
        if status not in ATTEMPT_STATUSES:
            raise ValueError(f"Invalid validation attempt status: {status}")
        stage_status = {
            int(stage): str(value)
            for stage, value in dict(self.stage_status or {}).items()
        }
        object.__setattr__(self, "attempt_id", str(self.attempt_id or "").strip())
        object.__setattr__(self, "node_id", str(self.node_id or "").strip())
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "branch_run_dir",
            str(self.branch_run_dir or "").strip(),
        )
        object.__setattr__(self, "stage_status", stage_status)
        object.__setattr__(self, "metrics", dict(self.metrics or {}))
        object.__setattr__(
            self,
            "artifacts",
            [str(artifact) for artifact in list(self.artifacts or [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "node_id": self.node_id,
            "status": self.status,
            "branch_run_dir": self.branch_run_dir,
            "workspace_path": self.workspace_path,
            "agent_session_name": self.agent_session_name,
            "stage_status": {
                str(stage): status
                for stage, status in sorted(self.stage_status.items())
            },
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "decision": self.decision,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ValidationAttempt:
        data = data if isinstance(data, dict) else {}
        return cls(
            attempt_id=data.get("attempt_id") or "",
            node_id=data.get("node_id") or "",
            status=data.get("status") or "queued",
            branch_run_dir=data.get("branch_run_dir") or "",
            workspace_path=data.get("workspace_path"),
            agent_session_name=data.get("agent_session_name"),
            stage_status=data.get("stage_status") or {},
            metrics=data.get("metrics") or {},
            artifacts=data.get("artifacts") or [],
            decision=data.get("decision"),
            error=data.get("error"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
        )


class HypothesisStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.tree_dir = self.run_dir / TREE_DIRNAME
        self.nodes_dir = self.tree_dir / "nodes"
        self.events_path = self.tree_dir / "events.jsonl"

    def _next_node_id(self) -> str:
        max_number = 0
        if self.nodes_dir.exists():
            for child in self.nodes_dir.iterdir():
                match = _NODE_ID_RE.match(child.name)
                if match:
                    max_number = max(max_number, int(match.group(1)))
        return f"h-{max_number + 1:03d}"

    def _node_dir(self, node_id: str) -> Path:
        return self.nodes_dir / node_id

    def _attempt_dir(self, node_id: str) -> Path:
        return self._node_dir(node_id) / "attempts" / node_id

    def _attempt_path(self, attempt_id: str) -> Path:
        node_id = attempt_id.split("/", 1)[0]
        return self._node_dir(node_id) / "attempts" / Path(attempt_id).with_suffix(".json")

    def _next_attempt_id(self, node_id: str) -> str:
        max_number = 0
        attempt_dir = self._attempt_dir(node_id)
        if attempt_dir.exists():
            for child in attempt_dir.iterdir():
                match = _ATTEMPT_ID_RE.match(child.name)
                if match:
                    max_number = max(max_number, int(match.group(1)))
        return f"{node_id}/attempt-{max_number + 1:03d}"

    def _read_attempt(self, attempt_id: str) -> ValidationAttempt:
        path = self._attempt_path(attempt_id)
        if not path.exists():
            raise ValueError(f"Unknown validation attempt: {attempt_id}")
        return ValidationAttempt.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _append_event(
        self,
        *,
        event_type: str,
        node_id: str | None,
        data: dict[str, Any],
        timestamp: str,
    ) -> None:
        self.tree_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_type": event_type,
            "node_id": node_id,
            "data": data,
            "timestamp": timestamp,
        }
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            fh.write("\n")

    def create_node(
        self,
        *,
        statement: str,
        prediction: str = "",
        falsification: str = "",
        rationale: str = "",
        baselines: tuple[str, ...] = (),
        source: str = "stage8_batch",
        parent_id: str | None = None,
        created_at: str | None = None,
    ) -> HypothesisNode:
        created_at = created_at or _utcnow_iso()
        node = HypothesisNode(
            id=self._next_node_id(),
            statement=statement,
            prediction=prediction,
            falsification=falsification,
            rationale=rationale,
            baselines=baselines,
            source=source,
            parent_id=parent_id,
            created_at=created_at,
        )
        node_dir = self.nodes_dir / node.id
        node_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(node_dir / "node.json", node.to_dict())
        _atomic_write_text(node_dir / "hypothesis.md", _node_markdown(node))
        self._append_event(
            event_type="node_proposed",
            node_id=node.id,
            data={
                "parent_id": parent_id,
                "source": node.source,
                "hypothesis_hash": node.hypothesis_hash,
            },
            timestamp=created_at,
        )
        return node

    def add_attempt(
        self,
        *,
        node_id: str,
        branch_run_dir: str,
        workspace_path: str | None = None,
        agent_session_name: str | None = None,
        created_at: str | None = None,
    ) -> ValidationAttempt:
        if not (self._node_dir(node_id) / "node.json").exists():
            raise ValueError(f"Unknown hypothesis node: {node_id}")
        created_at = created_at or _utcnow_iso()
        attempt = ValidationAttempt(
            attempt_id=self._next_attempt_id(node_id),
            node_id=node_id,
            branch_run_dir=branch_run_dir,
            workspace_path=workspace_path,
            agent_session_name=agent_session_name,
        )
        _atomic_write_json(self._attempt_path(attempt.attempt_id), attempt.to_dict())
        self._append_event(
            event_type="attempt_queued",
            node_id=node_id,
            data={
                "attempt_id": attempt.attempt_id,
                "branch_run_dir": attempt.branch_run_dir,
            },
            timestamp=created_at,
        )
        return attempt

    def update_attempt(
        self,
        attempt_id: str,
        *,
        status: str | None = None,
        branch_run_dir: str | None = None,
        workspace_path: str | None = None,
        agent_session_name: str | None = None,
        stage_status: dict[int, str] | None = None,
        metrics: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        decision: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> ValidationAttempt:
        current = self._read_attempt(attempt_id)
        updated = ValidationAttempt(
            attempt_id=current.attempt_id,
            node_id=current.node_id,
            status=status or current.status,
            branch_run_dir=branch_run_dir or current.branch_run_dir,
            workspace_path=(
                current.workspace_path if workspace_path is None else workspace_path
            ),
            agent_session_name=(
                current.agent_session_name
                if agent_session_name is None
                else agent_session_name
            ),
            stage_status=(
                current.stage_status if stage_status is None else stage_status
            ),
            metrics=current.metrics if metrics is None else metrics,
            artifacts=current.artifacts if artifacts is None else artifacts,
            decision=current.decision if decision is None else decision,
            error=current.error if error is None else error,
            started_at=current.started_at if started_at is None else started_at,
            finished_at=current.finished_at if finished_at is None else finished_at,
        )
        _atomic_write_json(self._attempt_path(updated.attempt_id), updated.to_dict())
        if updated.status in {"succeeded", "failed", "abandoned"}:
            self._append_event(
                event_type="attempt_finished",
                node_id=updated.node_id,
                data={
                    "attempt_id": updated.attempt_id,
                    "status": updated.status,
                    "decision": updated.decision,
                    "error": updated.error,
                },
                timestamp=updated.finished_at or _utcnow_iso(),
            )
        return updated
