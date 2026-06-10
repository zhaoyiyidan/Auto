"""Per-hypothesis validation store models.

This module is additive to the legacy ``hypothesis_tree`` sidecars.  It keeps
immutable hypothesis science separate from runtime validation attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from researchclaw.experiment.protocol import HypothesisSpec


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
