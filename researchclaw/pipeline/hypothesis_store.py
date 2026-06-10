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
