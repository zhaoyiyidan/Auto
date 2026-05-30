"""Agent-declared execution contract models for workspace experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


CONTRACT_SCHEMA_VERSION = "researchclaw.execution_contract.v1"
SUCCESS_STATUSES = ("completed", "complete", "succeeded", "success", "done", "passed")
INCOMPLETE_STATUSES = ("queued", "pending", "running", "submitted")


@dataclass(frozen=True)
class ArtifactCheck:
    path: str
    type: str = "file"
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(self.path))
        object.__setattr__(self, "type", str(self.type or "file"))
        object.__setattr__(self, "required", bool(self.required))

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "type": self.type, "required": self.required}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactCheck:
        return cls(
            path=str(data.get("path", "")),
            type=str(data.get("type") or "file"),
            required=bool(data.get("required", True)),
        )


@dataclass(frozen=True)
class MetricCheck:
    name: str
    type: str = "number"
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "type", str(self.type or "number"))
        object.__setattr__(self, "required", bool(self.required))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "required": self.required}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricCheck:
        return cls(
            name=str(data.get("name", "")),
            type=str(data.get("type") or "number"),
            required=bool(data.get("required", True)),
        )


@dataclass(frozen=True)
class PrimaryMetric:
    name: str = "primary_metric"
    direction: str = "maximize"
    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name or "primary_metric"))
        object.__setattr__(self, "direction", str(self.direction or "maximize"))
        object.__setattr__(self, "unit", str(self.unit or ""))

    def to_dict(self) -> dict[str, Any]:
        payload = {"name": self.name, "direction": self.direction}
        if self.unit:
            payload["unit"] = self.unit
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrimaryMetric:
        return cls(
            name=str(data.get("name") or "primary_metric"),
            direction=str(data.get("direction") or "maximize"),
            unit=str(data.get("unit") or ""),
        )


@dataclass(frozen=True)
class MetricsContract:
    primary: PrimaryMetric = field(default_factory=PrimaryMetric)
    required: tuple[MetricCheck, ...] = ()
    allow_extra: bool = True

    def __post_init__(self) -> None:
        primary = self.primary
        if not isinstance(primary, PrimaryMetric):
            primary = PrimaryMetric.from_dict(primary if isinstance(primary, dict) else {})
        object.__setattr__(self, "primary", primary)
        object.__setattr__(
            self,
            "required",
            tuple(
                item
                if isinstance(item, MetricCheck)
                else MetricCheck.from_dict(item if isinstance(item, dict) else {})
                for item in self.required
            ),
        )
        object.__setattr__(self, "allow_extra", bool(self.allow_extra))

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.to_dict(),
            "required": [item.to_dict() for item in self.required],
            "allow_extra": self.allow_extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricsContract:
        primary_data = data.get("primary")
        required_data = data.get("required") or ()
        if not isinstance(primary_data, dict):
            primary_data = {}
        if not isinstance(required_data, list | tuple):
            required_data = ()
        return cls(
            primary=PrimaryMetric.from_dict(primary_data),
            required=tuple(
                MetricCheck.from_dict(item)
                for item in required_data
                if isinstance(item, dict)
            ),
            allow_extra=bool(data.get("allow_extra", True)),
        )


@dataclass(frozen=True)
class CompletionContract:
    success_statuses: tuple[str, ...] = SUCCESS_STATUSES
    incomplete_statuses: tuple[str, ...] = INCOMPLETE_STATUSES
    require_any_metric: bool = True
    require_any_artifact: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "success_statuses",
            tuple(str(item).strip().lower() for item in self.success_statuses),
        )
        object.__setattr__(
            self,
            "incomplete_statuses",
            tuple(str(item).strip().lower() for item in self.incomplete_statuses),
        )
        object.__setattr__(self, "require_any_metric", bool(self.require_any_metric))
        object.__setattr__(self, "require_any_artifact", bool(self.require_any_artifact))

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_statuses": list(self.success_statuses),
            "incomplete_statuses": list(self.incomplete_statuses),
            "require_any_metric": self.require_any_metric,
            "require_any_artifact": self.require_any_artifact,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionContract:
        success = data.get("success_statuses") or SUCCESS_STATUSES
        incomplete = data.get("incomplete_statuses") or INCOMPLETE_STATUSES
        return cls(
            success_statuses=tuple(success) if isinstance(success, list | tuple) else SUCCESS_STATUSES,
            incomplete_statuses=(
                tuple(incomplete)
                if isinstance(incomplete, list | tuple)
                else INCOMPLETE_STATUSES
            ),
            require_any_metric=bool(data.get("require_any_metric", True)),
            require_any_artifact=bool(data.get("require_any_artifact", False)),
        )


@dataclass(frozen=True)
class AgentDeclaredChecks:
    enabled: bool = False
    location: str = ""
    field: str = ""
    required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "location", str(self.location or ""))
        object.__setattr__(self, "field", str(self.field or ""))
        object.__setattr__(self, "required", bool(self.required))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "location": self.location,
            "field": self.field,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentDeclaredChecks:
        return cls(
            enabled=bool(data.get("enabled", False)),
            location=str(data.get("location") or ""),
            field=str(data.get("field") or ""),
            required=bool(data.get("required", False)),
        )


@dataclass(frozen=True)
class ExecutionContract:
    schema_version: str = CONTRACT_SCHEMA_VERSION
    result_artifacts: tuple[ArtifactCheck, ...] = ()
    metrics: MetricsContract = field(default_factory=MetricsContract)
    completion: CompletionContract = field(default_factory=CompletionContract)
    agent_declared: AgentDeclaredChecks = field(default_factory=AgentDeclaredChecks)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema_version", str(self.schema_version or CONTRACT_SCHEMA_VERSION)
        )
        object.__setattr__(
            self,
            "result_artifacts",
            tuple(
                item
                if isinstance(item, ArtifactCheck)
                else ArtifactCheck.from_dict(item if isinstance(item, dict) else {})
                for item in self.result_artifacts
            ),
        )
        if not isinstance(self.metrics, MetricsContract):
            object.__setattr__(
                self,
                "metrics",
                MetricsContract.from_dict(self.metrics if isinstance(self.metrics, dict) else {}),
            )
        if not isinstance(self.completion, CompletionContract):
            object.__setattr__(
                self,
                "completion",
                CompletionContract.from_dict(
                    self.completion if isinstance(self.completion, dict) else {}
                ),
            )
        if not isinstance(self.agent_declared, AgentDeclaredChecks):
            object.__setattr__(
                self,
                "agent_declared",
                AgentDeclaredChecks.from_dict(
                    self.agent_declared if isinstance(self.agent_declared, dict) else {}
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "result_artifacts": [item.to_dict() for item in self.result_artifacts],
            "metrics": self.metrics.to_dict(),
            "completion": self.completion.to_dict(),
            "agent_declared": self.agent_declared.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionContract:
        result_artifacts = data.get("result_artifacts") or ()
        metrics_data = data.get("metrics") or {}
        completion_data = data.get("completion") or {}
        agent_declared_data = data.get("agent_declared") or {}
        return cls(
            schema_version=str(data.get("schema_version") or CONTRACT_SCHEMA_VERSION),
            result_artifacts=tuple(
                ArtifactCheck.from_dict(item)
                for item in result_artifacts
                if isinstance(item, dict)
            )
            if isinstance(result_artifacts, list | tuple)
            else (),
            metrics=MetricsContract.from_dict(metrics_data if isinstance(metrics_data, dict) else {}),
            completion=CompletionContract.from_dict(
                completion_data if isinstance(completion_data, dict) else {}
            ),
            agent_declared=AgentDeclaredChecks.from_dict(
                agent_declared_data if isinstance(agent_declared_data, dict) else {}
            ),
        )


def default_contract(
    *,
    primary_metric: str,
    metric_direction: str,
    expected_outputs: list[str] | tuple[str, ...],
) -> ExecutionContract:
    """Build the compatibility contract matching the pre-contract route rules."""

    return ExecutionContract(
        result_artifacts=tuple(
            ArtifactCheck(path=str(path), type="file", required=False)
            for path in expected_outputs
        ),
        metrics=MetricsContract(
            primary=PrimaryMetric(
                name=str(primary_metric or "primary_metric"),
                direction=str(metric_direction or "maximize"),
            ),
            required=(),
            allow_extra=True,
        ),
        completion=CompletionContract(
            success_statuses=SUCCESS_STATUSES,
            incomplete_statuses=INCOMPLETE_STATUSES,
            require_any_metric=True,
            require_any_artifact=False,
        ),
        agent_declared=AgentDeclaredChecks(enabled=False),
    )


def metrics_have_numeric(metrics: Any) -> bool:
    """Return True if a metrics mapping contains any finite numeric value."""

    if not isinstance(metrics, dict):
        return False

    def _finite(value: Any) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )

    for value in metrics.values():
        if _finite(value):
            return True
        if isinstance(value, dict):
            for nested in value.values():
                if _finite(nested):
                    return True
    return False
