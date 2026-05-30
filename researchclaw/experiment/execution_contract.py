"""Agent-declared execution contract models for workspace experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


CONTRACT_SCHEMA_VERSION = "researchclaw.execution_contract.v1"
EVIDENCE_SCHEMA_VERSION = "researchclaw.contract_evidence.v1"
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


@dataclass(frozen=True)
class ArtifactCheckEvidence:
    path: str
    required: bool
    exists: bool
    type_ok: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(self.path))
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "exists", bool(self.exists))
        object.__setattr__(self, "type_ok", bool(self.type_ok))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "required": self.required,
            "exists": self.exists,
            "type_ok": self.type_ok,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactCheckEvidence:
        return cls(
            path=str(data.get("path", "")),
            required=bool(data.get("required", False)),
            exists=bool(data.get("exists", False)),
            type_ok=bool(data.get("type_ok", True)),
        )


@dataclass(frozen=True)
class MetricCheckEvidence:
    name: str
    present: bool
    type_ok: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "present", bool(self.present))
        object.__setattr__(self, "type_ok", bool(self.type_ok))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "present": self.present,
            "type_ok": self.type_ok,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricCheckEvidence:
        return cls(
            name=str(data.get("name", "")),
            present=bool(data.get("present", False)),
            type_ok=bool(data.get("type_ok", True)),
        )


@dataclass(frozen=True)
class AgentCheckEvidence:
    requested: bool = False
    found: bool = False
    location: str = ""
    field: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested", bool(self.requested))
        object.__setattr__(self, "found", bool(self.found))
        object.__setattr__(self, "location", str(self.location or ""))
        object.__setattr__(self, "field", str(self.field or ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "found": self.found,
            "location": self.location,
            "field": self.field,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCheckEvidence:
        return cls(
            requested=bool(data.get("requested", False)),
            found=bool(data.get("found", False)),
            location=str(data.get("location") or ""),
            field=str(data.get("field") or ""),
        )


@dataclass(frozen=True)
class ContractEvidence:
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    completion_status: str = "failed"
    metrics_present: bool = False
    metrics_have_numeric: bool = False
    artifact_checks: tuple[ArtifactCheckEvidence, ...] = ()
    artifacts_declared: bool = False
    any_artifact_exists: bool = False
    metric_checks: tuple[MetricCheckEvidence, ...] = ()
    agent_checks: AgentCheckEvidence = field(default_factory=AgentCheckEvidence)
    violations: tuple[str, ...] = ()
    ok: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema_version", str(self.schema_version or EVIDENCE_SCHEMA_VERSION)
        )
        object.__setattr__(self, "completion_status", str(self.completion_status))
        object.__setattr__(self, "metrics_present", bool(self.metrics_present))
        object.__setattr__(
            self, "metrics_have_numeric", bool(self.metrics_have_numeric)
        )
        object.__setattr__(
            self,
            "artifact_checks",
            tuple(
                item
                if isinstance(item, ArtifactCheckEvidence)
                else ArtifactCheckEvidence.from_dict(
                    item if isinstance(item, dict) else {}
                )
                for item in self.artifact_checks
            ),
        )
        object.__setattr__(self, "artifacts_declared", bool(self.artifacts_declared))
        object.__setattr__(self, "any_artifact_exists", bool(self.any_artifact_exists))
        object.__setattr__(
            self,
            "metric_checks",
            tuple(
                item
                if isinstance(item, MetricCheckEvidence)
                else MetricCheckEvidence.from_dict(item if isinstance(item, dict) else {})
                for item in self.metric_checks
            ),
        )
        if not isinstance(self.agent_checks, AgentCheckEvidence):
            object.__setattr__(
                self,
                "agent_checks",
                AgentCheckEvidence.from_dict(
                    self.agent_checks if isinstance(self.agent_checks, dict) else {}
                ),
            )
        object.__setattr__(
            self, "violations", tuple(str(item) for item in self.violations)
        )
        object.__setattr__(self, "ok", bool(self.ok))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "completion_status": self.completion_status,
            "metrics_present": self.metrics_present,
            "metrics_have_numeric": self.metrics_have_numeric,
            "artifact_checks": [item.to_dict() for item in self.artifact_checks],
            "artifacts_declared": self.artifacts_declared,
            "any_artifact_exists": self.any_artifact_exists,
            "metric_checks": [item.to_dict() for item in self.metric_checks],
            "agent_checks": self.agent_checks.to_dict(),
            "violations": list(self.violations),
            "ok": self.ok,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContractEvidence:
        artifact_checks = data.get("artifact_checks") or ()
        metric_checks = data.get("metric_checks") or ()
        agent_checks = data.get("agent_checks") or {}
        violations = data.get("violations") or ()
        return cls(
            schema_version=str(data.get("schema_version") or EVIDENCE_SCHEMA_VERSION),
            completion_status=str(data.get("completion_status") or "failed"),
            metrics_present=bool(data.get("metrics_present", False)),
            metrics_have_numeric=bool(data.get("metrics_have_numeric", False)),
            artifact_checks=tuple(
                ArtifactCheckEvidence.from_dict(item)
                for item in artifact_checks
                if isinstance(item, dict)
            )
            if isinstance(artifact_checks, list | tuple)
            else (),
            artifacts_declared=bool(data.get("artifacts_declared", False)),
            any_artifact_exists=bool(data.get("any_artifact_exists", False)),
            metric_checks=tuple(
                MetricCheckEvidence.from_dict(item)
                for item in metric_checks
                if isinstance(item, dict)
            )
            if isinstance(metric_checks, list | tuple)
            else (),
            agent_checks=AgentCheckEvidence.from_dict(
                agent_checks if isinstance(agent_checks, dict) else {}
            ),
            violations=tuple(violations) if isinstance(violations, list | tuple) else (),
            ok=bool(data.get("ok", False)),
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


def evaluate_contract(
    contract: ExecutionContract,
    execution_record: dict[str, Any],
    result_artifacts: dict[str, Any],
) -> ContractEvidence:
    """Evaluate run evidence against an agent-declared contract."""

    final_status = str(execution_record.get("final_status", "")).strip().lower()
    violations: list[str] = []
    if final_status in contract.completion.incomplete_statuses:
        completion_status = "incomplete"
        violations.append(f"completion:incomplete:final_status={final_status}")
    elif final_status in contract.completion.success_statuses:
        completion_status = "complete"
    else:
        completion_status = "failed"
        violations.append(
            f"completion:failed:final_status={final_status or 'missing'}"
        )

    metrics = execution_record.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    metrics_present = bool(metrics)
    numeric_present = metrics_have_numeric(metrics)
    if contract.completion.require_any_metric and not metrics_present:
        violations.append("metrics:empty")

    artifact_rows = _artifact_rows(result_artifacts)
    artifacts_declared = bool(artifact_rows)
    any_artifact_exists = any(_artifact_exists(row) for row in artifact_rows)
    if artifacts_declared and not any_artifact_exists:
        violations.append("artifacts:all_missing")
    if contract.completion.require_any_artifact and not any_artifact_exists:
        violations.append("artifacts:all_missing")

    artifact_checks: list[ArtifactCheckEvidence] = []
    for check in contract.result_artifacts:
        exists = _artifact_path_exists(check.path, artifact_rows)
        type_ok = _artifact_type_ok(check, exists)
        artifact_checks.append(
            ArtifactCheckEvidence(
                path=check.path,
                required=check.required,
                exists=exists,
                type_ok=type_ok,
            )
        )
        if check.required and not exists:
            violations.append(f"artifact:{check.path}:missing")
        elif check.required and not type_ok:
            violations.append(f"artifact:{check.path}:wrong_type")

    metric_checks: list[MetricCheckEvidence] = []
    for check in contract.metrics.required:
        present = check.name in metrics
        type_ok = _metric_type_ok(metrics.get(check.name), check.type) if present else True
        metric_checks.append(
            MetricCheckEvidence(name=check.name, present=present, type_ok=type_ok)
        )
        if check.required and not present:
            violations.append(f"metric:{check.name}:missing")
        elif check.required and present and not type_ok:
            violations.append(f"metric:{check.name}:wrong_type")

    agent = contract.agent_declared
    agent_found = bool(
        agent.enabled
        and agent.field
        and isinstance(metrics, dict)
        and agent.field in metrics
    )
    agent_checks = AgentCheckEvidence(
        requested=agent.enabled,
        found=agent_found,
        location=agent.location,
        field=agent.field,
    )
    if agent.enabled and agent.required and not agent_found:
        field = agent.field or "agent_declared"
        violations.append(f"agent_declared:{field}:missing")

    return ContractEvidence(
        completion_status=completion_status,
        metrics_present=metrics_present,
        metrics_have_numeric=numeric_present,
        artifact_checks=tuple(artifact_checks),
        artifacts_declared=artifacts_declared,
        any_artifact_exists=any_artifact_exists,
        metric_checks=tuple(metric_checks),
        agent_checks=agent_checks,
        violations=tuple(violations),
        ok=not violations,
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


def _artifact_rows(result_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result_artifacts.get("artifacts") if isinstance(result_artifacts, dict) else None
    if not isinstance(rows, list):
        return []
    return [item for item in rows if isinstance(item, dict)]


def _artifact_exists(row: dict[str, Any]) -> bool:
    return bool(row.get("exists"))


def _artifact_path_exists(path: str, rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if str(row.get("path", "")) == path and _artifact_exists(row):
            return True
    return False


def _artifact_type_ok(check: ArtifactCheck, exists: bool) -> bool:
    if not exists:
        return True
    return check.type.strip().lower() in {"", "file", "json", "metrics_json"}


def _metric_type_ok(value: Any, type_name: str) -> bool:
    expected = type_name.strip().lower()
    if expected == "number":
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    if expected == "int":
        return not isinstance(value, bool) and isinstance(value, int)
    if expected == "string":
        return isinstance(value, str)
    if expected == "bool":
        return isinstance(value, bool)
    return True
