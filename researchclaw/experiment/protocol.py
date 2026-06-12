"""Deterministic hypothesis parsing helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_DIRECTION_MINIMIZE = {"minimize", "min", "lower", "lower_is_better", "down", "decrease"}
_DIRECTION_MAXIMIZE = {"maximize", "max", "higher", "higher_is_better", "up", "increase"}
_COMPARATORS = {"gt", "gte", "lt", "lte", "delta_gt", "delta_lt", "abs_delta_lt", "within_pct"}


def _coerce_direction(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _DIRECTION_MINIMIZE:
        return "minimize"
    if text in _DIRECTION_MAXIMIZE:
        return "maximize"
    return "maximize"


def _coerce_tuple(value: Any, *, split_commas: bool = False) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = re.split(r"[,;]\s*", value) if split_commas else (value,)
    elif isinstance(value, list | tuple | set):
        items = value
    else:
        items = (value,)
    return tuple(str(item).strip() for item in items if str(item).strip())


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_prefixed_id(value: Any, prefix: str, default_index: int = 1) -> str:
    text = str(value or "").strip()
    match = re.search(r"(\d+)", text)
    if match:
        return f"{prefix}{int(match.group(1))}"
    if text.upper().startswith(prefix):
        return text.upper()
    return f"{prefix}{default_index}"


@dataclass(frozen=True)
class HypothesisSpec:
    id: str
    statement: str
    prediction: str = ""
    falsification: str = ""
    rationale: str = ""
    baselines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _normalize_prefixed_id(self.id, "H"))
        object.__setattr__(self, "statement", str(self.statement or "").strip())
        object.__setattr__(self, "prediction", str(self.prediction or "").strip())
        object.__setattr__(self, "falsification", str(self.falsification or "").strip())
        object.__setattr__(self, "rationale", str(self.rationale or "").strip())
        object.__setattr__(
            self,
            "baselines",
            _coerce_tuple(self.baselines, split_commas=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "prediction": self.prediction,
            "falsification": self.falsification,
            "rationale": self.rationale,
            "baselines": list(self.baselines),
        }

    @classmethod
    def from_dict(cls, data: Any) -> HypothesisSpec:
        data = data if isinstance(data, dict) else {}
        return cls(
            id=data.get("id", "H1"),
            statement=data.get("statement") or data.get("hypothesis") or "",
            prediction=data.get("prediction") or data.get("measurable_prediction") or "",
            falsification=(
                data.get("falsification")
                or data.get("failure_condition")
                or data.get("failure")
                or ""
            ),
            rationale=data.get("rationale") or "",
            baselines=data.get("baselines") or data.get("required_baselines") or (),
        )


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str = "maximize"
    unit: str = ""
    description: str = ""
    hypothesis_ids: tuple[str, ...] = ()
    is_primary: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name or "primary_metric").strip())
        object.__setattr__(self, "direction", _coerce_direction(self.direction))
        object.__setattr__(self, "unit", str(self.unit or "").strip())
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(
            self,
            "hypothesis_ids",
            tuple(_normalize_prefixed_id(item, "H") for item in _coerce_tuple(self.hypothesis_ids)),
        )
        object.__setattr__(self, "is_primary", bool(self.is_primary))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "unit": self.unit,
            "description": self.description,
            "hypothesis_ids": list(self.hypothesis_ids),
            "is_primary": self.is_primary,
        }

    @classmethod
    def from_dict(cls, data: Any) -> MetricSpec:
        data = data if isinstance(data, dict) else {}
        return cls(
            name=data.get("name") or data.get("key") or "primary_metric",
            direction=data.get("direction") or "maximize",
            unit=data.get("unit") or "",
            description=data.get("description") or "",
            hypothesis_ids=data.get("hypothesis_ids") or (),
            is_primary=data.get("is_primary", False),
        )


@dataclass(frozen=True)
class ComparisonSpec:
    id: str
    kind: str = "baseline_vs_treatment"
    baseline: str = ""
    treatment: str = ""
    conditions: tuple[str, ...] = ()
    metric: str = ""
    hypothesis_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _normalize_prefixed_id(self.id, "C"))
        object.__setattr__(self, "kind", str(self.kind or "baseline_vs_treatment").strip())
        object.__setattr__(self, "baseline", str(self.baseline or "").strip())
        object.__setattr__(self, "treatment", str(self.treatment or "").strip())
        object.__setattr__(self, "conditions", _coerce_tuple(self.conditions, split_commas=True))
        object.__setattr__(self, "metric", str(self.metric or "").strip())
        object.__setattr__(
            self,
            "hypothesis_ids",
            tuple(_normalize_prefixed_id(item, "H") for item in _coerce_tuple(self.hypothesis_ids)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "baseline": self.baseline,
            "treatment": self.treatment,
            "conditions": list(self.conditions),
            "metric": self.metric,
            "hypothesis_ids": list(self.hypothesis_ids),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ComparisonSpec:
        data = data if isinstance(data, dict) else {}
        return cls(
            id=data.get("id", "C1"),
            kind=data.get("kind") or "baseline_vs_treatment",
            baseline=data.get("baseline") or "",
            treatment=data.get("treatment") or "",
            conditions=data.get("conditions") or (),
            metric=data.get("metric") or "",
            hypothesis_ids=data.get("hypothesis_ids") or (),
        )


@dataclass(frozen=True)
class DecisionRule:
    hypothesis_id: str
    metric: str
    comparator: str = "gt"
    threshold: float = 0.0
    baseline_metric: str = ""
    supported_if: str = "pass"
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypothesis_id", _normalize_prefixed_id(self.hypothesis_id, "H"))
        object.__setattr__(self, "metric", str(self.metric or "").strip())
        comparator = str(self.comparator or "gt").strip().lower()
        if comparator not in _COMPARATORS:
            comparator = "gt"
        object.__setattr__(self, "comparator", comparator)
        object.__setattr__(self, "threshold", _coerce_float(self.threshold))
        object.__setattr__(self, "baseline_metric", str(self.baseline_metric or "").strip())
        supported_if = str(self.supported_if or "pass").strip().lower()
        object.__setattr__(self, "supported_if", supported_if if supported_if == "fail" else "pass")
        object.__setattr__(self, "description", str(self.description or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "metric": self.metric,
            "comparator": self.comparator,
            "threshold": self.threshold,
            "baseline_metric": self.baseline_metric,
            "supported_if": self.supported_if,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Any) -> DecisionRule:
        data = data if isinstance(data, dict) else {}
        return cls(
            hypothesis_id=data.get("hypothesis_id") or data.get("hypothesis") or "H1",
            metric=data.get("metric") or "",
            comparator=data.get("comparator") or "gt",
            threshold=data.get("threshold", 0.0),
            baseline_metric=data.get("baseline_metric") or "",
            supported_if=data.get("supported_if") or "pass",
            description=data.get("description") or "",
        )


_HYPOTHESIS_HEADING_RE = re.compile(
    r"^#{1,6}\s*((?:H\s*\d+)|(?:Hypothesis\s*\d+))\b[^\n]*\n?",
    re.IGNORECASE | re.MULTILINE,
)
_TRAILER_RE = re.compile(r"^#{1,6}\s*(Generated|Notes)\b.*$", re.IGNORECASE | re.MULTILINE)
_LABEL_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?"
    r"(?P<label>hypothesis\s+statement|statement|measurable\s+prediction|prediction|"
    r"failure\s+condition|falsification|falsifi\w*|rationale|required\s+baselines|baselines)"
    r"(?:\*\*)?\s*:\s*(?P<value>.*)$",
    re.IGNORECASE,
)


def parse_hypotheses_md(text: str) -> tuple[HypothesisSpec, ...]:
    """Parse Stage 8 markdown into normalized hypothesis specs without LLM calls."""

    source = str(text or "").strip()
    if not source:
        return ()
    matches = list(_HYPOTHESIS_HEADING_RE.finditer(source))
    if not matches:
        stripped = _strip_trailer(source).strip()
        return (HypothesisSpec(id="H1", statement=stripped),) if stripped else ()

    hypotheses: list[HypothesisSpec] = []
    for index, match in enumerate(matches, start=1):
        next_start = matches[index].start() if index < len(matches) else len(source)
        section = _strip_trailer(source[match.end() : next_start]).strip()
        if not section:
            continue
        spec = _parse_hypothesis_section(
            section,
            _normalize_prefixed_id(match.group(1), "H", default_index=index),
        )
        if spec.statement:
            hypotheses.append(spec)
    return tuple(hypotheses)


def _strip_trailer(text: str) -> str:
    match = _TRAILER_RE.search(text)
    return text[: match.start()] if match else text


def _parse_hypothesis_section(section: str, hypothesis_id: str) -> HypothesisSpec:
    fields: dict[str, str] = {}
    unlabeled: list[str] = []
    current: str | None = None

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            current = None
            continue
        label_match = _LABEL_RE.match(line.replace("**", ""))
        if label_match:
            current = _canonical_label(label_match.group("label"))
            fields[current] = label_match.group("value").strip()
            continue
        if current is not None:
            fields[current] = (fields.get(current, "") + " " + line).strip()
        else:
            unlabeled.append(_strip_markdown_bullet(line))

    statement = fields.get("statement", "").strip()
    if not statement:
        statement = " ".join(item for item in unlabeled if item).strip()
    return HypothesisSpec(
        id=hypothesis_id,
        statement=statement,
        prediction=fields.get("prediction", ""),
        falsification=fields.get("falsification", ""),
        rationale=fields.get("rationale", ""),
        baselines=fields.get("baselines", ()),
    )


def _canonical_label(label: str) -> str:
    label = re.sub(r"\s+", " ", label.strip().lower())
    if label in {"hypothesis statement", "statement"}:
        return "statement"
    if label in {"measurable prediction", "prediction"}:
        return "prediction"
    if label in {"failure condition", "falsification"} or label.startswith("falsif"):
        return "falsification"
    if label == "rationale":
        return "rationale"
    return "baselines"


def _strip_markdown_bullet(line: str) -> str:
    line = re.sub(r"^\s*[-*]\s*", "", line)
    line = line.strip().strip("*").strip()
    return line
