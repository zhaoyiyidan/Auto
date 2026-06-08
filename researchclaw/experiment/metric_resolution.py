"""Protocol-first primary metric resolution for experiment runs."""

from __future__ import annotations

from pathlib import Path

from researchclaw.experiment.protocol import ExperimentProtocol
from researchclaw.experiment.workspace import TaskSpec
from researchclaw.pipeline._helpers import _find_prior_file


SAFE_DEFAULT_METRIC = ("primary_metric", "maximize")


def resolve_experiment_metric(run_dir: Path) -> tuple[str, str]:
    """Resolve ``(primary_metric_name, direction)`` for an experiment run.

    Resolution is intentionally artifact-only: protocol first, legacy task spec
    second, then a safe default. Scientific metrics no longer fall back to
    engineering config.
    """

    run_dir = Path(run_dir)

    protocol_path = _find_prior_file(run_dir, "experiment_protocol.json")
    if protocol_path is not None:
        protocol = ExperimentProtocol.from_path(protocol_path)
        primary = protocol.primary_metric()
        if primary.name:
            return primary.name, primary.direction

    task_spec_path = _find_prior_file(run_dir, "task_spec.yaml")
    if task_spec_path is not None:
        try:
            spec = TaskSpec.from_path(task_spec_path)
        except Exception:  # noqa: BLE001
            spec = None
        if spec is not None:
            contract = spec.execution_contract
            if contract is not None:
                primary = contract.metrics.primary
                if primary.name:
                    return primary.name, primary.direction
            if spec.primary_metric:
                return spec.primary_metric, spec.metric_direction

    return SAFE_DEFAULT_METRIC
