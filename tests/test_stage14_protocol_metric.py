from __future__ import annotations

import json
from pathlib import Path

from researchclaw.config import RCConfig
from researchclaw.experiment.protocol import ExperimentProtocol, MetricSpec


def _config(tmp_path: Path) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "stage14-protocol-test", "mode": "docs-first"},
            "research": {"topic": "protocol metric routing"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
            },
            "experiment": {},
        },
        project_root=tmp_path,
        check_paths=False,
    )


def test_build_experiment_summary_uses_protocol_metric(tmp_path: Path) -> None:
    from researchclaw.pipeline.stage_impls._analysis import _build_experiment_summary

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stage9 = run_dir / "stage-09"
    stage9.mkdir()
    (stage9 / "experiment_protocol.json").write_text(
        ExperimentProtocol(
            metrics=(MetricSpec(name="loss", direction="minimize", is_primary=True),)
        ).to_json(),
        encoding="utf-8",
    )
    stage12 = run_dir / "stage-12"
    stage12.mkdir()
    (stage12 / "execution_record.json").write_text(
        json.dumps(
            {
                "code_commit": "better",
                "metrics": {"loss": 0.2, "accuracy": 0.6},
                "final_status": "completed",
            }
        ),
        encoding="utf-8",
    )
    stage12_v1 = run_dir / "stage-12_v1"
    stage12_v1.mkdir()
    (stage12_v1 / "execution_record.json").write_text(
        json.dumps(
            {
                "code_commit": "worse",
                "metrics": {"loss": 0.5, "accuracy": 0.9},
                "final_status": "completed",
            }
        ),
        encoding="utf-8",
    )

    summary, _provenance = _build_experiment_summary(run_dir, _config(tmp_path))

    assert summary["primary_metric"] == "loss"
    assert summary["metric_direction"] == "minimize"
    assert summary["best_metric"] == 0.2
    assert summary["best_commit"] == "better"
