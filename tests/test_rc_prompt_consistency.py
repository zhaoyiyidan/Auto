"""Tests for cross-domain prompt catalog consistency."""

from __future__ import annotations

from researchclaw.prompts.consistency import (
    CANONICAL_STAGES,
    check_domain_consistency,
)
from researchclaw.prompts.manager import SUPPORTED_DOMAINS


def test_all_domains_implement_canonical_stages() -> None:
    report = check_domain_consistency()

    assert len(CANONICAL_STAGES) == 20
    assert set(report.missing_stages) == set(SUPPORTED_DOMAINS)
    assert all(not missing for missing in report.missing_stages.values())


def test_json_mode_consistent_across_domains() -> None:
    report = check_domain_consistency()

    assert report.json_mode_mismatches == {}


def test_required_vars_consistent_across_domains() -> None:
    report = check_domain_consistency()

    assert report.required_var_mismatches == {}


def test_partial_domain_inherits_ml_passes() -> None:
    report = check_domain_consistency(domains=("biology_metabolic",))

    assert report.missing_stages == {"biology_metabolic": ()}
    assert report.json_mode_mismatches == {}
    assert report.required_var_mismatches == {}
    assert report.meta_gaps == {}
