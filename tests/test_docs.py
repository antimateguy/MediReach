"""Phase 9: the headline numbers quoted in the docs must match the artifacts.

Guards v1 defect D8 (docs that contradicted the metrics). If a pipeline change
moves a number, these tests say which document needs updating.
"""
from __future__ import annotations

import os

import pytest

from medireach import artifacts as files
from medireach.artifacts import ArtifactStore
from medireach.config import PROJECT_ROOT, Settings

_STORE = ArtifactStore(Settings().artifacts_dir)
pytestmark = pytest.mark.skipif(
    not os.path.exists(_STORE.path(files.IMPACT_SUMMARY)), reason="run `python -m medireach.pipelines.run_all` first",
)


def _read(*parts: str) -> str:
    with open(os.path.join(PROJECT_ROOT, *parts), "r", encoding="utf-8") as handle:
        return handle.read()


def _missing(document: str, expected: dict[str, str]) -> list[str]:
    return [f"{label}: {value}" for label, value in expected.items() if value not in document]


def _guide_missing(expected: dict[str, str]) -> list[str]:
    """The interview guide is a personal document kept outside the repository; check it only where it exists."""
    path = os.path.join(PROJECT_ROOT, "docs", "INTERVIEW_GUIDE.md")
    return _missing(_read("docs", "INTERVIEW_GUIDE.md"), expected) if os.path.exists(path) else []


def test_readme_and_guide_quote_the_model_metrics() -> None:
    metrics = _STORE.read_json(files.METRICS)
    expected = {}
    for name, results in metrics["models"].items():
        expected[f"{name} test ROC-AUC"] = f"{results['test']['roc_auc']:.3f}"
        expected[f"{name} precision at 10%"] = f"{results['test']['precision_at_10']:.1%}"
    segments = _STORE.read_csv(files.SEGMENT_METRICS)
    selected = segments[(segments["model"] == metrics["selected_model"]) & (segments["value"] == "booked_ahead")]
    expected["ROC-AUC among appointments booked ahead"] = f"{selected['roc_auc'].iloc[0]:.3f}"
    assert not _missing(_read("README.md"), expected)
    guide_subset = {label: value for label, value in expected.items() if "random_forest precision" not in label}
    assert not _guide_missing(guide_subset)


def test_readme_and_guide_quote_the_impact_numbers() -> None:
    comparison = _STORE.read_csv(files.IMPACT_COMPARISON).set_index("strategy")
    summary = _STORE.read_json(files.IMPACT_SUMMARY)
    readme = {f"net benefit, {strategy}": f"{comparison.loc[strategy, 'net']:,.0f}"
              for strategy in ("Random (no targeting)", "Lead-time baseline", "Logistic regression", "Oracle (perfect knowledge)")}
    shared = {
        "break-even call effectiveness": f"{summary['break_even_call_effectiveness']:.1%}",
        "pilot patients per arm": f"{summary['pilot_plan']['sample_size_per_arm']:,}",
    }
    assert not _missing(_read("README.md"), {**readme, **shared})
    assert not _guide_missing(shared)


def test_data_notes_quote_the_cleaning_report() -> None:
    report = _STORE.read_json(files.CLEANING_REPORT)
    expected = {
        "rows in": f"{report['rows_in']:,}", "rows out": f"{report['rows_out']:,}",
        "patients": f"{report['patients']:,}", "same-day rows": f"{report['same_day_rows']:,}",
        "patients with varying attributes": f"{report['patients_with_varying_attributes']:,}",
        "same-day no-show rate": f"{report['same_day_no_show_rate']:.1%}",
    }
    assert not _missing(_read("docs", "DATA_NOTES.md"), expected)


def test_insights_quote_each_hypothesis_verdict_and_headline_rates() -> None:
    insights = _read("docs", "INSIGHTS.md")
    results = {result["hypothesis_id"]: result for result in _STORE.read_json(files.HYPOTHESES)}
    for hypothesis_id, result in results.items():
        assert f"**Verdict:** {result['verdict']}" in insights, hypothesis_id
    details = {**results["H1"]["details"], **results["H2"]["details"], **results["H3"]["details"]}
    expected = {
        "H1 long-lead rate": f"{details['long_vs_1_7_rate_1']:.1%}", "H1 same-day rate": f"{details['same_day_rate']:.1%}",
        "H2 prior no-show rate": f"{details['prior_vs_none_rate_1']:.1%}",
        "H3 naive gap": f"{100 * details['naive_difference']:+.1f} pts",
        "H3 stratified gap": f"{100 * details['adjusted_difference']:+.1f} pts",
    }
    assert not _missing(insights, expected)


def test_readme_and_guide_quote_the_feature_study_and_the_replay_validation() -> None:
    ablation = _STORE.read_csv(files.FEATURE_ABLATION)
    steps = ablation[(ablation["study"] == "added one at a time") & (ablation["label"] != "+ Clinic location (encoded)")]
    replay = _STORE.read_json(files.REPLAY_VALIDATION)
    expected = {"lead time alone": f"{steps.iloc[0]['roc_auc']:.3f}", "all groups": f"{steps.iloc[-1]['roc_auc']:.3f}"}
    for source in ("historical", "simulated"):
        part = replay[source]
        low, high = part["reduction_ci"]
        expected[f"{source} reduction"] = f"{100 * part['relative_reduction']:+.1f}%"
        expected[f"{source} CI"] = f"{100 * low:+.1f}% to {100 * high:+.1f}%"
    assert not _missing(_read("README.md"), expected)
    assert not _guide_missing(expected)
