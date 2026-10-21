"""Phase 5: ranking metrics, gains, calibration and segment slices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medireach.ml.evaluation import ModelEvaluator, expected_hits_in_top


@pytest.fixture()
def evaluator() -> ModelEvaluator:
    return ModelEvaluator()


@pytest.fixture()
def informative() -> tuple[np.ndarray, np.ndarray]:
    """1,000 rows where the score really is the probability of a no-show."""
    rng = np.random.default_rng(5)
    score = rng.uniform(0.02, 0.6, 1000)
    return (rng.random(1000) < score).astype(int), score


def test_perfect_scores_give_auc_one(evaluator: ModelEvaluator) -> None:
    y = np.array([0] * 80 + [1] * 20)
    summary = evaluator.summary(y, y.astype(float))
    assert summary["roc_auc"] == 1.0 and summary["pr_auc"] == pytest.approx(1.0)
    assert summary["precision_at_10"] == 1.0 and summary["capture_at_20"] == 1.0
    assert summary["base_rate"] == 0.2 and summary["lift_top_decile"] == pytest.approx(5.0)


def test_uninformative_scores_give_auc_about_half(evaluator: ModelEvaluator) -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.2).astype(int)
    assert evaluator.summary(y, np.full(5000, 0.2))["roc_auc"] == 0.5
    assert evaluator.summary(y, rng.random(5000))["roc_auc"] == pytest.approx(0.5, abs=0.03)


def test_tied_scores_do_not_depend_on_row_order(evaluator: ModelEvaluator) -> None:
    # Every row has the same score, so "the top 10%" is an arbitrary pick.
    # The fair answer is the base rate, whichever way the rows are ordered.
    y = np.array([1] * 30 + [0] * 70)
    constant = np.full(100, 0.3)
    assert evaluator.precision_at_k(y, constant, 0.10) == pytest.approx(0.30)
    assert evaluator.precision_at_k(y[::-1], constant, 0.10) == pytest.approx(0.30)
    assert evaluator.capture_rate_at_k(y, constant, 0.10) == pytest.approx(0.10)


def test_expected_hits_mixes_clear_winners_with_a_tied_group() -> None:
    y = np.array([1, 1, 1, 0, 0, 0])
    score = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1])
    # Top 3 = the 0.9 row (a no-show) plus 2 of the four tied rows (half are no-shows).
    assert expected_hits_in_top(y, score, 3) == pytest.approx(1 + 2 * 0.5)
    assert expected_hits_in_top(y, score, 0) == 0.0
    assert expected_hits_in_top(y, score, 99) == pytest.approx(3.0)


def test_precision_and_capture_by_hand(evaluator: ModelEvaluator) -> None:
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    score = np.linspace(1.0, 0.1, 10)
    assert evaluator.precision_at_k(y, score, 0.20) == 1.0
    assert evaluator.precision_at_k(y, score, 0.30) == pytest.approx(2 / 3)
    assert evaluator.capture_rate_at_k(y, score, 0.30) == pytest.approx(2 / 3)
    assert evaluator.capture_rate_at_k(y, score, 1.0) == 1.0


def test_gains_table_is_cumulative_and_lifts_average_to_one(evaluator: ModelEvaluator, informative) -> None:
    y, score = informative
    gains = evaluator.gains_table(y, score)
    assert list(gains["decile"]) == list(range(1, 11))
    assert int(gains["n"].sum()) == len(y)
    assert gains["no_shows"].sum() == pytest.approx(y.sum())
    assert gains["cumulative_capture"].is_monotonic_increasing
    assert gains["cumulative_capture"].iloc[-1] == pytest.approx(1.0)
    assert gains["lift"].mean() == pytest.approx(1.0)
    assert gains["lift"].iloc[0] > 1 > gains["lift"].iloc[-1]
    assert gains["cumulative_capture"].iloc[0] == pytest.approx(evaluator.capture_rate_at_k(y, score, 0.10))


def test_calibration_table_for_a_calibrated_and_a_biased_score(evaluator: ModelEvaluator, informative) -> None:
    y, score = informative
    table = evaluator.calibration_table(y, score)
    assert len(table) == 10 and int(table["n"].sum()) == len(y)
    assert table["mean_predicted"].is_monotonic_increasing
    assert table["gap"].abs().max() < 0.12
    inflated = evaluator.calibration_table(y, np.clip(score + 0.2, 0, 1))
    assert (inflated["gap"] > 0.05).all()


def test_calibration_keeps_ten_bins_when_scores_repeat(evaluator: ModelEvaluator) -> None:
    y = np.array([0, 1] * 50)
    table = evaluator.calibration_table(y, np.repeat([0.1, 0.4], 50))
    assert len(table) == 10 and (table["n"] == 10).all()


def test_segment_table_skips_small_and_single_class_segments(informative) -> None:
    y, score = informative
    segments = pd.DataFrame({
        "gender": np.where(np.arange(1000) < 600, "F", "M"),
        "site": np.where(np.arange(1000) < 950, "main", "tiny"),
    })
    table = ModelEvaluator(min_segment_rows=200).segment_table(y, score, segments)
    assert list(table.columns) == ["segment", "value", "n", "base_rate", "roc_auc"]
    assert set(zip(table["segment"], table["value"])) == {("gender", "F"), ("gender", "M"), ("site", "main")}
    assert table["roc_auc"].between(0.5, 1.0).all()
    all_show = ModelEvaluator(min_segment_rows=1).segment_table(np.zeros(10), np.linspace(0, 1, 10), segments.head(10))
    assert all_show.empty


def test_summary_has_every_headline_metric(evaluator: ModelEvaluator, informative) -> None:
    summary = evaluator.summary(*informative)
    expected = {"n", "base_rate", "roc_auc", "pr_auc", "brier", "lift_top_decile"}
    expected |= {f"{kind}_at_{k}" for kind in ("precision", "capture") for k in (5, 10, 20, 30)}
    assert set(summary) == expected
    assert summary["precision_at_5"] >= summary["precision_at_30"] > summary["base_rate"]
    assert summary["capture_at_5"] < summary["capture_at_30"]
