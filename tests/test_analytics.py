"""Phase 3: statistics helpers, analytical views, hypothesis checks."""
from __future__ import annotations

import json
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from medireach.config import Settings
from medireach.data.analytics import VIEW_NAMES, AnalyticsQueries, compare_rates, wilson_ci, with_wilson_ci
from medireach.data.cleaning import RawDataLoader
from medireach.data.database import IN_MEMORY, SqliteDatabase
from medireach.data.hypotheses import NOT_SUPPORTED, PARTLY, SUPPORTED, HypothesisTests, _verdict
from medireach.pipelines.build_db import populate
from tests.factories import TIE_PATIENT_INDEX, make_raw_appointments, patient_id_for


@pytest.fixture(scope="module")
def analytics() -> Iterator[AnalyticsQueries]:
    """A larger synthetic database, so that clinics clear the n >= 100 filter."""
    settings = Settings()
    patients, appointments, _ = RawDataLoader(settings).clean(make_raw_appointments(120, 900, seed=1))
    database = SqliteDatabase(IN_MEMORY)
    populate(database, settings, patients, appointments)
    yield AnalyticsQueries(database)
    database.close()


# ------------------------------------------------------------- stats helpers

@pytest.mark.parametrize("k, n, low, high", [
    (5, 10, 0.2366, 0.7634),
    (0, 10, 0.0, 0.2775),
    (10, 10, 0.7225, 1.0),
    (20, 100, 0.1334, 0.2888),
])
def test_wilson_ci_known_values(k: int, n: int, low: float, high: float) -> None:
    assert wilson_ci(k, n) == pytest.approx((low, high), abs=5e-4)


def test_wilson_ci_edge_cases() -> None:
    assert wilson_ci(0, 0) == (0.0, 1.0)
    narrow, wide = wilson_ci(200, 1000), wilson_ci(20, 100)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]


def test_compare_rates_known_value() -> None:
    result = compare_rates(50, 100, 30, 100)
    assert result.difference == pytest.approx(0.20)
    assert result.z == pytest.approx(2.8868, abs=1e-3)
    assert result.p_value == pytest.approx(0.00389, abs=1e-4)
    assert result.ci_low < 0.20 < result.ci_high and result.ci_low > 0


def test_compare_rates_is_symmetric() -> None:
    forward, backward = compare_rates(40, 200, 90, 300), compare_rates(90, 300, 40, 200)
    assert forward.difference == pytest.approx(-backward.difference)
    assert forward.p_value == pytest.approx(backward.p_value)
    assert (forward.ci_low, forward.ci_high) == pytest.approx((-backward.ci_high, -backward.ci_low))


def test_compare_rates_equal_groups_is_not_significant() -> None:
    result = compare_rates(20, 100, 40, 200)
    assert result.difference == 0 and result.p_value == pytest.approx(1.0)


def test_with_wilson_ci_adds_columns_without_mutating() -> None:
    frame = pd.DataFrame({"n": [100, 10], "no_shows": [20, 5]})
    result = with_wilson_ci(frame)
    assert list(frame.columns) == ["n", "no_shows"]
    assert (result["ci_low"] < result["no_shows"] / result["n"]).all()
    assert (result["ci_high"] > result["no_shows"] / result["n"]).all()


# --------------------------------------------------------------------- views

@pytest.mark.parametrize("view", VIEW_NAMES)
def test_every_view_returns_rows(analytics: AnalyticsQueries, view: str) -> None:
    frame = getattr(analytics, view.removeprefix("v_"))()
    assert len(frame) > 0


def test_grouped_views_account_for_every_appointment(analytics: AnalyticsQueries) -> None:
    no_shows, n = analytics.overall()
    for frame in (analytics.lead_time(), analytics.age_group(), analytics.weekday(), analytics.sms_by_lead()):
        assert int(frame["n"].sum()) == n
        assert int(frame["no_shows"].sum()) == no_shows
        assert frame["no_show_rate"].between(0, 1).all()


def test_sql_buckets_match_the_settings_edges(analytics: AnalyticsQueries) -> None:
    settings, base = Settings(), analytics.appointment_base()
    for value, edges, labels, column in (
        ("lead_days", settings.lead_bucket_edges, settings.lead_bucket_labels, "lead_bucket"),
        ("age", settings.age_group_edges, settings.age_group_labels, "age_group"),
    ):
        expected = pd.cut(base[value], bins=[-1, *edges, np.inf], labels=list(labels)).astype(str)
        assert (base[column] == expected).all()
        assert set(base[column]) <= set(labels)


def test_same_day_flag_and_weekday(analytics: AnalyticsQueries) -> None:
    base = analytics.appointment_base()
    assert ((base["lead_days"] == 0) == (base["is_same_day"] == 1)).all()
    assert (pd.to_datetime(base["appointment_date"]).dt.day_name() == base["weekday"]).all()


def test_patient_history_uses_strictly_earlier_dates(analytics: AnalyticsQueries) -> None:
    history = analytics.patient_history()
    # Brute force: pair every appointment with the same patient's appointments
    # on strictly earlier dates, then count them.
    pairs = history.merge(history, on="patient_id", suffixes=("", "_other"))
    earlier = pairs[pairs["appointment_date_other"] < pairs["appointment_date"]]
    expected = earlier.groupby("appointment_id")["no_show_other"].agg(["count", "sum"])
    expected = expected.reindex(history["appointment_id"], fill_value=0)
    assert list(history["prior_appointments"]) == list(expected["count"])
    assert list(history["prior_no_shows"]) == list(expected["sum"])

    # The fixture gives patient 0 two visits on one day: they must share one history.
    patient = history[history["patient_id"] == patient_id_for(TIE_PATIENT_INDEX)]
    per_day = patient.groupby("appointment_date")
    assert per_day.size().max() >= 2
    assert (per_day["prior_appointments"].nunique() == 1).all()
    assert (per_day["prior_no_shows"].nunique() == 1).all()
    assert sorted(patient["visit_number"]) == list(range(1, len(patient) + 1))
    assert history["previous_no_show"].isna().sum() == history["patient_id"].nunique()


def test_history_effect_only_counts_appointments_with_history(analytics: AnalyticsQueries) -> None:
    history = analytics.patient_history()
    assert int(analytics.history_effect()["n"].sum()) == int((history["prior_appointments"] > 0).sum())


def test_clinic_view_ranks_and_compares_to_the_true_overall_rate(analytics: AnalyticsQueries) -> None:
    areas = analytics.clinic()
    no_shows, n = analytics.overall()
    assert (areas["n"] >= 100).all()
    assert areas["overall_rate"].iloc[0] == pytest.approx(no_shows / n)
    assert areas["rate_rank"].min() == 1
    assert areas.sort_values("rate_rank")["no_show_rate"].is_monotonic_decreasing


def test_weekly_trend_changes_are_differences_to_the_previous_week(analytics: AnalyticsQueries) -> None:
    weeks = analytics.weekly_trend()
    assert pd.isna(weeks["rate_change"].iloc[0])
    assert weeks["n_change"].iloc[1] == weeks["n"].iloc[1] - weeks["n"].iloc[0]
    assert (pd.to_datetime(weeks["week_start"]).dt.day_name() == "Monday").all()


# ---------------------------------------------------------------- hypotheses

def test_all_six_hypotheses_produce_a_serialisable_result(analytics: AnalyticsQueries) -> None:
    results = HypothesisTests(analytics).run_all()
    assert [result.hypothesis_id for result in results] == ["H1", "H2", "H3", "H4", "H5", "H6"]
    for result in results:
        assert result.verdict in {SUPPORTED, PARTLY, NOT_SUPPORTED}
        assert result.summary and not result.table.empty
        json.dumps(result.to_dict())


def test_verdict_needs_significance_direction_and_size() -> None:
    assert _verdict(compare_rates(3000, 10000, 2000, 10000), +1) == SUPPORTED
    assert _verdict(compare_rates(3000, 10000, 2000, 10000), -1) == NOT_SUPPORTED
    assert _verdict(compare_rates(2100, 10000, 2000, 10000), +1) == NOT_SUPPORTED  # p = 0.08
    assert _verdict(compare_rates(21000, 100000, 20000, 100000), +1) == PARTLY     # real but 1 pt
    assert _verdict(compare_rates(5, 20, 4, 20), +1) == NOT_SUPPORTED


class _ConfoundedSms:
    """SMS goes mostly to the risky long-lead bucket, but lowers risk inside each bucket."""

    def sms_by_lead(self) -> pd.DataFrame:
        rows = [("1-3", 2, 0, 9000, 0.10), ("1-3", 2, 1, 1000, 0.08),
                ("15-30", 5, 0, 1000, 0.40), ("15-30", 5, 1, 9000, 0.34),
                ("0", 1, 0, 5000, 0.05)]
        frame = pd.DataFrame(rows, columns=["lead_bucket", "sort_key", "sms_received", "n", "rate"])
        frame["no_shows"] = (frame["n"] * frame["rate"]).round().astype(int)
        frame["no_show_rate"] = frame["no_shows"] / frame["n"]
        return frame.drop(columns="rate")


def test_h3_separates_the_naive_gap_from_the_stratified_one() -> None:
    result = HypothesisTests(_ConfoundedSms()).h3_sms()  # type: ignore[arg-type]
    assert result.details["naive_difference"] > 0          # SMS looks harmful overall
    assert result.details["adjusted_difference"] < 0       # and helpful within a bucket
    assert result.details["buckets_compared"] == 2         # the same-day bucket has no SMS rows
    assert result.details["buckets_sms_lower"] == 2
    assert result.verdict == PARTLY
