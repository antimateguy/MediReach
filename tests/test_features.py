"""Phase 4: the feature definition, with the leakage tripwire and SQL/pandas parity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medireach.config import Settings
from medireach.data.analytics import AnalyticsQueries
from medireach.data.database import SqliteDatabase
from medireach.data.repositories import SqliteAppointmentRepository
from medireach.features.builder import (
    AGE_GROUP_COLUMNS,
    CONTINUOUS_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    LEAD_BUCKET_COLUMNS,
    FeatureBuilder,
    bucket,
)
from tests.doubles import InMemoryAppointmentRepository
from tests.factories import TIE_PATIENT_INDEX, patient_id_for

PRIOR_RATE = 0.2


@pytest.fixture()
def appointments(cleaned) -> pd.DataFrame:
    patients, rows, _ = cleaned
    return InMemoryAppointmentRepository(patients, rows).get_appointments()


@pytest.fixture()
def builder(settings: Settings) -> FeatureBuilder:
    return FeatureBuilder(settings, prior_rate=PRIOR_RATE)


def _by_appointment(builder: FeatureBuilder, appointments: pd.DataFrame) -> pd.DataFrame:
    """Features indexed by appointment_id, so rows can be compared across builds."""
    features, meta = builder.build(appointments)
    return features.set_index(meta["appointment_id"])


# ------------------------------------------------------------------- leakage

def test_leakage_tripwire_own_label_never_changes_own_features(builder: FeatureBuilder, appointments: pd.DataFrame) -> None:
    original = _by_appointment(builder, appointments)
    for position in range(len(appointments)):
        flipped = appointments.copy()
        label_column = flipped.columns.get_loc("no_show")
        flipped.iloc[position, label_column] = 1 - flipped.iloc[position, label_column]
        appointment_id = flipped.iloc[position]["appointment_id"]
        rebuilt = _by_appointment(builder, flipped)
        pd.testing.assert_series_equal(rebuilt.loc[appointment_id], original.loc[appointment_id])


def test_a_label_only_affects_the_same_patients_later_dates(builder: FeatureBuilder, appointments: pd.DataFrame) -> None:
    patient = appointments[appointments["patient_id"] == patient_id_for(TIE_PATIENT_INDEX)]
    target = patient.iloc[0]
    flipped = appointments.copy()
    flipped.loc[flipped["appointment_id"] == target["appointment_id"], "no_show"] = 1 - target["no_show"]
    changed = (_by_appointment(builder, appointments) != _by_appointment(builder, flipped)).any(axis=1)
    changed_ids = set(changed[changed].index)
    later = patient[patient["appointment_date"] > target["appointment_date"]]
    assert changed_ids == set(later["appointment_id"]) and len(changed_ids) > 0


def test_same_day_appointments_share_one_history(builder: FeatureBuilder, appointments: pd.DataFrame) -> None:
    features, meta = builder.build(appointments)
    history = features[["has_history", "prior_appt_count", "prior_no_show_rate_smoothed", "had_prior_no_show"]]
    per_patient_day = history.join(meta[["patient_id", "appointment_date"]]).groupby(["patient_id", "appointment_date"])
    assert per_patient_day.size().max() >= 2
    assert (per_patient_day.nunique() == 1).all().all()


def test_unknown_outcomes_are_not_counted_as_history(builder: FeatureBuilder, appointments: pd.DataFrame) -> None:
    # Scoring tomorrow's list: tomorrow's outcomes do not exist yet.
    last_day = appointments["appointment_date"].max()
    unknown = appointments.astype({"no_show": "float64"})
    unknown.loc[unknown["appointment_date"] == last_day, "no_show"] = np.nan
    on_last_day = appointments.loc[appointments["appointment_date"] == last_day, "appointment_id"]
    original, rebuilt = _by_appointment(builder, appointments), _by_appointment(builder, unknown)
    pd.testing.assert_frame_equal(rebuilt.loc[on_last_day], original.loc[on_last_day])
    features, meta = builder.build(appointments.drop(columns=["no_show"]))
    assert "no_show" not in meta.columns and (features["has_history"] == 0).all()


# -------------------------------------------------------------------- values

def test_first_visit_has_no_history_and_the_average_rate(builder: FeatureBuilder, appointments: pd.DataFrame) -> None:
    features, meta = builder.build(appointments)
    first_date = meta.groupby("patient_id")["appointment_date"].transform("min")
    first = features[meta["appointment_date"] == first_date]
    assert (first["has_history"] == 0).all() and (first["prior_appt_count"] == 0).all()
    assert (first["had_prior_no_show"] == 0).all()
    assert first["prior_no_show_rate_smoothed"].to_numpy() == pytest.approx(PRIOR_RATE)


def test_history_values_stay_in_range(builder: FeatureBuilder, appointments: pd.DataFrame, settings: Settings) -> None:
    features, _ = builder.build(appointments)
    assert features["prior_no_show_rate_smoothed"].between(0, 1).all()
    assert features["prior_appt_count"].between(0, settings.prior_appt_cap).all()
    assert (features["has_history"] == (features["prior_appt_count"] > 0)).all()
    assert (features.loc[features["had_prior_no_show"] == 1, "has_history"] == 1).all()
    assert features["has_history"].sum() > 0 and features["had_prior_no_show"].sum() > 0


def test_smoothed_rate_formula(settings: Settings) -> None:
    builder = FeatureBuilder(settings, prior_rate=0.25)
    dates = pd.to_datetime(["2016-05-02", "2016-05-03", "2016-05-04", "2016-05-05"])
    rows = pd.DataFrame({
        "appointment_id": [1, 2, 3, 4], "patient_id": 7, "appointment_date": dates, "no_show": [1, 0, 1, 0],
        "lead_days": 2, "age": 40, "gender": "F", "handicap": 0, "sms_received": 0,
        "scholarship": 0, "hypertension": 0, "diabetes": 0, "alcoholism": 0,
    })
    features, _ = builder.build(rows)
    # (prior no-shows + 2 * 0.25) / (prior appointments + 2)
    assert list(features["prior_no_show_rate_smoothed"]) == pytest.approx([0.25, 1.5 / 3, 1.5 / 4, 2.5 / 5])
    assert list(features["prior_appt_count"]) == [0, 1, 2, 3]
    assert list(features["had_prior_no_show"]) == [0, 1, 1, 1]
    # 2 days ahead and age 40 are the reference categories: every one-hot column is 0.
    assert (features[list(LEAD_BUCKET_COLUMNS + AGE_GROUP_COLUMNS)] == 0).all().all()


def test_columns_order_types_and_no_missing_values(builder: FeatureBuilder, appointments: pd.DataFrame) -> None:
    features, meta = builder.build(appointments)
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert not features.isna().any().any()
    assert all(str(dtype) == "float64" for dtype in features.dtypes)
    assert features.index.equals(meta.index) and len(features) == len(appointments)
    assert list(meta.columns) == ["appointment_id", "patient_id", "appointment_date", "no_show"]
    binary = [column for column in FEATURE_COLUMNS if column not in CONTINUOUS_COLUMNS]
    assert features[binary].isin([0, 1]).all().all()
    assert (features[list(LEAD_BUCKET_COLUMNS)].sum(axis=1) <= 1).all()
    assert (features[list(AGE_GROUP_COLUMNS)].sum(axis=1) <= 1).all()


def test_flags_are_derived_from_the_raw_columns(builder: FeatureBuilder, appointments: pd.DataFrame) -> None:
    features, _ = builder.build(appointments)
    assert (features["has_handicap"] == (appointments["handicap"] > 0)).all()
    assert (features["gender_male"] == (appointments["gender"] == "M")).all()
    assert (features["lead_bucket_0"] == (appointments["lead_days"] == 0)).all()
    assert (features["is_weekend"] == 0).all()  # the fixture only has business days
    saturday = appointments.head(1).assign(appointment_date=pd.Timestamp("2016-05-14"))
    assert builder.build(saturday)[0]["is_weekend"].iloc[0] == 1


def test_every_feature_is_described_and_nothing_forbidden_is_a_feature() -> None:
    assert set(FEATURE_DESCRIPTIONS) == set(FEATURE_COLUMNS)
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS) == 21
    assert not {"no_show", "appointment_id", "patient_id"} & set(FEATURE_COLUMNS)


def test_feature_columns_match_the_settings_buckets(settings: Settings) -> None:
    lead = {f"lead_bucket_{label}" for label in settings.lead_bucket_labels if label != settings.lead_bucket_reference}
    age = {f"age_group_{label}" for label in settings.age_group_labels if label != settings.age_group_reference}
    assert lead == set(LEAD_BUCKET_COLUMNS) and age == set(AGE_GROUP_COLUMNS)


def test_bucket_edges_are_inclusive_upper_bounds(settings: Settings) -> None:
    values = pd.Series([0, 1, 3, 4, 7, 8, 14, 15, 30, 31, 179])
    labels = bucket(values, settings.lead_bucket_edges, settings.lead_bucket_labels)
    assert list(labels) == ["0", "1-3", "1-3", "4-7", "4-7", "8-14", "8-14", "15-30", "15-30", "31+", "31+"]


# ----------------------------------------------------------------- behaviour

def test_build_does_not_modify_its_input(builder: FeatureBuilder, appointments: pd.DataFrame) -> None:
    before = appointments.copy(deep=True)
    builder.build(appointments)
    pd.testing.assert_frame_equal(appointments, before)


def test_row_order_does_not_change_any_appointments_features(builder: FeatureBuilder, appointments: pd.DataFrame) -> None:
    shuffled = appointments.sample(frac=1.0, random_state=3)
    original, rebuilt = _by_appointment(builder, appointments), _by_appointment(builder, shuffled)
    pd.testing.assert_frame_equal(rebuilt.loc[original.index], original)


def test_bad_input_is_rejected(builder: FeatureBuilder, appointments: pd.DataFrame, settings: Settings) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        builder.build(appointments.drop(columns=["lead_days"]))
    with pytest.raises(ValueError, match="prior_rate"):
        FeatureBuilder(settings, prior_rate=1.2)


# -------------------------------------------------------- SQL / pandas parity

def test_history_counts_match_the_sql_view(builder: FeatureBuilder, database: SqliteDatabase) -> None:
    appointments = SqliteAppointmentRepository(database).get_appointments()
    from_pandas = builder.history_counts(appointments).set_index(appointments["appointment_id"])
    from_sql = AnalyticsQueries(database).patient_history().set_index("appointment_id").loc[from_pandas.index]
    assert list(from_pandas["prior_appts"]) == list(from_sql["prior_appointments"])
    assert list(from_pandas["prior_no_shows"]) == list(from_sql["prior_no_shows"])
    assert from_pandas["prior_appts"].sum() > 0
