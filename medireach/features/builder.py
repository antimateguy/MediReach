"""FeatureBuilder, FEATURE_COLUMNS and the chronological split."""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

from medireach.config import Settings

# Reference categories (lead 1-3 days, age 35-49) have no column: a row in the
# reference group has zeros in every column of that block.
LEAD_BUCKET_COLUMNS: tuple[str, ...] = (
    "lead_bucket_0", "lead_bucket_4-7", "lead_bucket_8-14", "lead_bucket_15-30", "lead_bucket_31+",
)
AGE_GROUP_COLUMNS: tuple[str, ...] = (
    "age_group_0-17", "age_group_18-34", "age_group_50-64", "age_group_65+",
)
FLAG_COLUMNS: tuple[str, ...] = (
    "sms_received", "scholarship", "hypertension", "diabetes", "alcoholism",
    "has_handicap", "gender_male", "is_weekend",
)
HISTORY_COLUMNS: tuple[str, ...] = (
    "has_history", "prior_appt_count", "prior_no_show_rate_smoothed", "had_prior_no_show",
)
# The only non-binary columns; the logistic model standardises exactly these.
CONTINUOUS_COLUMNS: tuple[str, ...] = ("prior_appt_count", "prior_no_show_rate_smoothed")
FEATURE_COLUMNS: tuple[str, ...] = LEAD_BUCKET_COLUMNS + AGE_GROUP_COLUMNS + FLAG_COLUMNS + HISTORY_COLUMNS

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "lead_bucket_0": "Booked on the day of the appointment (vs 1-3 days ahead)",
    "lead_bucket_4-7": "Booked 4-7 days ahead (vs 1-3 days ahead)",
    "lead_bucket_8-14": "Booked 8-14 days ahead (vs 1-3 days ahead)",
    "lead_bucket_15-30": "Booked 15-30 days ahead (vs 1-3 days ahead)",
    "lead_bucket_31+": "Booked 31 or more days ahead (vs 1-3 days ahead)",
    "age_group_0-17": "Child, 0-17 (vs age 35-49)",
    "age_group_18-34": "Younger adult, 18-34 (vs age 35-49)",
    "age_group_50-64": "Age 50-64 (vs age 35-49)",
    "age_group_65+": "Age 65 or older (vs age 35-49)",
    "sms_received": "An SMS reminder was sent for this appointment",
    "scholarship": "Enrolled in the welfare programme (supportive outreach only)",
    "hypertension": "Has hypertension",
    "diabetes": "Has diabetes",
    "alcoholism": "Alcoholism recorded",
    "has_handicap": "Any handicap recorded (supportive outreach only)",
    "gender_male": "Male (vs female)",
    "is_weekend": "Appointment falls on a Saturday or Sunday",
    "has_history": "Patient has at least one earlier appointment on record",
    "prior_appt_count": "Number of earlier appointments (capped at 5)",
    "prior_no_show_rate_smoothed": "Share of earlier appointments missed, pulled towards the average when history is short",
    "had_prior_no_show": "Missed at least one earlier appointment",
}

META_COLUMNS: tuple[str, ...] = ("appointment_id", "patient_id", "appointment_date")
_REQUIRED: tuple[str, ...] = META_COLUMNS + (
    "lead_days", "age", "gender", "handicap", "sms_received", "scholarship",
    "hypertension", "diabetes", "alcoholism",
)


def bucket(values: pd.Series, edges: Sequence[int], labels: Sequence[str]) -> pd.Series:
    """Label each value with its bucket. Each edge is the inclusive upper bound of a bucket."""
    bins = [-np.inf, *edges, np.inf]
    return pd.cut(values, bins=bins, labels=list(labels)).astype(str)


class FeatureBuilder:
    """The single definition of model features, used by training and by scoring.

    `prior_rate` is the no-show rate of the training period. It is passed in
    (not computed here) so that scoring uses exactly the value training used.
    """

    def __init__(self, settings: Settings, prior_rate: float) -> None:
        if not 0.0 <= prior_rate <= 1.0:
            raise ValueError("prior_rate must be between 0 and 1")
        self._settings = settings
        self._prior_rate = prior_rate

    def build(self, appointments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (X, meta). Both use a fresh 0..n-1 index in the input's row order."""
        missing = sorted(set(_REQUIRED) - set(appointments.columns))
        if missing:
            raise ValueError(f"appointments are missing columns: {missing}")
        rows = appointments.reset_index(drop=True)
        blocks = [self._lead_buckets(rows), self._age_groups(rows), self._flags(rows), self._history(rows)]
        features = pd.concat(blocks, axis=1)[list(FEATURE_COLUMNS)].astype("float64")
        if features.isna().any().any():
            raise ValueError("feature matrix contains missing values")
        meta_columns = list(META_COLUMNS) + (["no_show"] if "no_show" in rows.columns else [])
        return features, rows[meta_columns].copy()

    def history_counts(self, appointments: pd.DataFrame) -> pd.DataFrame:
        """Per row: appointments and no-shows the patient had on strictly earlier dates.

        Steps: (1) one row per patient and date with that day's totals,
        (2) running total within the patient, (3) shift by one date so each
        date sees only earlier dates. Two appointments on the same day share
        one history, and no appointment ever sees its own outcome.
        Appointments without a known outcome are not counted as history.
        """
        if "no_show" in appointments.columns:
            outcome = appointments["no_show"].astype("float64")
        else:
            outcome = pd.Series(np.nan, index=appointments.index)
        keys = ["patient_id", "appointment_date"]
        daily = pd.DataFrame({
            "patient_id": appointments["patient_id"],
            "appointment_date": appointments["appointment_date"],
            "prior_appts": outcome.notna().astype("int64"),
            "prior_no_shows": outcome.fillna(0).astype("int64"),
        }).groupby(keys, sort=True).sum()
        running = daily.groupby(level="patient_id").cumsum()
        before = running.groupby(level="patient_id").shift(1, fill_value=0)
        return appointments[keys].join(before, on=keys)[["prior_appts", "prior_no_shows"]]

    def _lead_buckets(self, rows: pd.DataFrame) -> pd.DataFrame:
        labels = bucket(rows["lead_days"], self._settings.lead_bucket_edges, self._settings.lead_bucket_labels)
        return pd.DataFrame({column: labels == column.removeprefix("lead_bucket_") for column in LEAD_BUCKET_COLUMNS})

    def _age_groups(self, rows: pd.DataFrame) -> pd.DataFrame:
        labels = bucket(rows["age"], self._settings.age_group_edges, self._settings.age_group_labels)
        return pd.DataFrame({column: labels == column.removeprefix("age_group_") for column in AGE_GROUP_COLUMNS})

    @staticmethod
    def _flags(rows: pd.DataFrame) -> pd.DataFrame:
        flags = rows[["sms_received", "scholarship", "hypertension", "diabetes", "alcoholism"]].copy()
        flags["has_handicap"] = rows["handicap"] > 0
        flags["gender_male"] = rows["gender"] == "M"
        # Monday is 0, so 5 and 6 are Saturday and Sunday.
        flags["is_weekend"] = rows["appointment_date"].dt.dayofweek >= 5
        return flags

    def _history(self, rows: pd.DataFrame) -> pd.DataFrame:
        counts = self.history_counts(rows)
        strength = self._settings.history_prior_strength
        # Smoothing: with no history the rate equals the training average; each
        # real appointment pulls it towards the patient's own record.
        smoothed = (counts["prior_no_shows"] + strength * self._prior_rate) / (counts["prior_appts"] + strength)
        return pd.DataFrame({
            "has_history": counts["prior_appts"] > 0,
            "prior_appt_count": counts["prior_appts"].clip(upper=self._settings.prior_appt_cap),
            "prior_no_show_rate_smoothed": smoothed,
            "had_prior_no_show": counts["prior_no_shows"] > 0,
        })


class Split(NamedTuple):
    """Boolean row masks for the three periods, aligned to the frame that was split."""

    train: pd.Series
    validation: pd.Series
    test: pd.Series


def time_split(meta: pd.DataFrame, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> Split:
    """Chronological split: earliest dates train, latest dates test.

    Walks through the dates in order and cuts where the running share of rows
    reaches the train and train+validation fractions. A date is never split,
    so no day appears in two parts.
    """
    dates = meta["appointment_date"]
    running_share = dates.value_counts().sort_index().cumsum() / len(dates)
    train_end = running_share.index[(running_share >= fractions[0]).argmax()]
    validation_end = running_share.index[(running_share >= fractions[0] + fractions[1] - 1e-9).argmax()]
    split = Split(dates <= train_end, (dates > train_end) & (dates <= validation_end), dates > validation_end)
    if not all(mask.any() for mask in split):
        raise ValueError("not enough distinct dates to make three non-empty periods")
    return split
