"""RawDataLoader: raw CSV to cleaned frames plus a CleaningReport."""
from __future__ import annotations

import logging

import pandas as pd

from medireach.config import Settings
from medireach.domain import CleaningReport
from medireach.errors import DataAccessError

logger = logging.getLogger(__name__)

# Raw Kaggle name -> our name. Fixes the typos in the source file.
COLUMN_MAP: dict[str, str] = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_ts",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    # The data dictionary: "where the appointment takes place" (the health unit).
    "Neighbourhood": "clinic",
    "Scholarship": "scholarship",
    "Hipertension": "hypertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",
    "SMS_received": "sms_received",
    "No-show": "no_show",
}
PATIENT_ATTRIBUTES: tuple[str, ...] = (
    "gender", "age", "scholarship", "hypertension", "diabetes", "alcoholism", "handicap",
)
PATIENT_COLUMNS: tuple[str, ...] = ("patient_id",) + PATIENT_ATTRIBUTES
APPOINTMENT_TABLE_COLUMNS: tuple[str, ...] = (
    "appointment_id", "patient_id", "clinic_id", "scheduled_ts", "scheduled_date", "appointment_date",
    "lead_days", "sms_received", "no_show",
)
_RECENCY_ORDER = ["appointment_date", "scheduled_ts", "appointment_id"]


class RawDataLoader:
    """Turns the raw Kaggle file into a patients frame and an appointments frame.

    Knows nothing about databases. Every removed row is counted against the
    rule that removed it, so the cleaning can be audited.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load(self, csv_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
        """Read the CSV and clean it. Returns (patients, appointments, report)."""
        path = csv_path or self._settings.raw_csv_path
        try:
            raw = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise DataAccessError("read_raw_csv", f"{exc} (see README for the download link)") from exc
        return self.clean(raw)

    def clean(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
        """Apply the cleaning rules in order. Returns (patients, appointments, report)."""
        df = self._rename(raw)
        collisions = self._fix_patient_id(df)
        self._encode_target(df)
        self._add_dates(df)
        rows_in, base_rate_before = len(df), float(df["no_show"].mean())
        v1_filter_drops = int(df.pop("_v1_filter_drops").sum())

        dropped: dict[str, int] = {}
        # Dates, not timestamps: a same-day booking has lead 0 and is valid.
        df = self._drop(df, df["lead_days"] < 0, "negative_lead_days", dropped)
        df = self._drop(df, ~df["age"].between(0, self._settings.max_age), "invalid_age", dropped)
        df = self._drop(df, df["appointment_id"].duplicated(keep="first"), "duplicate_appointment_id", dropped)

        self._add_clinic_ids(df)
        varying = self._varying_attributes(df)
        same_day = df["lead_days"] == 0
        report = CleaningReport(
            rows_in=rows_in,
            rows_out=len(df),
            dropped=dropped,
            patients=int(df["patient_id"].nunique()),
            clinics=int(df["clinic_id"].nunique()),
            patients_with_varying_attributes=int(varying.any(axis=1).sum()),
            varying_by_attribute={name: int(count) for name, count in varying.sum().items()},
            patient_id_collisions=collisions,
            base_rate_before=base_rate_before,
            base_rate_after=float(df["no_show"].mean()),
            same_day_rows=int(same_day.sum()),
            v1_timestamp_filter_drops=v1_filter_drops,
            same_day_share=float(same_day.mean()),
            same_day_no_show_rate=float(df.loc[same_day, "no_show"].mean()) if same_day.any() else 0.0,
        )
        logger.info("Cleaning kept %d of %d rows; dropped %s", report.rows_out, rows_in, dropped)
        return self._patients(df), self._appointments(df), report

    @staticmethod
    def _rename(raw: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(COLUMN_MAP) - set(raw.columns))
        if missing:
            raise DataAccessError("clean_raw_data", f"raw file is missing columns: {missing}")
        return raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)

    @staticmethod
    def _fix_patient_id(df: pd.DataFrame) -> int:
        """Float ids (the file uses scientific notation) to integers. Returns collisions."""
        as_float = df["patient_id"].astype("float64")
        as_int = as_float.round().astype("int64")
        # A collision would mean rounding merged two different patients.
        collisions = int(as_float.nunique() - as_int.nunique())
        df["patient_id"] = as_int
        return collisions

    @staticmethod
    def _encode_target(df: pd.DataFrame) -> None:
        encoded = df["no_show"].map({"Yes": 1, "No": 0})
        if encoded.isna().any():
            bad = sorted(df.loc[encoded.isna(), "no_show"].astype(str).unique())
            raise DataAccessError("clean_raw_data", f"unexpected No-show values: {bad}")
        df["no_show"] = encoded.astype("int64")

    @staticmethod
    def _add_dates(df: pd.DataFrame) -> None:
        scheduled = pd.to_datetime(df["scheduled_ts"], utc=True).dt.tz_localize(None)
        appointment = pd.to_datetime(df["appointment_day"], utc=True).dt.tz_localize(None)
        # AppointmentDay carries no time (always midnight), so both sides are
        # reduced to dates before subtracting. This is the fix for the v1 bug.
        df["scheduled_date"] = scheduled.dt.normalize()
        df["appointment_date"] = appointment.dt.normalize()
        df["scheduled_ts"] = scheduled.dt.strftime("%Y-%m-%dT%H:%M:%S")
        df["lead_days"] = (df["appointment_date"] - df["scheduled_date"]).dt.days.astype("int64")
        # Kept only to document the v1 bug: comparing the timestamp with midnight.
        df["_v1_filter_drops"] = scheduled > appointment

    @staticmethod
    def _drop(df: pd.DataFrame, bad: pd.Series, rule: str, dropped: dict[str, int]) -> pd.DataFrame:
        dropped[rule] = int(bad.sum())
        return df.loc[~bad].copy()

    @staticmethod
    def _add_clinic_ids(df: pd.DataFrame) -> None:
        """Number the clinics 1..k in alphabetical order, so the ids are stable between builds."""
        names = sorted(df["clinic"].unique())
        df["clinic_id"] = df["clinic"].map({name: index for index, name in enumerate(names, start=1)}).astype("int64")

    @staticmethod
    def _varying_attributes(df: pd.DataFrame) -> pd.DataFrame:
        """Per patient and attribute: does the value change across that patient's rows?"""
        return df.groupby("patient_id")[list(PATIENT_ATTRIBUTES)].nunique() > 1

    @staticmethod
    def _patients(df: pd.DataFrame) -> pd.DataFrame:
        """One row per patient, attributes taken from the most recent appointment."""
        latest = df.sort_values(_RECENCY_ORDER).groupby("patient_id", sort=True).tail(1)
        return latest[list(PATIENT_COLUMNS)].sort_values("patient_id").reset_index(drop=True)

    @staticmethod
    def _appointments(df: pd.DataFrame) -> pd.DataFrame:
        """Appointment rows with their clinic id, plus the clinic name for convenience."""
        ordered = df.sort_values(["appointment_date", "appointment_id"])
        return ordered[[*APPOINTMENT_TABLE_COLUMNS, "clinic"]].reset_index(drop=True)


def clinic_table(appointments: pd.DataFrame) -> pd.DataFrame:
    """One row per clinic (clinic_id, name), taken from the cleaned appointments."""
    clinics = appointments[["clinic_id", "clinic"]].drop_duplicates().rename(columns={"clinic": "name"})
    return clinics.sort_values("clinic_id").reset_index(drop=True)
