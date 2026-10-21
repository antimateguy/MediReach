"""Test doubles: in-memory stand-ins for the repository and model interfaces."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from medireach.data.repositories import (
    APPOINTMENT_COLUMNS,
    QUEUE_COLUMNS,
    TIMESTAMP_FORMAT,
    AppointmentRepository,
    QueueRepository,
)
from medireach.domain import QueueRun
from medireach.ml.models import ExplainableRiskModel, RiskModel


class InMemoryAppointmentRepository(AppointmentRepository):
    """Serves appointments from two DataFrames. No database involved."""

    def __init__(self, patients: pd.DataFrame, appointments: pd.DataFrame) -> None:
        joined = appointments.merge(patients, on="patient_id", how="inner", validate="many_to_one")
        joined["scheduled_ts"] = pd.to_datetime(joined["scheduled_ts"], format=TIMESTAMP_FORMAT)
        self._rows = joined[list(APPOINTMENT_COLUMNS)]

    def get_appointments(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        rows = self._rows
        if start is not None:
            rows = rows[rows["appointment_date"] >= pd.Timestamp(start)]
        if end is not None:
            rows = rows[rows["appointment_date"] <= pd.Timestamp(end)]
        return rows.sort_values(["appointment_date", "appointment_id"]).reset_index(drop=True)

    def date_range(self) -> tuple[date, date]:
        dates = self._rows["appointment_date"]
        return dates.min().date(), dates.max().date()

    def appointment_dates(self) -> list[date]:
        return sorted({value.date() for value in self._rows["appointment_date"]})


class ShuffledAppointmentRepository(AppointmentRepository):
    """Wraps a repository and returns its rows in a scrambled order with a scrambled index.

    Used to prove that nothing downstream depends on row position.
    """

    def __init__(self, inner: AppointmentRepository, seed: int = 11) -> None:
        self._inner, self._seed = inner, seed

    def get_appointments(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        return self._inner.get_appointments(start, end).sample(frac=1.0, random_state=self._seed)

    def date_range(self) -> tuple[date, date]:
        return self._inner.date_range()

    def appointment_dates(self) -> list[date]:
        return self._inner.appointment_dates()


class InMemoryQueueRepository(QueueRepository):
    """Keeps runs in a list. Append-only, like the real one."""

    def __init__(self) -> None:
        self._runs: list[tuple[QueueRun, pd.DataFrame]] = []

    def save_run(self, run: QueueRun, rows: pd.DataFrame) -> None:
        self._runs.append((run, rows[list(QUEUE_COLUMNS)].copy()))

    def get_run(self, run_id: str) -> pd.DataFrame:
        for run, rows in self._runs:
            if run.run_id == run_id:
                return rows.sort_values(["appointment_date", "day_rank", "appointment_id"]).reset_index(drop=True)
        return pd.DataFrame(columns=list(QUEUE_COLUMNS))

    def latest_run_id(self) -> str | None:
        return max(self._runs, key=lambda item: (item[0].created_at, item[0].run_id))[0].run_id if self._runs else None

    def list_runs(self) -> pd.DataFrame:
        return pd.DataFrame([run.__dict__ for run, _ in reversed(self._runs)])


class StubRiskModel(RiskModel):
    """A fixed, transparent scoring rule, so tests can predict every score by hand."""

    name = "stub"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StubRiskModel":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.expected(X).to_numpy()

    @staticmethod
    def expected(X: pd.DataFrame) -> pd.Series:
        return 0.05 + 0.30 * X["has_history"] + 0.20 * X["lead_bucket_8-14"] + 0.10 * X["gender_male"] + 0.02 * X["prior_appt_count"]


class FixedContributionModel(ExplainableRiskModel):
    """Returns whatever contributions it was given. For testing the reason logic alone."""

    name = "fixed_contributions"

    def __init__(self, contributions: pd.DataFrame) -> None:
        self._contributions = contributions

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FixedContributionModel":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.5)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._contributions.loc[X.index]
