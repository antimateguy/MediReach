"""Repository interfaces and their SQLite implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import astuple
from datetime import date, datetime
from typing import Callable

import pandas as pd

from medireach.data.database import SqliteDatabase
from medireach.domain import QueueRun
from medireach.errors import DataAccessError

# The contract every AppointmentRepository returns: one row per appointment
# with the patient's attributes and the clinic joined on.
APPOINTMENT_COLUMNS: tuple[str, ...] = (
    "appointment_id", "patient_id", "clinic_id", "clinic", "scheduled_ts", "scheduled_date", "appointment_date",
    "lead_days", "sms_received", "no_show", "gender", "age", "scholarship",
    "hypertension", "diabetes", "alcoholism", "handicap",
)
QUEUE_COLUMNS: tuple[str, ...] = (
    "appointment_id", "patient_id", "appointment_date", "risk_score", "day_rank",
    "priority", "action", "reasons",
)
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


class AppointmentRepository(ABC):
    """Read access to appointments. Callers never see SQL or a connection."""

    @abstractmethod
    def get_appointments(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """Appointments with `start <= appointment_date <= end` (either bound optional).

        Columns are APPOINTMENT_COLUMNS, dates and timestamps are datetime64, rows
        are ordered by appointment_date then appointment_id. `no_show` is 0/1, or
        missing (NaN) when the outcome is not known yet (see AsOfAppointmentRepository).
        """

    @abstractmethod
    def date_range(self) -> tuple[date, date]:
        """First and last appointment date available."""

    @abstractmethod
    def appointment_dates(self) -> list[date]:
        """Every date that has at least one appointment, in order (the clinic days)."""


class QueueRepository(ABC):
    """Stores outreach runs. Append-only: history is never deleted."""

    @abstractmethod
    def save_run(self, run: QueueRun, rows: pd.DataFrame) -> None:
        """Persist one run and its rows (QUEUE_COLUMNS) as a single transaction."""

    @abstractmethod
    def get_run(self, run_id: str) -> pd.DataFrame:
        """Rows of one run, ordered by appointment_date then day_rank."""

    @abstractmethod
    def latest_run_id(self) -> str | None:
        """Id of the most recently created run, or None if there are no runs."""

    @abstractmethod
    def list_runs(self) -> pd.DataFrame:
        """One row per run, newest first."""


class SqliteAppointmentRepository(AppointmentRepository):
    """Appointments read from the SQLite database."""

    _SELECT = """
        SELECT a.appointment_id, a.patient_id, a.clinic_id, c.name AS clinic,
               a.scheduled_ts, a.scheduled_date, a.appointment_date, a.lead_days, a.sms_received, a.no_show,
               p.gender, p.age, p.scholarship,
               p.hypertension, p.diabetes, p.alcoholism, p.handicap
        FROM appointments AS a
        JOIN patients AS p ON p.patient_id = a.patient_id
        JOIN clinics AS c ON c.clinic_id = a.clinic_id
        WHERE (:start IS NULL OR a.appointment_date >= :start)
          AND (:end IS NULL OR a.appointment_date <= :end)
        ORDER BY a.appointment_date, a.appointment_id
    """

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def get_appointments(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        params = {
            "start": start.strftime(DATE_FORMAT) if start else None,
            "end": end.strftime(DATE_FORMAT) if end else None,
        }
        frame = self._database.query("select_appointments", self._SELECT, params)
        for column in ("scheduled_date", "appointment_date"):
            frame[column] = pd.to_datetime(frame[column], format=DATE_FORMAT)
        frame["scheduled_ts"] = pd.to_datetime(frame["scheduled_ts"], format=TIMESTAMP_FORMAT)
        return frame[list(APPOINTMENT_COLUMNS)]

    def date_range(self) -> tuple[date, date]:
        sql = "SELECT MIN(appointment_date) AS first, MAX(appointment_date) AS last FROM appointments"
        row = self._database.query("select_date_range", sql).iloc[0]
        if row["first"] is None:
            raise DataAccessError("select_date_range", "the appointments table is empty")
        return date.fromisoformat(row["first"]), date.fromisoformat(row["last"])

    def appointment_dates(self) -> list[date]:
        sql = "SELECT DISTINCT appointment_date FROM appointments ORDER BY appointment_date"
        frame = self._database.query("select_appointment_dates", sql)
        return [date.fromisoformat(value) for value in frame["appointment_date"]]


class AsOfAppointmentRepository(AppointmentRepository):
    """The data as it looked at one moment: what had been booked, and which outcomes were known.

    Wraps any AppointmentRepository without changing it (the Decorator pattern).
    A booking appears once its `scheduled_ts` has passed. Its outcome appears
    only after the appointment day has ended; until then `no_show` is NaN.
    `as_of` is called on every read, so the view moves with the clock.
    """

    def __init__(self, inner: AppointmentRepository, as_of: Callable[[], datetime]) -> None:
        self._inner = inner
        self._as_of = as_of

    def get_appointments(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        moment = pd.Timestamp(self._as_of())
        rows = self._inner.get_appointments(start, end)
        visible = rows[rows["scheduled_ts"] <= moment].copy()
        unknown = visible["appointment_date"] >= moment.normalize()
        visible["no_show"] = visible["no_show"].astype("float64").mask(unknown)
        return visible.reset_index(drop=True)

    def date_range(self) -> tuple[date, date]:
        rows = self.get_appointments()
        if rows.empty:
            raise DataAccessError("select_date_range", "nothing had been booked at this moment")
        return rows["appointment_date"].min().date(), rows["appointment_date"].max().date()

    def appointment_dates(self) -> list[date]:
        rows = self.get_appointments()
        return sorted({value.date() for value in rows["appointment_date"]})

    def view_at(self, as_of: datetime) -> "AsOfAppointmentRepository":
        """The same data, frozen at another moment (for example the moment a plan is made)."""
        return AsOfAppointmentRepository(self._inner, lambda: as_of)


class SqliteQueueRepository(QueueRepository):
    """Outreach runs stored in outreach_runs and outreach_queue."""

    _INSERT_RUN = """
        INSERT INTO outreach_runs
            (run_id, created_at, window_start, window_end, model_name, model_version,
             policy_name, n_rows, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    _INSERT_ROW = """
        INSERT INTO outreach_queue
            (run_id, appointment_id, patient_id, appointment_date, risk_score, day_rank,
             priority, action, reasons)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def save_run(self, run: QueueRun, rows: pd.DataFrame) -> None:
        missing = sorted(set(QUEUE_COLUMNS) - set(rows.columns))
        if missing:
            raise DataAccessError("insert_outreach_run", f"queue rows are missing columns: {missing}")
        payload = rows[list(QUEUE_COLUMNS)].copy()
        payload["appointment_date"] = pd.to_datetime(payload["appointment_date"]).dt.strftime(DATE_FORMAT)
        # astype(object) turns numpy scalars into plain Python values, which
        # is what sqlite3 can bind.
        queue_rows = [(run.run_id, *values) for values in payload.astype(object).values.tolist()]
        # The run header and its rows succeed or fail together.
        self._database.write(
            "insert_outreach_run",
            [(self._INSERT_RUN, [astuple(run)]), (self._INSERT_ROW, queue_rows)],
        )

    def get_run(self, run_id: str) -> pd.DataFrame:
        sql = f"""
            SELECT {", ".join(QUEUE_COLUMNS)}
            FROM outreach_queue
            WHERE run_id = ?
            ORDER BY appointment_date, day_rank, appointment_id
        """
        frame = self._database.query("select_outreach_run", sql, (run_id,))
        frame["appointment_date"] = pd.to_datetime(frame["appointment_date"], format=DATE_FORMAT)
        return frame

    def latest_run_id(self) -> str | None:
        sql = "SELECT run_id FROM outreach_runs ORDER BY created_at DESC, run_id DESC LIMIT 1"
        frame = self._database.query("select_latest_run", sql)
        return None if frame.empty else str(frame.iloc[0]["run_id"])

    def list_runs(self) -> pd.DataFrame:
        sql = "SELECT * FROM outreach_runs ORDER BY created_at DESC, run_id DESC"
        return self._database.query("select_outreach_runs", sql)
