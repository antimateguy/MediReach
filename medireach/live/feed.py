"""BookingFeed: incremental, idempotent intake of new bookings, with checks and a log."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time

import pandas as pd

from medireach.data.database import SqliteDatabase
from medireach.data.repositories import TIMESTAMP_FORMAT, AppointmentRepository
from medireach.live.clock import ReplayMoment

_REQUIRED = ("appointment_id", "patient_id", "clinic_id", "scheduled_ts", "appointment_date", "lead_days")


@dataclass(frozen=True)
class IngestionBatch:
    """One run of the feed: which bookings arrived between two data times, and how the checks went."""

    cycle_start: date
    from_ts: datetime
    to_ts: datetime
    rows_in: int
    check_details: dict[str, int] = field(default_factory=dict)
    ran_at: str = ""

    @property
    def checks_failed(self) -> int:
        return sum(1 for count in self.check_details.values() if count > 0)


class IngestionChecks:
    """Quality checks on each batch. A failure is reported, never silently dropped."""

    def run(self, rows: pd.DataFrame) -> dict[str, int]:
        """Number of offending rows per check (0 means the check passed)."""
        return {
            "duplicate_ids": int(rows["appointment_id"].duplicated().sum()),
            "negative_lead": int((rows["appointment_date"] < rows["scheduled_ts"].dt.normalize()).sum()),
            "missing_values": int(rows[list(_REQUIRED)].isna().any(axis=1).sum()),
        }


class IngestionLog(ABC):
    """Where batches are recorded. The watermark of a cycle is the end of its last batch."""

    @abstractmethod
    def record(self, batch: IngestionBatch) -> None:
        """Append one batch."""

    @abstractmethod
    def watermark(self, cycle_start: date) -> datetime | None:
        """Data time up to which bookings have been taken in for this cycle, or None."""

    @abstractmethod
    def batches(self, cycle_start: date) -> pd.DataFrame:
        """All batches of one cycle, oldest first."""


class SqliteIngestionLog(IngestionLog):
    """Batches stored in ops.db (table ingestion_runs)."""

    _INSERT = """
        INSERT INTO ingestion_runs (cycle_start, from_ts, to_ts, rows_in, checks_failed, check_details, ran_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def record(self, batch: IngestionBatch) -> None:
        row = (batch.cycle_start.isoformat(), batch.from_ts.strftime(TIMESTAMP_FORMAT),
               batch.to_ts.strftime(TIMESTAMP_FORMAT), batch.rows_in, batch.checks_failed,
               json.dumps(batch.check_details, sort_keys=True), batch.ran_at)
        self._database.write("insert_ingestion_run", [(self._INSERT, [row])])

    def watermark(self, cycle_start: date) -> datetime | None:
        sql = "SELECT MAX(to_ts) AS watermark FROM ingestion_runs WHERE cycle_start = ?"
        value = self._database.query("select_watermark", sql, (cycle_start.isoformat(),)).iloc[0]["watermark"]
        return None if value is None else datetime.strptime(value, TIMESTAMP_FORMAT)

    def batches(self, cycle_start: date) -> pd.DataFrame:
        sql = "SELECT * FROM ingestion_runs WHERE cycle_start = ? ORDER BY to_ts"
        return self._database.query("select_ingestion_runs", sql, (cycle_start.isoformat(),))


class BookingFeed:
    """Takes in the bookings made since the last run, checks them and logs the batch.

    The source plays the hospital's booking system. Everything booked before the
    replayed week is history the warehouse already holds; within the week, a
    booking arrives at its real booking time (`scheduled_ts`). Running the feed
    twice at the same moment takes in nothing the second time (idempotent).
    """

    def __init__(self, source: AppointmentRepository, log: IngestionLog, checks: IngestionChecks,
                 week_start: date) -> None:
        self._source = source
        self._log = log
        self._checks = checks
        self._baseline = datetime.combine(week_start, time())
        self._bookings: pd.DataFrame | None = None

    def _all_bookings(self) -> pd.DataFrame:
        # The source does not change while the app runs, so it is read once.
        if self._bookings is None:
            self._bookings = self._source.get_appointments()
        return self._bookings

    def bookings_between(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Bookings made after `start` and up to and including `end` (data time)."""
        rows = self._all_bookings()
        made = rows["scheduled_ts"]
        return rows[(made > pd.Timestamp(start)) & (made <= pd.Timestamp(end))].reset_index(drop=True)

    def catch_up(self, moment: ReplayMoment) -> IngestionBatch | None:
        """Take in everything booked since the watermark. None if nothing new can have arrived."""
        start = self._log.watermark(moment.cycle_start) or self._baseline
        # The log keeps whole seconds, so compare at that precision: a second run within the same
        # second then takes in nothing instead of logging an empty window.
        end = moment.data_now.replace(microsecond=0)
        if end <= start:
            return None
        rows = self.bookings_between(start, end)
        batch = IngestionBatch(
            cycle_start=moment.cycle_start, from_ts=start, to_ts=end, rows_in=len(rows),
            check_details=self._checks.run(rows), ran_at=moment.real_now.isoformat(timespec="seconds"),
        )
        self._log.record(batch)
        return batch
