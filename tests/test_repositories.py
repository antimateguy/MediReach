"""Phase 2: schema integrity, repository contract, append-only queue."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from medireach.data.database import SqliteDatabase
from medireach.data.repositories import (
    APPOINTMENT_COLUMNS,
    QUEUE_COLUMNS,
    AppointmentRepository,
    SqliteAppointmentRepository,
    SqliteQueueRepository,
)
from medireach.domain import QueueRun
from medireach.errors import DataAccessError
from tests.doubles import InMemoryAppointmentRepository


# ---------------------------------------------------------------- integrity

def test_build_leaves_no_orphan_foreign_keys(database: SqliteDatabase, cleaned) -> None:
    patients, appointments, _ = cleaned
    assert database.query("fk_check", "PRAGMA foreign_key_check").empty
    counts = database.query("counts", "SELECT (SELECT COUNT(*) FROM patients) AS p, (SELECT COUNT(*) FROM appointments) AS a")
    assert int(counts["p"].iloc[0]) == len(patients)
    assert int(counts["a"].iloc[0]) == len(appointments)


def test_foreign_keys_and_checks_are_enforced(database: SqliteDatabase) -> None:
    insert = (
        "INSERT INTO appointments (appointment_id, patient_id, clinic_id, scheduled_ts, scheduled_date, "
        "appointment_date, lead_days, sms_received, no_show) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)"
    )
    unknown_patient = (1, -999, "2016-05-02T08:00:00", "2016-05-02", "2016-05-02", 0, 0, 0)
    with pytest.raises(DataAccessError, match="insert_orphan"):
        database.write("insert_orphan", [(insert, [unknown_patient])])
    known = int(database.query("one", "SELECT patient_id FROM patients LIMIT 1")["patient_id"].iloc[0])
    negative_lead = (2, known, "2016-05-03T08:00:00", "2016-05-03", "2016-05-02", -1, 0, 0)
    with pytest.raises(DataAccessError, match="insert_negative_lead"):
        database.write("insert_negative_lead", [(insert, [negative_lead])])


def test_failed_statement_is_wrapped_with_its_name(database: SqliteDatabase) -> None:
    with pytest.raises(DataAccessError, match="select_nothing"):
        database.query("select_nothing", "SELECT * FROM table_that_does_not_exist")


# ------------------------------------------------- appointment repo contract

@pytest.fixture(params=["sqlite", "in_memory"])
def repository(request: pytest.FixtureRequest, database: SqliteDatabase, cleaned) -> AppointmentRepository:
    """The same tests run against the real repository and the test double (Liskov)."""
    if request.param == "sqlite":
        return SqliteAppointmentRepository(database)
    patients, appointments, _ = cleaned
    return InMemoryAppointmentRepository(patients, appointments)


def test_contract_returns_every_appointment_in_the_agreed_shape(repository: AppointmentRepository, cleaned) -> None:
    _, appointments, _ = cleaned
    frame = repository.get_appointments()
    assert tuple(frame.columns) == APPOINTMENT_COLUMNS
    assert len(frame) == len(appointments)
    assert frame["appointment_id"].is_unique
    assert pd.api.types.is_datetime64_any_dtype(frame["appointment_date"])
    assert pd.api.types.is_datetime64_any_dtype(frame["scheduled_date"])
    assert not frame.isna().any().any()


def test_contract_orders_by_date_then_appointment_id(repository: AppointmentRepository) -> None:
    frame = repository.get_appointments()
    expected = frame.sort_values(["appointment_date", "appointment_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected)


def test_contract_date_bounds_are_inclusive(repository: AppointmentRepository) -> None:
    everything = repository.get_appointments()
    day = everything["appointment_date"].iloc[len(everything) // 2].date()
    on_day = repository.get_appointments(start=day, end=day)
    assert len(on_day) == int((everything["appointment_date"] == pd.Timestamp(day)).sum()) > 0
    up_to_day = repository.get_appointments(end=day)
    assert len(up_to_day) == int((everything["appointment_date"] <= pd.Timestamp(day)).sum())
    from_day = repository.get_appointments(start=day)
    assert len(from_day) == int((everything["appointment_date"] >= pd.Timestamp(day)).sum())


def test_contract_empty_window_keeps_the_columns(repository: AppointmentRepository) -> None:
    frame = repository.get_appointments(start=date(1999, 1, 1), end=date(1999, 1, 2))
    assert frame.empty and tuple(frame.columns) == APPOINTMENT_COLUMNS


def test_contract_date_range(repository: AppointmentRepository) -> None:
    everything = repository.get_appointments()
    first, last = repository.date_range()
    assert isinstance(first, date) and isinstance(last, date)
    assert first == everything["appointment_date"].min().date()
    assert last == everything["appointment_date"].max().date()


def test_both_implementations_return_identical_data(database: SqliteDatabase, cleaned) -> None:
    patients, appointments, _ = cleaned
    from_sqlite = SqliteAppointmentRepository(database).get_appointments()
    from_memory = InMemoryAppointmentRepository(patients, appointments).get_appointments()
    pd.testing.assert_frame_equal(from_sqlite, from_memory, check_dtype=False)


# ------------------------------------------------------------ queue repository

def _run(run_id: str, created_at: str, n_rows: int) -> QueueRun:
    return QueueRun(
        run_id=run_id, created_at=created_at, window_start="2016-05-02", window_end="2016-05-03",
        model_name="stub", model_version="v0", policy_name="capacity", n_rows=n_rows, notes="test",
    )


def _queue_rows(database: SqliteDatabase, n: int = 5) -> pd.DataFrame:
    source = SqliteAppointmentRepository(database).get_appointments().head(n)
    return pd.DataFrame({
        "appointment_id": source["appointment_id"],
        "patient_id": source["patient_id"],
        "appointment_date": source["appointment_date"],
        "risk_score": [0.9, 0.7, 0.5, 0.3, 0.1][:n],
        "day_rank": list(range(1, n + 1)),
        "priority": ["HIGH", "MEDIUM", "LOW", "LOW", "LOW"][:n],
        "action": ["Call", "SMS", "None", "None", "None"][:n],
        "reasons": ["Booked 8-14 days ahead"] * n,
    })


def test_queue_starts_empty(database: SqliteDatabase) -> None:
    queue = SqliteQueueRepository(database)
    assert queue.latest_run_id() is None
    assert queue.list_runs().empty


def test_saved_run_reads_back_unchanged(database: SqliteDatabase) -> None:
    queue, rows = SqliteQueueRepository(database), _queue_rows(database)
    queue.save_run(_run("run-1", "2016-06-01T18:00:00", len(rows)), rows)
    stored = queue.get_run("run-1")
    assert tuple(stored.columns) == QUEUE_COLUMNS
    expected = rows.sort_values(["appointment_date", "day_rank", "appointment_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(stored, expected, check_dtype=False)


def test_queue_is_append_only(database: SqliteDatabase) -> None:
    queue, rows = SqliteQueueRepository(database), _queue_rows(database)
    queue.save_run(_run("run-1", "2016-06-01T18:00:00", len(rows)), rows)
    queue.save_run(_run("run-2", "2016-06-02T18:00:00", 3), rows.head(3))
    assert len(queue.get_run("run-1")) == len(rows)
    assert len(queue.get_run("run-2")) == 3
    assert queue.latest_run_id() == "run-2"
    assert list(queue.list_runs()["run_id"]) == ["run-2", "run-1"]


def test_failed_save_leaves_nothing_behind(database: SqliteDatabase) -> None:
    queue, rows = SqliteQueueRepository(database), _queue_rows(database)
    bad = rows.assign(priority="URGENT")
    with pytest.raises(DataAccessError, match="insert_outreach_run"):
        queue.save_run(_run("run-bad", "2016-06-01T18:00:00", len(bad)), bad)
    # The run header was inserted first; the rollback must have removed it.
    assert queue.list_runs().empty
    with pytest.raises(DataAccessError, match="missing columns"):
        queue.save_run(_run("run-x", "2016-06-01T18:00:00", 1), rows.drop(columns=["reasons"]))


def test_run_id_cannot_be_reused_and_scores_need_a_real_appointment(database: SqliteDatabase) -> None:
    queue, rows = SqliteQueueRepository(database), _queue_rows(database)
    queue.save_run(_run("run-1", "2016-06-01T18:00:00", len(rows)), rows)
    with pytest.raises(DataAccessError):
        queue.save_run(_run("run-1", "2016-06-02T18:00:00", len(rows)), rows)
    with pytest.raises(DataAccessError):
        queue.save_run(_run("run-2", "2016-06-02T18:00:00", len(rows)), rows.assign(appointment_id=-1))
    assert list(queue.list_runs()["run_id"]) == ["run-1"]
