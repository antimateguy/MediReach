"""v3 G1: the replay clock and calendar, the as-of view (leakage tripwire), the booking feed."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from medireach.data.database import IN_MEMORY, SqliteDatabase, read_sql_file
from medireach.data.repositories import AsOfAppointmentRepository
from medireach.features.builder import FeatureBuilder
from medireach.live.clock import ReplayCalendar, ReplayClock
from medireach.live.feed import BookingFeed, IngestionBatch, IngestionChecks, SqliteIngestionLog
from medireach.live.status import LiveMonitor
from tests.doubles import InMemoryAppointmentRepository

WEEK_START = date(2016, 6, 2)                     # the replayed week starts on a Thursday
CLINIC_DAYS = [date(2016, 6, 1), date(2016, 6, 2), date(2016, 6, 3),
               date(2016, 6, 6), date(2016, 6, 7), date(2016, 6, 8)]


def _clock(at: datetime) -> ReplayClock:
    return ReplayClock(WEEK_START, "Asia/Kolkata", fixed_now=at)


# ---------------------------------------------------------------- clock

@pytest.mark.parametrize("real, data", [
    (date(2026, 10, 19), date(2016, 6, 6)),   # Monday
    (date(2026, 10, 20), date(2016, 6, 7)),   # Tuesday
    (date(2026, 10, 21), date(2016, 6, 8)),   # Wednesday
    (date(2026, 10, 22), date(2016, 6, 2)),   # Thursday: a new replay week starts
    (date(2026, 10, 23), date(2016, 6, 3)),   # Friday
    (date(2026, 10, 24), date(2016, 6, 4)),   # Saturday (no appointments in the data)
    (date(2026, 10, 25), date(2016, 6, 5)),   # Sunday
])
def test_each_real_date_maps_to_the_same_weekday_of_the_replayed_week(real: date, data: date) -> None:
    clock = _clock(datetime(2026, 10, 19))
    assert clock.data_date_for(real) == data
    assert real.weekday() == data.weekday()
    assert clock.real_date_for(data, clock.cycle_start_for(real)) == real


def test_cycle_starts_on_thursday_and_time_of_day_is_kept() -> None:
    moment = _clock(datetime(2026, 10, 19, 14, 30)).moment()
    assert moment.cycle_start == date(2026, 10, 15)
    assert moment.data_now == datetime(2016, 6, 6, 14, 30)
    assert _clock(datetime(2026, 10, 22, 0, 5)).moment().cycle_start == date(2026, 10, 22)


def test_clock_works_in_the_clinic_time_zone() -> None:
    moment = _clock(datetime(2026, 10, 22, 4, 45, tzinfo=timezone.utc)).moment()
    assert moment.real_now.utcoffset().total_seconds() == 5.5 * 3600
    assert moment.data_now == datetime(2016, 6, 2, 10, 15)


def test_next_week_is_read_one_week_earlier_in_data_time() -> None:
    wednesday = _clock(datetime(2026, 10, 21, 10, 0))
    moment = wednesday.moment()
    assert wednesday.as_of_for(date(2026, 10, 21), moment) == datetime(2016, 6, 8, 10, 0)
    assert wednesday.as_of_for(date(2026, 10, 22), moment) == datetime(2016, 6, 1, 10, 0)


# ------------------------------------------------------------- calendar

def test_calendar_open_days_and_next_clinic_day() -> None:
    calendar = ReplayCalendar(_clock(datetime(2026, 10, 19)), CLINIC_DAYS, plan_hour=8)
    assert calendar.is_open(date(2026, 10, 22)) and not calendar.is_open(date(2026, 10, 24))
    assert calendar.next_clinic_day(date(2026, 10, 23)) == date(2026, 10, 26)   # Friday -> Monday
    assert calendar.next_clinic_day(date(2026, 10, 24)) == date(2026, 10, 26)   # Saturday -> Monday
    assert calendar.next_clinic_day(date(2026, 10, 21)) == date(2026, 10, 22)   # Wednesday -> Thursday


def test_plan_is_made_at_plan_hour_on_the_previous_clinic_day_in_data_time() -> None:
    calendar = ReplayCalendar(_clock(datetime(2026, 10, 19)), CLINIC_DAYS, plan_hour=8)
    assert calendar.plan_as_of(date(2026, 10, 22)) == datetime(2016, 6, 1, 8, 0)   # Thursday's plan
    assert calendar.plan_as_of(date(2026, 10, 26)) == datetime(2016, 6, 3, 8, 0)   # Monday's plan, made Friday


# ------------------------------------------------- as-of view (tripwire)

@pytest.fixture()
def source(cleaned) -> InMemoryAppointmentRepository:
    patients, appointments, _ = cleaned
    return InMemoryAppointmentRepository(patients, appointments)


@pytest.mark.parametrize("as_of", [datetime(2016, 5, 9, 0, 0), datetime(2016, 5, 16, 11, 30), datetime(2016, 5, 27, 23, 59)])
def test_as_of_view_never_shows_a_future_booking_or_an_unknown_outcome(source, as_of: datetime) -> None:
    full = source.get_appointments().set_index("appointment_id")
    view = AsOfAppointmentRepository(source, lambda: as_of).get_appointments()
    assert (view["scheduled_ts"] <= pd.Timestamp(as_of)).all()
    expected_ids = set(full.index[full["scheduled_ts"] <= pd.Timestamp(as_of)])
    assert set(view["appointment_id"]) == expected_ids
    unknown = view["appointment_date"] >= pd.Timestamp(as_of).normalize()
    assert view.loc[unknown, "no_show"].isna().all()
    known = view[~unknown].set_index("appointment_id")["no_show"]
    assert (known == full.loc[known.index, "no_show"]).all()


def test_as_of_view_follows_the_clock_and_can_be_frozen(source) -> None:
    times = iter([datetime(2016, 5, 10), datetime(2016, 5, 20)])
    current = {"now": next(times)}
    view = AsOfAppointmentRepository(source, lambda: current["now"])
    early = len(view.get_appointments())
    frozen = view.view_at(current["now"])
    current["now"] = next(times)
    assert len(view.get_appointments()) > early == len(frozen.get_appointments())
    assert view.appointment_dates() == sorted({d.date() for d in view.get_appointments()["appointment_date"]})


def test_features_built_from_the_as_of_view_match_the_full_data_for_that_day(source, settings) -> None:
    # Planning for a day sees every earlier outcome, so history features are unchanged.
    day = datetime(2016, 5, 18, 8, 0)
    builder = FeatureBuilder(settings, prior_rate=0.2)
    view = AsOfAppointmentRepository(source, lambda: day).get_appointments(start=day.date(), end=None)
    full_x, full_meta = builder.build(source.get_appointments())
    as_of_x, as_of_meta = builder.build(AsOfAppointmentRepository(source, lambda: day).get_appointments())
    on_day = as_of_meta["appointment_date"] == pd.Timestamp(day.date())
    ids = as_of_meta.loc[on_day, "appointment_id"]
    assert len(ids) > 0 and len(view) >= len(ids)
    left = as_of_x[on_day.to_numpy()].set_axis(ids.to_numpy())
    right = full_x.set_axis(full_meta["appointment_id"].to_numpy()).loc[ids.to_numpy()]
    pd.testing.assert_frame_equal(left, right)


# ------------------------------------------------------------------ feed

@pytest.fixture()
def ops(settings) -> Iterator[SqliteDatabase]:
    database = SqliteDatabase(IN_MEMORY)
    schema = read_sql_file(settings.ops_schema_path, "read_ops_schema_file")
    database.run_script("create_ops_schema", schema)
    database.run_script("create_ops_schema_again", schema)   # idempotent
    yield database
    database.close()


def _moment(at: datetime, week_start: date):
    return ReplayClock(week_start, "Asia/Kolkata", fixed_now=at).moment()


def test_feed_is_incremental_and_idempotent(source, ops) -> None:
    week_start = date(2016, 5, 12)                                     # a Thursday in the fixture data
    log = SqliteIngestionLog(ops)
    feed = BookingFeed(source, log, IngestionChecks(), week_start)
    morning = _moment(datetime(2026, 10, 22, 9, 0), week_start)        # Thursday of a new cycle
    evening = _moment(datetime(2026, 10, 22, 19, 0), week_start)
    first = feed.catch_up(morning)
    assert first is not None and first.from_ts == datetime(2016, 5, 12, 0, 0)
    assert feed.catch_up(morning) is None                              # same moment: nothing new
    second = feed.catch_up(evening)
    assert second is not None and second.from_ts == first.to_ts
    everything = feed.bookings_between(datetime(2016, 5, 12), datetime(2016, 5, 12, 19, 0))
    assert first.rows_in + second.rows_in == len(everything)
    assert len(log.batches(morning.cycle_start)) == 2
    assert log.watermark(morning.cycle_start) == datetime(2016, 5, 12, 19, 0)


def test_two_runs_within_the_same_second_take_in_nothing_the_second_time(source, ops) -> None:
    # The log stores whole seconds; a second run 0.7 s later must not log an empty window (it used to
    # fail the database's from_ts < to_ts check when two requests came in quickly on the real clock).
    week_start = date(2016, 5, 12)
    feed = BookingFeed(source, SqliteIngestionLog(ops), IngestionChecks(), week_start)
    assert feed.catch_up(_moment(datetime(2026, 10, 22, 9, 0, 5, 100_000), week_start)) is not None
    assert feed.catch_up(_moment(datetime(2026, 10, 22, 9, 0, 5, 800_000), week_start)) is None
    later = feed.catch_up(_moment(datetime(2026, 10, 22, 9, 0, 6, 200_000), week_start))
    assert later is not None and later.from_ts == datetime(2016, 5, 12, 9, 0, 5)


def test_each_cycle_has_its_own_watermark(source, ops) -> None:
    week_start = date(2016, 5, 12)
    log = SqliteIngestionLog(ops)
    feed = BookingFeed(source, log, IngestionChecks(), week_start)
    feed.catch_up(_moment(datetime(2026, 10, 22, 9, 0), week_start))
    next_week = feed.catch_up(_moment(datetime(2026, 10, 29, 9, 0), week_start))
    assert next_week is not None and next_week.from_ts == datetime(2016, 5, 12, 0, 0)


def test_checks_report_bad_rows_without_dropping_them() -> None:
    rows = pd.DataFrame({
        "appointment_id": [1, 1, 3, 4],
        "patient_id": [10, 11, 12, np.nan],
        "clinic_id": [1, 1, 2, 2],
        "scheduled_ts": pd.to_datetime(["2016-06-02 08:00", "2016-06-02 09:00", "2016-06-03 09:00", "2016-06-02 10:00"]),
        "appointment_date": pd.to_datetime(["2016-06-02", "2016-06-02", "2016-06-02", "2016-06-06"]),
        "lead_days": [0, 0, -1, 4],
    })
    details = IngestionChecks().run(rows)
    assert details == {"duplicate_ids": 1, "negative_lead": 1, "missing_values": 1}
    batch = IngestionBatch(date(2026, 10, 22), datetime(2016, 6, 2), datetime(2016, 6, 3), len(rows), details)
    assert batch.checks_failed == 3


def test_monitor_snapshot_counts_only_what_is_known(source, ops) -> None:
    week_start = date(2016, 5, 12)
    clock = ReplayClock(week_start, "Asia/Kolkata", fixed_now=datetime(2026, 10, 22, 12, 0))
    calendar = ReplayCalendar(clock, source.appointment_dates(), plan_hour=8)
    live = AsOfAppointmentRepository(source, lambda: clock.moment().data_now)
    log = SqliteIngestionLog(ops)
    monitor = LiveMonitor(clock, calendar, live, BookingFeed(source, log, IngestionChecks(), week_start), log)
    snapshot = monitor.refresh()
    full = source.get_appointments()
    noon = pd.Timestamp(datetime(2016, 5, 12, 12, 0))
    today = full[(full["appointment_date"] == pd.Timestamp(2016, 5, 12)) & (full["scheduled_ts"] <= noon)]
    assert snapshot.is_open and snapshot.appointments_today == len(today)
    assert snapshot.next_clinic_day == date(2026, 10, 23)
    tomorrow = full[(full["appointment_date"] == pd.Timestamp(2016, 5, 13)) & (full["scheduled_ts"] <= noon)]
    assert snapshot.appointments_next_day_known == len(tomorrow)
    assert snapshot.plan_due and snapshot.batches_this_cycle == 1
