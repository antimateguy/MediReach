"""LiveMonitor: what the clinic knows right now, for the Today page and the plan."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from medireach.data.repositories import AsOfAppointmentRepository
from medireach.live.clock import ReplayCalendar, ReplayClock, ReplayMoment
from medireach.live.feed import BookingFeed, IngestionBatch, IngestionLog


@dataclass(frozen=True)
class LiveSnapshot:
    """The numbers the operations manager sees at one moment."""

    moment: ReplayMoment
    is_open: bool
    appointments_today: int
    same_day_bookings_so_far: int
    next_clinic_day: date
    appointments_next_day_known: int
    plan_as_of: datetime
    plan_due: bool
    batches_this_cycle: int
    last_batch: IngestionBatch | None


class LiveMonitor:
    """Runs the feed and summarises the as-of view. Reads only what was knowable at the moment."""

    def __init__(self, clock: ReplayClock, calendar: ReplayCalendar, live: AsOfAppointmentRepository,
                 feed: BookingFeed, log: IngestionLog) -> None:
        self._clock = clock
        self._calendar = calendar
        self._live = live
        self._feed = feed
        self._log = log

    def refresh(self) -> LiveSnapshot:
        """Catch the feed up to now (a cron job would do this on a server), then summarise."""
        moment = self._clock.moment()
        batch = self._feed.catch_up(moment)
        return self._snapshot(moment, batch)

    def _snapshot(self, moment: ReplayMoment, batch: IngestionBatch | None) -> LiveSnapshot:
        today = pd.Timestamp(moment.data_date)
        next_real = self._calendar.next_clinic_day(moment.real_date)
        next_data = pd.Timestamp(self._clock.data_date_for(next_real))
        rows = self._live.view_at(moment.data_now).get_appointments(start=moment.data_date, end=moment.data_date)
        booked_today = rows["scheduled_ts"].dt.normalize() == today
        # The next clinic day may sit in the next replay week; read it at the matching data time.
        next_view = self._live.view_at(self._clock.as_of_for(next_real, moment))
        next_rows = next_view.get_appointments(start=next_data.date(), end=next_data.date())
        plan_as_of = self._calendar.plan_as_of(next_real)
        return LiveSnapshot(
            moment=moment,
            is_open=self._calendar.is_open(moment.real_date),
            appointments_today=len(rows),
            same_day_bookings_so_far=int(booked_today.sum()),
            next_clinic_day=next_real,
            appointments_next_day_known=len(next_rows),
            plan_as_of=plan_as_of,
            plan_due=self._clock.as_of_for(next_real, moment) >= plan_as_of,
            batches_this_cycle=len(self._log.batches(moment.cycle_start)),
            last_batch=batch,
        )
