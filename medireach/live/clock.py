"""ReplayClock and ReplayCalendar: map the real calendar onto the replayed clinic week."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

WEEK = 7


@dataclass(frozen=True)
class ReplayMoment:
    """One instant, seen two ways: the real clock and the replayed data clock."""

    real_now: datetime   # time-zone aware, in the clinic's time zone
    data_now: datetime   # naive, in the data's own time (same time of day, replayed date)
    cycle_start: date    # real date on which the current replay week began

    @property
    def real_date(self) -> date:
        return self.real_now.date()

    @property
    def data_date(self) -> date:
        return self.data_now.date()


class ReplayClock:
    """Maps each real date to the day of the replayed week with the same weekday.

    The data holds one complete clinic week, so Monday maps to its Monday,
    Thursday to its Thursday, and so on. A new replay cycle begins every time
    the real calendar reaches the weekday the data week starts on.
    """

    def __init__(self, week_start: date, timezone: str,
                 now: Callable[[], datetime] | None = None, fixed_now: datetime | None = None) -> None:
        self._week_start = week_start
        self._zone = ZoneInfo(timezone)
        self._now = now or (lambda: datetime.now(self._zone))
        self._fixed_now = fixed_now

    def real_now(self) -> datetime:
        """Now in the clinic's time zone (a naive fixed time is read as clinic time)."""
        current = self._fixed_now or self._now()
        if current.tzinfo is None:
            return current.replace(tzinfo=self._zone)
        return current.astimezone(self._zone)

    def moment(self) -> ReplayMoment:
        real_now = self.real_now()
        data_date = self.data_date_for(real_now.date())
        data_now = datetime.combine(data_date, real_now.time().replace(tzinfo=None))
        return ReplayMoment(real_now, data_now, self.cycle_start_for(real_now.date()))

    def _offset(self, real_date: date) -> int:
        return (real_date.weekday() - self._week_start.weekday()) % WEEK

    def data_date_for(self, real_date: date) -> date:
        """The replayed day with the same weekday as `real_date`."""
        return self._week_start + timedelta(days=self._offset(real_date))

    def cycle_start_for(self, real_date: date) -> date:
        """The real date on which the replay week containing `real_date` began."""
        return real_date - timedelta(days=self._offset(real_date))

    def as_of_for(self, target_real_date: date, moment: ReplayMoment) -> datetime:
        """What is knowable now about `target_real_date`, expressed in data time.

        Within the current replay week that is simply the moment's data time.
        A target in the next week (Wednesday looking at Thursday) is replayed
        one week earlier in the data, so the same instant is a week earlier too.
        """
        weeks_ahead = (self.cycle_start_for(target_real_date) - moment.cycle_start).days // WEEK
        return moment.data_now - timedelta(days=WEEK * weeks_ahead)

    def real_date_for(self, data_date: date, cycle_start: date) -> date:
        """The real date that shows `data_date` within the cycle that began on `cycle_start`."""
        return cycle_start + (data_date - self._week_start)


class ReplayCalendar:
    """Clinic days and plan times on the real calendar, using the clock and the data's clinic days."""

    def __init__(self, clock: ReplayClock, clinic_days: Iterable[date], plan_hour: int) -> None:
        self._clock = clock
        self._clinic_days = sorted(set(clinic_days))
        self._plan_hour = plan_hour

    def is_open(self, real_date: date) -> bool:
        """A real date is a clinic day if its replayed day had appointments (weekends had none)."""
        return self._clock.data_date_for(real_date) in self._clinic_days

    def next_clinic_day(self, real_date: date) -> date:
        """The first clinic day strictly after `real_date` (Friday -> Monday)."""
        for step in range(1, WEEK + 1):
            candidate = real_date + timedelta(days=step)
            if self.is_open(candidate):
                return candidate
        raise ValueError("the replayed week has no clinic days")

    def previous_clinic_days(self, data_date: date, count: int) -> list[date]:
        """The last `count` clinic days strictly before `data_date` (data dates, oldest first)."""
        return [day for day in self._clinic_days if day < data_date][-count:]

    def plan_day_for(self, target_real_date: date) -> date:
        """The real date on which the plan for `target_real_date` is made (and its calls are made)."""
        gap = self._clock.data_date_for(target_real_date) - self.plan_as_of(target_real_date).date()
        return target_real_date - gap

    def plan_as_of(self, target_real_date: date) -> datetime:
        """Data time at which the plan for `target_real_date` is made.

        The plan is made at plan_hour on the clinic day before the target, so it
        sees exactly the bookings that existed by then and no outcome of the
        target day. Both are stated in data time, which keeps the rule the same
        whether or not the previous clinic day falls inside the replayed week.
        """
        target = self._clock.data_date_for(target_real_date)
        earlier = [day for day in self._clinic_days if day < target]
        if not earlier:
            raise ValueError(f"no clinic day before {target} in the data")
        return datetime.combine(earlier[-1], time(hour=self._plan_hour))
