"""DemoControls and OperationsCycle: move the demo clock, simulate outcomes, reset, refresh."""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Callable

import numpy as np

from medireach.data.database import SqliteDatabase, read_sql_file
from medireach.domain import Event, Outcome, TaskKind, TaskStatus, User
from medireach.live.clock import ReplayCalendar, ReplayClock
from medireach.live.status import LiveMonitor, LiveSnapshot
from medireach.ops.outcomes import (
    DEFAULT_PLANTED_EFFECT, HISTORICAL, OUTCOME_SOURCE_KEY, PLANTED_EFFECT_KEY, SIMULATED, DayCloser,
)
from medireach.ops.repositories import EventLog
from medireach.ops.tasks import TaskService
from medireach.ops.workflow_store import PlanRepository, StateStore

DEMO_NOW_KEY = "demo_now"
# Simulated receptionist results when a call connects: most patients confirm.
_REACHED_MIX = ((Outcome.CONFIRMED, 0.90), (Outcome.RESCHEDULE, 0.06), (Outcome.CANCEL, 0.04))
_WORKFLOW_TABLES = ("revealed_outcomes", "tasks", "plan_items", "daily_plans", "ingestion_runs", "ops_state", "events")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def demo_now_provider(state: StateStore, fallback: datetime | None,
                      real_now: Callable[[], datetime]) -> Callable[[], datetime]:
    """The clock's source of "now": the demo time if one is set, else a fixed time, else the real clock."""
    def now() -> datetime:
        override = state.get(DEMO_NOW_KEY)
        if override:
            return datetime.fromisoformat(override)
        return fallback or real_now()
    return now


class DemoControls:
    """Everything the admin needs to run a replay week quickly in a demo. Every action is logged."""

    def __init__(self, state: StateStore, clock: ReplayClock, calendar: ReplayCalendar, plans: PlanRepository,
                 task_service: TaskService, database: SqliteDatabase, schema_path: str, events: EventLog,
                 plan_hour: int, now: Callable[[], str] = _utc_now) -> None:
        self._state, self._clock, self._calendar, self._plans = state, clock, calendar, plans
        self._task_service, self._database, self._schema_path = task_service, database, schema_path
        self._events, self._plan_hour, self._now = events, plan_hour, now

    # -------------------------------------------------------------- time
    def demo_now(self) -> datetime | None:
        value = self._state.get(DEMO_NOW_KEY)
        return datetime.fromisoformat(value) if value else None

    def set_demo_now(self, value: datetime, actor: User) -> None:
        self._state.set(DEMO_NOW_KEY, value.replace(tzinfo=None).isoformat(timespec="minutes"))
        self._log(actor, "DEMO_TIME_SET", {"now": value.isoformat(timespec="minutes")})

    def clear_demo_now(self, actor: User) -> None:
        self._state.delete(DEMO_NOW_KEY)
        self._log(actor, "DEMO_TIME_CLEARED", {})

    def advance_to_next_clinic_day(self, actor: User) -> datetime:
        """Jump to plan time (08:00) on the next clinic day: the next step of the daily cycle."""
        target = self._calendar.next_clinic_day(self._clock.moment().real_date)
        moment = datetime.combine(target, time(hour=self._plan_hour))
        self.set_demo_now(moment, actor)
        return moment

    # ---------------------------------------------------------- outcomes
    def outcome_settings(self) -> tuple[str, float]:
        return (self._state.get(OUTCOME_SOURCE_KEY, HISTORICAL) or HISTORICAL,
                float(self._state.get(PLANTED_EFFECT_KEY, str(DEFAULT_PLANTED_EFFECT)) or DEFAULT_PLANTED_EFFECT))

    def set_outcome_source(self, source: str, effect: float, actor: User) -> None:
        if source not in (HISTORICAL, SIMULATED) or not 0 <= effect <= 1:
            raise ValueError("source must be historical or simulated, and effect between 0 and 1")
        self._state.set(OUTCOME_SOURCE_KEY, source)
        self._state.set(PLANTED_EFFECT_KEY, str(effect))
        self._log(actor, "OUTCOME_SOURCE_SET", {"source": source, "effect": effect})

    def auto_complete(self, plan_id: int, actor: User, reach_rate: float, limit: int | None = None) -> int:
        """Simulate the receptionists finishing open tasks of a plan (all, or the `limit` highest priority).

        Returns tasks completed. Without a limit the order and the random draws are unchanged.
        """
        rng = np.random.default_rng(self._plans.get(plan_id).seed)
        completed = 0
        for _ in range(4):  # first calls, retries, SMS fallbacks
            pending = self._task_service.tasks_for(plan_id)
            pending = pending[pending["status"] == TaskStatus.PENDING.value]
            if limit is not None:
                pending = pending.sort_values(["priority_rank", "attempt", "task_id"]).head(limit - completed)
            if pending.empty:
                break
            outcomes = {}
            for _, task in pending.iterrows():
                outcome = Outcome.SMS_SENT
                if task["kind"] == TaskKind.CALL.value:
                    outcome = self._reached_outcome(rng) if rng.random() < reach_rate else Outcome.NO_ANSWER
                outcomes[int(task["task_id"])] = outcome
            self._task_service.record_many(outcomes, actor, notes="Simulated")
            completed += len(outcomes)
        self._log(actor, "TASKS_SIMULATED", {"plan_id": plan_id, "completed": completed, "reach_rate": reach_rate})
        return completed

    @staticmethod
    def _reached_outcome(rng: np.random.Generator) -> Outcome:
        draw, total = rng.random(), 0.0
        for outcome, share in _REACHED_MIX:
            total += share
            if draw < total:
                return outcome
        return Outcome.CONFIRMED

    # ------------------------------------------------------------- reset
    def reset(self, actor: User) -> None:
        """Start the demo again: plans, tasks, outcomes, the feed log, demo settings and the audit trail.

        Logins are kept. Dropping the tables is the one way the append-only
        audit trail ends; the reset itself is the first event of the new one.
        """
        drops = "".join(f"DROP TABLE IF EXISTS {table};\n" for table in _WORKFLOW_TABLES)
        self._database.run_script("reset_workflow_tables", drops)
        self._database.run_script("create_ops_schema", read_sql_file(self._schema_path, "read_ops_schema_file"))
        self._log(actor, "DEMO_RESET", {})

    def _log(self, actor: User, event_type: str, payload: dict[str, object]) -> None:
        self._events.append(Event(self._now(), actor.username, event_type, "demo", "demo", payload))


class OperationsCycle:
    """One call per page load: catch the booking feed up, then close any day that is over."""

    def __init__(self, monitor: LiveMonitor, closer: DayCloser, clock: ReplayClock) -> None:
        self._monitor, self._closer, self._clock = monitor, closer, clock

    def refresh(self) -> tuple[LiveSnapshot, list[str]]:
        snapshot = self._monitor.refresh()
        return snapshot, self._closer.catch_up(self._clock.real_now())
