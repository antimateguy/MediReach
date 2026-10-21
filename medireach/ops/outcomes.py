"""Outcome sources and DayCloser: what happened after a planned day, revealed when the day is over."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from medireach.data.repositories import AppointmentRepository
from medireach.domain import REACHED_OUTCOMES, Arm, Event, PlanStatus, TaskKind
from medireach.ops.repositories import EventLog
from medireach.ops.tasks import TaskService
from medireach.ops.workflow_store import PlanRepository, StateStore, TaskRepository

OUTCOME_SOURCE_KEY = "outcome_source"
PLANTED_EFFECT_KEY = "planted_effect"
HISTORICAL, SIMULATED = "historical", "simulated"
DEFAULT_PLANTED_EFFECT = 0.30


class OutcomeSource(ABC):
    """Decides each appointment's no-show once its day is over."""

    name: str

    @abstractmethod
    def observe(self, items: pd.DataFrame, reached_patients: set[int], seed: int) -> pd.Series:
        """0/1 no-show per row of `items` (columns appointment_id, patient_id, arm, historical_no_show)."""


class HistoricalOutcomes(OutcomeSource):
    """The real 2016 outcome. Calls made in the demo cannot change it, so any measured effect is noise."""

    name = HISTORICAL

    def observe(self, items: pd.DataFrame, reached_patients: set[int], seed: int) -> pd.Series:
        return items["historical_no_show"].astype(int)


class PlantedEffectOutcomes(OutcomeSource):
    """Simulation: a reached patient in the call arm who really missed now attends with probability `effect`.

    Used to check that the analysis recovers a known effect. Always labelled as a simulation.
    """

    name = SIMULATED

    def __init__(self, effect: float) -> None:
        if not 0 <= effect <= 1:
            raise ValueError("effect must be between 0 and 1")
        self.effect = effect

    def observe(self, items: pd.DataFrame, reached_patients: set[int], seed: int) -> pd.Series:
        # Draw in appointment_id order so the result does not depend on row order.
        order = items["appointment_id"].argsort(kind="stable").to_numpy()
        draws = np.empty(len(items))
        draws[order] = np.random.default_rng(seed).random(len(items))
        eligible = ((items["arm"] == Arm.CALL.value) & items["patient_id"].isin(reached_patients)
                    & (items["historical_no_show"] == 1)).to_numpy()
        flipped = eligible & (draws < self.effect)
        return pd.Series(np.where(flipped, 0, items["historical_no_show"].astype(int)), index=items.index)


def current_source(state: StateStore) -> OutcomeSource:
    """The outcome source chosen in the demo controls (historical unless switched)."""
    if state.get(OUTCOME_SOURCE_KEY, HISTORICAL) == SIMULATED:
        return PlantedEffectOutcomes(float(state.get(PLANTED_EFFECT_KEY, str(DEFAULT_PLANTED_EFFECT))))
    return HistoricalOutcomes()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DayCloser:
    """Closes the working day and reveals attendance, driven by the (replay) clock.

    After the plan day: open tasks become NOT_ATTEMPTED. After the target day:
    attendance is revealed for every appointment in the plan and the plan
    closes. Safe to call on every page load.
    """

    def __init__(self, plans: PlanRepository, tasks: TaskRepository, task_service: TaskService,
                 appointments: AppointmentRepository, state: StateStore, events: EventLog,
                 clock: Callable[[], str] = _utc_now) -> None:
        self._plans = plans
        self._tasks = tasks
        self._task_service = task_service
        self._appointments = appointments
        self._state = state
        self._events = events
        self._clock = clock

    def catch_up(self, real_now: datetime) -> list[str]:
        """Apply every step that is due at `real_now`; returns a line per step taken."""
        notes = []
        for plan in self._plans.list_plans([PlanStatus.APPROVED]):
            if real_now.date() > plan.plan_day:
                expired = self._task_service.expire_open(plan.plan_id)
                if expired:
                    notes.append(f"Plan {plan.plan_id}: {expired} open task(s) marked not attempted")
            if real_now.date() > plan.target_real_date:
                source = self.reveal(plan.plan_id)
                notes.append(f"Plan {plan.plan_id}: attendance revealed ({source})")
        return notes

    def reveal(self, plan_id: int) -> str:
        """Record attendance for a plan's appointments and close the plan. Returns the source used."""
        plan = self._plans.get(plan_id)
        items = self._plans.items(plan_id)
        actual = self._appointments.get_appointments(plan.target_data_date, plan.target_data_date)
        truth = actual.set_index("appointment_id")["no_show"].astype(int)
        items = items[items["appointment_id"].isin(truth.index)].copy()
        items["historical_no_show"] = items["appointment_id"].map(truth)
        source = current_source(self._state)
        observed = source.observe(items, self._reached_patients(plan_id), seed=plan.seed)
        revealed_at = self._clock()
        self._plans.save_outcomes(plan_id, pd.DataFrame({
            "appointment_id": items["appointment_id"], "no_show": observed,
            "historical_no_show": items["historical_no_show"], "source": source.name, "revealed_at": revealed_at,
        }))
        self._plans.update_status(plan_id, PlanStatus.CLOSED, closed_at=revealed_at)
        self._events.append(Event(revealed_at, "system", "PLAN_CLOSED", "plan", str(plan_id),
                                  {"source": source.name, "appointments": len(items)}))
        return source.name

    def _reached_patients(self, plan_id: int) -> set[int]:
        tasks = self._tasks.for_plans([plan_id])
        calls = tasks[(tasks["kind"] == TaskKind.CALL.value) & tasks["outcome"].isin({o.value for o in REACHED_OUTCOMES})]
        return set(calls["patient_id"].astype(int))
