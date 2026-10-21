"""DailyPlanner: propose tomorrow's outreach plan, let the admin edit it, approve it with a hold-out."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from medireach.data.repositories import AsOfAppointmentRepository
from medireach.domain import (
    HOLDOUT_SHARES, Action, Arm, Event, HoldoutMode, ImpactAssumptions, Plan, PlanStatus, Role, Task, TaskKind,
    TaskStatus, User,
)
from medireach.errors import AccessDenied
from medireach.live.clock import ReplayCalendar, ReplayClock
from medireach.ops.measurement import Recalibrator
from medireach.ops.repositories import EventLog, UserRepository
from medireach.ops.workflow_store import PlanRepository, TaskRepository
from medireach.outreach.policies import FixedCountPolicy
from medireach.outreach.service import OutreachService

OVERRIDE_REASONS: tuple[str, ...] = (
    "Patient already confirmed", "Known contact problem", "Clinical priority", "Capacity changed", "Other",
)
LATE_BOOKING_REASON = "Late booking"
RECALIBRATION_DAYS = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HoldoutAllocator:
    """Seeded randomisation of the call list into a call arm and a hold-out arm, by patient."""

    def __init__(self, seed: int) -> None:
        self._seed = seed

    def holdout(self, patient_ids: Sequence[int], share: float) -> set[int]:
        ordered = np.array(sorted(set(int(p) for p in patient_ids)))
        shuffled = np.random.default_rng(self._seed).permutation(ordered)
        return set(int(p) for p in shuffled[: int(round(share * len(ordered)))])


def assign_round_robin(count: int, receptionists: Sequence[str]) -> list[str]:
    """Deal tasks (already in priority order) to receptionists in turn, so everyone gets a mix."""
    if not receptionists:
        raise ValueError("no active receptionist account to assign tasks to")
    return [receptionists[index % len(receptionists)] for index in range(count)]


def plan_numbers(items: pd.DataFrame) -> dict[str, float]:
    """Headline numbers of a plan, per patient where it matters (one contact per patient)."""
    patients = items.groupby("patient_id").agg(action=("action", "first"), risk=("risk_calibrated", "max"))
    on_call_list = patients.loc[patients["action"] == Action.CALL.value, "risk"].sum()
    expected = float(items["risk_calibrated"].sum())
    return {
        "appointments": float(len(items)), "patients": float(len(patients)),
        "calls": float((patients["action"] == Action.CALL.value).sum()),
        "sms": float((patients["action"] == Action.SMS.value).sum()),
        "expected_no_shows_raw": float(items["risk_score"].sum()), "expected_no_shows": expected,
        "expected_on_call_list": float(on_call_list),
        "call_list_capture": float(on_call_list / expected) if expected else 0.0,
    }


def holdout_cost(items: pd.DataFrame, share: float, assumptions: ImpactAssumptions) -> float:
    """Expected prevented no-shows given up by not calling the hold-out (illustrative assumptions)."""
    numbers = plan_numbers(items)
    return share * numbers["expected_on_call_list"] * assumptions.call_reach_rate * assumptions.call_effectiveness


class DailyPlanner:
    """Turns the engine's scores into an editable plan, and an approved plan into tasks."""

    def __init__(self, scorer: OutreachService, live: AsOfAppointmentRepository, clock: ReplayClock,
                 calendar: ReplayCalendar, plans: PlanRepository, tasks: TaskRepository, users: UserRepository,
                 events: EventLog, now: Callable[[], str] = _utc_now) -> None:
        self._scorer, self._live, self._clock, self._calendar = scorer, live, clock, calendar
        self._plans, self._tasks, self._users, self._events, self._now = plans, tasks, users, events, now

    # ------------------------------------------------------------ timing
    def next_target(self) -> date:
        return self._calendar.next_clinic_day(self._clock.moment().real_date)

    def current_plan(self, target: date | None = None) -> Plan | None:
        target = target or self.next_target()
        return self._plans.find(self._clock.cycle_start_for(target), target)

    def can_propose(self, target: date) -> tuple[bool, str]:
        moment = self._clock.moment()
        if self.current_plan(target) is not None:
            return False, "a plan for this day already exists"
        if self._clock.as_of_for(target, moment) < self._calendar.plan_as_of(target):
            plan_day = self._calendar.plan_day_for(target)
            return False, f"the plan can be made from {self._calendar.plan_as_of(target):%H:%M} on {plan_day:%A %d %B}"
        return True, ""

    # ----------------------------------------------------------- propose
    def propose(self, target: date, call_capacity: int, sms_capacity: int, actor: User) -> Plan:
        """Score the target day as it was known at plan time and store the proposal."""
        self._require_admin(actor)
        allowed, why = self.can_propose(target)
        if not allowed:
            raise ValueError(why)
        as_of = self._calendar.plan_as_of(target)
        view = self._live.view_at(as_of)
        day = self._clock.data_date_for(target)
        scored = self._scorer.score_appointments(day, day, source=view)
        details = view.get_appointments(day, day)[["appointment_id", "clinic", "age", "lead_days"]]
        recalibrator = self._recalibrator(view, as_of)
        items = self._items(scored.merge(details, on="appointment_id", validate="one_to_one"),
                            recalibrator, FixedCountPolicy(call_capacity, sms_capacity))
        plan = Plan(plan_id=0, cycle_start=self._clock.cycle_start_for(target), target_real_date=target,
                    target_data_date=day, plan_day=self._calendar.plan_day_for(target), as_of=as_of,
                    model_version=self._scorer.model_version, status=PlanStatus.PROPOSED,
                    call_capacity=call_capacity, sms_capacity=sms_capacity, holdout_mode=None, holdout_share=None,
                    seed=int(target.strftime("%Y%m%d")), recalibration_shift=recalibrator.shift,
                    created_by=actor.username, created_at=self._now())
        stored = self._plans.create(plan, items)
        self._log(actor, "PLAN_PROPOSED", stored.plan_id, plan_numbers(items))
        return stored

    def _recalibrator(self, view: AsOfAppointmentRepository, as_of: datetime) -> Recalibrator:
        days = self._calendar.previous_clinic_days(as_of.date(), RECALIBRATION_DAYS)
        if not days:
            return Recalibrator(0.0)
        recent = self._scorer.score_appointments(days[0], days[-1], source=view)
        recent = recent[recent["no_show"].notna()]
        return Recalibrator.fit(recent["risk_score"].to_numpy(), recent["no_show"].to_numpy())

    @staticmethod
    def _items(scored: pd.DataFrame, recalibrator: Recalibrator, policy: FixedCountPolicy) -> pd.DataFrame:
        """One row per appointment; the decision is taken per patient (one contact per patient)."""
        best = scored.sort_values(["risk_score", "appointment_id"], ascending=[False, True]).drop_duplicates("patient_id")
        units = policy.assign(best[["appointment_id", "patient_id", "appointment_date", "risk_score"]])
        decision = units.set_index("patient_id")[["action", "day_rank"]]
        items = scored.join(decision, on="patient_id")
        items["recommended_action"] = items["action"]
        items["risk_calibrated"] = recalibrator.apply(items["risk_score"])
        items["arm"], items["override_reason"], items["late_booking"] = None, None, 0
        return items.sort_values(["day_rank", "appointment_id"]).reset_index(drop=True)

    # -------------------------------------------------------------- edit
    def late_bookings(self, plan: Plan) -> pd.DataFrame:
        """Appointments for the target day booked after the plan was made, and not in it yet."""
        view = self._live.view_at(self._clock.as_of_for(plan.target_real_date, self._clock.moment()))
        rows = view.get_appointments(plan.target_data_date, plan.target_data_date)
        known = set(self._plans.items(plan.plan_id)["appointment_id"])
        late = rows[~rows["appointment_id"].isin(known)]
        return late[["appointment_id", "patient_id", "scheduled_ts", "clinic", "age", "lead_days"]].reset_index(drop=True)

    def add_late_booking(self, plan_id: int, appointment_id: int, action: Action, actor: User) -> None:
        plan = self._editable(plan_id, actor)
        view = self._live.view_at(self._clock.as_of_for(plan.target_real_date, self._clock.moment()))
        scored = self._scorer.score_appointments(plan.target_data_date, plan.target_data_date, source=view)
        details = view.get_appointments(plan.target_data_date, plan.target_data_date)
        row = scored.merge(details[["appointment_id", "clinic", "age", "lead_days"]], on="appointment_id")
        row = row[row["appointment_id"] == appointment_id]
        if row.empty or appointment_id in set(self._plans.items(plan_id)["appointment_id"]):
            raise ValueError(f"appointment {appointment_id} is not a late booking for this plan")
        items = self._plans.items(plan_id)
        item = row.assign(action=action.value, recommended_action=Action.NONE.value, day_rank=int(items["day_rank"].max()) + 1,
                          risk_calibrated=Recalibrator(plan.recalibration_shift).apply(row["risk_score"]),
                          arm=None, override_reason=LATE_BOOKING_REASON, late_booking=1)
        self._plans.add_items(plan_id, item)
        self._set_patient_action(plan_id, int(item["patient_id"].iloc[0]), action, LATE_BOOKING_REASON)
        self._log(actor, "LATE_BOOKING_ADDED", plan_id, {"appointment_id": appointment_id, "action": action.value})

    def override(self, plan_id: int, patient_id: int, action: Action, reason: str, actor: User) -> None:
        """Change what a patient gets. Every override needs a reason and is logged."""
        if not reason.strip():
            raise ValueError("an override needs a reason")
        self._editable(plan_id, actor)
        before = self._set_patient_action(plan_id, patient_id, action, reason.strip())
        self._log(actor, "PLAN_OVERRIDE", plan_id, {"patient_id": patient_id, "from": before, "to": action.value,
                                                    "reason": reason.strip()})

    def _set_patient_action(self, plan_id: int, patient_id: int, action: Action, reason: str) -> str:
        items = self._plans.items(plan_id)
        rows = items[items["patient_id"] == patient_id]
        if rows.empty:
            raise ValueError(f"patient {patient_id} is not in this plan")
        self._plans.update_items(plan_id, rows.assign(action=action.value, arm=None, override_reason=reason))
        return str(rows["action"].iloc[0])

    # ----------------------------------------------------------- approve
    def approve(self, plan_id: int, mode: HoldoutMode, actor: User) -> Plan:
        """Randomise the hold-out, create the tasks and deal them to the receptionists."""
        plan = self._editable(plan_id, actor)
        items = self._plans.items(plan_id)
        patients = items.groupby("patient_id").agg(action=("action", "first"), rank=("day_rank", "min")).sort_values("rank")
        share = HOLDOUT_SHARES[mode]
        call_patients = patients.index[patients["action"] == Action.CALL.value]
        held_out = HoldoutAllocator(plan.seed).holdout(call_patients, share)
        arm = {Action.CALL.value: Arm.CALL.value, Action.SMS.value: Arm.SMS.value, Action.NONE.value: Arm.NONE.value}
        patients["arm"] = [Arm.HOLDOUT.value if pid in held_out else arm[action] for pid, action in patients["action"].items()]
        self._plans.update_items(plan_id, items.assign(arm=items["patient_id"].map(patients["arm"])))
        tasks = self._tasks.add(self._tasks_for(plan, patients))
        self._plans.update_status(plan_id, PlanStatus.APPROVED, holdout_mode=mode, holdout_share=share,
                                  approved_by=actor.username, approved_at=self._now())
        counts = patients["arm"].value_counts().to_dict()
        self._log(actor, "PLAN_APPROVED", plan_id, {"tasks": len(tasks), **{str(k): int(v) for k, v in counts.items()}})
        return self._plans.get(plan_id)

    def _tasks_for(self, plan: Plan, patients: pd.DataFrame) -> list[Task]:
        contact = patients[patients["arm"].isin([Arm.CALL.value, Arm.SMS.value])]
        receptionists = sorted(user.username for user in self._users.list_users()
                               if user.role == Role.RECEPTIONIST and user.active)
        owners = assign_round_robin(len(contact), receptionists)
        return [Task(task_id=0, plan_id=plan.plan_id, patient_id=int(pid),
                     kind=TaskKind.CALL if row["arm"] == Arm.CALL.value else TaskKind.SMS,
                     assigned_to=owner, status=TaskStatus.PENDING, attempt=1, priority_rank=int(row["rank"]))
                for (pid, row), owner in zip(contact.iterrows(), owners)]

    # ----------------------------------------------------------- helpers
    def _editable(self, plan_id: int, actor: User) -> Plan:
        self._require_admin(actor)
        plan = self._plans.get(plan_id)
        if plan.status != PlanStatus.PROPOSED:
            raise ValueError(f"plan {plan_id} is {plan.status.value}; only a proposed plan can change")
        return plan

    @staticmethod
    def _require_admin(actor: User) -> None:
        if actor.role != Role.ADMIN or not actor.active:
            raise AccessDenied("only the admin can plan")

    def _log(self, actor: User, event_type: str, plan_id: int, payload: dict[str, object]) -> None:
        self._events.append(Event(self._now(), actor.username, event_type, "plan", str(plan_id), payload))
