"""TaskService: the receptionist's task state machine, with retries and an SMS fallback."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from typing import Callable, Mapping, Sequence

import pandas as pd

from medireach.domain import Event, Outcome, Plan, PlanStatus, Role, Task, TaskKind, TaskStatus, User
from medireach.errors import AccessDenied
from medireach.ops.repositories import EventLog
from medireach.ops.workflow_store import PlanRepository, TaskRepository

CALL_OUTCOMES: frozenset[Outcome] = frozenset({
    Outcome.CONFIRMED, Outcome.RESCHEDULE, Outcome.CANCEL, Outcome.NO_ANSWER, Outcome.WRONG_NUMBER,
})
SMS_OUTCOMES: frozenset[Outcome] = frozenset({Outcome.SMS_SENT})
MAX_CALL_ATTEMPTS = 2
SKIP_REASONS: tuple[str, ...] = ("Patient asked not to be contacted", "Already confirmed by another channel",
                                 "Out of time today", "Other")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TaskService:
    """Records what happened on each task. Only open tasks of an approved plan can change.

    An unanswered call gets one retry; if that is unanswered too, an SMS task
    replaces it. A wrong number ends the attempts.
    """

    def __init__(self, tasks: TaskRepository, plans: PlanRepository, events: EventLog,
                 clock: Callable[[], str] = _utc_now) -> None:
        self._tasks = tasks
        self._plans = plans
        self._events = events
        self._clock = clock

    def tasks_for(self, plan_id: int, username: str | None = None) -> pd.DataFrame:
        tasks = self._tasks.for_plans([plan_id])
        return tasks if username is None else tasks[tasks["assigned_to"] == username].reset_index(drop=True)

    def plan_for_day(self, day: date) -> Plan | None:
        """The approved plan whose calls are made on `day` (the clinic day before its target)."""
        return next((plan for plan in self._plans.list_plans([PlanStatus.APPROVED]) if plan.plan_day == day), None)

    def worklist(self, plan_id: int, username: str | None = None) -> pd.DataFrame:
        """Tasks with what the receptionist needs about the patient: clinic, age, appointments, reasons."""
        tasks = self.tasks_for(plan_id, username)
        items = self._plans.items(plan_id).sort_values(["risk_calibrated", "appointment_id"], ascending=[False, True])
        patients = items.groupby("patient_id").agg(
            clinic=("clinic", "first"), age=("age", "first"), appointments=("appointment_id", "size"),
            risk=("risk_calibrated", "max"), reasons=("reasons", "first"), lead_days=("lead_days", "max"),
        )
        board = tasks.join(patients, on="patient_id")
        return board.sort_values(["priority_rank", "attempt", "task_id"]).reset_index(drop=True)

    def _editable(self, task_id: int, actor: User) -> Task:
        task = self._tasks.get(task_id)
        open_plans = {task.plan_id} if self._plans.get(task.plan_id).status == PlanStatus.APPROVED else set()
        self._check(task, actor, open_plans)
        return task

    @staticmethod
    def _check(task: Task, actor: User, open_plans: set[int]) -> None:
        if actor.role != Role.ADMIN and task.assigned_to != actor.username:
            raise AccessDenied("this task is assigned to someone else")
        if task.status != TaskStatus.PENDING:
            raise ValueError(f"task {task.task_id} is already {task.status.value}")
        if task.plan_id not in open_plans:
            raise ValueError("the plan for this task is closed")

    def record(self, task_id: int, outcome: Outcome, actor: User, notes: str = "") -> list[Task]:
        """Save an outcome; returns any follow-up task created (a retry or an SMS)."""
        return self.record_many({task_id: outcome}, actor, notes)

    def record_many(self, outcomes: Mapping[int, Outcome], actor: User, notes: str = "") -> list[Task]:
        """Save several outcomes at once (same rules as one at a time); returns the follow-up tasks."""
        tasks = self._tasks.get_many(list(outcomes))
        if len(tasks) != len(outcomes):
            raise ValueError("unknown task id")
        open_plans = {plan_id for plan_id in {task.plan_id for task in tasks}
                      if self._plans.get(plan_id).status == PlanStatus.APPROVED}
        stamp, finished, events = self._clock(), [], []
        for task in tasks:
            self._check(task, actor, open_plans)
            outcome = outcomes[task.task_id]
            if outcome not in (CALL_OUTCOMES if task.kind == TaskKind.CALL else SMS_OUTCOMES):
                raise ValueError(f"{outcome.value} is not a valid outcome for a {task.kind.value} task")
            done = replace(task, status=TaskStatus.DONE, outcome=outcome, notes=notes.strip() or None, updated_at=stamp)
            finished.append(done)
            events.append(self._event(actor, "TASK_DONE", done, {"outcome": outcome.value, "attempt": task.attempt}))
        self._tasks.update_many(finished)
        self._events.append_many(events)
        follow_ups = [follow for follow in (self._follow_up(task) for task in finished) if follow is not None]
        return self._tasks.add(follow_ups)

    def _follow_up(self, task: Task) -> Task | None:
        if task.kind != TaskKind.CALL or task.outcome != Outcome.NO_ANSWER:
            return None
        retry = task.attempt < MAX_CALL_ATTEMPTS
        return Task(task_id=0, plan_id=task.plan_id, patient_id=task.patient_id,
                    kind=TaskKind.CALL if retry else TaskKind.SMS, assigned_to=task.assigned_to,
                    status=TaskStatus.PENDING, attempt=task.attempt + 1 if retry else 1,
                    priority_rank=task.priority_rank, parent_task_id=task.task_id)

    def skip(self, task_id: int, reason: str, actor: User) -> None:
        if not reason.strip():
            raise ValueError("a reason is needed to skip a task")
        task = self._editable(task_id, actor)
        skipped = replace(task, status=TaskStatus.SKIPPED, notes=reason.strip(), updated_at=self._clock())
        self._tasks.update(skipped)
        self._log(actor, "TASK_SKIPPED", skipped, {"reason": reason.strip()})

    def reassign(self, task_ids: Sequence[int], to_username: str, actor: User) -> int:
        """Move open tasks to another receptionist (admin only). Returns how many moved."""
        if actor.role != Role.ADMIN:
            raise AccessDenied("only the admin can reassign tasks")
        moved = 0
        for task_id in task_ids:
            task = self._editable(task_id, actor)
            if task.assigned_to != to_username:
                self._tasks.update(replace(task, assigned_to=to_username, updated_at=self._clock()))
                self._log(actor, "TASK_REASSIGNED", task, {"from": task.assigned_to, "to": to_username})
                moved += 1
        return moved

    def expire_open(self, plan_id: int, actor: str = "system") -> int:
        """At the end of the working day, open tasks become NOT_ATTEMPTED. Returns how many."""
        tasks = self._tasks.for_plans([plan_id])
        open_ids = tasks.loc[tasks["status"] == TaskStatus.PENDING.value, "task_id"].astype(int).tolist()
        stamp = self._clock()
        self._tasks.update_many([replace(task, status=TaskStatus.NOT_ATTEMPTED, updated_at=stamp)
                                 for task in self._tasks.get_many(open_ids)])
        if open_ids:
            self._events.append(Event(self._clock(), actor, "TASKS_EXPIRED", "plan", str(plan_id), {"count": len(open_ids)}))
        return len(open_ids)

    def _event(self, actor: User, event_type: str, task: Task, payload: dict[str, object]) -> Event:
        return Event(self._clock(), actor.username, event_type, "task", str(task.task_id),
                     {"plan_id": task.plan_id, "patient_id": task.patient_id, **payload})

    def _log(self, actor: User, event_type: str, task: Task, payload: dict[str, object]) -> None:
        self._events.append(self._event(actor, event_type, task, payload))
