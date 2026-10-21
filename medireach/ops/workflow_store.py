"""Repositories for plans, plan items, tasks, revealed outcomes and the demo state (ops.db)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Sequence

import pandas as pd

from medireach.data.database import SqliteDatabase
from medireach.data.repositories import TIMESTAMP_FORMAT
from medireach.domain import HoldoutMode, Outcome, Plan, PlanStatus, Task, TaskKind, TaskStatus
from medireach.errors import DataAccessError

ITEM_COLUMNS: tuple[str, ...] = (
    "appointment_id", "patient_id", "clinic", "age", "lead_days", "risk_score", "risk_calibrated",
    "day_rank", "recommended_action", "action", "arm", "override_reason", "late_booking", "reasons",
)
OUTCOME_COLUMNS: tuple[str, ...] = ("appointment_id", "no_show", "historical_no_show", "source", "revealed_at")


def _rows(frame: pd.DataFrame, columns: Sequence[str]) -> list[list[Any]]:
    """Plain Python values (sqlite3 cannot bind numpy scalars); NaN becomes NULL."""
    subset = frame[list(columns)].astype(object)
    return subset.where(subset.notna(), None).values.tolist()


class PlanRepository(ABC):
    """Daily plans, their items and their revealed outcomes."""

    @abstractmethod
    def create(self, plan: Plan, items: pd.DataFrame) -> Plan:
        """Store a new plan (its plan_id is assigned here) with all its items, in one transaction."""

    @abstractmethod
    def get(self, plan_id: int) -> Plan:
        """One plan; DataAccessError if it does not exist."""

    @abstractmethod
    def find(self, cycle_start: date, target_real_date: date) -> Plan | None:
        """The plan for one target day of one replay week, if any."""

    @abstractmethod
    def list_plans(self, statuses: Sequence[PlanStatus] = ()) -> list[Plan]:
        """Plans, oldest target first, optionally only some statuses."""

    @abstractmethod
    def items(self, plan_id: int) -> pd.DataFrame:
        """Every item of a plan (ITEM_COLUMNS), by day rank."""

    @abstractmethod
    def add_items(self, plan_id: int, items: pd.DataFrame) -> None:
        """Add items (late bookings)."""

    @abstractmethod
    def update_items(self, plan_id: int, items: pd.DataFrame) -> None:
        """Set action, arm and override_reason of the given appointment_ids."""

    @abstractmethod
    def update_status(self, plan_id: int, status: PlanStatus, **fields: Any) -> None:
        """Move a plan to `status`, also setting any of the approval or closing fields."""

    @abstractmethod
    def save_outcomes(self, plan_id: int, outcomes: pd.DataFrame) -> None:
        """Store the revealed attendance of a plan's appointments (OUTCOME_COLUMNS)."""

    @abstractmethod
    def outcomes(self, plan_ids: Sequence[int]) -> pd.DataFrame:
        """Revealed outcomes of the given plans, with plan_id."""


class TaskRepository(ABC):
    """Contact tasks."""

    @abstractmethod
    def add(self, tasks: Sequence[Task]) -> list[Task]:
        """Store tasks (task_id assigned here) and return them."""

    @abstractmethod
    def get(self, task_id: int) -> Task:
        """One task; DataAccessError if it does not exist."""

    @abstractmethod
    def get_many(self, task_ids: Sequence[int]) -> list[Task]:
        """Several tasks at once, in task_id order."""

    @abstractmethod
    def for_plans(self, plan_ids: Sequence[int]) -> pd.DataFrame:
        """All tasks of the given plans, by priority then attempt."""

    @abstractmethod
    def update(self, task: Task) -> None:
        """Save the task's status, outcome, notes, assignee and timestamp."""

    @abstractmethod
    def update_many(self, tasks: Sequence[Task]) -> None:
        """Save several tasks in one transaction."""


def _plan(row: pd.Series) -> Plan:
    def optional(name: str) -> Any:
        value = row[name]
        return None if value is None or (isinstance(value, float) and pd.isna(value)) else value

    mode = optional("holdout_mode")
    return Plan(
        plan_id=int(row["plan_id"]), cycle_start=date.fromisoformat(row["cycle_start"]),
        target_real_date=date.fromisoformat(row["target_real_date"]),
        target_data_date=date.fromisoformat(row["target_data_date"]), plan_day=date.fromisoformat(row["plan_day"]),
        as_of=datetime.strptime(row["as_of_ts"], TIMESTAMP_FORMAT), model_version=str(row["model_version"]),
        status=PlanStatus(row["status"]), call_capacity=int(row["call_capacity"]), sms_capacity=int(row["sms_capacity"]),
        holdout_mode=HoldoutMode(mode) if mode else None,
        holdout_share=float(row["holdout_share"]) if optional("holdout_share") is not None else None,
        seed=int(row["seed"]), recalibration_shift=float(row["recalibration_shift"]),
        created_by=str(row["created_by"]), created_at=str(row["created_at"]),
        approved_by=optional("approved_by"), approved_at=optional("approved_at"), closed_at=optional("closed_at"),
    )


class SqlitePlanRepository(PlanRepository):
    """Plans stored in ops.db."""

    _PLAN_INSERT = """
        INSERT INTO daily_plans (cycle_start, target_real_date, target_data_date, plan_day, as_of_ts,
            model_version, status, call_capacity, sms_capacity, seed, recalibration_shift, created_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    _ITEM_INSERT = f"INSERT INTO plan_items (plan_id, {', '.join(ITEM_COLUMNS)}) VALUES ({', '.join('?' * (len(ITEM_COLUMNS) + 1))})"

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def create(self, plan: Plan, items: pd.DataFrame) -> Plan:
        row = (plan.cycle_start.isoformat(), plan.target_real_date.isoformat(), plan.target_data_date.isoformat(),
               plan.plan_day.isoformat(), plan.as_of.strftime(TIMESTAMP_FORMAT), plan.model_version, plan.status.value,
               plan.call_capacity, plan.sms_capacity, plan.seed, plan.recalibration_shift, plan.created_by, plan.created_at)
        self._database.write("insert_plan", [(self._PLAN_INSERT, [row])])
        stored = self.find(plan.cycle_start, plan.target_real_date)
        if stored is None:
            raise DataAccessError("insert_plan", "the plan was not stored")
        self.add_items(stored.plan_id, items)
        return stored

    def _select(self, where: str, params: Sequence[Any]) -> list[Plan]:
        frame = self._database.query("select_plans", f"SELECT * FROM daily_plans WHERE {where} ORDER BY target_real_date, plan_id", params)
        return [_plan(row) for _, row in frame.iterrows()]

    def get(self, plan_id: int) -> Plan:
        found = self._select("plan_id = ?", (plan_id,))
        if not found:
            raise DataAccessError("select_plan", f"no plan {plan_id}")
        return found[0]

    def find(self, cycle_start: date, target_real_date: date) -> Plan | None:
        found = self._select("cycle_start = ? AND target_real_date = ?", (cycle_start.isoformat(), target_real_date.isoformat()))
        return found[0] if found else None

    def list_plans(self, statuses: Sequence[PlanStatus] = ()) -> list[Plan]:
        if not statuses:
            return self._select("1 = 1", ())
        marks = ", ".join("?" for _ in statuses)
        return self._select(f"status IN ({marks})", [status.value for status in statuses])

    def items(self, plan_id: int) -> pd.DataFrame:
        sql = f"SELECT {', '.join(ITEM_COLUMNS)} FROM plan_items WHERE plan_id = ? ORDER BY day_rank, appointment_id"
        return self._database.query("select_plan_items", sql, (plan_id,))

    def add_items(self, plan_id: int, items: pd.DataFrame) -> None:
        rows = [[plan_id, *values] for values in _rows(items, ITEM_COLUMNS)]
        self._database.write("insert_plan_items", [(self._ITEM_INSERT, rows)])

    def update_items(self, plan_id: int, items: pd.DataFrame) -> None:
        sql = "UPDATE plan_items SET action = ?, arm = ?, override_reason = ? WHERE plan_id = ? AND appointment_id = ?"
        rows = [[*values, plan_id, appointment_id] for appointment_id, values in
                zip(items["appointment_id"].tolist(), _rows(items, ("action", "arm", "override_reason")))]
        self._database.write("update_plan_items", [(sql, rows)])

    def update_status(self, plan_id: int, status: PlanStatus, **fields: Any) -> None:
        allowed = {"holdout_mode", "holdout_share", "approved_by", "approved_at", "closed_at"}
        unknown = set(fields) - allowed
        if unknown:
            raise DataAccessError("update_plan_status", f"unknown fields {sorted(unknown)}")
        # Column names come from the fixed set above; values are bound.
        assignments = ", ".join(["status = ?", *(f"{name} = ?" for name in fields)])
        values = [status.value, *(value.value if hasattr(value, "value") else value for value in fields.values())]
        self._database.write("update_plan_status", [(f"UPDATE daily_plans SET {assignments} WHERE plan_id = ?", [[*values, plan_id]])])

    def save_outcomes(self, plan_id: int, outcomes: pd.DataFrame) -> None:
        sql = f"INSERT INTO revealed_outcomes (plan_id, {', '.join(OUTCOME_COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?)"
        self._database.write("insert_revealed_outcomes", [(sql, [[plan_id, *values] for values in _rows(outcomes, OUTCOME_COLUMNS)])])

    def outcomes(self, plan_ids: Sequence[int]) -> pd.DataFrame:
        if not plan_ids:
            return pd.DataFrame(columns=["plan_id", *OUTCOME_COLUMNS])
        marks = ", ".join("?" for _ in plan_ids)
        sql = f"SELECT plan_id, {', '.join(OUTCOME_COLUMNS)} FROM revealed_outcomes WHERE plan_id IN ({marks})"
        return self._database.query("select_revealed_outcomes", sql, list(plan_ids))


def _task(row: pd.Series) -> Task:
    outcome = row["outcome"]
    parent = row["parent_task_id"]
    return Task(
        task_id=int(row["task_id"]), plan_id=int(row["plan_id"]), patient_id=int(row["patient_id"]),
        kind=TaskKind(row["kind"]), assigned_to=str(row["assigned_to"]), status=TaskStatus(row["status"]),
        attempt=int(row["attempt"]), priority_rank=int(row["priority_rank"]),
        outcome=Outcome(outcome) if isinstance(outcome, str) else None,
        notes=row["notes"] if isinstance(row["notes"], str) else None,
        parent_task_id=None if parent is None or pd.isna(parent) else int(parent),
        updated_at=row["updated_at"] if isinstance(row["updated_at"], str) else None,
    )


class SqliteTaskRepository(TaskRepository):
    """Tasks stored in ops.db."""

    _COLUMNS = ("plan_id", "patient_id", "kind", "assigned_to", "status", "attempt", "priority_rank",
                "outcome", "notes", "parent_task_id", "updated_at")

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def add(self, tasks: Sequence[Task]) -> list[Task]:
        if not tasks:
            return []
        before = int(self._database.query("max_task_id", "SELECT COALESCE(MAX(task_id), 0) AS m FROM tasks").iloc[0]["m"])
        sql = f"INSERT INTO tasks ({', '.join(self._COLUMNS)}) VALUES ({', '.join('?' * len(self._COLUMNS))})"
        rows = [[task.plan_id, task.patient_id, task.kind.value, task.assigned_to, task.status.value, task.attempt,
                 task.priority_rank, task.outcome.value if task.outcome else None, task.notes, task.parent_task_id,
                 task.updated_at] for task in tasks]
        self._database.write("insert_tasks", [(sql, rows)])
        frame = self._database.query("select_new_tasks", "SELECT * FROM tasks WHERE task_id > ? ORDER BY task_id", (before,))
        return [_task(row) for _, row in frame.iterrows()]

    def get(self, task_id: int) -> Task:
        frame = self._database.query("select_task", "SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        if frame.empty:
            raise DataAccessError("select_task", f"no task {task_id}")
        return _task(frame.iloc[0])

    def get_many(self, task_ids: Sequence[int]) -> list[Task]:
        if not task_ids:
            return []
        marks = ", ".join("?" for _ in task_ids)
        frame = self._database.query("select_tasks", f"SELECT * FROM tasks WHERE task_id IN ({marks}) ORDER BY task_id",
                                     [int(task_id) for task_id in task_ids])
        return [_task(row) for _, row in frame.iterrows()]

    def for_plans(self, plan_ids: Sequence[int]) -> pd.DataFrame:
        if not plan_ids:
            return pd.DataFrame(columns=["task_id", *self._COLUMNS])
        marks = ", ".join("?" for _ in plan_ids)
        sql = f"SELECT * FROM tasks WHERE plan_id IN ({marks}) ORDER BY plan_id, priority_rank, attempt, task_id"
        return self._database.query("select_plan_tasks", sql, list(plan_ids))

    _UPDATE = "UPDATE tasks SET status = ?, outcome = ?, notes = ?, assigned_to = ?, updated_at = ? WHERE task_id = ?"

    @staticmethod
    def _update_row(task: Task) -> list[Any]:
        return [task.status.value, task.outcome.value if task.outcome else None, task.notes, task.assigned_to,
                task.updated_at, task.task_id]

    def update(self, task: Task) -> None:
        self._database.write("update_task", [(self._UPDATE, [self._update_row(task)])])

    def update_many(self, tasks: Sequence[Task]) -> None:
        self._database.write("update_tasks", [(self._UPDATE, [self._update_row(task) for task in tasks])])


class StateStore:
    """Small key-value settings shared by every user of the demo (table ops_state)."""

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def get(self, key: str, default: str | None = None) -> str | None:
        frame = self._database.query("select_state", "SELECT value FROM ops_state WHERE key = ?", (key,))
        return default if frame.empty else str(frame.iloc[0]["value"])

    def set(self, key: str, value: str) -> None:
        sql = "INSERT INTO ops_state (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value"
        self._database.write("upsert_state", [(sql, [(key, value)])])

    def delete(self, key: str) -> None:
        self._database.write("delete_state", [("DELETE FROM ops_state WHERE key = ?", [(key,)])])
