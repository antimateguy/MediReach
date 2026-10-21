"""Enums and value objects shared across the package."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class Priority(str, Enum):
    """Outreach priority assigned by a policy."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Role(str, Enum):
    """Who is signed in. Public pages need no role."""

    ADMIN = "ADMIN"
    RECEPTIONIST = "RECEPTIONIST"


class Action(str, Enum):
    """Recommended contact type. A recommendation only: nothing is ever sent."""

    CALL = "Call"
    SMS = "SMS"
    NONE = "None"


class PlanStatus(str, Enum):
    """A daily plan is proposed by the engine, approved by the admin, closed after the day."""

    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    CLOSED = "CLOSED"


class HoldoutMode(str, Enum):
    """Share of the call list kept as a randomised comparison group."""

    PILOT = "PILOT"            # 50%: enough patients to measure the effect quickly
    MONITORING = "MONITORING"  # 10%: keeps measuring at a lower cost


class Arm(str, Enum):
    """What happens to a patient in an approved plan."""

    CALL = "CALL"
    HOLDOUT = "HOLDOUT"   # on the call list, randomly not called (usual care)
    SMS = "SMS"
    NONE = "NONE"


class TaskKind(str, Enum):
    CALL = "CALL"
    SMS = "SMS"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    DONE = "DONE"
    SKIPPED = "SKIPPED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"   # still open when the working day ended


class Outcome(str, Enum):
    """What the receptionist recorded."""

    CONFIRMED = "CONFIRMED"       # reached; will attend
    RESCHEDULE = "RESCHEDULE"     # reached; wants another date (slot released)
    CANCEL = "CANCEL"             # reached; cancels (slot released)
    NO_ANSWER = "NO_ANSWER"
    WRONG_NUMBER = "WRONG_NUMBER"
    SMS_SENT = "SMS_SENT"


REACHED_OUTCOMES: frozenset[Outcome] = frozenset({Outcome.CONFIRMED, Outcome.RESCHEDULE, Outcome.CANCEL})
RELEASED_OUTCOMES: frozenset[Outcome] = frozenset({Outcome.RESCHEDULE, Outcome.CANCEL})
HOLDOUT_SHARES: dict[HoldoutMode, float] = {HoldoutMode.PILOT: 0.5, HoldoutMode.MONITORING: 0.1}


@dataclass(frozen=True)
class Plan:
    """One day's outreach plan (see sql/ops_schema.sql, table daily_plans)."""

    plan_id: int
    cycle_start: date
    target_real_date: date
    target_data_date: date
    plan_day: date            # real date on which the plan is made and the calls are made
    as_of: datetime           # data time whose bookings the plan sees
    model_version: str
    status: PlanStatus
    call_capacity: int
    sms_capacity: int
    holdout_mode: HoldoutMode | None
    holdout_share: float | None
    seed: int
    recalibration_shift: float
    created_by: str
    created_at: str
    approved_by: str | None = None
    approved_at: str | None = None
    closed_at: str | None = None


@dataclass(frozen=True)
class Task:
    """One contact attempt for one patient (see table tasks)."""

    task_id: int
    plan_id: int
    patient_id: int
    kind: TaskKind
    assigned_to: str
    status: TaskStatus
    attempt: int
    priority_rank: int
    outcome: Outcome | None = None
    notes: str | None = None
    parent_task_id: int | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class User:
    """A signed-in person. Never carries the password or its hash."""

    user_id: int
    username: str
    display_name: str
    role: Role
    active: bool = True


@dataclass(frozen=True)
class Event:
    """One line of the audit trail: who did what, to which thing, when."""

    at: str
    actor: str
    event_type: str
    entity: str
    entity_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CleaningReport:
    """What the cleaning step did to the raw file, rule by rule."""

    rows_in: int
    rows_out: int
    dropped: dict[str, int]
    patients: int
    clinics: int
    patients_with_varying_attributes: int
    varying_by_attribute: dict[str, int]
    patient_id_collisions: int
    base_rate_before: float
    base_rate_after: float
    same_day_rows: int
    same_day_share: float
    same_day_no_show_rate: float
    # Rows the v1 filter (booking timestamp <= appointment midnight) would have dropped.
    v1_timestamp_filter_drops: int = 0

    def __post_init__(self) -> None:
        # Every removed row must be attributed to exactly one rule.
        if self.rows_in - sum(self.dropped.values()) != self.rows_out:
            raise ValueError("dropped counts do not add up to rows_in - rows_out")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelMetadata:
    """Everything needed to trust and reload a saved model bundle."""

    version: str
    model_name: str
    feature_columns: list[str]
    trained_at: str
    train_range: tuple[str, str]
    validation_range: tuple[str, str]
    test_range: tuple[str, str]
    sklearn_version: str
    python_version: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    selection_reason: str
    # Training-period no-show rate used to smooth the history feature. Scoring
    # must build features with this same value.
    history_prior_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelMetadata":
        # JSON turns tuples into lists; convert the date ranges back.
        fixed = dict(raw)
        for key in ("train_range", "validation_range", "test_range"):
            fixed[key] = tuple(raw[key])
        return cls(**fixed)


@dataclass(frozen=True)
class QueueRun:
    """One scoring run: which window, which model, which policy."""

    run_id: str
    created_at: str
    window_start: str
    window_end: str
    model_name: str
    model_version: str
    policy_name: str
    n_rows: int
    notes: str = ""


@dataclass(frozen=True)
class ImpactAssumptions:
    """Illustrative inputs to the impact estimate. Amounts are in INR."""

    call_reach_rate: float = 0.60
    call_effectiveness: float = 0.30
    sms_reach_rate: float = 0.90
    sms_effectiveness: float = 0.10
    cost_per_call: float = 40.0
    cost_per_sms: float = 1.0
    value_per_kept_visit: float = 500.0

    def __post_init__(self) -> None:
        rates = ("call_reach_rate", "call_effectiveness", "sms_reach_rate", "sms_effectiveness")
        for name in rates:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("cost_per_call", "cost_per_sms", "value_per_kept_visit"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True)
class ImpactResult:
    """Outcome of one strategy in the backtest, under one set of assumptions."""

    strategy: str
    n_appointments: int
    n_calls: int
    n_sms: int
    no_shows_total: int
    no_shows_called: float
    no_shows_sms: float
    prevented: float
    cost: float
    benefit: float
    net: float
    cost_per_prevented_no_show: float
    precision_in_contacted: float
    capture_rate: float
    lift_vs_random: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PilotPlan:
    """Design of a randomised pilot that would measure the real effect of calls."""

    p_control: float
    relative_reduction: float
    alpha: float
    power: float
    sample_size_per_arm: int
    total_sample: int
    daily_high_volume: float
    estimated_days: int
    primary_metric: str = "no-show rate"
    guardrails: tuple[str, ...] = field(default_factory=tuple)
