"""v3 Block H: planner, hold-out, tasks, day closing, outcome sources, measurement, demo controls."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from medireach.config import Settings
from medireach.data.cleaning import RawDataLoader
from medireach.data.database import IN_MEMORY, SqliteDatabase, read_sql_file
from medireach.data.repositories import AsOfAppointmentRepository
from medireach.domain import (
    HOLDOUT_SHARES, Action, Arm, HoldoutMode, ImpactAssumptions, Outcome, PlanStatus, TaskKind, TaskStatus, User,
)
from medireach.errors import AccessDenied
from medireach.features.builder import FeatureBuilder
from medireach.impact.pilot import PilotDesigner
from medireach.live.clock import ReplayCalendar, ReplayClock
from medireach.ml.models import LogisticRiskModel
from medireach.ops.auth import AuthService, PasswordHasher
from medireach.ops.demo import DemoControls, demo_now_provider
from medireach.ops.measurement import ExperimentAnalyzer, Recalibrator, relative_reduction
from medireach.ops.outcomes import SIMULATED, DayCloser, HistoricalOutcomes, PlantedEffectOutcomes
from medireach.ops.plans import LATE_BOOKING_REASON, DailyPlanner, HoldoutAllocator, assign_round_robin, plan_numbers
from medireach.ops.repositories import SqliteEventLog, SqliteUserRepository
from medireach.ops.tasks import TaskService
from medireach.ops.workflow_store import SqlitePlanRepository, SqliteTaskRepository, StateStore
from medireach.outreach.policies import CapacityPolicy
from medireach.outreach.reasons import ReasonExplainer
from medireach.outreach.service import OutreachService
from tests.doubles import InMemoryAppointmentRepository, InMemoryQueueRepository
from tests.factories import make_raw_appointments

WEEK_START = date(2016, 5, 12)            # a Thursday in the synthetic data
WEDNESDAY_PLAN_DAY = datetime(2026, 10, 21, 8, 30)   # plans Thursday 2026-10-22 = data 2016-05-12
PASSWORD = "demo"


@dataclass
class Workflow:
    """Everything bootstrap would wire, on synthetic data and an in-memory ops.db."""

    source: InMemoryAppointmentRepository
    state: StateStore
    clock: ReplayClock
    calendar: ReplayCalendar
    plans: SqlitePlanRepository
    tasks: SqliteTaskRepository
    task_service: TaskService
    planner: DailyPlanner
    closer: DayCloser
    analyzer: ExperimentAnalyzer
    demo: DemoControls
    events: SqliteEventLog
    users: SqliteUserRepository
    admin: User
    reception1: User
    reception2: User


@pytest.fixture(scope="module")
def data() -> tuple[InMemoryAppointmentRepository, LogisticRiskModel]:
    settings = Settings()
    patients, appointments, _ = RawDataLoader(settings).clean(make_raw_appointments(400, 4000, seed=3))
    source = InMemoryAppointmentRepository(patients, appointments)
    features, meta = FeatureBuilder(settings, 0.2).build(source.get_appointments())
    return source, LogisticRiskModel().fit(features, meta["no_show"])


@pytest.fixture()
def flow(data) -> Iterator[Workflow]:
    source, model = data
    settings = replace(Settings(), replay_week_start=WEEK_START.isoformat())
    database = SqliteDatabase(IN_MEMORY)
    database.run_script("create_ops_schema", read_sql_file(settings.ops_schema_path, "read_ops_schema_file"))
    state = StateStore(database)
    clock = ReplayClock(WEEK_START, "Asia/Kolkata", now=demo_now_provider(state, WEDNESDAY_PLAN_DAY, datetime.now))
    calendar = ReplayCalendar(clock, source.appointment_dates(), plan_hour=8)
    live = AsOfAppointmentRepository(source, lambda: clock.moment().data_now)
    users, events = SqliteUserRepository(database), SqliteEventLog(database)
    auth = AuthService(users, PasswordHasher(iterations=1_000), events)
    auth.seed_demo_users(PASSWORD)
    plans, tasks = SqlitePlanRepository(database), SqliteTaskRepository(database)
    task_service = TaskService(tasks, plans, events)
    service = OutreachService(source, InMemoryQueueRepository(), FeatureBuilder(settings, 0.2), model, "v-test",
                              CapacityPolicy(), ReasonExplainer(model))
    planner = DailyPlanner(service, live, clock, calendar, plans, tasks, users, events)
    closer = DayCloser(plans, tasks, task_service, source, state, events)
    analyzer = ExperimentAnalyzer(plans, tasks, PilotDesigner(), ImpactAssumptions())
    demo = DemoControls(state, clock, calendar, plans, task_service, database, settings.ops_schema_path, events, 8)
    yield Workflow(source, state, clock, calendar, plans, tasks, task_service, planner, closer, analyzer, demo,
                   events, users, auth.login("admin", PASSWORD), auth.login("reception1", PASSWORD),
                   auth.login("reception2", PASSWORD))
    database.close()


def _proposed(flow: Workflow, calls: int = 20, sms: int = 30):
    target = flow.planner.next_target()
    return flow.planner.propose(target, calls, sms, flow.admin)


def _approved(flow: Workflow, mode: HoldoutMode = HoldoutMode.PILOT):
    return flow.planner.approve(_proposed(flow).plan_id, mode, flow.admin)


# -------------------------------------------------------------- planner

def test_plan_is_refused_before_plan_time_and_made_once(flow: Workflow) -> None:
    flow.demo.set_demo_now(datetime(2026, 10, 21, 7, 0), flow.admin)
    target = flow.planner.next_target()
    assert target == date(2026, 10, 22)
    allowed, why = flow.planner.can_propose(target)
    assert not allowed and "08:00" in why
    flow.demo.set_demo_now(WEDNESDAY_PLAN_DAY, flow.admin)
    plan = _proposed(flow)
    assert plan.status == PlanStatus.PROPOSED and plan.as_of == datetime(2016, 5, 11, 8, 0)
    with pytest.raises(ValueError, match="already exists"):
        flow.planner.propose(target, 20, 30, flow.admin)


def test_plan_sees_only_bookings_made_by_plan_time(flow: Workflow) -> None:
    plan = _proposed(flow)
    items = flow.plans.items(plan.plan_id)
    everything = flow.source.get_appointments(plan.target_data_date, plan.target_data_date)
    known = everything[everything["scheduled_ts"] <= pd.Timestamp(plan.as_of)]
    assert set(items["appointment_id"]) == set(known["appointment_id"]) and len(known) < len(everything)


def test_one_decision_per_patient_and_capacity_is_exact(flow: Workflow) -> None:
    items = flow.plans.items(_proposed(flow, calls=20, sms=30).plan_id)
    assert (items.groupby("patient_id")["action"].nunique() == 1).all()
    numbers = plan_numbers(items)
    assert numbers["calls"] == 20 and numbers["sms"] == 30
    calls = items.groupby("patient_id").agg(action=("action", "first"), risk=("risk_score", "max"))
    assert calls.loc[calls["action"] == "Call", "risk"].min() >= calls.loc[calls["action"] != "Call", "risk"].max()
    assert items["risk_calibrated"].between(0, 1).all()


def test_overrides_need_a_reason_an_admin_and_an_open_plan(flow: Workflow) -> None:
    plan = _proposed(flow)
    patient = int(flow.plans.items(plan.plan_id).iloc[0]["patient_id"])
    with pytest.raises(ValueError, match="reason"):
        flow.planner.override(plan.plan_id, patient, Action.SMS, "  ", flow.admin)
    with pytest.raises(AccessDenied):
        flow.planner.override(plan.plan_id, patient, Action.SMS, "Other", flow.reception1)
    flow.planner.override(plan.plan_id, patient, Action.SMS, "Patient already confirmed", flow.admin)
    rows = flow.plans.items(plan.plan_id).query("patient_id == @patient")
    assert set(rows["action"]) == {"SMS"} and set(rows["override_reason"]) == {"Patient already confirmed"}
    assert "PLAN_OVERRIDE" in set(flow.events.list_events(entity="plan")["event_type"])
    flow.planner.approve(plan.plan_id, HoldoutMode.PILOT, flow.admin)
    with pytest.raises(ValueError, match="only a proposed plan"):
        flow.planner.override(plan.plan_id, patient, Action.NONE, "Other", flow.admin)


def test_late_bookings_can_be_added_after_plan_time(flow: Workflow) -> None:
    plan = _proposed(flow)
    assert flow.planner.late_bookings(plan).empty      # nothing booked since plan time yet
    flow.demo.set_demo_now(datetime(2026, 10, 21, 19, 0), flow.admin)    # later that day
    late = flow.planner.late_bookings(plan)
    assert len(late) > 0 and (late["scheduled_ts"] > pd.Timestamp(plan.as_of)).all()
    appointment_id = int(late.iloc[0]["appointment_id"])
    flow.planner.add_late_booking(plan.plan_id, appointment_id, Action.CALL, flow.admin)
    row = flow.plans.items(plan.plan_id).query("appointment_id == @appointment_id").iloc[0]
    assert row["late_booking"] == 1 and row["action"] == "Call" and row["override_reason"] == LATE_BOOKING_REASON
    assert appointment_id not in set(flow.planner.late_bookings(plan)["appointment_id"])


# --------------------------------------------------------------- hold-out

def test_holdout_is_seeded_and_the_right_size() -> None:
    patients = list(range(100, 300))
    first, again = HoldoutAllocator(7).holdout(patients, 0.5), HoldoutAllocator(7).holdout(list(reversed(patients)), 0.5)
    assert first == again and len(first) == 100
    assert HoldoutAllocator(8).holdout(patients, 0.5) != first
    assert len(HoldoutAllocator(7).holdout(patients, HOLDOUT_SHARES[HoldoutMode.MONITORING])) == 20


def test_approval_creates_tasks_only_for_the_call_arm_and_sms(flow: Workflow) -> None:
    plan = _approved(flow)
    items = flow.plans.items(plan.plan_id)
    patients = items.groupby("patient_id").agg(arm=("arm", "first"), action=("action", "first"))
    on_list = patients[patients["action"] == "Call"]
    assert (on_list["arm"] == Arm.HOLDOUT.value).sum() == round(0.5 * len(on_list))
    tasks = flow.task_service.tasks_for(plan.plan_id)
    tasked = set(tasks["patient_id"])
    assert tasked == set(patients.index[patients["arm"].isin(["CALL", "SMS"])])
    assert not tasked & set(patients.index[patients["arm"] == "HOLDOUT"])
    per_user = tasks["assigned_to"].value_counts()
    assert set(per_user.index) == {"reception1", "reception2"} and per_user.max() - per_user.min() <= 1
    assert plan.status == PlanStatus.APPROVED and plan.holdout_share == 0.5
    with pytest.raises(ValueError):
        flow.planner.approve(plan.plan_id, HoldoutMode.PILOT, flow.admin)


def test_round_robin_needs_someone_to_assign_to() -> None:
    assert assign_round_robin(5, ["a", "b"]) == ["a", "b", "a", "b", "a"]
    with pytest.raises(ValueError):
        assign_round_robin(1, [])


# ------------------------------------------------------------------ tasks

def _first_task(flow: Workflow, plan_id: int, kind: TaskKind, owner: str) -> int:
    tasks = flow.task_service.tasks_for(plan_id, owner)
    return int(tasks[(tasks["kind"] == kind.value) & (tasks["status"] == "PENDING")].iloc[0]["task_id"])


def test_unanswered_call_gets_one_retry_then_an_sms(flow: Workflow) -> None:
    plan = _approved(flow)
    task_id = _first_task(flow, plan.plan_id, TaskKind.CALL, "reception1")
    retry = flow.task_service.record(task_id, Outcome.NO_ANSWER, flow.reception1)
    assert len(retry) == 1 and retry[0].kind == TaskKind.CALL and retry[0].attempt == 2
    fallback = flow.task_service.record(retry[0].task_id, Outcome.NO_ANSWER, flow.reception1)
    assert len(fallback) == 1 and fallback[0].kind == TaskKind.SMS and fallback[0].parent_task_id == retry[0].task_id
    assert flow.task_service.record(fallback[0].task_id, Outcome.SMS_SENT, flow.reception1) == []


def test_task_rules(flow: Workflow) -> None:
    plan = _approved(flow)
    call = _first_task(flow, plan.plan_id, TaskKind.CALL, "reception1")
    with pytest.raises(AccessDenied):
        flow.task_service.record(call, Outcome.CONFIRMED, flow.reception2)
    assert flow.task_service.record(call, Outcome.WRONG_NUMBER, flow.reception1) == []
    with pytest.raises(ValueError, match="already DONE"):
        flow.task_service.record(call, Outcome.CONFIRMED, flow.reception1)
    sms = _first_task(flow, plan.plan_id, TaskKind.SMS, "reception2")
    with pytest.raises(ValueError, match="not a valid outcome"):
        flow.task_service.record(sms, Outcome.CONFIRMED, flow.reception2)
    with pytest.raises(ValueError, match="reason"):
        flow.task_service.skip(sms, "", flow.reception2)
    with pytest.raises(AccessDenied):
        flow.task_service.reassign([sms], "reception1", flow.reception2)
    assert flow.task_service.reassign([sms], "reception1", flow.admin) == 1
    assert flow.tasks.get(sms).assigned_to == "reception1"


# ------------------------------------------------------------ day closing

def test_day_closer_expires_open_tasks_then_reveals_attendance(flow: Workflow) -> None:
    plan = _approved(flow)
    assert flow.closer.catch_up(datetime(2026, 10, 21, 20, 0)) == []          # still the plan day
    flow.closer.catch_up(datetime(2026, 10, 22, 9, 0))                         # target day: tasks expire
    tasks = flow.task_service.tasks_for(plan.plan_id)
    assert (tasks["status"] == TaskStatus.NOT_ATTEMPTED.value).all()
    assert flow.plans.get(plan.plan_id).status == PlanStatus.APPROVED
    notes = flow.closer.catch_up(datetime(2026, 10, 23, 8, 0))               # after the target day
    assert notes and flow.plans.get(plan.plan_id).status == PlanStatus.CLOSED
    outcomes = flow.plans.outcomes([plan.plan_id])
    truth = flow.source.get_appointments(plan.target_data_date, plan.target_data_date).set_index("appointment_id")["no_show"]
    assert len(outcomes) == len(flow.plans.items(plan.plan_id))
    assert (outcomes.set_index("appointment_id")["no_show"] == truth.loc[outcomes["appointment_id"]].to_numpy()).all()
    assert flow.closer.catch_up(datetime(2026, 10, 24, 8, 0)) == []           # nothing left to do


def test_outcome_sources() -> None:
    items = pd.DataFrame({"appointment_id": range(8), "patient_id": [1, 2, 3, 4, 5, 6, 7, 8],
                          "arm": ["CALL"] * 4 + ["HOLDOUT"] * 2 + ["SMS", "CALL"], "historical_no_show": [1, 1, 0, 1, 1, 0, 1, 1]})
    reached = {1, 2, 3, 5, 7}
    assert list(HistoricalOutcomes().observe(items, reached, 1)) == [1, 1, 0, 1, 1, 0, 1, 1]
    assert list(PlantedEffectOutcomes(0.0).observe(items, reached, 1)) == [1, 1, 0, 1, 1, 0, 1, 1]
    # Effect 1: every reached call-arm patient who missed now attends; nobody else changes.
    assert list(PlantedEffectOutcomes(1.0).observe(items, reached, 1)) == [0, 0, 0, 1, 1, 0, 1, 1]
    shuffled = items.sample(frac=1.0, random_state=2)
    half = PlantedEffectOutcomes(0.5)
    assert half.observe(items, reached, 9).sort_index().equals(half.observe(shuffled, reached, 9).sort_index())


# ------------------------------------------------------------ measurement

def test_recalibration_matches_the_observed_rate_and_keeps_the_order() -> None:
    rng = np.random.default_rng(0)
    predicted = rng.uniform(0.05, 0.6, 2000)
    observed = (rng.random(2000) < predicted * 0.8).astype(int)
    recalibrator = Recalibrator.fit(predicted, observed)
    adjusted = recalibrator.apply(predicted)
    assert recalibrator.shift < 0 and adjusted.mean() == pytest.approx(observed.mean(), abs=1e-6)
    assert (np.argsort(adjusted) == np.argsort(predicted)).all()
    assert Recalibrator.fit(np.array([]), np.array([])).shift == 0.0


def test_relative_reduction_recovers_a_known_effect() -> None:
    rng = np.random.default_rng(1)
    n, p_holdout, true_reduction = 20_000, 0.38, 0.18
    k_holdout = int((rng.random(n) < p_holdout).sum())
    k_call = int((rng.random(n) < p_holdout * (1 - true_reduction)).sum())
    estimate, (low, high) = relative_reduction(k_call, n, k_holdout, n)
    assert estimate == pytest.approx(true_reduction, abs=0.03) and low < true_reduction < high
    estimate, (low, high) = relative_reduction(int((rng.random(n) < 0.38).sum()), n, k_holdout, n)
    assert low < 0 < high
    assert relative_reduction(3, 10, 0, 10) is None
    estimate, (low, high) = relative_reduction(0, 10, 3, 10)
    assert 0.5 < estimate < 1.0 and low < estimate < high


def test_unequal_arms_sample_size() -> None:
    equal = PilotDesigner.sample_size_per_arm(0.388, 0.18)
    assert PilotDesigner.sample_sizes(0.388, 0.18, ratio=1.0) == (equal, equal)
    called, held_out = PilotDesigner.sample_sizes(0.388, 0.18, ratio=9.0)
    assert held_out < equal < called and called + held_out > 2 * equal


def test_analyzer_matches_a_manual_count_after_a_simulated_week(flow: Workflow) -> None:
    flow.demo.set_outcome_source(SIMULATED, 1.0, flow.admin)
    for _ in range(3):
        flow.closer.catch_up(flow.clock.real_now())
        plan = _approved(flow)
        flow.demo.auto_complete(plan.plan_id, flow.admin, reach_rate=1.0)
        flow.demo.advance_to_next_clinic_day(flow.admin)
    flow.demo.advance_to_next_clinic_day(flow.admin)
    flow.closer.catch_up(flow.clock.real_now())
    summary = flow.analyzer.summary()
    assert summary.closed_days == 3 and summary.source == SIMULATED
    assert summary.reach == 1.0 and summary.no_show_call == 0.0     # effect 1, everyone reached
    assert summary.no_show_holdout > 0 and summary.relative_reduction > 0.8
    assert sum(summary.outcome_mix.values()) > 0 and summary.required_holdout is not None


# ----------------------------------------------------------------- demo

def test_demo_time_moves_the_clock_and_reset_keeps_logins(flow: Workflow) -> None:
    flow.demo.set_demo_now(datetime(2026, 10, 23, 15, 0), flow.admin)    # Friday afternoon
    assert flow.clock.moment().data_date == date(2016, 5, 13)
    assert flow.demo.advance_to_next_clinic_day(flow.admin) == datetime(2026, 10, 26, 8, 0)
    plan = _approved(flow)
    assert flow.demo.auto_complete(plan.plan_id, flow.admin, reach_rate=0.6) > 0
    assert (flow.task_service.tasks_for(plan.plan_id)["status"] != "PENDING").all()
    flow.demo.reset(flow.admin)
    assert flow.plans.list_plans() == [] and len(flow.users.list_users()) == 3
    assert list(flow.events.list_events()["event_type"]) == ["DEMO_RESET"]
    assert flow.demo.demo_now() is None


def test_a_batch_with_one_invalid_task_changes_nothing(flow: Workflow) -> None:
    plan = _approved(flow)
    tasks = flow.task_service.tasks_for(plan.plan_id)
    calls = tasks[tasks["kind"] == "CALL"]["task_id"].astype(int).tolist()[:3]
    sms = int(tasks[tasks["kind"] == "SMS"].iloc[0]["task_id"])
    events_before = len(flow.events.list_events())
    with pytest.raises(ValueError, match="not a valid outcome"):
        flow.task_service.record_many({**{task: Outcome.CONFIRMED for task in calls}, sms: Outcome.CONFIRMED}, flow.admin)
    after = flow.task_service.tasks_for(plan.plan_id)
    assert (after["status"] == "PENDING").all() and len(flow.events.list_events()) == events_before
    flow.task_service.record_many({task: Outcome.CONFIRMED for task in calls}, flow.admin)
    assert len(flow.events.list_events()) == events_before + 3
