"""Payloads: what each screen of the web app shows, built from the container. No HTTP here.

Every function returns plain JSON-ready data (numbers stay numbers; dates come with a display label),
so the browser only formats and lays out. Business rules stay in the engine.
"""
from __future__ import annotations

import math
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from medireach import artifacts as files
from medireach.assist.briefing import Briefing, briefing_facts
from medireach.artifacts import ArtifactStore
from medireach.bootstrap import Container
from medireach.data.repositories import AppointmentRepository
from medireach.domain import (
    HOLDOUT_SHARES, Action, HoldoutMode, ImpactAssumptions, Outcome, Plan, PlanStatus, Role, TaskKind, TaskStatus, User,
)
from medireach.impact.pilot import PilotDesigner
from medireach.impact.simulator import ImpactSimulator
from medireach.live.status import LiveSnapshot
from medireach.ops.outcomes import SIMULATED
from medireach.ops.plans import OVERRIDE_REASONS, holdout_cost, plan_numbers
from medireach.ops.tasks import SKIP_REASONS
from medireach.outreach.policies import CapacityPolicy
from medireach.pipelines.impact import STRATEGIES

REPLAY_LABEL = "Replay of 2016 data"
PLAN_PAGE_ROWS = 50
TEAM_PAGE_ROWS = 40
DIRECTORY_ROWS = 300
BACKTEST_ROWS = 12
UP_NEXT_ROWS = 6
CALL_OUTCOME_LABELS: dict[str, str] = {
    Outcome.CONFIRMED.value: "Will attend", Outcome.RESCHEDULE.value: "Wants another date",
    Outcome.CANCEL.value: "Cancels", Outcome.NO_ANSWER.value: "No answer", Outcome.WRONG_NUMBER.value: "Wrong number",
}
OUTCOME_LABELS: dict[str, str] = {**CALL_OUTCOME_LABELS, Outcome.SMS_SENT.value: "SMS sent"}
MODEL_LABELS: dict[str, str] = {
    "lead_time_baseline": "Lead-time baseline", "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}


# ---------------------------------------------------------------- formatting

def clean(value: Any) -> Any:
    """Recursively turn numpy/pandas values into JSON-safe Python values (NaN becomes None)."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def day_label(day: date) -> str:
    return f"{day:%a %d %b %Y}"


def _local_time(stamp: str | None, zone: ZoneInfo) -> str | None:
    """HH:MM in the clinic's time zone for a stored UTC timestamp."""
    if not stamp:
        return None
    moment = datetime.fromisoformat(stamp)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return f"{moment.astimezone(zone):%H:%M}"


def user_payload(user: User) -> dict[str, Any]:
    initials = "".join(part[0] for part in user.display_name.split()[:2]).upper() or user.username[:2].upper()
    return {"username": user.username, "name": user.display_name, "role": user.role.value,
            "role_label": "Administrator" if user.role == Role.ADMIN else "Reception", "initials": initials}


# -------------------------------------------------------------------- status

def status(container: Container, user: User | None, default_password_in_use: bool) -> dict[str, Any]:
    """Shown on every screen: who is signed in, the replay clock and the labels."""
    moment = container.clock.moment()
    source, effect = container.demo.outcome_settings()
    return clean({
        "user": user_payload(user) if user else None,
        "replay_label": REPLAY_LABEL,
        "simulated": source == SIMULATED,
        "planted_effect": effect,
        "now_label": f"{moment.real_now:%a %d %b %Y, %H:%M}",
        "replaying_label": f"{moment.data_now:%a %d %b %Y}",
        "demo_time": container.demo.demo_now() is not None,
        "timezone": container.settings.timezone,
        "ai_available": container.briefing.ai_available,
        "demo_accounts": [{"username": u.username, **user_payload(u)} for u in container.users.list_users()],
        "demo_password_hint": container.settings.demo_password if default_password_in_use else None,
    })


# ------------------------------------------------------------- public story

def _hypothesis_bars(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The groups behind each hypothesis, as label / rate / n (from its stored table)."""
    table = pd.DataFrame(result["table"])
    hypothesis = result["hypothesis_id"]
    if hypothesis == "H3":
        details = result["details"]
        return [{"label": "Got an SMS", "rate": details["naive_rate_1"]},
                {"label": "No SMS", "rate": details["naive_rate_2"]}]
    if hypothesis == "H6":
        ordered = table.sort_values("no_show_rate")
        picks = pd.concat([ordered.tail(2).iloc[::-1], ordered.head(2)])
        return [{"label": str(row["clinic"]).title(), "rate": row["no_show_rate"], "n": row["n"]}
                for _, row in picks.iterrows()]
    label = {"H1": "lead_bucket", "H2": "prior_no_show_bucket", "H4": "age_group", "H5": "weekday"}[hypothesis]
    prefix = {"H1": "Booked ", "H2": "Earlier no-shows: ", "H4": "Age ", "H5": ""}[hypothesis]
    suffix = {"H1": " days ahead", "H2": "", "H4": "", "H5": ""}[hypothesis]
    rows = table.sort_values("sort_key")
    if hypothesis == "H5":
        rows = rows[~rows["low_n"].astype(bool)]
    bars = []
    for _, row in rows.iterrows():
        text = str(row[label])
        if hypothesis == "H1" and text == "0":
            text = "Same day"
        else:
            text = f"{prefix}{text}{suffix}"
        bars.append({"label": text, "rate": row["no_show_rate"], "n": row["n"]})
    return bars


def public_summary(store: ArtifactStore, appointments: AppointmentRepository) -> dict[str, Any]:
    """Everything the public story needs. Read from the artifacts the pipelines wrote."""
    report = store.read_json(files.CLEANING_REPORT)
    metrics = store.read_json(files.METRICS)
    models = store.read_csv(files.MODEL_COMPARISON)
    ablation = store.read_csv(files.FEATURE_ABLATION)
    impact = store.read_json(files.IMPACT_SUMMARY)
    comparison = store.read_csv(files.IMPACT_COMPARISON).set_index("strategy")
    replay = store.read_json(files.REPLAY_VALIDATION)
    selected = metrics["selected_model"]
    test = models[(models["model"] == selected) & (models["period"] == "test")].iloc[0]
    first, last = appointments.date_range()

    added = ablation[ablation["study"] == "added one at a time"]
    left_out = ablation[ablation["study"] == "left out one at a time"]
    grid = ablation[ablation["study"] == "model x information"]
    hypotheses = [{
        "id": result["hypothesis_id"], "statement": result["statement"], "verdict": result["verdict"],
        "summary": result["summary"], "bars": _hypothesis_bars(result),
    } for result in store.read_json(files.HYPOTHESES)]

    return clean({
        "data": {
            "rows_in": report["rows_in"], "rows_out": report["rows_out"], "dropped": report["dropped"],
            "patients": report["patients"], "clinics": report["clinics"],
            "no_show_rate": report["base_rate_after"], "same_day_rows": report["same_day_rows"],
            "same_day_share": report["same_day_share"], "same_day_no_show_rate": report["same_day_no_show_rate"],
            "v1_timestamp_filter_drops": report["v1_timestamp_filter_drops"],
            "first_date": f"{first:%d %b %Y}", "last_date": f"{last:%d %b %Y}",
            "appointment_days": len(appointments.appointment_dates()),
        },
        "split": {name: {"start": f"{pd.Timestamp(part['start']):%d %b}", "end": f"{pd.Timestamp(part['end']):%d %b}",
                         "n": part["n"], "base_rate": part["base_rate"]}
                  for name, part in metrics["split"].items()},
        "selected_model": selected,
        "selection_reason": metrics["selection_reason"],
        "model_version": metrics["model_version"],
        "headline": {"roc_auc": test["roc_auc"], "precision_at_10": test["precision_at_10"],
                     "capture_at_30": test["capture_at_30"], "lift_top_decile": test["lift_top_decile"],
                     "base_rate": test["base_rate"]},
        "models": [{"model": row["model"], "label": MODEL_LABELS.get(row["model"], row["model"]),
                    "period": row["period"], "selected": bool(row["selected"]),
                    **{key: row[key] for key in ("n", "base_rate", "roc_auc", "pr_auc", "precision_at_10",
                                                 "capture_at_10", "capture_at_30", "lift_top_decile")}}
                   for _, row in models.iterrows()],
        "features": {
            "added": [{"label": row["label"], "roc_auc": row["roc_auc"], "gain": row["gain_roc_auc"]}
                      for _, row in added.iterrows()],
            "left_out": [{"label": str(row["label"]).replace("without ", ""), "loss": -row["gain_roc_auc"],
                          "roc_auc": row["roc_auc"]} for _, row in left_out.iterrows()],
            "grid": [{"label": row["label"], "model": MODEL_LABELS.get(row["model"], row["model"]),
                      "roc_auc": row["roc_auc"]} for _, row in grid.iterrows()],
        },
        "hypotheses": hypotheses,
        "impact": {
            "assumptions": impact["assumptions"], "break_even": impact["break_even_call_effectiveness"],
            "selected_net": comparison.loc[MODEL_LABELS[selected], "net"],
            "random_net": comparison.loc["Random (no targeting)", "net"],
            "selected_prevented": comparison.loc[MODEL_LABELS[selected], "prevented"],
            "selected_cost_per_prevented": comparison.loc[MODEL_LABELS[selected], "cost_per_prevented_no_show"],
            "n_calls": comparison.loc[MODEL_LABELS[selected], "n_calls"],
            "n_sms": comparison.loc[MODEL_LABELS[selected], "n_sms"],
        },
        "pilot": impact["pilot_plan"],
        "replay": {source: {key: replay[source][key] for key in ("relative_reduction", "reduction_ci", "patients_call",
                                                                 "patients_holdout", "closed_days")}
                   for source in ("historical", "simulated")} | {"settings": replay["settings"]},
    })


def impact(simulator: ImpactSimulator, policy: CapacityPolicy, scores: pd.DataFrame,
           assumptions: ImpactAssumptions) -> dict[str, Any]:
    """The strategy comparison on the test week under the given assumptions (same budget for all)."""
    table = simulator.compare(scores, STRATEGIES, policy, assumptions)
    rows = [{
        "strategy": row["strategy"], "prevented": row["prevented"], "net": row["net"],
        "cost_per_prevented": row["cost_per_prevented_no_show"] if row["prevented"] > 0 else None,
        "n_calls": row["n_calls"], "n_sms": row["n_sms"], "capture_rate": row["capture_rate"],
    } for _, row in table.iterrows()]
    return clean({"strategies": rows, "assumptions": asdict(assumptions)})


def pilot(designer: PilotDesigner, p_control: float, reduction: float, holdout_share: float,
          daily_volume: float) -> dict[str, Any]:
    """Patients needed to detect `reduction` with a hold-out of `holdout_share` (80% power, 5% two-sided)."""
    called, held = designer.sample_sizes(p_control, reduction, ratio=(1 - holdout_share) / holdout_share)
    return clean({"total": called + held, "called": called, "held_out": held,
                  "days": math.ceil((called + held) / daily_volume), "daily_volume": daily_volume,
                  "p_control": p_control})


# --------------------------------------------------------------------- admin

def _plan_summary(container: Container, plan: Plan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    numbers = plan_numbers(container.plans.items(plan.plan_id))
    return {"plan_id": plan.plan_id, "status": plan.status.value, "target_label": day_label(plan.target_real_date),
            "calls": numbers["calls"], "sms": numbers["sms"], "expected_no_shows": numbers["expected_no_shows"],
            "holdout_mode": plan.holdout_mode.value if plan.holdout_mode else None,
            "holdout_share": plan.holdout_share, "approved_by": plan.approved_by,
            "approved_at": _local_time(plan.approved_at, ZoneInfo(container.settings.timezone))}


def _progress(tasks: pd.DataFrame | None) -> dict[str, Any]:
    if tasks is None or tasks.empty:
        return {"done": 0, "total": 0, "rows": []}
    finished = tasks["status"] != TaskStatus.PENDING.value
    rows = [{"label": "Calls", "done": int(finished[tasks["kind"] == TaskKind.CALL.value].sum()),
             "total": int((tasks["kind"] == TaskKind.CALL.value).sum())},
            {"label": "SMS", "done": int(finished[tasks["kind"] == TaskKind.SMS.value].sum()),
             "total": int((tasks["kind"] == TaskKind.SMS.value).sum())}]
    for owner, group in tasks.groupby("assigned_to"):
        rows.append({"label": str(owner), "done": int((group["status"] != TaskStatus.PENDING.value).sum()),
                     "total": len(group)})
    return {"done": int(finished.sum()), "total": len(tasks), "rows": rows}


def _briefing_payload(briefing: Briefing) -> dict[str, Any]:
    return {"headline": briefing.headline, "bullets": briefing.bullets, "source": briefing.source,
            "model": briefing.model, "note": briefing.note}


def today_facts(container: Container, snapshot: LiveSnapshot) -> dict[str, str]:
    """The numbers behind the daily briefing (also what the AI is allowed to use)."""
    next_plan = container.planner.current_plan(snapshot.next_clinic_day) if container.planner else None
    working = container.task_service.plan_for_day(snapshot.moment.real_date)
    tasks = container.task_service.tasks_for(working.plan_id) if working else None
    return briefing_facts(
        snapshot, next_plan, plan_numbers(container.plans.items(next_plan.plan_id)) if next_plan else None,
        int((tasks["status"] != TaskStatus.PENDING.value).sum()) if tasks is not None else 0,
        len(tasks) if tasks is not None else 0, container.analyzer.summary(),
    )


def today(container: Container, snapshot: LiveSnapshot, notes: list[str]) -> dict[str, Any]:
    moment = snapshot.moment
    next_plan = container.planner.current_plan(snapshot.next_clinic_day) if container.planner else None
    working = container.task_service.plan_for_day(moment.real_date)
    tasks = container.task_service.tasks_for(working.plan_id) if working else None
    facts = today_facts(container, snapshot)
    briefing = container.briefing.cached(facts) or container.briefing.template(facts)
    batch = snapshot.last_batch
    return clean({
        "notes": notes,
        "now_label": f"{moment.real_now:%A %d %B %Y}", "time_label": f"{moment.real_now:%H:%M}",
        "replaying_label": f"{moment.data_now:%A %d %B %Y}",
        "is_open": snapshot.is_open, "next_clinic_day_label": day_label(snapshot.next_clinic_day),
        "appointments_today": snapshot.appointments_today,
        "same_day_bookings_so_far": snapshot.same_day_bookings_so_far,
        "appointments_next_day_known": snapshot.appointments_next_day_known,
        "plan_due": snapshot.plan_due, "plan_hour": container.settings.plan_hour,
        "feed": {"batches": snapshot.batches_this_cycle, "last_rows": batch.rows_in if batch else None,
                 "last_failed_checks": batch.checks_failed if batch else None},
        "next_plan": _plan_summary(container, next_plan),
        "working_plan": _plan_summary(container, working),
        "progress": _progress(tasks),
        "briefing": _briefing_payload(briefing), "facts": facts, "ai_available": container.briefing.ai_available,
    })


def plan_page(container: Container, offset: int = 0) -> dict[str, Any]:
    planner = container.planner
    target = planner.next_target()
    plan = planner.current_plan(target)
    base = {"target_label": day_label(target), "replays_label": day_label(container.clock.data_date_for(target)),
            "override_reasons": list(OVERRIDE_REASONS),
            "holdout_modes": [{"mode": mode.value, "share": HOLDOUT_SHARES[mode]} for mode in HoldoutMode]}
    if plan is None:
        allowed, why = planner.can_propose(target)
        return clean({**base, "plan": None, "can_propose": allowed, "why_not": why,
                      "defaults": {"receptionists": 4, "calls_per_receptionist": 110, "sms_budget": 880}})
    items = container.plans.items(plan.plan_id)
    numbers = plan_numbers(items)
    patients = (items.sort_values(["day_rank", "appointment_id"])
                .groupby("patient_id", sort=False)
                .agg(rank=("day_rank", "min"), risk=("risk_calibrated", "max"),
                     recommended=("recommended_action", "first"), action=("action", "first"), arm=("arm", "first"),
                     reasons=("reasons", "first"), clinic=("clinic", "first"), age=("age", "first"),
                     lead_days=("lead_days", "max"), appointments=("appointment_id", "size"),
                     override=("override_reason", "first"), late=("late_booking", "max"))
                .reset_index().sort_values("rank"))
    page = patients.iloc[offset: offset + PLAN_PAGE_ROWS]
    owners: dict[int, str] = {}
    if plan.status != PlanStatus.PROPOSED:
        tasks = container.task_service.tasks_for(plan.plan_id)
        first = tasks.sort_values(["attempt", "task_id"]).drop_duplicates("patient_id")
        owners = dict(zip(first["patient_id"].astype(int), first["assigned_to"]))
    names = {user.username: user.display_name for user in container.users.list_users()}
    rows = [{"patient_id": int(row["patient_id"]), "rank": int(row["rank"]), "risk": row["risk"],
             "recommended": row["recommended"], "action": row["action"], "arm": row["arm"],
             "reasons": str(row["reasons"]).split("; "), "clinic": str(row["clinic"]).title(), "age": row["age"],
             "lead_days": row["lead_days"], "appointments": row["appointments"], "override": row["override"],
             "late": bool(row["late"]), "assignee": names.get(owners.get(int(row["patient_id"]), ""), None)}
            for _, row in page.iterrows()]
    arms = patients["arm"].value_counts().to_dict() if plan.status != PlanStatus.PROPOSED else {}
    late = planner.late_bookings(plan) if plan.status == PlanStatus.PROPOSED else pd.DataFrame()
    return clean({
        **base, "plan": {**_plan_summary(container, plan), "as_of_label": f"{plan.as_of:%a %d %b %H:%M}",
                         "model_version": plan.model_version, "created_by": plan.created_by},
        "numbers": numbers, "rows": rows, "offset": offset, "page_size": PLAN_PAGE_ROWS, "patients": len(patients),
        "arms": arms,
        "holdout_cost": {mode.value: holdout_cost(items, HOLDOUT_SHARES[mode], container.settings.impact)
                         for mode in HoldoutMode},
        "late_bookings": [{"appointment_id": int(row["appointment_id"]), "patient_id": int(row["patient_id"]),
                           "clinic": str(row["clinic"]).title(), "age": row["age"], "lead_days": row["lead_days"],
                           "booked_label": f"{pd.Timestamp(row['scheduled_ts']):%a %H:%M}"}
                          for _, row in late.head(50).iterrows()],
        "late_count": len(late),
    })


def team(container: Container, snapshot: LiveSnapshot) -> dict[str, Any]:
    plan = container.task_service.plan_for_day(snapshot.moment.real_date)
    next_plan = container.planner.current_plan(snapshot.next_clinic_day) if container.planner else None
    receptionists = sorted((user for user in container.users.list_users() if user.role == Role.RECEPTIONIST),
                           key=lambda user: user.username)
    if plan is None:
        return clean({"plan": None, "next_plan": _plan_summary(container, next_plan),
                      "team": [{**user_payload(user), "done": 0, "total": 0, "tasks": []} for user in receptionists]})
    board = container.task_service.worklist(plan.plan_id)
    people = []
    for user in receptionists:
        mine = board[board["assigned_to"] == user.username]
        listed = pd.concat([mine[mine["status"] == TaskStatus.PENDING.value],
                            mine[mine["status"] != TaskStatus.PENDING.value]]).head(TEAM_PAGE_ROWS)
        people.append({
            **user_payload(user), "done": int((mine["status"] != TaskStatus.PENDING.value).sum()), "total": len(mine),
            "open": int((mine["status"] == TaskStatus.PENDING.value).sum()),
            "tasks": [{"task_id": int(row["task_id"]), "rank": int(row["priority_rank"]),
                       "patient_id": int(row["patient_id"]), "kind": row["kind"], "attempt": int(row["attempt"]),
                       "status": row["status"], "outcome": row["outcome"],
                       "outcome_label": OUTCOME_LABELS.get(row["outcome"]) if row["outcome"] else None,
                       "notes": row["notes"]}
                      for _, row in listed.iterrows()],
        })
    return clean({"plan": _plan_summary(container, plan), "next_plan": _plan_summary(container, next_plan),
                  "team": people, "page_size": TEAM_PAGE_ROWS})


def results(container: Container) -> dict[str, Any]:
    summary = container.analyzer.summary()
    source, effect = container.demo.outcome_settings()
    comparison = summary.comparison
    return clean({
        "source": summary.source, "setting": source, "planted_effect": effect, "closed_days": summary.closed_days,
        "patients_call": summary.patients_call, "patients_holdout": summary.patients_holdout,
        "no_show_call": summary.no_show_call, "no_show_holdout": summary.no_show_holdout,
        "difference": comparison.difference if comparison else None,
        "difference_ci": [comparison.ci_low, comparison.ci_high] if comparison else None,
        "p_value": comparison.p_value if comparison else None,
        "relative_reduction": summary.relative_reduction, "reduction_ci": summary.reduction_ci,
        "required_call": summary.required_call, "required_holdout": summary.required_holdout,
        "reach": summary.reach, "patients_reached": summary.patients_reached,
        "patients_attempted": summary.patients_attempted, "released_slots": summary.released_slots,
        "outcome_mix": {OUTCOME_LABELS.get(key, key): value for key, value in summary.outcome_mix.items()},
        "overrides": summary.overrides,
        "assumed_reach": container.settings.impact.call_reach_rate,
    })


def directory(container: Container, snapshot: LiveSnapshot, query: str, day: str) -> dict[str, Any]:
    """Appointments as the clinic knows them now (the as-of view): today and later, read-only."""
    moment = snapshot.moment
    rows = container.live_appointments.get_appointments(start=moment.data_date)
    next_data = container.clock.data_date_for(snapshot.next_clinic_day)
    real_day = rows["appointment_date"].map(lambda value: container.clock.real_date_for(value.date(), moment.cycle_start))
    rows = rows.assign(real_day=real_day)
    if day == "today":
        rows = rows[rows["appointment_date"].dt.date == moment.data_date]
    elif day == "next":
        rows = rows[rows["appointment_date"].dt.date == next_data]
    text = query.strip().lower()
    if text:
        haystack = (rows["patient_id"].astype(str) + " " + rows["appointment_id"].astype(str) + " "
                    + rows["clinic"].astype(str).str.lower())
        rows = rows[haystack.str.contains(text, regex=False)]
    rows = rows.sort_values(["appointment_date", "clinic", "appointment_id"])
    listed = [{"appointment_id": int(row["appointment_id"]), "patient_id": int(row["patient_id"]),
               "day_label": f"{row['real_day']:%a %d %b}", "clinic": str(row["clinic"]).title(), "age": row["age"],
               "lead_days": row["lead_days"], "sms": bool(row["sms_received"]),
               "status": "Not yet known" if pd.isna(row["no_show"]) else ("No-show" if row["no_show"] == 1 else "Attended")}
              for _, row in rows.head(DIRECTORY_ROWS).iterrows()]
    return clean({"rows": listed, "count": len(rows), "shown": len(listed),
                  "tabs": [{"key": "all", "label": "From today"},
                           {"key": "today", "label": f"Today, {moment.real_now:%a %d %b}"},
                           {"key": "next", "label": f"Next clinic day, {snapshot.next_clinic_day:%a %d %b}"}]})


def demo(container: Container, snapshot: LiveSnapshot) -> dict[str, Any]:
    source, effect = container.demo.outcome_settings()
    working = container.task_service.plan_for_day(snapshot.moment.real_date)
    tasks = container.task_service.tasks_for(working.plan_id) if working else None
    open_tasks = int((tasks["status"] == TaskStatus.PENDING.value).sum()) if tasks is not None else 0
    return clean({
        "time_label": f"{snapshot.moment.real_now:%H:%M}", "date_label": f"{snapshot.moment.real_now:%a %d %b %Y}",
        "replaying_label": f"{snapshot.moment.data_now:%a %d %b %Y}",
        "demo_time": container.demo.demo_now() is not None,
        "source": source, "planted_effect": effect,
        "working_plan": _plan_summary(container, working), "open_tasks": open_tasks,
        "total_tasks": len(tasks) if tasks is not None else 0,
        "assumed_reach": container.settings.impact.call_reach_rate,
    })


def backtest(container: Container, queue: pd.DataFrame, run: dict[str, Any], day: str | None) -> dict[str, Any]:
    """The stored historical queue for one test day, with what really happened (possible only in the past)."""
    days = sorted(queue["appointment_date"].dt.date.unique())
    chosen = date.fromisoformat(day) if day else days[0]
    if chosen not in days:
        raise ValueError(f"{chosen} is not a day of the stored queue")
    scored = queue[queue["appointment_date"].dt.date == chosen].drop(columns=["day_rank", "priority", "action"])
    assigned = CapacityPolicy(container.settings.call_share, container.settings.sms_share).assign(scored)
    assigned = assigned.sort_values("day_rank")
    called = assigned["action"] == Action.CALL.value
    texted = assigned["action"] == Action.SMS.value
    top = assigned[called].head(BACKTEST_ROWS)
    return clean({
        "days": [{"key": value.isoformat(), "label": f"{value:%a %d}"} for value in days],
        "day": chosen.isoformat(), "day_label": day_label(chosen),
        "run": {"run_id": run["run_id"], "model_version": run["model_version"]},
        "visits": len(assigned), "calls": int(called.sum()), "sms": int(texted.sum()),
        "call_rate": float(assigned.loc[called, "no_show"].mean()) if called.any() else None,
        "day_rate": float(assigned["no_show"].mean()),
        "rows": [{"rank": int(row["day_rank"]), "risk": row["risk_score"], "patient_id": int(row["patient_id"]),
                  "clinic": str(row["clinic"]).title(), "lead_days": row["lead_days"],
                  "reasons": str(row["reasons"]).split("; "),
                  "outcome": "No-show" if row["no_show"] == 1 else "Attended"}
                 for _, row in top.iterrows()],
    })


# ----------------------------------------------------------------- reception

def my_tasks(container: Container, snapshot: LiveSnapshot, user: User) -> dict[str, Any]:
    plan = container.task_service.plan_for_day(snapshot.moment.real_date)
    if plan is None:
        return clean({"plan": None, "date_label": f"{snapshot.moment.real_now:%a %d %b %Y}"})
    board = container.task_service.worklist(plan.plan_id, user.username)
    pending = board[board["status"] == TaskStatus.PENDING.value]
    done = board[board["status"] != TaskStatus.PENDING.value]
    payload: dict[str, Any] = {
        "plan": _plan_summary(container, plan), "date_label": f"{snapshot.moment.real_now:%a %d %b %Y}",
        "total": len(board), "done": len(done), "open": len(pending), "skip_reasons": list(SKIP_REASONS),
        "ai_available": container.script_writer.ai_available, "next": None, "up_next": [],
    }
    if not pending.empty:
        task = pending.iloc[0]
        kind = TaskKind(task["kind"])
        appointments = int(task["appointments"])
        script = (container.scripts.call_script(str(task["reasons"]), plan.target_real_date, str(task["clinic"]), appointments)
                  if kind == TaskKind.CALL else
                  container.scripts.sms_text(plan.target_real_date, str(task["clinic"]), appointments))
        outcomes = (list(CALL_OUTCOME_LABELS.items()) if kind == TaskKind.CALL
                    else [(Outcome.SMS_SENT.value, OUTCOME_LABELS[Outcome.SMS_SENT.value])])
        payload["next"] = {
            "task_id": int(task["task_id"]), "kind": kind.value, "attempt": int(task["attempt"]),
            "position": len(done) + 1, "patient_id": int(task["patient_id"]), "age": task["age"],
            "clinic": str(task["clinic"]).title(), "appointments": appointments, "risk": task["risk"],
            "lead_days": task["lead_days"], "reasons": str(task["reasons"]).split("; "), "script": script,
            "visit_label": day_label(plan.target_real_date),
            "outcomes": [{"value": value, "label": label} for value, label in outcomes],
        }
        payload["up_next"] = [{"task_id": int(row["task_id"]), "rank": int(row["priority_rank"]),
                               "patient_id": int(row["patient_id"]), "kind": row["kind"], "attempt": int(row["attempt"]),
                               "clinic": str(row["clinic"]).title()}
                              for _, row in pending.iloc[1: 1 + UP_NEXT_ROWS].iterrows()]
    return clean(payload)


def my_day(container: Container, snapshot: LiveSnapshot, user: User) -> dict[str, Any]:
    plan = container.task_service.plan_for_day(snapshot.moment.real_date)
    if plan is None:
        return clean({"plan": None, "date_label": f"{snapshot.moment.real_now:%a %d %b %Y}"})
    board = container.task_service.worklist(plan.plan_id, user.username)
    done = board[board["status"].isin([TaskStatus.DONE.value, TaskStatus.SKIPPED.value])]
    zone = ZoneInfo(container.settings.timezone)
    log = done.sort_values("updated_at", ascending=False)
    return clean({
        "plan": _plan_summary(container, plan), "date_label": f"{snapshot.moment.real_now:%a %d %b %Y}",
        "total": len(board), "done": int((board["status"] != TaskStatus.PENDING.value).sum()),
        "calls": int(((board["kind"] == TaskKind.CALL.value) & (board["status"] == TaskStatus.DONE.value)).sum()),
        "confirmed": int((board["outcome"] == Outcome.CONFIRMED.value).sum()),
        "sms": int((board["outcome"] == Outcome.SMS_SENT.value).sum()),
        "skipped": int((board["status"] == TaskStatus.SKIPPED.value).sum()),
        "log": [{"time": _local_time(row["updated_at"], zone), "patient_id": int(row["patient_id"]),
                 "kind": row["kind"], "rank": int(row["priority_rank"]), "status": row["status"],
                 "label": OUTCOME_LABELS.get(row["outcome"], "Skipped") if row["outcome"] else "Skipped",
                 "notes": row["notes"]}
                for _, row in log.head(100).iterrows()],
    })


def impact_assumptions(defaults: ImpactAssumptions, values: dict[str, float]) -> ImpactAssumptions:
    """Defaults with the given fields replaced; rates in 0-1, money in INR. Raises on nonsense."""
    known = {key: float(value) for key, value in values.items() if key in asdict(defaults)}
    for key, value in known.items():
        if value < 0 or (not key.startswith(("cost", "value")) and value > 1):
            raise ValueError(f"{key} is out of range")
    return replace(defaults, **known)
