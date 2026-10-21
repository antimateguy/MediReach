"""Pipeline: replay the test week through the full workflow twice and save what the hold-out shows.

Once with the real 2016 outcomes (the true effect is zero, so the interval should contain zero) and
once with a planted effect (the analysis should find it). Runs on a throwaway ops.db, so the app's own
workflow state is never touched. Writes artifacts/replay_validation.json.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta

from medireach import artifacts as files
from medireach.artifacts import ArtifactStore
from medireach.bootstrap import build_container, configure_logging
from medireach.config import Settings
from medireach.domain import HoldoutMode
from medireach.ops.outcomes import HISTORICAL, SIMULATED

logger = logging.getLogger(__name__)

CALLS_PER_DAY, SMS_PER_DAY, REACH_PER_ATTEMPT, PLANTED_EFFECT = 440, 880, 0.60, 0.30
# A fixed real week (a Thursday), so the seeded randomisation and the results are the same on every run.
REPLAY_WEEK = date(2026, 10, 22)


def replay_week(settings: Settings, source: str, ops_db_path: str) -> dict[str, object]:
    """Plan, approve, simulate the receptionists and close every clinic day of one replay week."""
    container = build_container(replace(settings, ops_db_path=ops_db_path, demo_now=""))
    admin = container.auth.login("admin", settings.demo_password)
    if admin is None or container.planner is None:
        raise RuntimeError("the replay needs the demo admin account and a trained model")
    container.demo.set_outcome_source(source, PLANTED_EFFECT, admin)
    # Start on the Wednesday before the replay week begins, at plan time.
    container.demo.set_demo_now(datetime.combine(REPLAY_WEEK - timedelta(days=1), time(settings.plan_hour)), admin)
    for _ in range(sum(container.calendar.is_open(REPLAY_WEEK + timedelta(days=d)) for d in range(7))):
        container.cycle.refresh()
        plan = container.planner.propose(container.planner.next_target(), CALLS_PER_DAY, SMS_PER_DAY, admin)
        container.planner.approve(plan.plan_id, HoldoutMode.PILOT, admin)
        container.demo.auto_complete(plan.plan_id, admin, REACH_PER_ATTEMPT)
        container.demo.advance_to_next_clinic_day(admin)
    container.demo.advance_to_next_clinic_day(admin)
    container.cycle.refresh()
    summary = asdict(container.analyzer.summary())
    summary["comparison"] = None if summary["comparison"] is None else dict(container.analyzer.summary().comparison._asdict())
    return summary


def write_replay_validation(settings: Settings) -> dict[str, object]:
    results: dict[str, object] = {
        "settings": {"replay_week": REPLAY_WEEK.isoformat(), "calls_per_day": CALLS_PER_DAY,
                     "sms_per_day": SMS_PER_DAY, "holdout": "PILOT (50%)",
                     "reach_per_attempt": REACH_PER_ATTEMPT, "planted_effect": PLANTED_EFFECT},
    }
    with tempfile.TemporaryDirectory(prefix="medireach_replay_") as folder:
        for source in (HISTORICAL, SIMULATED):
            results[source] = replay_week(settings, source, os.path.join(folder, f"{source}.db"))
            logger.info("Replay with %s outcomes done", source)
    ArtifactStore(settings.artifacts_dir).write_json(files.REPLAY_VALIDATION, results)
    return results


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings)
    results = write_replay_validation(settings)
    for source in (HISTORICAL, SIMULATED):
        part = results[source]
        print(f"{source}: call {part['no_show_call']:.4f} (n {part['patients_call']}), hold-out "
              f"{part['no_show_holdout']:.4f} (n {part['patients_holdout']}), reduction {part['relative_reduction']:.4f} "
              f"CI {part['reduction_ci']}")


if __name__ == "__main__":
    main()
