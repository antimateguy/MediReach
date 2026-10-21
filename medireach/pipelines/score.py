"""Pipeline: score a date window and append a run to the queue."""
from __future__ import annotations

import argparse
from datetime import date

from medireach.bootstrap import Container, build_container, configure_logging
from medireach.config import Settings
from medireach.domain import QueueRun
from medireach.errors import ModelBundleError


def score_window(container: Container, start: date | None = None, end: date | None = None,
                 notes: str = "") -> QueueRun:
    """Score [start, end]. Defaults to the model's test period: 'tomorrow's list, replayed on history'."""
    if container.outreach is None or container.metadata is None:
        raise ModelBundleError("no trained model is loaded; run the train pipeline first")
    test_start, test_end = container.metadata.test_range
    window_start = start or date.fromisoformat(test_start)
    window_end = end or date.fromisoformat(test_end)
    return container.outreach.run(window_start, window_end, notes or "test period replay")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score appointments and append an outreach run.")
    parser.add_argument("--start", type=date.fromisoformat, default=None, help="first date, YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="last date, YYYY-MM-DD")
    parser.add_argument("--notes", default="", help="free text stored with the run")
    arguments = parser.parse_args()

    settings = Settings.from_env()
    configure_logging(settings)
    run = score_window(build_container(settings), arguments.start, arguments.end, arguments.notes)
    print(f"{run.run_id}: {run.n_rows} appointments, {run.window_start} to {run.window_end}, "
          f"model {run.model_name} ({run.model_version}), policy {run.policy_name}")


if __name__ == "__main__":
    main()
