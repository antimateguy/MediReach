"""Pipeline: build_db, insights, train, score, impact, the replay validation and the web export, in order."""
from __future__ import annotations

import logging

from medireach.bootstrap import build_container, configure_logging
from medireach.config import Settings
from medireach.pipelines.build_db import build_database
from medireach.pipelines.export_web import write_public_bundle
from medireach.pipelines.impact import write_impact
from medireach.pipelines.insights import write_hypotheses
from medireach.pipelines.replay_validation import write_replay_validation
from medireach.pipelines.score import score_window
from medireach.pipelines.train import print_summary, train

logger = logging.getLogger(__name__)


def run_all(settings: Settings) -> None:
    """Everything from the raw CSV to a stored outreach run. Safe to repeat."""
    report = build_database(settings)
    logger.info("Step 1/7 database: %d rows kept of %d", report.rows_out, report.rows_in)

    results = write_hypotheses(settings)
    logger.info("Step 2/7 insights: %s", ", ".join(f"{r.hypothesis_id} {r.verdict}" for r in results))

    # The model does not exist yet, so the first container is built without one.
    untrained = build_container(settings, require_model=False)
    metadata = train(settings, untrained.appointments, untrained.model_store, untrained.evaluator)
    logger.info("Step 3/7 training: selected %s (version %s)", metadata.model_name, metadata.version)

    trained = build_container(settings)
    run = score_window(trained)
    logger.info("Step 4/7 scoring: %s with %d appointments", run.run_id, run.n_rows)

    comparison = write_impact(trained)
    logger.info("Step 5/7 impact: %d strategies compared on the test period", len(comparison))

    # The full daily workflow on a throwaway ops.db: real outcomes, then a planted effect.
    write_replay_validation(settings)
    logger.info("Step 6/7 replay validation: written")

    # The public story's numbers for the static site (no server needed to show them).
    logger.info("Step 7/7 web export: %s", write_public_bundle(settings))


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings)
    run_all(settings)
    print_summary(settings.artifacts_dir)


if __name__ == "__main__":
    main()
