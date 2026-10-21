"""Pipeline: run the six hypothesis checks and write them to artifacts/hypotheses.json."""
from __future__ import annotations

import logging

from medireach.artifacts import HYPOTHESES, ArtifactStore
from medireach.bootstrap import configure_logging
from medireach.config import Settings
from medireach.data.analytics import AnalyticsQueries
from medireach.data.database import SqliteDatabase
from medireach.data.hypotheses import HypothesisResult, HypothesisTests

logger = logging.getLogger(__name__)

def write_hypotheses(settings: Settings) -> list[HypothesisResult]:
    """Test H1-H6 against the built database and save the results for the docs."""
    results = HypothesisTests(AnalyticsQueries(SqliteDatabase(settings.db_path))).run_all()
    store = ArtifactStore(settings.artifacts_dir)
    store.write_json(HYPOTHESES, [result.to_dict() for result in results])
    logger.info("Wrote %d hypothesis results to %s", len(results), store.path(HYPOTHESES))
    return results


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings)
    for result in write_hypotheses(settings):
        print(f"{result.hypothesis_id} [{result.verdict}] {result.summary}")


if __name__ == "__main__":
    main()
