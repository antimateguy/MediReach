"""Pipeline: write web/data/public.json, so the public story loads with no server at all.

The static site (Cloudflare Pages, or the app itself) shows these numbers at once; the server is
needed only for the live calculators and the clinic portal. Same payload functions as the API,
so the file and the API can never disagree (tests/test_web_api.py checks it).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from medireach import artifacts as files
from medireach.artifacts import ArtifactStore
from medireach.bootstrap import configure_logging
from medireach.config import PROJECT_ROOT, Settings
from medireach.data.database import SqliteDatabase
from medireach.data.repositories import SqliteAppointmentRepository
from medireach.impact.pilot import PilotDesigner
from medireach.impact.simulator import ImpactSimulator
from medireach.outreach.policies import CapacityPolicy
from medireach.web import payloads

logger = logging.getLogger(__name__)

PUBLIC_JSON = os.path.join(PROJECT_ROOT, "web", "data", "public.json")
DEFAULT_HOLDOUT_SHARE = 0.5


def public_bundle(settings: Settings) -> dict[str, Any]:
    """The story's numbers, the impact comparison and the pilot size at the default assumptions."""
    store = ArtifactStore(settings.artifacts_dir)
    summary = payloads.public_summary(store, SqliteAppointmentRepository(SqliteDatabase(settings.db_path)))
    policy = CapacityPolicy(settings.call_share, settings.sms_share)
    plan = summary["pilot"]
    return {
        "summary": summary,
        "impact": payloads.impact(ImpactSimulator(), policy, store.read_csv(files.TEST_SCORES), settings.impact),
        "pilot": payloads.pilot(PilotDesigner(), plan["p_control"], plan["relative_reduction"],
                                DEFAULT_HOLDOUT_SHARE, plan["daily_high_volume"]),
    }


def write_public_bundle(settings: Settings, path: str = PUBLIC_JSON) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(public_bundle(settings), handle, separators=(",", ":"))
        handle.write("\n")
    return path


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings)
    logger.info("Wrote %s", write_public_bundle(settings))


if __name__ == "__main__":
    main()
