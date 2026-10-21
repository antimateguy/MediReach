"""Settings: every path, seed, bucket edge and default assumption in one place."""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field, fields
from datetime import date, datetime
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from medireach.domain import ImpactAssumptions
from medireach.errors import ConfigError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Window functions (used by the analytical views) arrived in SQLite 3.25 and
# RANGE frames with offsets in 3.28.
MIN_SQLITE_VERSION = (3, 28, 0)

# Environment variable -> (Settings field, converter)
_ENV_FIELDS: dict[str, tuple[str, type]] = {
    "MEDIREACH_RAW_CSV": ("raw_csv_path", str),
    "MEDIREACH_DB_PATH": ("db_path", str),
    "MEDIREACH_MODELS_DIR": ("models_dir", str),
    "MEDIREACH_ARTIFACTS_DIR": ("artifacts_dir", str),
    "MEDIREACH_SEED": ("seed", int),
    "MEDIREACH_CALL_SHARE": ("call_share", float),
    "MEDIREACH_SMS_SHARE": ("sms_share", float),
    "MEDIREACH_LOG_LEVEL": ("log_level", str),
    "MEDIREACH_OPS_DB_PATH": ("ops_db_path", str),
    "MEDIREACH_TIMEZONE": ("timezone", str),
    "MEDIREACH_PLAN_HOUR": ("plan_hour", int),
    "MEDIREACH_DEMO_NOW": ("demo_now", str),
    "MEDIREACH_DEMO_PASSWORD": ("demo_password", str),
    "MEDIREACH_GEMINI_MODEL": ("gemini_model", str),
    "MEDIREACH_GEMINI_FALLBACK_MODEL": ("gemini_fallback_model", str),
    "MEDIREACH_GEMINI_TIMEOUT": ("gemini_timeout", float),
}
_PATH_FIELDS = ("raw_csv_path", "db_path", "models_dir", "artifacts_dir", "ops_db_path")


def _in_root(*parts: str) -> str:
    # normpath so "data/x.db" from an env var gets the platform's separators.
    return os.path.normpath(os.path.join(PROJECT_ROOT, *parts))


@dataclass(frozen=True)
class Settings:
    """Immutable configuration. Build it with `Settings.from_env()`."""

    # Paths
    raw_csv_path: str = _in_root("data", "raw", "KaggleV2-May-2016.csv")
    db_path: str = _in_root("data", "medireach.db")
    schema_path: str = _in_root("sql", "schema.sql")
    views_path: str = _in_root("sql", "views.sql")
    models_dir: str = _in_root("models")
    artifacts_dir: str = _in_root("artifacts")
    # Workflow state (plans, tasks, logins, ingestion log). Written by the app,
    # never committed; the analytical database above stays read-only.
    ops_db_path: str = _in_root("data", "ops.db")
    ops_schema_path: str = _in_root("sql", "ops_schema.sql")

    # Reproducibility
    seed: int = 42
    log_level: str = "INFO"

    # Buckets: each edge is the inclusive upper bound of its bucket; the last
    # label covers everything above the last edge. sql/views.sql repeats these
    # edges in CASE expressions, and a test checks the two stay in step.
    lead_bucket_edges: tuple[int, ...] = (0, 3, 7, 14, 30)
    lead_bucket_labels: tuple[str, ...] = ("0", "1-3", "4-7", "8-14", "15-30", "31+")
    lead_bucket_reference: str = "1-3"
    age_group_edges: tuple[int, ...] = (17, 34, 49, 64)
    age_group_labels: tuple[str, ...] = ("0-17", "18-34", "35-49", "50-64", "65+")
    age_group_reference: str = "35-49"
    max_age: int = 110

    # Features and split
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    history_prior_strength: float = 2.0
    prior_appt_cap: int = 5

    # Model selection: the simpler model wins unless beaten by more than this.
    selection_margin: float = 0.01
    min_segment_rows: int = 200
    min_clinic_rows: int = 100

    # Outreach policies
    call_share: float = 0.10
    sms_share: float = 0.20
    threshold_high: float = 0.50
    threshold_medium: float = 0.35

    # Live replay: the test week is replayed on the real calendar, weekday by
    # weekday. replay_week_start is the first day of that week (a Thursday).
    replay_week_start: str = "2016-06-02"
    timezone: str = "Asia/Kolkata"
    plan_hour: int = 8
    # Optional fixed "now" for demos and tests, e.g. "2026-10-22T10:15" (clinic time).
    demo_now: str = ""
    # Shared password of the seeded demo accounts. Demo-grade only; see .env.example.
    demo_password: str = "medireach-demo"

    # AI text on the free Gemini tier. The key itself is never a setting: it is read from the
    # GEMINI_API_KEY environment variable or from secrets.toml in the project root (git-ignored).
    gemini_model: str = "gemini-flash-lite-latest"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    gemini_timeout: float = 60.0
    secrets_path: str = _in_root("secrets.toml")

    # Impact (illustrative defaults, editable in the app)
    impact: ImpactAssumptions = field(default_factory=ImpactAssumptions)

    def __post_init__(self) -> None:
        if len(self.lead_bucket_labels) != len(self.lead_bucket_edges) + 1:
            raise ConfigError("lead buckets need one more label than edges")
        if len(self.age_group_labels) != len(self.age_group_edges) + 1:
            raise ConfigError("age groups need one more label than edges")
        if abs(sum(self.split_fractions) - 1.0) > 1e-9:
            raise ConfigError("split_fractions must sum to 1")
        if not (0 < self.call_share and 0 <= self.sms_share and self.call_share + self.sms_share <= 1):
            raise ConfigError("call_share and sms_share must be shares that sum to at most 1")
        try:
            ZoneInfo(self.timezone)
            date.fromisoformat(self.replay_week_start)
            if self.demo_now:
                datetime.fromisoformat(self.demo_now)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ConfigError(f"invalid timezone, replay_week_start or demo_now: {exc}") from exc
        if not 0 <= self.plan_hour <= 23:
            raise ConfigError("plan_hour must be between 0 and 23")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Defaults, overridden by any MEDIREACH_* variables that are set."""
        source = os.environ if env is None else env
        overrides: dict[str, object] = {}
        for variable, (name, convert) in _ENV_FIELDS.items():
            raw = source.get(variable)
            if raw is None or raw.strip() == "":
                continue
            try:
                value: object = convert(raw.strip())
            except ValueError as exc:
                raise ConfigError(f"{variable}={raw!r} is not a valid {convert.__name__}") from exc
            if name in _PATH_FIELDS and not os.path.isabs(str(value)):
                value = _in_root(str(value))
            overrides[name] = value
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in overrides.items() if k in known})


def check_sqlite_version(minimum: tuple[int, int, int] = MIN_SQLITE_VERSION) -> str:
    """Fail early, with a clear message, if SQLite cannot run the window-function views."""
    found = sqlite3.sqlite_version_info
    if found < minimum:
        needed = ".".join(str(part) for part in minimum)
        raise ConfigError(f"SQLite {needed}+ is required, found {sqlite3.sqlite_version}")
    return sqlite3.sqlite_version
