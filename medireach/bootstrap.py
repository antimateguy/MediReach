"""Composition root: the only module that chooses concrete classes."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from medireach.artifacts import ArtifactStore
from medireach.assist.briefing import BriefingWriter
from medireach.assist.llm import GeminiGenerator, TextCache, TextGenerator, read_api_key
from medireach.assist.scripts import ScriptWriter, TemplateScripts
from medireach.config import Settings
from medireach.data.analytics import AnalyticsQueries
from medireach.data.database import SqliteDatabase, read_sql_file
from medireach.data.hypotheses import HypothesisTests
from medireach.data.repositories import (
    AppointmentRepository,
    AsOfAppointmentRepository,
    QueueRepository,
    SqliteAppointmentRepository,
    SqliteQueueRepository,
)
from medireach.domain import ModelMetadata
from medireach.features.builder import FeatureBuilder
from medireach.impact.pilot import PilotDesigner
from medireach.impact.simulator import ImpactSimulator
from medireach.live.clock import ReplayCalendar, ReplayClock
from medireach.live.feed import BookingFeed, IngestionChecks, IngestionLog, SqliteIngestionLog
from medireach.live.status import LiveMonitor
from medireach.ml.evaluation import ModelEvaluator
from medireach.ml.models import ExplainableRiskModel, RiskModel
from medireach.ml.store import ModelStore
from medireach.ops.auth import AuthService, PasswordHasher
from medireach.ops.demo import DemoControls, OperationsCycle, demo_now_provider
from medireach.ops.measurement import ExperimentAnalyzer
from medireach.ops.outcomes import DayCloser
from medireach.ops.plans import DailyPlanner
from medireach.ops.tasks import TaskService
from medireach.ops.workflow_store import (
    PlanRepository,
    SqlitePlanRepository,
    SqliteTaskRepository,
    StateStore,
    TaskRepository,
)
from medireach.ops.repositories import EventLog, SqliteEventLog, SqliteUserRepository, UserRepository
from medireach.outreach.policies import CapacityPolicy
from medireach.outreach.reasons import ReasonExplainer
from medireach.outreach.service import OutreachService

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(settings: Settings) -> None:
    """Set up logging once for the whole process. Later calls change nothing."""
    logging.basicConfig(level=settings.log_level.upper(), format=_LOG_FORMAT)


@dataclass(frozen=True)
class Container:
    """Every wired-up object the pipelines and the app need.

    The model-dependent parts are None until a model has been trained.
    """

    settings: Settings
    appointments: AppointmentRepository
    queue: QueueRepository
    analytics: AnalyticsQueries
    hypotheses: HypothesisTests
    model_store: ModelStore
    evaluator: ModelEvaluator
    artifacts: ArtifactStore
    simulator: ImpactSimulator
    pilot: PilotDesigner
    state: StateStore
    clock: ReplayClock
    calendar: ReplayCalendar
    live_appointments: AsOfAppointmentRepository
    ingestion_log: IngestionLog
    monitor: LiveMonitor
    users: UserRepository
    events: EventLog
    auth: AuthService
    plans: PlanRepository
    tasks: TaskRepository
    task_service: TaskService
    day_closer: DayCloser
    analyzer: ExperimentAnalyzer
    demo: DemoControls
    cycle: OperationsCycle
    scripts: TemplateScripts
    script_writer: ScriptWriter
    briefing: BriefingWriter
    model: RiskModel | None = None
    metadata: ModelMetadata | None = None
    outreach: OutreachService | None = None
    planner: DailyPlanner | None = None


def build_container(settings: Settings | None = None, require_model: bool = True) -> Container:
    """Build the object graph. Swapping SQLite for another engine means changing this function only."""
    settings = settings or Settings.from_env()
    database = SqliteDatabase(settings.db_path)
    analytics = AnalyticsQueries(database)
    appointments = SqliteAppointmentRepository(database)
    queue = SqliteQueueRepository(database)
    store = ModelStore(settings.models_dir)
    ops = open_ops_database(settings)
    live = _live_parts(settings, appointments, ops)
    parts = dict(
        settings=settings, appointments=appointments, queue=queue, analytics=analytics,
        hypotheses=HypothesisTests(analytics), model_store=store,
        evaluator=ModelEvaluator(min_segment_rows=settings.min_segment_rows),
        artifacts=ArtifactStore(settings.artifacts_dir), simulator=ImpactSimulator(), pilot=PilotDesigner(),
        **live, **_ops_parts(settings, ops, appointments, live),
    )
    if not require_model:
        return Container(**parts)

    model, metadata = store.load()
    service = OutreachService(
        appointments=appointments,
        queue=queue,
        # The smoothing prior comes from the bundle, so scoring builds features
        # exactly the way training did.
        feature_builder=FeatureBuilder(settings, metadata.history_prior_rate),
        model=model,
        model_version=metadata.version,
        policy=CapacityPolicy(settings.call_share, settings.sms_share),
        # Only models that can explain themselves get reason codes.
        explainer=ReasonExplainer(model) if isinstance(model, ExplainableRiskModel) else None,
    )
    planner = DailyPlanner(service, live["live_appointments"], live["clock"], live["calendar"],
                           parts["plans"], parts["tasks"], parts["users"], parts["events"])
    return Container(**parts, model=model, metadata=metadata, outreach=service, planner=planner)


def open_ops_database(settings: Settings) -> SqliteDatabase:
    """The workflow database: created on first use, schema applied on every start (idempotent)."""
    folder = os.path.dirname(settings.ops_db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    database = SqliteDatabase(settings.ops_db_path)
    database.run_script("create_ops_schema", read_sql_file(settings.ops_schema_path, "read_ops_schema_file"))
    return database


def _live_parts(settings: Settings, appointments: AppointmentRepository, ops: SqliteDatabase) -> dict[str, object]:
    """Demo state, clock, calendar, as-of view, feed and monitor for the live replay."""
    week_start = date.fromisoformat(settings.replay_week_start)
    state = StateStore(ops)
    fixed_now = datetime.fromisoformat(settings.demo_now) if settings.demo_now else None
    zone = ZoneInfo(settings.timezone)
    # "Now" is the demo time set in the app if any, else MEDIREACH_DEMO_NOW, else the real clock.
    clock = ReplayClock(week_start, settings.timezone, now=demo_now_provider(state, fixed_now, lambda: datetime.now(zone)))
    calendar = ReplayCalendar(clock, appointments.appointment_dates(), settings.plan_hour)
    # Every page reads through this view, so nothing booked or decided later can leak in.
    live = AsOfAppointmentRepository(appointments, lambda: clock.moment().data_now)
    log = SqliteIngestionLog(ops)
    feed = BookingFeed(appointments, log, IngestionChecks(), week_start)
    monitor = LiveMonitor(clock, calendar, live, feed, log)
    return dict(state=state, clock=clock, calendar=calendar, live_appointments=live, ingestion_log=log, monitor=monitor)


def _ops_parts(settings: Settings, ops: SqliteDatabase, appointments: AppointmentRepository,
               live: dict[str, object]) -> dict[str, object]:
    """Logins, audit trail, plans, tasks, day closing, measurement and demo controls."""
    users = SqliteUserRepository(ops)
    events = SqliteEventLog(ops)
    auth = AuthService(users, PasswordHasher(), events)
    auth.seed_demo_users(settings.demo_password)
    plans, tasks = SqlitePlanRepository(ops), SqliteTaskRepository(ops)
    task_service = TaskService(tasks, plans, events)
    state = live["state"]
    clock, calendar, monitor = live["clock"], live["calendar"], live["monitor"]
    closer = DayCloser(plans, tasks, task_service, appointments, state, events)
    demo = DemoControls(state, clock, calendar, plans, task_service, ops, settings.ops_schema_path, events,
                        settings.plan_hour)
    generator, cache, templates = _ai_generator(settings), TextCache(ops), TemplateScripts()
    return dict(
        users=users, events=events, auth=auth, plans=plans, tasks=tasks, task_service=task_service,
        day_closer=closer, analyzer=ExperimentAnalyzer(plans, tasks, PilotDesigner(), settings.impact),
        demo=demo, cycle=OperationsCycle(monitor, closer, clock), scripts=templates,
        script_writer=ScriptWriter(generator, cache, templates), briefing=BriefingWriter(generator, cache),
    )


def _ai_generator(settings: Settings) -> TextGenerator | None:
    """Gemini when a key is available (environment, or secrets.toml); otherwise None (templates)."""
    key = read_api_key(os.environ, settings.secrets_path)
    if key is None:
        return None
    return GeminiGenerator(key, (settings.gemini_model, settings.gemini_fallback_model), settings.gemini_timeout)
