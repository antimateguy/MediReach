"""The backend: a small JSON API over the container (and, in one-service mode, the static frontend).

Routes are thin: read the request, check the role, call the engine, return a payload from
payloads.py. Every operations request first runs the operations cycle (feed catch-up and
day closing), so the replayed clinic is always current. One lock serialises the workflow,
because the demo state is shared by every user.

Two ways to deploy, same code:
- one service: this app serves web/ and the API from one address (local runs, Docker);
- split: web/ on a static host (Cloudflare Pages), this API elsewhere; MEDIREACH_CORS_ORIGINS
  lists the static site's address. Login is a bearer token, so no cross-site cookie is needed.
"""
from __future__ import annotations

import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import fields
from datetime import datetime, time, timedelta
from typing import Any, Awaitable, Callable

import pandas as pd
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from medireach import artifacts as files
from medireach.bootstrap import Container, build_container, configure_logging
from medireach.config import PROJECT_ROOT, Settings
from medireach.domain import Action, HoldoutMode, Outcome, Role, User
from medireach.errors import AccessDenied, DataAccessError
from medireach.ops.auth import require_role
from medireach.ops.outcomes import HISTORICAL, SIMULATED
from medireach.outreach.policies import CapacityPolicy
from medireach.web import payloads
from medireach.web.sessions import SessionSigner

logger = logging.getLogger(__name__)

WEB_DIR = os.path.join(PROJECT_ROOT, "web")
_DEFAULT_PASSWORD = next(field.default for field in fields(Settings) if field.name == "demo_password")

Handler = Callable[[Request, dict[str, Any], User | None], Any]


class WebApp:
    """Holds the container, the session signer and the workflow lock; builds the routes."""

    def __init__(self, container: Container, signer: SessionSigner) -> None:
        self.container = container
        self.signer = signer
        self.lock = threading.RLock()
        self._summary: dict[str, Any] | None = None
        self._scores: pd.DataFrame | None = None
        self._queue: tuple[dict[str, Any], pd.DataFrame] | None = None

    # ------------------------------------------------------------ plumbing
    def current_user(self, request: Request) -> User | None:
        """The user named by a valid `Authorization: Bearer <token>` header, if still active."""
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        username = self.signer.username(token.strip()) if scheme.lower() == "bearer" else None
        if username is None:
            return None
        found = self.container.users.find(username)
        if found is None or not found[0].active:
            return None
        return found[0]

    def endpoint(self, handler: Handler, roles: tuple[Role, ...] = (), write: bool = False) -> Callable[[Request], Awaitable[Response]]:
        """Wrap a handler: parse JSON for writes, check the role, run it off the event loop, map errors."""
        async def run(request: Request) -> Response:
            body: dict[str, Any] = {}
            if write:
                if "application/json" not in request.headers.get("content-type", ""):
                    return JSONResponse({"error": "send JSON"}, status_code=415)
                try:
                    body = await request.json()
                except ValueError:
                    return JSONResponse({"error": "the body is not valid JSON"}, status_code=400)
                if not isinstance(body, dict):
                    return JSONResponse({"error": "the body must be a JSON object"}, status_code=400)
            user = self.current_user(request)

            def work() -> Any:
                if roles:
                    require_role(user, *roles)
                    with self.lock:
                        return handler(request, body, user)
                return handler(request, body, user)
            try:
                result = await run_in_threadpool(work)
            except AccessDenied as exc:
                return JSONResponse({"error": str(exc)}, status_code=403 if user else 401)
            except KeyError as exc:
                return JSONResponse({"error": f"missing field: {exc.args[0]}"}, status_code=400)
            except (ValueError, TypeError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            except DataAccessError as exc:
                logger.exception("data access failed")
                return JSONResponse({"error": str(exc)}, status_code=500)
            return result if isinstance(result, Response) else JSONResponse(result)
        return run

    def refresh(self) -> tuple[Any, list[str]]:
        return self.container.cycle.refresh()

    def planner(self):
        if self.container.planner is None:
            raise ValueError("no trained model is loaded; run the pipelines first")
        return self.container.planner

    # -------------------------------------------------------------- public
    def status(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        default = self.container.settings.demo_password == _DEFAULT_PASSWORD
        return payloads.status(self.container, user, default)

    def login(self, request: Request, body: dict[str, Any], user: User | None) -> Response:
        signed_in = self.container.auth.login(str(body.get("username", "")).strip().lower(), str(body.get("password", "")))
        if signed_in is None:
            return JSONResponse({"error": "Unknown account or wrong password."}, status_code=401)
        # The browser keeps the token for this tab only (sessionStorage) and sends it as a header.
        return JSONResponse({"user": payloads.user_payload(signed_in), "token": self.signer.sign(signed_in.username),
                             "expires_in": self.signer.max_age})

    def health(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        """Cheap liveness check; the static site calls it on load to wake a sleeping free host."""
        return {"ok": True}

    def summary(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        if self._summary is None:
            self._summary = payloads.public_summary(self.container.artifacts, self.container.appointments)
        return self._summary

    def scores(self) -> pd.DataFrame:
        if self._scores is None:
            self._scores = self.container.artifacts.read_csv(files.TEST_SCORES)
        return self._scores

    def impact(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        assumptions = payloads.impact_assumptions(self.container.settings.impact, dict(request.query_params))
        policy = CapacityPolicy(self.container.settings.call_share, self.container.settings.sms_share)
        return payloads.impact(self.container.simulator, policy, self.scores(), assumptions)

    def pilot(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        plan = self.summary(request, body, user)["pilot"]
        reduction = float(request.query_params.get("reduction", plan["relative_reduction"]))
        holdout = float(request.query_params.get("holdout", 0.5))
        if not 0.01 <= reduction <= 0.9 or not 0.05 <= holdout <= 0.95:
            raise ValueError("reduction must be 1-90% and the hold-out share 5-95%")
        return payloads.pilot(self.container.pilot, plan["p_control"], reduction, holdout, plan["daily_high_volume"])

    # --------------------------------------------------------------- admin
    def today(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        snapshot, notes = self.refresh()
        return payloads.today(self.container, snapshot, notes)

    def write_briefing(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        snapshot, _ = self.refresh()
        written = self.container.briefing.write(payloads.today_facts(self.container, snapshot))
        return {"briefing": payloads._briefing_payload(written)}

    def plan(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        self.refresh()
        self.planner()
        return payloads.plan_page(self.container, max(0, int(request.query_params.get("offset", 0))))

    def propose(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        self.refresh()
        planner = self.planner()
        calls, sms = int(body["calls"]), int(body["sms"])
        if not 1 <= calls <= 5000 or not 0 <= sms <= 20000:
            raise ValueError("calls must be 1-5,000 and SMS 0-20,000")
        planner.propose(planner.next_target(), calls, sms, user)
        return payloads.plan_page(self.container)

    def override(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        self.planner().override(int(body["plan_id"]), int(body["patient_id"]), Action(body["action"]),
                                str(body.get("reason", "")), user)
        return payloads.plan_page(self.container, max(0, int(body.get("offset", 0))))

    def add_late(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        self.planner().add_late_booking(int(body["plan_id"]), int(body["appointment_id"]), Action(body["action"]), user)
        return payloads.plan_page(self.container, max(0, int(body.get("offset", 0))))

    def approve(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        self.planner().approve(int(body["plan_id"]), HoldoutMode(body["mode"]), user)
        return payloads.plan_page(self.container)

    def team(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        snapshot, _ = self.refresh()
        return payloads.team(self.container, snapshot)

    def reassign(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        moved = self.container.task_service.reassign([int(task) for task in body["task_ids"]], str(body["to"]), user)
        snapshot, _ = self.refresh()
        return {"moved": moved, **payloads.team(self.container, snapshot)}

    def results(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        self.refresh()
        return payloads.results(self.container)

    def directory(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        snapshot, _ = self.refresh()
        day = request.query_params.get("day", "all")
        if day not in ("all", "today", "next"):
            raise ValueError("day must be all, today or next")
        return payloads.directory(self.container, snapshot, request.query_params.get("q", ""), day)

    def demo(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        snapshot, _ = self.refresh()
        return payloads.demo(self.container, snapshot)

    def demo_time(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        demo, action = self.container.demo, body.get("action")
        now = self.container.clock.real_now().replace(tzinfo=None)
        if action == "next_day":
            demo.advance_to_next_clinic_day(user)
        elif action == "real":
            demo.clear_demo_now(user)
        elif action == "shift":
            minutes = int(body["minutes"])
            if abs(minutes) > 24 * 60:
                raise ValueError("shift by at most a day")
            demo.set_demo_now(now + timedelta(minutes=minutes), user)
        elif action == "set":
            demo.set_demo_now(datetime.combine(now.date(), time.fromisoformat(str(body["time"]))), user)
        else:
            raise ValueError("action must be next_day, real, shift or set")
        snapshot, notes = self.refresh()
        return {"notes": notes, **payloads.demo(self.container, snapshot)}

    def simulate(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        snapshot, _ = self.refresh()
        plan = self.container.task_service.plan_for_day(snapshot.moment.real_date)
        if plan is None:
            raise ValueError("no approved plan is being worked on today")
        reach = float(body.get("reach", self.container.settings.impact.call_reach_rate))
        limit = body.get("limit")
        if not 0 <= reach <= 1 or (limit is not None and int(limit) < 1):
            raise ValueError("reach must be 0-1 and the limit at least 1")
        done = self.container.demo.auto_complete(plan.plan_id, user, reach, None if limit is None else int(limit))
        snapshot, _ = self.refresh()
        return {"completed": done, **payloads.demo(self.container, snapshot)}

    def outcomes(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        source = str(body.get("source"))
        if source not in (HISTORICAL, SIMULATED):
            raise ValueError("source must be historical or simulated")
        self.container.demo.set_outcome_source(source, float(body["effect"]), user)
        snapshot, _ = self.refresh()
        return payloads.demo(self.container, snapshot)

    def reset(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        if body.get("confirm") is not True:
            raise ValueError("confirm the reset")
        self.container.demo.reset(user)
        snapshot, _ = self.refresh()
        return payloads.demo(self.container, snapshot)

    def backtest(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        if self._queue is None:
            self._queue = latest_queue(self.container)
        run, queue = self._queue
        return payloads.backtest(self.container, queue, run, request.query_params.get("day"))

    # ----------------------------------------------------------- reception
    def my_tasks(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        snapshot, notes = self.refresh()
        return {"notes": notes, **payloads.my_tasks(self.container, snapshot, user)}

    def record(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        task_id = int(request.path_params["task_id"])
        self.container.task_service.record(task_id, Outcome(body["outcome"]), user, str(body.get("notes", ""))[:500])
        snapshot, _ = self.refresh()
        return payloads.my_tasks(self.container, snapshot, user)

    def skip(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        self.container.task_service.skip(int(request.path_params["task_id"]), str(body.get("reason", "")), user)
        snapshot, _ = self.refresh()
        return payloads.my_tasks(self.container, snapshot, user)

    def polish(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        snapshot, _ = self.refresh()
        page = payloads.my_tasks(self.container, snapshot, user)
        task = page.get("next")
        if task is None or task["task_id"] != int(request.path_params["task_id"]) or task["kind"] != "CALL":
            raise ValueError("only your current call task can be polished")
        plan = self.container.task_service.plan_for_day(snapshot.moment.real_date)
        text, source, note = self.container.script_writer.polished(
            "; ".join(task["reasons"]), plan.target_real_date, task["clinic"], task["appointments"])
        return {"script": text, "source": source, "note": note}

    def my_day(self, request: Request, body: dict[str, Any], user: User | None) -> dict[str, Any]:
        snapshot, _ = self.refresh()
        return payloads.my_day(self.container, snapshot, user)

    # -------------------------------------------------------------- routes
    def routes(self, web_dir: str) -> list[Any]:
        admin, reception = (Role.ADMIN,), (Role.RECEPTIONIST,)
        api = [
            ("/api/health", self.health, (), "GET"),
            ("/api/status", self.status, (), "GET"), ("/api/login", self.login, (), "POST"),
            ("/api/public/summary", self.summary, (), "GET"), ("/api/public/impact", self.impact, (), "GET"),
            ("/api/public/pilot", self.pilot, (), "GET"),
            ("/api/admin/today", self.today, admin, "GET"), ("/api/admin/briefing", self.write_briefing, admin, "POST"),
            ("/api/admin/plan", self.plan, admin, "GET"), ("/api/admin/plan/propose", self.propose, admin, "POST"),
            ("/api/admin/plan/override", self.override, admin, "POST"),
            ("/api/admin/plan/late", self.add_late, admin, "POST"),
            ("/api/admin/plan/approve", self.approve, admin, "POST"),
            ("/api/admin/team", self.team, admin, "GET"), ("/api/admin/team/reassign", self.reassign, admin, "POST"),
            ("/api/admin/results", self.results, admin, "GET"),
            ("/api/admin/directory", self.directory, admin, "GET"),
            ("/api/admin/demo", self.demo, admin, "GET"), ("/api/admin/demo/time", self.demo_time, admin, "POST"),
            ("/api/admin/demo/simulate", self.simulate, admin, "POST"),
            ("/api/admin/demo/outcomes", self.outcomes, admin, "POST"),
            ("/api/admin/demo/reset", self.reset, admin, "POST"),
            ("/api/admin/backtest", self.backtest, admin, "GET"),
            ("/api/me/tasks", self.my_tasks, reception, "GET"),
            ("/api/me/tasks/{task_id:int}/outcome", self.record, reception, "POST"),
            ("/api/me/tasks/{task_id:int}/skip", self.skip, reception, "POST"),
            ("/api/me/tasks/{task_id:int}/polish", self.polish, reception, "POST"),
            ("/api/me/day", self.my_day, reception, "GET"),
        ]
        routes: list[Any] = [Route(path, self.endpoint(handler, roles, write=method == "POST"), methods=[method])
                             for path, handler, roles, method in api]
        routes.append(Route("/api/{rest:path}", _not_found, methods=["GET", "POST"]))
        routes.append(Mount("/", CachedStaticFiles(directory=web_dir, html=True), name="web"))
        return routes

    def warm_up(self) -> None:
        """Do the slow first-request work at start-up: public numbers, scores, the feed catch-up."""
        try:
            self.summary(None, {}, None)  # type: ignore[arg-type]
            self.scores()
            with self.lock:
                self.refresh()
            logger.info("Warm-up done")
        except Exception:  # noqa: BLE001 - warm-up is best effort; requests still work without it
            logger.exception("warm-up failed")


class CachedStaticFiles(StaticFiles):
    """Static files with cache headers: fonts and images for a week, pages and code revalidated."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            long_lived = path.replace("\\", "/").startswith("assets/")
            response.headers["Cache-Control"] = "public, max-age=604800" if long_lived else "no-cache"
        return response


async def _not_found(request: Request) -> Response:
    return JSONResponse({"error": "no such API route"}, status_code=404)


def latest_queue(container: Container) -> tuple[dict[str, Any], pd.DataFrame]:
    """The most recent stored outreach run, joined to appointment details and the actual outcome."""
    run_id = container.queue.latest_run_id()
    if run_id is None:
        raise ValueError("no outreach run is stored; run the score pipeline first")
    runs = container.queue.list_runs()
    run = runs[runs["run_id"] == run_id].iloc[0].to_dict()
    rows = container.queue.get_run(str(run_id))
    window = container.appointments.get_appointments(pd.Timestamp(run["window_start"]).date(),
                                                     pd.Timestamp(run["window_end"]).date())
    details = window[["appointment_id", "lead_days", "age", "gender", "clinic", "no_show"]]
    return run, rows.merge(details, on="appointment_id", how="left", validate="one_to_one")


def _session_secret() -> bytes:
    """MEDIREACH_SESSION_SECRET if set (keeps logins across restarts); else a fresh random one."""
    value = os.environ.get("MEDIREACH_SESSION_SECRET", "")
    return value.encode("utf-8") if len(value) >= 16 else secrets.token_bytes(32)


def _cors_origins() -> list[str]:
    """Addresses of a separately hosted frontend, e.g. https://medireach.pages.dev (comma-separated)."""
    return [origin.strip().rstrip("/") for origin in os.environ.get("MEDIREACH_CORS_ORIGINS", "").split(",")
            if origin.strip()]


def create_app(container: Container | None = None, web_dir: str = WEB_DIR, warm_up: bool = True) -> Starlette:
    """Build the ASGI app. Tests pass their own container; the server builds one from the environment."""
    if container is None:
        settings = Settings.from_env()
        configure_logging(settings)
        container = build_container(settings)
    web = WebApp(container, SessionSigner(_session_secret()))

    @asynccontextmanager
    async def lifespan(app: Starlette):
        if warm_up:
            threading.Thread(target=web.warm_up, name="warm-up", daemon=True).start()
        yield

    middleware = [Middleware(GZipMiddleware, minimum_size=1024)]
    origins = _cors_origins()
    if origins:
        middleware.append(Middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                                     allow_headers=["Authorization", "Content-Type"], max_age=600))
    app = Starlette(routes=web.routes(web_dir), middleware=middleware, lifespan=lifespan)
    app.state.web = web
    return app
