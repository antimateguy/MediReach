"""The web app's API: sessions, roles, the public numbers, and the whole daily workflow over HTTP.

Uses the built database, model and artifacts (skipped before the first `run_all`), a fresh
workflow database in a temporary folder, a fixed Thursday morning and no AI key, so nothing
here touches the network.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import replace

import pandas as pd
import pytest
from starlette.testclient import TestClient

from medireach import artifacts as files
from medireach.artifacts import ArtifactStore
from medireach.bootstrap import build_container
from medireach.config import Settings
from medireach.web import payloads
from medireach.pipelines.export_web import PUBLIC_JSON, public_bundle
from medireach.web.server import create_app
from medireach.web.sessions import SessionSigner

_SETTINGS = Settings()
_BUILT = os.path.exists(_SETTINGS.db_path) and os.path.exists(os.path.join(_SETTINGS.models_dir, "LATEST"))
needs_build = pytest.mark.skipif(not _BUILT, reason="run `python -m medireach.pipelines.run_all` first")
PASSWORD = _SETTINGS.demo_password


@pytest.fixture(scope="module")
def app():
    folder = tempfile.mkdtemp(prefix="medireach_web_")
    settings = replace(Settings.from_env(), ops_db_path=os.path.join(folder, "ops.db"), demo_now="2026-10-22T10:15",
                       secrets_path=os.path.join(folder, "no_secrets.toml"))
    saved = os.environ.pop("GEMINI_API_KEY", None)
    try:
        container = build_container(settings)
    finally:
        if saved is not None:
            os.environ["GEMINI_API_KEY"] = saved
    return create_app(container, warm_up=False)


def _client(app, username: str | None = None) -> TestClient:
    client = TestClient(app)
    if username:
        response = client.post("/api/login", json={"username": username, "password": PASSWORD})
        assert response.status_code == 200, response.text
        client.headers["Authorization"] = f"Bearer {response.json()['token']}"
    return client


# ------------------------------------------------------------------ sessions

def test_session_token_round_trips_and_rejects_tampering_and_expiry() -> None:
    now = [1_000_000.0]
    signer = SessionSigner(b"x" * 32, max_age_seconds=60, clock=lambda: now[0])
    token = signer.sign("admin")
    assert signer.username(token) == "admin"
    assert signer.username(token[:-2] + ("A" if token[-2] != "A" else "B") + token[-1]) is None
    assert signer.username(SessionSigner(b"y" * 32).sign("admin")) is None
    assert signer.username("garbage") is None and signer.username(None) is None
    now[0] += 61
    assert signer.username(token) is None


def test_session_secret_must_be_long_enough() -> None:
    with pytest.raises(ValueError):
        SessionSigner(b"short")


def test_clean_makes_values_json_safe() -> None:
    import numpy as np
    cleaned = payloads.clean({"a": np.int64(3), "b": float("nan"), "c": [np.float64(0.5), pd.Timestamp("2016-06-02")]})
    assert cleaned == {"a": 3, "b": None, "c": [0.5, "2016-06-02T00:00:00"]}


# -------------------------------------------------------------------- public

@needs_build
def test_status_for_a_visitor_shows_the_labels_and_the_demo_hint(app) -> None:
    status = _client(app).get("/api/status").json()
    assert status["user"] is None
    assert status["replay_label"] == "Replay of 2016 data"
    assert status["replaying_label"] == "Thu 02 Jun 2016"
    assert {account["username"] for account in status["demo_accounts"]} == {"admin", "reception1", "reception2"}
    assert status["ai_available"] is False


@needs_build
def test_public_summary_quotes_the_artifacts(app) -> None:
    summary = _client(app).get("/api/public/summary").json()
    store = ArtifactStore(_SETTINGS.artifacts_dir)
    models = store.read_csv(files.MODEL_COMPARISON)
    selected = models[(models["selected"] == 1) & (models["period"] == "test")].iloc[0]
    assert summary["headline"]["roc_auc"] == pytest.approx(selected["roc_auc"])
    assert summary["data"]["rows_out"] == store.read_json(files.CLEANING_REPORT)["rows_out"]
    hypotheses = store.read_json(files.HYPOTHESES)
    assert [h["verdict"] for h in summary["hypotheses"]] == [h["verdict"] for h in hypotheses]
    assert all(h["bars"] for h in summary["hypotheses"])
    assert summary["replay"]["simulated"]["relative_reduction"] == pytest.approx(
        store.read_json(files.REPLAY_VALIDATION)["simulated"]["relative_reduction"])


@needs_build
def test_impact_at_default_assumptions_matches_the_pipeline(app) -> None:
    strategies = _client(app).get("/api/public/impact").json()["strategies"]
    stored = ArtifactStore(_SETTINGS.artifacts_dir).read_csv(files.IMPACT_COMPARISON).set_index("strategy")
    for row in strategies:
        assert row["net"] == pytest.approx(stored.loc[row["strategy"], "net"], abs=0.01)


@needs_build
def test_impact_rejects_impossible_assumptions(app) -> None:
    response = _client(app).get("/api/public/impact?call_reach_rate=1.5")
    assert response.status_code == 400


@needs_build
def test_pilot_default_matches_the_stored_plan_and_a_small_holdout_needs_more_patients(app) -> None:
    client = _client(app)
    equal = client.get("/api/public/pilot?reduction=0.18&holdout=0.5").json()
    assert equal["called"] == equal["held_out"] == 735
    small = client.get("/api/public/pilot?reduction=0.18&holdout=0.1").json()
    assert small["held_out"] < equal["held_out"] and small["total"] > equal["total"]


@needs_build
def test_static_frontend_and_unknown_api_routes(app) -> None:
    client = _client(app)
    assert client.get("/").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert client.get("/api/nothing").status_code == 404
    assert client.get("/api/health").json() == {"ok": True}


@needs_build
def test_static_files_are_cached_sensibly_and_responses_compressed(app) -> None:
    client = _client(app)
    assert "max-age=604800" in client.get("/assets/fonts.css").headers["cache-control"]
    assert client.get("/app.js").headers["cache-control"] == "no-cache"
    summary = client.get("/api/public/summary", headers={"Accept-Encoding": "gzip"})
    assert summary.headers.get("content-encoding") == "gzip"


@needs_build
def test_the_exported_public_file_is_current() -> None:
    """web/data/public.json must equal what the API would serve (rerun the export_web pipeline if not)."""
    import json
    with open(PUBLIC_JSON, encoding="utf-8") as handle:
        assert json.load(handle) == json.loads(json.dumps(public_bundle(_SETTINGS)))


@needs_build
def test_cors_allows_only_the_configured_frontend(app, monkeypatch) -> None:
    monkeypatch.setenv("MEDIREACH_CORS_ORIGINS", "https://frontend.example")
    split = TestClient(create_app(app.state.web.container, warm_up=False))
    allowed = split.options("/api/status", headers={"Origin": "https://frontend.example", "Access-Control-Request-Method": "GET",
                                                     "Access-Control-Request-Headers": "authorization"})
    assert allowed.headers.get("access-control-allow-origin") == "https://frontend.example"
    other = split.get("/api/status", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers
    assert "access-control-allow-origin" not in _client(app).get("/api/status", headers={"Origin": "https://frontend.example"}).headers


# ---------------------------------------------------------------------- auth

@needs_build
def test_login_returns_a_bearer_token_and_wrong_passwords_fail(app) -> None:
    client = TestClient(app)
    assert client.post("/api/login", json={"username": "admin", "password": "wrong"}).status_code == 401
    response = client.post("/api/login", json={"username": "admin", "password": PASSWORD})
    assert response.status_code == 200 and "set-cookie" not in response.headers
    token = response.json()["token"]
    assert client.get("/api/admin/today").status_code == 401
    assert client.get("/api/admin/today", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert client.get("/api/admin/today", headers={"Authorization": f"Bearer {token}x"}).status_code == 401
    assert client.get("/api/admin/today", headers={"Authorization": token}).status_code == 401
    me = client.get("/api/status", headers={"Authorization": f"Bearer {token}"}).json()["user"]
    assert me["username"] == "admin" and me["role"] == "ADMIN"


@needs_build
def test_roles_are_enforced(app) -> None:
    visitor, admin, reception = _client(app), _client(app, "admin"), _client(app, "reception1")
    assert visitor.get("/api/admin/plan").status_code == 401
    assert visitor.get("/api/me/tasks").status_code == 401
    assert reception.get("/api/admin/plan").status_code == 403
    assert admin.get("/api/me/tasks").status_code == 403
    assert reception.post("/api/admin/demo/reset", json={"confirm": True}).status_code == 403


@needs_build
def test_writes_must_be_json(app) -> None:
    admin = _client(app, "admin")
    assert admin.post("/api/admin/demo/reset", content="confirm").status_code == 415
    assert admin.post("/api/admin/demo/reset", json={}).status_code == 400


# ------------------------------------------------------------------ workflow

@needs_build
def test_the_whole_daily_cycle_over_the_api(app) -> None:
    admin, reception = _client(app, "admin"), _client(app, "reception1")
    assert admin.post("/api/admin/demo/reset", json={"confirm": True}).status_code == 200

    page = admin.get("/api/admin/plan").json()
    assert page["plan"] is None and page["can_propose"] is True
    page = admin.post("/api/admin/plan/propose", json={"calls": 440, "sms": 880}).json()
    plan_id, first = page["plan"]["plan_id"], page["rows"][0]
    assert page["plan"]["status"] == "PROPOSED" and page["numbers"]["calls"] == 440

    no_reason = admin.post("/api/admin/plan/override", json={"plan_id": plan_id, "patient_id": first["patient_id"],
                                                             "action": "SMS", "reason": ""})
    assert no_reason.status_code == 400
    changed = admin.post("/api/admin/plan/override", json={"plan_id": plan_id, "patient_id": first["patient_id"],
                                                           "action": "SMS", "reason": "Clinical priority"}).json()
    assert changed["rows"][0]["action"] == "SMS" and changed["rows"][0]["override"] == "Clinical priority"

    approved = admin.post("/api/admin/plan/approve", json={"plan_id": plan_id, "mode": "PILOT"}).json()
    assert approved["plan"]["status"] == "APPROVED"
    assert approved["arms"]["CALL"] + approved["arms"]["HOLDOUT"] == 439
    assert abs(approved["arms"]["CALL"] - approved["arms"]["HOLDOUT"]) <= 1
    assert admin.post("/api/admin/plan/approve", json={"plan_id": plan_id, "mode": "PILOT"}).status_code == 400

    tasks = reception.get("/api/me/tasks").json()
    task = tasks["next"]
    assert tasks["open"] == tasks["total"] > 0 and task["script"]
    assert reception.post(f"/api/me/tasks/{task['task_id']}/outcome", json={"outcome": "NOPE"}).status_code == 400
    valid = task["outcomes"][0]["value"]
    after = reception.post(f"/api/me/tasks/{task['task_id']}/outcome", json={"outcome": valid}).json()
    assert after["done"] == 1
    assert reception.post(f"/api/me/tasks/{task['task_id']}/outcome", json={"outcome": valid}).status_code == 400
    other = _client(app, "reception2").get("/api/me/tasks").json()["next"]["task_id"]
    assert reception.post(f"/api/me/tasks/{other}/outcome", json={"outcome": "SMS_SENT"}).status_code == 403

    skipped = reception.post(f"/api/me/tasks/{after['next']['task_id']}/skip", json={"reason": "Out of time today"})
    assert skipped.status_code == 200
    day = reception.get("/api/me/day").json()
    assert day["done"] == 2 and day["skipped"] == 1 and len(day["log"]) == 2

    team = admin.get("/api/admin/team").json()
    open_task = next(t for t in team["team"][0]["tasks"] if t["status"] == "PENDING")
    moved = admin.post("/api/admin/team/reassign", json={"task_ids": [open_task["task_id"]], "to": "reception2"}).json()
    assert moved["moved"] == 1

    five = admin.post("/api/admin/demo/simulate", json={"limit": 5}).json()
    assert five["completed"] == 5
    rest = admin.post("/api/admin/demo/simulate", json={}).json()
    assert rest["open_tasks"] == 0

    assert admin.post("/api/admin/demo/time", json={"action": "next_day"}).json()["time_label"] == "08:00"
    closed = admin.post("/api/admin/demo/time", json={"action": "next_day"}).json()
    assert closed["notes"], "moving past the target day should close it"
    results = admin.get("/api/admin/results").json()
    assert results["closed_days"] == 1 and results["source"] == "historical"
    assert results["patients_call"] > 0 and results["patients_holdout"] > 0
    assert results["reduction_ci"][0] <= results["relative_reduction"] <= results["reduction_ci"][1]

    assert admin.post("/api/admin/demo/time", json={"action": "shift", "minutes": 15}).json()["time_label"] == "08:15"
    assert admin.post("/api/admin/demo/time", json={"action": "set", "time": "17:00"}).json()["time_label"] == "17:00"
    assert admin.post("/api/admin/demo/time", json={"action": "fly"}).status_code == 400


@needs_build
def test_admin_read_pages_answer(app) -> None:
    admin = _client(app, "admin")
    assert admin.get("/api/admin/today").json()["briefing"]["source"] == "template"
    directory = admin.get("/api/admin/directory?day=next").json()
    assert directory["count"] >= directory["shown"]
    assert admin.get("/api/admin/directory?day=yesterday").status_code == 400
    backtest = admin.get("/api/admin/backtest").json()
    assert backtest["calls"] > 0 and len(backtest["rows"]) <= 12
    assert admin.get("/api/admin/backtest?day=2016-01-01").status_code == 400


@needs_build
def test_polish_without_ai_falls_back_to_the_template(app) -> None:
    admin, reception = _client(app, "admin"), _client(app, "reception1")
    admin.post("/api/admin/demo/reset", json={"confirm": True})
    plan_id = admin.post("/api/admin/plan/propose", json={"calls": 440, "sms": 880}).json()["plan"]["plan_id"]
    admin.post("/api/admin/plan/approve", json={"plan_id": plan_id, "mode": "MONITORING"})
    tasks = reception.get("/api/me/tasks").json()
    while tasks["next"]["kind"] != "CALL":
        tasks = reception.post(f"/api/me/tasks/{tasks['next']['task_id']}/outcome", json={"outcome": "SMS_SENT"}).json()
    polished = reception.post(f"/api/me/tasks/{tasks['next']['task_id']}/polish", json={}).json()
    assert polished["source"] == "template" and polished["script"] == tasks["next"]["script"]
    assert reception.post(f"/api/me/tasks/{tasks['next']['task_id'] + 99999}/polish", json={}).status_code == 400
