-- MediReach workflow database (SQLite), kept apart from the analytical database.
-- data/medireach.db is built by the pipelines and is read-only for the app.
-- data/ops.db holds what the running app writes. Created on first start; every
-- statement is idempotent (IF NOT EXISTS) so it can run on every start.

-- Write-ahead logging: readers and the one writer do not block each other.
PRAGMA journal_mode = WAL;

-- One row per ingestion batch of the live booking feed. The watermark of a
-- replay cycle is the latest to_ts recorded for it.
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_start   TEXT    NOT NULL,
    from_ts       TEXT    NOT NULL,
    to_ts         TEXT    NOT NULL,
    rows_in       INTEGER NOT NULL CHECK (rows_in >= 0),
    checks_failed INTEGER NOT NULL CHECK (checks_failed >= 0),
    check_details TEXT    NOT NULL,
    ran_at        TEXT    NOT NULL,
    CHECK (from_ts < to_ts)
);

CREATE INDEX IF NOT EXISTS idx_ingestion_cycle ON ingestion_runs (cycle_start, to_ts);

-- Demo accounts. Passwords are stored only as salted PBKDF2-SHA256 hashes.
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    display_name  TEXT    NOT NULL,
    role          TEXT    NOT NULL CHECK (role IN ('ADMIN', 'RECEPTIONIST')),
    password_hash TEXT    NOT NULL,
    salt          TEXT    NOT NULL,
    iterations    INTEGER NOT NULL CHECK (iterations > 0),
    active        INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1))
);

-- Audit trail: every login, approval, override, assignment and outcome.
-- Append-only, enforced by the database itself with the two triggers below.
CREATE TABLE IF NOT EXISTS events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    at         TEXT NOT NULL,
    actor      TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entity     TEXT NOT NULL,
    entity_id  TEXT NOT NULL,
    payload    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_entity ON events (entity, entity_id);

CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events are append-only');
END;

CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events are append-only');
END;

-- Demo controls (shared by every user of the demo): demo time, outcome source, planted effect.
CREATE TABLE IF NOT EXISTS ops_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One plan per target clinic day. The engine proposes it at 08:00 on the clinic day
-- before (plan_day); the admin edits and approves it; it closes after the target day.
CREATE TABLE IF NOT EXISTS daily_plans (
    plan_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_start         TEXT    NOT NULL,
    target_real_date    TEXT    NOT NULL,
    target_data_date    TEXT    NOT NULL,
    plan_day            TEXT    NOT NULL,
    as_of_ts            TEXT    NOT NULL,
    model_version       TEXT    NOT NULL,
    status              TEXT    NOT NULL CHECK (status IN ('PROPOSED', 'APPROVED', 'CLOSED')),
    call_capacity       INTEGER NOT NULL CHECK (call_capacity >= 0),
    sms_capacity        INTEGER NOT NULL CHECK (sms_capacity >= 0),
    holdout_mode        TEXT    CHECK (holdout_mode IN ('PILOT', 'MONITORING')),
    holdout_share       REAL    CHECK (holdout_share BETWEEN 0 AND 1),
    seed                INTEGER NOT NULL,
    recalibration_shift REAL    NOT NULL,
    created_by          TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    approved_by         TEXT,
    approved_at         TEXT,
    closed_at           TEXT,
    UNIQUE (cycle_start, target_real_date)
);

-- Every appointment of the target day, with the engine's recommendation and the final decision.
CREATE TABLE IF NOT EXISTS plan_items (
    plan_id            INTEGER NOT NULL REFERENCES daily_plans (plan_id),
    appointment_id     INTEGER NOT NULL,
    patient_id         INTEGER NOT NULL,
    clinic             TEXT    NOT NULL,
    age                INTEGER NOT NULL,
    lead_days          INTEGER NOT NULL,
    risk_score         REAL    NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    risk_calibrated    REAL    NOT NULL CHECK (risk_calibrated BETWEEN 0 AND 1),
    day_rank           INTEGER NOT NULL,
    recommended_action TEXT    NOT NULL CHECK (recommended_action IN ('Call', 'SMS', 'None')),
    action             TEXT    NOT NULL CHECK (action IN ('Call', 'SMS', 'None')),
    arm                TEXT    CHECK (arm IN ('CALL', 'HOLDOUT', 'SMS', 'NONE')),
    override_reason    TEXT,
    late_booking       INTEGER NOT NULL DEFAULT 0 CHECK (late_booking IN (0, 1)),
    reasons            TEXT,
    PRIMARY KEY (plan_id, appointment_id)
);

CREATE INDEX IF NOT EXISTS idx_plan_items_patient ON plan_items (plan_id, patient_id);

-- One task per patient and contact attempt. Retries and SMS fallbacks point to their parent.
CREATE TABLE IF NOT EXISTS tasks (
    task_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id         INTEGER NOT NULL REFERENCES daily_plans (plan_id),
    patient_id      INTEGER NOT NULL,
    kind            TEXT    NOT NULL CHECK (kind IN ('CALL', 'SMS')),
    assigned_to     TEXT    NOT NULL,
    status          TEXT    NOT NULL CHECK (status IN ('PENDING', 'DONE', 'SKIPPED', 'NOT_ATTEMPTED')),
    attempt         INTEGER NOT NULL CHECK (attempt >= 1),
    priority_rank   INTEGER NOT NULL,
    outcome         TEXT    CHECK (outcome IN ('CONFIRMED', 'RESCHEDULE', 'CANCEL', 'NO_ANSWER', 'WRONG_NUMBER', 'SMS_SENT')),
    notes           TEXT,
    parent_task_id  INTEGER REFERENCES tasks (task_id),
    updated_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_plan_user ON tasks (plan_id, assigned_to, status);

-- Attendance after the target day. no_show is what the analysis uses; historical_no_show
-- is the real 2016 outcome, kept so a simulated flip can always be told apart.
CREATE TABLE IF NOT EXISTS revealed_outcomes (
    plan_id            INTEGER NOT NULL REFERENCES daily_plans (plan_id),
    appointment_id     INTEGER NOT NULL,
    no_show            INTEGER NOT NULL CHECK (no_show IN (0, 1)),
    historical_no_show INTEGER NOT NULL CHECK (historical_no_show IN (0, 1)),
    source             TEXT    NOT NULL CHECK (source IN ('historical', 'simulated')),
    revealed_at        TEXT    NOT NULL,
    PRIMARY KEY (plan_id, appointment_id)
);

-- AI-written text, cached by what was asked: one call per distinct input, not per page load.
CREATE TABLE IF NOT EXISTS llm_outputs (
    cache_key  TEXT PRIMARY KEY,
    feature    TEXT NOT NULL,
    model      TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
