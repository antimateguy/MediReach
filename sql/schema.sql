-- MediReach schema (SQLite dialect).
-- Dates are stored as ISO text (YYYY-MM-DD) so they sort and compare correctly as strings.
-- Foreign keys are enforced per connection with PRAGMA foreign_keys = ON (see medireach/data/database.py).

-- One row per clinic (health unit). The source column is called Neighbourhood, but the data
-- dictionary says it is where the appointment takes place, not where the patient lives.
CREATE TABLE clinics (
    clinic_id INTEGER PRIMARY KEY,
    name      TEXT    NOT NULL UNIQUE
);

-- One row per patient. Attributes come from the patient's most recent appointment.
CREATE TABLE patients (
    patient_id    INTEGER PRIMARY KEY,
    gender        TEXT    NOT NULL CHECK (gender IN ('M', 'F')),
    age           INTEGER NOT NULL CHECK (age BETWEEN 0 AND 110),
    scholarship   INTEGER NOT NULL CHECK (scholarship IN (0, 1)),
    hypertension  INTEGER NOT NULL CHECK (hypertension IN (0, 1)),
    diabetes      INTEGER NOT NULL CHECK (diabetes IN (0, 1)),
    alcoholism    INTEGER NOT NULL CHECK (alcoholism IN (0, 1)),
    handicap      INTEGER NOT NULL CHECK (handicap BETWEEN 0 AND 4)
);

-- One row per appointment. lead_days = appointment_date - scheduled_date (0 = same-day booking).
CREATE TABLE appointments (
    appointment_id   INTEGER PRIMARY KEY,
    patient_id       INTEGER NOT NULL REFERENCES patients (patient_id),
    clinic_id        INTEGER NOT NULL REFERENCES clinics (clinic_id),
    scheduled_ts     TEXT    NOT NULL,
    scheduled_date   TEXT    NOT NULL,
    appointment_date TEXT    NOT NULL,
    lead_days        INTEGER NOT NULL CHECK (lead_days >= 0),
    sms_received     INTEGER NOT NULL CHECK (sms_received IN (0, 1)),
    no_show          INTEGER NOT NULL CHECK (no_show IN (0, 1))
);

CREATE INDEX idx_appointments_patient ON appointments (patient_id);
CREATE INDEX idx_appointments_date    ON appointments (appointment_date);
CREATE INDEX idx_appointments_clinic  ON appointments (clinic_id);

-- One row per scoring run. Runs are appended, never updated or deleted.
CREATE TABLE outreach_runs (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    window_start  TEXT NOT NULL,
    window_end    TEXT NOT NULL,
    model_name    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    policy_name   TEXT NOT NULL,
    n_rows        INTEGER NOT NULL,
    notes         TEXT
);

-- The scored list for a run. No outcome column on purpose: backtests join the
-- actual outcome from appointments, so a score can never be stored next to its label.
CREATE TABLE outreach_queue (
    run_id           TEXT    NOT NULL REFERENCES outreach_runs (run_id),
    appointment_id   INTEGER NOT NULL REFERENCES appointments (appointment_id),
    patient_id       INTEGER NOT NULL,
    appointment_date TEXT    NOT NULL,
    risk_score       REAL    NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    day_rank         INTEGER NOT NULL,
    priority         TEXT    NOT NULL CHECK (priority IN ('HIGH', 'MEDIUM', 'LOW')),
    action           TEXT    NOT NULL CHECK (action IN ('Call', 'SMS', 'None')),
    reasons          TEXT,
    PRIMARY KEY (run_id, appointment_id)
);

CREATE INDEX idx_queue_run_date ON outreach_queue (run_id, appointment_date);
