"""Pipeline: raw CSV to cleaned SQLite database with schema and views."""
from __future__ import annotations

import json
import logging
import os

import pandas as pd

from medireach.artifacts import CLEANING_REPORT, ArtifactStore
from medireach.bootstrap import configure_logging
from medireach.config import Settings
from medireach.data.cleaning import APPOINTMENT_TABLE_COLUMNS, PATIENT_COLUMNS, RawDataLoader, clinic_table
from medireach.data.database import SqliteDatabase, read_sql_file
from medireach.data.repositories import DATE_FORMAT
from medireach.domain import CleaningReport

logger = logging.getLogger(__name__)


def _insert_sql(table: str, columns: tuple[str, ...]) -> str:
    # Table and column names are constants from this codebase; values are bound.
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"


def populate(database: SqliteDatabase, settings: Settings,
             patients: pd.DataFrame, appointments: pd.DataFrame) -> None:
    """Create the schema, load the three tables in one transaction, then create the views."""
    database.run_script("create_schema", read_sql_file(settings.schema_path, "read_schema_file"))
    stored = appointments[list(APPOINTMENT_TABLE_COLUMNS)].copy()
    for column in ("scheduled_date", "appointment_date"):
        stored[column] = stored[column].dt.strftime(DATE_FORMAT)
    clinics = clinic_table(appointments)
    database.write("insert_clinics_patients_and_appointments", [
        # Clinics and patients first: appointments reference both.
        (_insert_sql("clinics", ("clinic_id", "name")), clinics.astype(object).values.tolist()),
        (_insert_sql("patients", PATIENT_COLUMNS),
         patients[list(PATIENT_COLUMNS)].astype(object).values.tolist()),
        (_insert_sql("appointments", APPOINTMENT_TABLE_COLUMNS), stored.astype(object).values.tolist()),
    ])
    database.run_script("create_views", read_sql_file(settings.views_path, "read_views_file"))


def build_database(settings: Settings, csv_path: str | None = None) -> CleaningReport:
    """Rebuild the database file from the raw CSV. Safe to run repeatedly."""
    patients, appointments, report = RawDataLoader(settings).load(csv_path)
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    # Build beside the real file and swap at the end, so a failed build never
    # leaves a half-written database behind.
    staging_path = settings.db_path + ".building"
    if os.path.exists(staging_path):
        os.remove(staging_path)
    populate(SqliteDatabase(staging_path), settings, patients, appointments)
    os.replace(staging_path, settings.db_path)
    ArtifactStore(settings.artifacts_dir).write_json(CLEANING_REPORT, report.to_dict())
    logger.info("Built %s: %d patients, %d appointments", settings.db_path, len(patients), len(appointments))
    return report


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings)
    report = build_database(settings)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
