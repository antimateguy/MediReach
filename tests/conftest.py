"""Shared pytest fixtures."""
from __future__ import annotations

from typing import Iterator

import pandas as pd
import pytest

from medireach.config import Settings
from medireach.data.cleaning import RawDataLoader
from medireach.data.database import IN_MEMORY, SqliteDatabase
from medireach.domain import CleaningReport
from medireach.pipelines.build_db import populate
from tests.factories import make_raw_appointments


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture()
def raw() -> pd.DataFrame:
    return make_raw_appointments()


@pytest.fixture()
def cleaned(settings: Settings, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """(patients, appointments, report) for the synthetic data."""
    return RawDataLoader(settings).clean(raw)


@pytest.fixture()
def database(settings: Settings, cleaned: tuple[pd.DataFrame, pd.DataFrame, CleaningReport]) -> Iterator[SqliteDatabase]:
    """An in-memory SQLite database with schema, synthetic data and views."""
    patients, appointments, _ = cleaned
    db = SqliteDatabase(IN_MEMORY)
    populate(db, settings, patients, appointments)
    yield db
    db.close()
