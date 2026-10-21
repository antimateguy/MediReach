"""Phase 1: settings defaults, environment overrides, enums and value objects."""
from __future__ import annotations

import os
from dataclasses import FrozenInstanceError

import pytest

from medireach.config import PROJECT_ROOT, Settings, check_sqlite_version
from medireach.domain import Action, CleaningReport, ImpactAssumptions, ModelMetadata, Priority
from medireach.errors import ConfigError, DataAccessError, MediReachError


def test_defaults_point_inside_the_project() -> None:
    settings = Settings.from_env({})
    assert settings.db_path == os.path.join(PROJECT_ROOT, "data", "medireach.db")
    assert settings.seed == 42
    assert settings.call_share == 0.10 and settings.sms_share == 0.20
    assert settings.split_fractions == (0.6, 0.2, 0.2)
    assert settings.lead_bucket_labels == ("0", "1-3", "4-7", "8-14", "15-30", "31+")


def test_env_overrides_are_typed_and_paths_are_resolved() -> None:
    env = {"MEDIREACH_SEED": "7", "MEDIREACH_CALL_SHARE": "0.25", "MEDIREACH_DB_PATH": "tmp/x.db"}
    settings = Settings.from_env(env)
    assert settings.seed == 7
    assert settings.call_share == 0.25
    assert settings.db_path == os.path.join(PROJECT_ROOT, "tmp", "x.db")


def test_blank_env_value_keeps_default() -> None:
    assert Settings.from_env({"MEDIREACH_SEED": " "}).seed == 42


def test_bad_env_value_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({"MEDIREACH_SEED": "forty-two"})
    with pytest.raises(ConfigError):
        Settings.from_env({"MEDIREACH_CALL_SHARE": "0.9", "MEDIREACH_SMS_SHARE": "0.5"})


def test_settings_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        Settings().seed = 1  # type: ignore[misc]


def test_sqlite_version_check() -> None:
    assert check_sqlite_version()
    with pytest.raises(ConfigError):
        check_sqlite_version(minimum=(99, 0, 0))


def test_enum_values_are_the_strings_stored_in_the_queue() -> None:
    assert [p.value for p in Priority] == ["HIGH", "MEDIUM", "LOW"]
    assert [a.value for a in Action] == ["Call", "SMS", "None"]
    assert Priority("HIGH") is Priority.HIGH


def test_cleaning_report_must_balance() -> None:
    common = dict(
        patients=1, clinics=1, patients_with_varying_attributes=0, varying_by_attribute={},
        patient_id_collisions=0, base_rate_before=0.2, base_rate_after=0.2,
        same_day_rows=3, same_day_share=0.3, same_day_no_show_rate=0.05,
    )
    CleaningReport(rows_in=10, rows_out=8, dropped={"a": 1, "b": 1}, **common)
    with pytest.raises(ValueError):
        CleaningReport(rows_in=10, rows_out=8, dropped={"a": 1}, **common)


def test_impact_assumptions_validate_ranges() -> None:
    assert ImpactAssumptions().call_effectiveness == 0.30
    with pytest.raises(ValueError):
        ImpactAssumptions(call_reach_rate=1.5)
    with pytest.raises(ValueError):
        ImpactAssumptions(cost_per_call=-1)


def test_model_metadata_round_trips_through_a_dict() -> None:
    meta = ModelMetadata(
        version="v1", model_name="logistic", feature_columns=["a"], trained_at="t",
        train_range=("2016-04-29", "2016-05-19"), validation_range=("a", "b"), test_range=("c", "d"),
        sklearn_version="1", python_version="3", validation_metrics={"roc_auc": 0.6},
        test_metrics={}, selection_reason="simpler", history_prior_rate=0.21,
    )
    as_json_would = {k: list(v) if isinstance(v, tuple) else v for k, v in meta.to_dict().items()}
    assert ModelMetadata.from_dict(as_json_would) == meta


def test_data_access_error_names_the_statement() -> None:
    error = DataAccessError("insert_patients", "disk full")
    assert isinstance(error, MediReachError)
    assert "insert_patients" in str(error)
