"""v3 I1: feature-group ablation and the leakage-safe clinic encoding."""
from __future__ import annotations

import pandas as pd
import pytest

from medireach.config import Settings
from medireach.data.cleaning import RawDataLoader
from medireach.features.builder import FEATURE_COLUMNS, FeatureBuilder, time_split
from medireach.ml.ablation import GROUPS, ClinicEncoder, FeatureAblation
from medireach.ml.evaluation import ModelEvaluator
from tests.doubles import InMemoryAppointmentRepository
from tests.factories import make_raw_appointments


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    settings = Settings()
    patients, appointments, _ = RawDataLoader(settings).clean(make_raw_appointments(300, 3000, seed=5))
    rows = InMemoryAppointmentRepository(patients, appointments).get_appointments()
    split = time_split(rows)
    features, meta = FeatureBuilder(settings, 0.2).build(rows)
    labels = meta["no_show"]
    clinics = rows["clinic"].reset_index(drop=True)
    return FeatureAblation(ModelEvaluator(min_segment_rows=10)).run(
        features[split.train], labels[split.train], features[split.validation], labels[split.validation],
        clinic_train=clinics[split.train], clinic_valid=clinics[split.validation],
    )


def test_groups_cover_every_feature_exactly_once() -> None:
    columns = [column for _, group in GROUPS for column in group]
    assert sorted(columns) == sorted(FEATURE_COLUMNS) and len(columns) == len(set(columns))


def test_stepwise_study_adds_groups_in_order_and_records_gains(table: pd.DataFrame) -> None:
    steps = table[(table["study"] == "added one at a time") & (table["step"] <= len(GROUPS))]
    assert list(steps["n_features"]) == [5, 9, 14, 19, 20, 21]
    assert pd.isna(steps["gain_roc_auc"].iloc[0])
    gains = steps["roc_auc"].diff().iloc[1:].to_numpy()
    assert gains == pytest.approx(steps["gain_roc_auc"].iloc[1:].to_numpy())


def test_leave_one_out_is_measured_against_the_full_model(table: pd.DataFrame) -> None:
    full = table[(table["study"] == "added one at a time") & (table["step"] == len(GROUPS))]["roc_auc"].iloc[0]
    left_out = table[table["study"] == "left out one at a time"]
    assert len(left_out) == len(GROUPS)
    assert (left_out["roc_auc"] - full).to_numpy() == pytest.approx(left_out["gain_roc_auc"].to_numpy())


def test_with_lead_time_only_the_logistic_model_ranks_like_the_baseline(table: pd.DataFrame) -> None:
    grid = table[table["study"] == "model x information"].set_index(["label", "model"])["roc_auc"]
    assert grid[("Lead time only", "logistic_regression")] == pytest.approx(grid[("Lead time only", "lead_time_baseline")], abs=0.01)
    assert len(table[table["label"] == "+ Clinic location (encoded)"]) == 1


def test_clinic_encoding_uses_training_labels_only() -> None:
    clinics = pd.Series(["A", "A", "B", "B", "B", "C"])
    labels = pd.Series([1, 1, 0, 0, 1, 0])
    encoder = ClinicEncoder(strength=0).fit(clinics, labels)
    assert encoder.transform(pd.Series(["A", "B"])).tolist() == pytest.approx([1.0, 1 / 3])
    assert encoder.transform(pd.Series(["unseen"])).iloc[0] == pytest.approx(labels.mean())
    smoothed = ClinicEncoder(strength=50).fit(clinics, labels).transform(pd.Series(["A"])).iloc[0]
    assert labels.mean() < smoothed < 1.0
