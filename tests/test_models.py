"""Phase 5: the three risk models, the model store and the selection rule."""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from medireach.config import Settings
from medireach.data.cleaning import RawDataLoader
from medireach.domain import ModelMetadata
from medireach.errors import ModelBundleError
from medireach.features.builder import CONTINUOUS_COLUMNS, FEATURE_COLUMNS, LEAD_BUCKET_COLUMNS, FeatureBuilder
from medireach.ml.evaluation import ModelEvaluator
from medireach.ml.models import (
    ExplainableRiskModel,
    LeadTimeBaselineModel,
    LogisticRiskModel,
    RandomForestRiskModel,
    RiskModel,
)
from medireach.ml.store import LATEST_FILE, MODEL_FILE, ModelStore
from medireach.pipelines.train import SCORE_COLUMNS, candidate_models, select_model, train
from tests.doubles import InMemoryAppointmentRepository
from tests.factories import make_raw_appointments


@pytest.fixture(scope="module")
def repository() -> InMemoryAppointmentRepository:
    patients, appointments, _ = RawDataLoader(Settings()).clean(make_raw_appointments(150, 1200, seed=2))
    return InMemoryAppointmentRepository(patients, appointments)


@pytest.fixture(scope="module")
def training_data(repository: InMemoryAppointmentRepository) -> tuple[pd.DataFrame, pd.Series]:
    features, meta = FeatureBuilder(Settings(), prior_rate=0.2).build(repository.get_appointments())
    return features, meta["no_show"]


def _metadata(version: str = "v-test", feature_columns: list[str] | None = None) -> ModelMetadata:
    return ModelMetadata(
        version=version, model_name="any", feature_columns=feature_columns or list(FEATURE_COLUMNS),
        trained_at="2016-06-01T00:00:00+00:00", train_range=("2016-05-02", "2016-05-13"),
        validation_range=("2016-05-16", "2016-05-20"), test_range=("2016-05-23", "2016-05-27"),
        sklearn_version="x", python_version="y", validation_metrics={}, test_metrics={},
        selection_reason="test", history_prior_rate=0.2,
    )


# ------------------------------------------------------------ all three models

@pytest.mark.parametrize("model", candidate_models(seed=42), ids=lambda model: model.name)
def test_model_fits_and_returns_probabilities(model: RiskModel, training_data) -> None:
    features, labels = training_data
    scores = model.fit(features, labels).predict_proba(features)
    assert isinstance(scores, np.ndarray) and scores.shape == (len(features),)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert len(np.unique(scores)) > 1
    # No class weighting: the average prediction on the training rows matches the base rate.
    assert scores.mean() == pytest.approx(labels.mean(), abs=0.02)


@pytest.mark.parametrize("model", candidate_models(seed=42), ids=lambda model: model.name)
def test_model_is_deterministic_and_ignores_column_order(model: RiskModel, training_data) -> None:
    features, labels = training_data
    first = model.fit(features, labels).predict_proba(features)
    again = type(model)().fit(features, labels).predict_proba(features)
    assert np.allclose(first, again)
    assert np.allclose(first, model.predict_proba(features[features.columns[::-1]]))


@pytest.mark.parametrize("model", candidate_models(seed=42), ids=lambda model: model.name)
def test_save_and_load_round_trip_gives_identical_predictions(model: RiskModel, training_data, tmp_path) -> None:
    features, labels = training_data
    model.fit(features, labels)
    store = ModelStore(str(tmp_path))
    version = store.save(model, _metadata("v-roundtrip"))
    loaded, metadata = store.load()
    assert version == store.latest_version() == metadata.version == "v-roundtrip"
    assert isinstance(loaded, type(model))
    # The forest averages its trees in parallel, so the summation order (and
    # the last bit of the result) can differ between two calls.
    assert np.allclose(loaded.predict_proba(features), model.predict_proba(features), rtol=0, atol=1e-12)


def test_no_model_uses_class_weights(training_data) -> None:
    features, labels = training_data
    logistic = LogisticRiskModel().fit(features, labels)
    forest = RandomForestRiskModel().fit(features, labels)
    assert logistic.pipeline_.named_steps["classify"].class_weight is None
    assert forest.forest_.class_weight is None


# ------------------------------------------------------------------- baseline

def test_baseline_predicts_the_training_rate_of_each_lead_bucket(training_data) -> None:
    features, labels = training_data
    scores = LeadTimeBaselineModel().fit(features, labels).predict_proba(features)
    same_day = features["lead_bucket_0"] == 1
    reference = features[list(LEAD_BUCKET_COLUMNS)].sum(axis=1) == 0
    assert scores[same_day.to_numpy()] == pytest.approx(labels[same_day].mean())
    assert scores[reference.to_numpy()] == pytest.approx(labels[reference].mean())
    assert len(np.unique(scores)) <= len(LEAD_BUCKET_COLUMNS) + 1


def test_baseline_falls_back_to_the_overall_rate_for_an_unseen_bucket(training_data) -> None:
    features, labels = training_data
    seen = features["lead_bucket_31+"] == 0
    model = LeadTimeBaselineModel().fit(features[seen], labels[seen])
    unseen_rows = features[~seen]
    assert len(unseen_rows) > 0
    assert model.predict_proba(unseen_rows) == pytest.approx(labels[seen].mean())


# ------------------------------------------------------------------- logistic

def test_logistic_contributions_plus_intercept_equal_the_logit(training_data) -> None:
    features, labels = training_data
    model = LogisticRiskModel().fit(features, labels)
    assert isinstance(model, ExplainableRiskModel)
    contributions = model.contributions(features)
    assert list(contributions.columns) == list(features.columns) and contributions.index.equals(features.index)
    probability = model.predict_proba(features)
    logit = np.log(probability / (1 - probability))
    assert contributions.sum(axis=1).to_numpy() + model.intercept == pytest.approx(logit, abs=1e-9)


def test_logistic_scales_only_the_continuous_columns(training_data) -> None:
    features, labels = training_data
    model = LogisticRiskModel().fit(features, labels)
    scaler = model.pipeline_.named_steps["scale"]
    assert list(scaler.transformers_[0][2]) == list(CONTINUOUS_COLUMNS)
    # For a binary column, contribution = coefficient when the flag is 1 and 0 when it is 0.
    contributions, coefficients = model.contributions(features), model.coefficients()
    flag = features["gender_male"] == 1
    assert contributions.loc[flag, "gender_male"].to_numpy() == pytest.approx(coefficients["gender_male"])
    assert (contributions.loc[~flag, "gender_male"] == 0).all()


def test_odds_ratio_table(training_data) -> None:
    features, labels = training_data
    table = LogisticRiskModel().fit(features, labels).odds_ratios()
    assert set(table["feature"]) == set(FEATURE_COLUMNS)
    assert table["odds_ratio"].to_numpy() == pytest.approx(np.exp(table["coefficient"].to_numpy()))
    assert table["coefficient"].abs().is_monotonic_decreasing
    per_sd = set(table.loc[table["unit"] == "per 1 SD", "feature"])
    assert per_sd == set(CONTINUOUS_COLUMNS)


# ---------------------------------------------------------------------- store

def test_loading_a_bundle_with_a_different_feature_list_is_refused(training_data, tmp_path) -> None:
    features, labels = training_data
    store = ModelStore(str(tmp_path))
    store.save(LeadTimeBaselineModel().fit(features, labels), _metadata("v-old", ["age_squared", "lead_days"]))
    with pytest.raises(ModelBundleError, match="different feature list"):
        store.load()


def test_missing_or_broken_bundles_raise_a_bundle_error(training_data, tmp_path) -> None:
    features, labels = training_data
    store = ModelStore(str(tmp_path))
    with pytest.raises(ModelBundleError, match="no trained model"):
        store.load()
    store.save(LeadTimeBaselineModel().fit(features, labels), _metadata("v-broken"))
    with open(os.path.join(str(tmp_path), "v-broken", MODEL_FILE), "wb") as handle:
        handle.write(b"not a model")
    with pytest.raises(ModelBundleError, match="cannot read model file"):
        store.load()
    with pytest.raises(ModelBundleError, match="cannot read model metadata"):
        store.load("v-missing")


def test_store_keeps_old_versions_and_latest_points_to_the_newest(training_data, tmp_path) -> None:
    features, labels = training_data
    store, model = ModelStore(str(tmp_path)), LeadTimeBaselineModel().fit(features, labels)
    store.save(model, _metadata("v-1"))
    store.save(model, _metadata("v-2"))
    assert store.latest_version() == "v-2"
    assert store.load("v-1")[1].version == "v-1"
    assert os.path.exists(os.path.join(str(tmp_path), LATEST_FILE))


# ------------------------------------------------------------------ selection

def test_selection_prefers_the_simpler_model_within_the_margin() -> None:
    name, reason = select_model({"baseline": 0.690, "logistic": 0.730, "forest": 0.735}, margin=0.01)
    assert name == "logistic" and "does not beat logistic" in reason


def test_selection_keeps_the_baseline_when_nothing_clearly_beats_it() -> None:
    assert select_model({"baseline": 0.700, "logistic": 0.705, "forest": 0.709}, margin=0.01)[0] == "baseline"
    # Exactly on the margin is not "more than" the margin.
    assert select_model({"baseline": 0.70, "logistic": 0.71}, margin=0.01)[0] == "baseline"


def test_selection_takes_the_complex_model_when_it_clearly_wins() -> None:
    assert select_model({"baseline": 0.69, "logistic": 0.73, "forest": 0.76}, margin=0.01)[0] == "forest"
    assert select_model({"baseline": 0.69, "logistic": 0.695, "forest": 0.76}, margin=0.01)[0] == "forest"


# ------------------------------------------------------------- train pipeline

def test_train_pipeline_writes_a_loadable_bundle_and_consistent_artifacts(repository, tmp_path) -> None:
    settings = replace(Settings(), artifacts_dir=str(tmp_path / "artifacts"), models_dir=str(tmp_path / "models"))
    store = ModelStore(settings.models_dir)
    metadata = train(settings, repository, store, ModelEvaluator(min_segment_rows=50))

    model, loaded = store.load()
    assert loaded == metadata and model.name == metadata.model_name
    assert metadata.train_range[1] < metadata.validation_range[0] <= metadata.validation_range[1] < metadata.test_range[0]

    expected = {"metrics.json", "model_comparison.csv", "gains_table.csv", "calibration.csv",
                "segment_metrics.csv", "test_scores.csv", "odds_ratios.csv", "history_block.csv"}
    assert expected <= set(os.listdir(settings.artifacts_dir))

    scores = pd.read_csv(os.path.join(settings.artifacts_dir, "test_scores.csv"))
    assert list(scores.columns) == ["appointment_id", "appointment_date", "no_show", *SCORE_COLUMNS.values()]
    assert scores["appointment_date"].min() >= metadata.test_range[0]
    assert scores["appointment_id"].is_unique

    with open(os.path.join(settings.artifacts_dir, "metrics.json"), "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["selected_model"] == metadata.model_name
    assert report["split"]["test"]["n"] == len(scores)
    assert report["models"][metadata.model_name]["test"] == metadata.test_metrics
    # The stored score column reproduces the stored test AUC: scores and metrics agree.
    column = SCORE_COLUMNS[metadata.model_name]
    recomputed = ModelEvaluator().summary(scores["no_show"], scores[column])["roc_auc"]
    assert recomputed == pytest.approx(metadata.test_metrics["roc_auc"], abs=1e-4)
