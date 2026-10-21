"""Pipeline: features, split, fit, evaluate, select, save."""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import sklearn

from medireach import artifacts as files
from medireach.artifacts import ArtifactStore
from medireach.bootstrap import build_container, configure_logging
from medireach.config import Settings
from medireach.data.repositories import DATE_FORMAT, AppointmentRepository
from medireach.domain import ModelMetadata
from medireach.features.builder import (
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    HISTORY_COLUMNS,
    FeatureBuilder,
    Split,
    bucket,
    time_split,
)
from medireach.ml.ablation import FeatureAblation
from medireach.ml.evaluation import ModelEvaluator
from medireach.ml.models import LeadTimeBaselineModel, LogisticRiskModel, RandomForestRiskModel, RiskModel
from medireach.ml.store import ModelStore

logger = logging.getLogger(__name__)

# Column names used for each model in test_scores.csv.
SCORE_COLUMNS: dict[str, str] = {
    "lead_time_baseline": "score_baseline",
    "logistic_regression": "score_lr",
    "random_forest": "score_rf",
}
COMPARISON_METRICS = ["roc_auc", "pr_auc", "brier", "precision_at_10", "capture_at_10", "capture_at_30", "lift_top_decile"]
PERIODS = ("train", "validation", "test")


@dataclass(frozen=True)
class TrainingData:
    """Features, labels and the chronological split, built once and shared by all models."""

    features: pd.DataFrame
    meta: pd.DataFrame
    segments: pd.DataFrame
    split: Split
    prior_rate: float
    clinics: pd.Series


def candidate_models(seed: int) -> list[RiskModel]:
    """The candidates, ordered from simplest to most complex (the order matters for selection)."""
    return [LeadTimeBaselineModel(), LogisticRiskModel(seed), RandomForestRiskModel(seed)]


def prepare(settings: Settings, repository: AppointmentRepository) -> TrainingData:
    appointments = repository.get_appointments()
    split = time_split(appointments, settings.split_fractions)
    # The smoothing prior uses training rows only, so validation and test
    # outcomes cannot reach the features through it.
    prior_rate = float(appointments.loc[split.train, "no_show"].mean())
    features, meta = FeatureBuilder(settings, prior_rate).build(appointments)
    segments = pd.DataFrame({
        "gender": appointments["gender"].to_numpy(),
        "age_group": bucket(appointments["age"], settings.age_group_edges, settings.age_group_labels).to_numpy(),
        # Same-day bookings are easy to rank (they rarely no-show), so the
        # overall AUC flatters the model. This slice shows it without them.
        "booking": appointments["lead_days"].map(lambda days: "same_day" if days == 0 else "booked_ahead").to_numpy(),
    })
    return TrainingData(features, meta, segments, split, prior_rate, appointments["clinic"].reset_index(drop=True))


def select_model(validation_auc: dict[str, float], margin: float) -> tuple[str, str]:
    """Simplicity rule: a more complex model replaces the current choice only if it
    beats it by more than `margin` ROC-AUC on validation. Keys must be ordered simplest first."""
    names = list(validation_auc)
    chosen, notes = names[0], []
    for candidate in names[1:]:
        # Rounded so that floating-point noise (0.71 - 0.70 = 0.010000000000000009)
        # cannot push a tie over the margin.
        gain = round(validation_auc[candidate] - validation_auc[chosen], 6)
        if gain > margin:
            notes.append(f"{candidate} ({validation_auc[candidate]:.4f}) beats {chosen} "
                         f"({validation_auc[chosen]:.4f}) by {gain:.4f}, more than {margin}")
            chosen = candidate
        else:
            notes.append(f"{candidate} ({validation_auc[candidate]:.4f}) does not beat {chosen} "
                         f"({validation_auc[chosen]:.4f}) by more than {margin} (gain {gain:+.4f})")
    return chosen, f"Selected {chosen} on validation ROC-AUC. " + "; ".join(notes) + "."


def _date_range(meta: pd.DataFrame, mask: pd.Series) -> tuple[str, str]:
    dates = meta.loc[mask, "appointment_date"]
    return dates.min().strftime(DATE_FORMAT), dates.max().strftime(DATE_FORMAT)


def _split_summary(data: TrainingData) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for period, mask in zip(PERIODS, data.split):
        start, end = _date_range(data.meta, mask)
        summary[period] = {
            "start": start, "end": end, "n": int(mask.sum()),
            "share_of_rows": float(mask.mean()),
            "base_rate": float(data.meta.loc[mask, "no_show"].mean()),
            "share_with_history": float(data.features.loc[mask, "has_history"].mean()),
        }
    return summary


def _test_tables(data: TrainingData, scores: dict[str, pd.Series], evaluator: ModelEvaluator) -> dict[str, pd.DataFrame]:
    """Per-model gains, calibration and segment tables on the test period, stacked."""
    test = data.split.test
    y_test = data.meta.loc[test, "no_show"]
    gains, calibration, segments = [], [], []
    for name, score in scores.items():
        gains.append(evaluator.gains_table(y_test, score[test]).assign(model=name))
        calibration.append(evaluator.calibration_table(y_test, score[test]).assign(model=name))
        segments.append(evaluator.segment_table(y_test, score[test], data.segments[test.to_numpy()]).assign(model=name))
    test_scores = data.meta.loc[test, ["appointment_id", "appointment_date", "no_show"]].copy()
    for name, score in scores.items():
        test_scores[SCORE_COLUMNS[name]] = score[test]
    return {
        files.GAINS_TABLE: pd.concat(gains, ignore_index=True),
        files.CALIBRATION: pd.concat(calibration, ignore_index=True),
        files.SEGMENT_METRICS: pd.concat(segments, ignore_index=True),
        files.TEST_SCORES: test_scores,
    }


def _comparison_table(metrics: dict[str, dict[str, dict[str, float]]], selected: str) -> pd.DataFrame:
    rows = []
    for name, by_period in metrics.items():
        for period in ("validation", "test"):
            row = {"model": name, "period": period, "selected": int(name == selected),
                   "n": int(by_period[period]["n"]), "base_rate": by_period[period]["base_rate"]}
            row.update({metric: by_period[period][metric] for metric in COMPARISON_METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def _odds_ratio_table(logistic: LogisticRiskModel) -> pd.DataFrame:
    table = logistic.odds_ratios()
    table.insert(1, "description", table["feature"].map(FEATURE_DESCRIPTIONS))
    return table


def _history_block_table(data: TrainingData, logistic: LogisticRiskModel) -> pd.DataFrame:
    """Net effect of the four history columns together, on the test period.

    The columns overlap, so their individual coefficients cannot be read one
    by one (one of them even has a negative sign). Their sum is what the model
    actually does with a patient's history.
    """
    test = data.split.test
    features = data.features[test]
    net = logistic.contributions(features)[list(HISTORY_COLUMNS)].sum(axis=1)
    group = pd.Series("Earlier visits, none missed", index=features.index)
    group[features["had_prior_no_show"] == 1] = "Earlier visits, at least one missed"
    group[features["has_history"] == 0] = "No earlier visits"
    frame = pd.DataFrame({"group": group, "net_log_odds": net, "no_show": data.meta.loc[test, "no_show"]})
    return frame.groupby("group", sort=True).agg(
        n=("no_show", "size"), observed_rate=("no_show", "mean"), mean_net_log_odds=("net_log_odds", "mean"),
    ).reset_index()


def train(settings: Settings, repository: AppointmentRepository, store: ModelStore,
          evaluator: ModelEvaluator) -> ModelMetadata:
    """Fit all candidates on train, select on validation, then look at test exactly once."""
    data = prepare(settings, repository)
    labels = data.meta["no_show"]
    models = {model.name: model for model in candidate_models(settings.seed)}
    scores: dict[str, pd.Series] = {}
    for name, model in models.items():
        model.fit(data.features[data.split.train], labels[data.split.train])
        scores[name] = pd.Series(model.predict_proba(data.features), index=data.features.index)
        logger.info("Fitted %s", name)

    validation = {name: evaluator.summary(labels[data.split.validation], score[data.split.validation])
                  for name, score in scores.items()}
    selected, reason = select_model({name: m["roc_auc"] for name, m in validation.items()}, settings.selection_margin)

    # Selection is final from here on. The test period is evaluated once, for reporting.
    test = {name: evaluator.summary(labels[data.split.test], score[data.split.test]) for name, score in scores.items()}
    metrics = {name: {"validation": validation[name], "test": test[name]} for name in models}

    trained_at = datetime.now(timezone.utc)
    metadata = ModelMetadata(
        version=trained_at.strftime("%Y%m%d-%H%M%S"), model_name=selected,
        feature_columns=list(FEATURE_COLUMNS), trained_at=trained_at.isoformat(timespec="seconds"),
        train_range=_date_range(data.meta, data.split.train),
        validation_range=_date_range(data.meta, data.split.validation),
        test_range=_date_range(data.meta, data.split.test),
        sklearn_version=sklearn.__version__, python_version=platform.python_version(),
        validation_metrics=validation[selected], test_metrics=test[selected],
        selection_reason=reason, history_prior_rate=data.prior_rate,
    )
    store.save(models[selected], metadata)
    _write_artifacts(ArtifactStore(settings.artifacts_dir), data, scores, evaluator, metrics, models, metadata, settings.seed)
    return metadata


def _ablation(data: TrainingData, evaluator: ModelEvaluator, seed: int) -> pd.DataFrame:
    """Feature-group studies on the validation period (the test period is not used)."""
    train, valid = data.split.train, data.split.validation
    labels = data.meta["no_show"]
    return FeatureAblation(evaluator, seed).run(
        data.features[train], labels[train], data.features[valid], labels[valid],
        clinic_train=data.clinics[train], clinic_valid=data.clinics[valid],
    )


def _write_artifacts(store: ArtifactStore, data: TrainingData, scores: dict[str, pd.Series], evaluator: ModelEvaluator,
                     metrics: dict[str, dict[str, dict[str, float]]], models: dict[str, RiskModel],
                     metadata: ModelMetadata, seed: int = 42) -> None:
    tables = _test_tables(data, scores, evaluator)
    tables[files.MODEL_COMPARISON] = _comparison_table(metrics, metadata.model_name)
    tables[files.FEATURE_ABLATION] = _ablation(data, evaluator, seed)
    for model in models.values():
        if isinstance(model, LogisticRiskModel):
            tables[files.ODDS_RATIOS] = _odds_ratio_table(model)
            tables[files.HISTORY_BLOCK] = _history_block_table(data, model)
    for name, table in tables.items():
        store.write_csv(name, table)
    report = {
        "selected_model": metadata.model_name, "selection_reason": metadata.selection_reason,
        "model_version": metadata.version, "history_prior_rate": data.prior_rate,
        "split": _split_summary(data), "models": metrics,
    }
    store.write_json(files.METRICS, report)
    logger.info("Wrote training artifacts to %s", store.path(""))


def print_summary(artifacts_dir: str) -> None:
    """Print the comparison table and the selection reason from the files just written."""
    store = ArtifactStore(artifacts_dir)
    comparison = store.read_csv(files.MODEL_COMPARISON)
    columns = ["model", "period", "roc_auc", "pr_auc", "precision_at_10", "lift_top_decile", "selected"]
    print(comparison[columns].round(4).to_string(index=False))
    print(store.read_json(files.METRICS)["selection_reason"])


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings)
    container = build_container(settings, require_model=False)
    train(settings, container.appointments, container.model_store, container.evaluator)
    print_summary(settings.artifacts_dir)


if __name__ == "__main__":
    main()
