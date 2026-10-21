"""FeatureAblation and ClinicEncoder: what each kind of information is worth (validation period only)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from medireach.features.builder import AGE_GROUP_COLUMNS, HISTORY_COLUMNS, LEAD_BUCKET_COLUMNS
from medireach.ml.evaluation import ModelEvaluator
from medireach.ml.models import LeadTimeBaselineModel, LogisticRiskModel, RandomForestRiskModel, RiskModel

CLINIC_COLUMN = "clinic_rate"
# Feature groups in the order they are added. Together they are the full feature list.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Lead time", LEAD_BUCKET_COLUMNS),
    ("Patient history", HISTORY_COLUMNS),
    ("Age and gender", AGE_GROUP_COLUMNS + ("gender_male",)),
    ("Health and welfare flags", ("scholarship", "hypertension", "diabetes", "alcoholism", "has_handicap")),
    ("SMS reminder", ("sms_received",)),
    ("Weekend", ("is_weekend",)),
)
COLUMNS = ["study", "step", "label", "model", "n_features", "roc_auc", "precision_at_10", "gain_roc_auc"]


class ClinicEncoder:
    """Leakage-safe target encoding of the clinic: each clinic's no-show rate on the training rows only,
    pulled towards the overall training rate when the clinic has few appointments."""

    def __init__(self, strength: float = 50.0) -> None:
        self._strength = strength

    def fit(self, clinics: pd.Series, labels: pd.Series) -> "ClinicEncoder":
        self.prior_ = float(labels.mean())
        stats = pd.DataFrame({"clinic": clinics.to_numpy(), "y": labels.to_numpy()}).groupby("clinic")["y"].agg(["sum", "count"])
        self.rates_ = (stats["sum"] + self._strength * self.prior_) / (stats["count"] + self._strength)
        return self

    def transform(self, clinics: pd.Series) -> pd.Series:
        """Encoded rate per row; clinics never seen in training get the overall rate."""
        return clinics.map(self.rates_).fillna(self.prior_).astype("float64")


class FeatureAblation:
    """Fits logistic regression on growing and shrinking feature sets and scores each on validation.

    The test period is never touched here, so these studies cannot influence the reported results.
    """

    def __init__(self, evaluator: ModelEvaluator, seed: int = 42) -> None:
        self._evaluator = evaluator
        self._seed = seed

    def _score(self, model: RiskModel, X_train: pd.DataFrame, y_train: pd.Series,
               X_valid: pd.DataFrame, y_valid: pd.Series) -> tuple[float, float]:
        scores = model.fit(X_train, y_train).predict_proba(X_valid)
        summary = self._evaluator.summary(y_valid, scores)
        return summary["roc_auc"], summary["precision_at_10"]

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series,
            clinic_train: pd.Series | None = None, clinic_valid: pd.Series | None = None) -> pd.DataFrame:
        rows = self._stepwise(X_train, y_train, X_valid, y_valid)
        rows += self._leave_one_out(X_train, y_train, X_valid, y_valid)
        rows += self._grid(X_train, y_train, X_valid, y_valid)
        if clinic_train is not None and clinic_valid is not None:
            rows += self._clinic(X_train, y_train, X_valid, y_valid, clinic_train, clinic_valid, rows)
        return pd.DataFrame(rows, columns=COLUMNS)

    def _stepwise(self, X_train, y_train, X_valid, y_valid) -> list[dict[str, object]]:
        rows, columns, previous = [], [], None
        for step, (label, group) in enumerate(GROUPS, start=1):
            columns += list(group)
            auc, precision = self._score(LogisticRiskModel(self._seed), X_train[columns], y_train, X_valid[columns], y_valid)
            rows.append(self._row("added one at a time", step, f"+ {label}" if step > 1 else label,
                                  "logistic_regression", len(columns), auc, precision, previous))
            previous = auc
        return rows

    def _leave_one_out(self, X_train, y_train, X_valid, y_valid) -> list[dict[str, object]]:
        full_auc, _ = self._score(LogisticRiskModel(self._seed), X_train, y_train, X_valid, y_valid)
        rows = []
        for step, (label, group) in enumerate(GROUPS, start=1):
            kept = [column for column in X_train.columns if column not in group]
            auc, precision = self._score(LogisticRiskModel(self._seed), X_train[kept], y_train, X_valid[kept], y_valid)
            rows.append(self._row("left out one at a time", step, f"without {label}", "logistic_regression",
                                  len(kept), auc, precision, full_auc))
        return rows

    def _grid(self, X_train, y_train, X_valid, y_valid) -> list[dict[str, object]]:
        """Does the gain come from a more complex model or from more information?"""
        lead = list(LEAD_BUCKET_COLUMNS)
        candidates: Sequence[tuple[str, RiskModel, list[str]]] = (
            ("Lead time only", LeadTimeBaselineModel(), lead),
            ("Lead time only", LogisticRiskModel(self._seed), lead),
            ("Lead time only", RandomForestRiskModel(self._seed), lead),
            ("All features", LogisticRiskModel(self._seed), list(X_train.columns)),
            ("All features", RandomForestRiskModel(self._seed), list(X_train.columns)),
        )
        rows = []
        for step, (label, model, columns) in enumerate(candidates, start=1):
            auc, precision = self._score(model, X_train[columns], y_train, X_valid[columns], y_valid)
            rows.append(self._row("model x information", step, label, model.name, len(columns), auc, precision, None))
        return rows

    def _clinic(self, X_train, y_train, X_valid, y_valid, clinic_train, clinic_valid,
                rows: list[dict[str, object]]) -> list[dict[str, object]]:
        """Would the clinic location earn a place? Encoded from training rows only."""
        encoder = ClinicEncoder().fit(clinic_train, y_train)
        train = X_train.assign(**{CLINIC_COLUMN: encoder.transform(clinic_train).to_numpy()})
        valid = X_valid.assign(**{CLINIC_COLUMN: encoder.transform(clinic_valid).to_numpy()})
        auc, precision = self._score(LogisticRiskModel(self._seed), train, y_train, valid, y_valid)
        full_auc = next(row["roc_auc"] for row in reversed(rows) if row["study"] == "added one at a time")
        return [self._row("added one at a time", len(GROUPS) + 1, "+ Clinic location (encoded)",
                          "logistic_regression", train.shape[1], auc, precision, full_auc)]

    @staticmethod
    def _row(study: str, step: int, label: str, model: str, n_features: int, auc: float, precision: float,
             reference: float | None) -> dict[str, object]:
        return {"study": study, "step": step, "label": label, "model": model, "n_features": n_features,
                "roc_auc": auc, "precision_at_10": precision,
                "gain_roc_auc": np.nan if reference is None else auc - reference}
