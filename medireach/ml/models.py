"""RiskModel interface and the three interchangeable models."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from medireach.features.builder import CONTINUOUS_COLUMNS, LEAD_BUCKET_COLUMNS


class RiskModel(ABC):
    """Anything that can learn from (X, y) and return P(no_show) per row."""

    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RiskModel":
        """Learn from the training rows and return self."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(no_show) for each row, shape (n,)."""


class ExplainableRiskModel(RiskModel, ABC):
    """A risk model that can also say how much each feature moved a score."""

    @abstractmethod
    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Signed log-odds contribution of each feature, same shape and columns as X."""


class LeadTimeBaselineModel(RiskModel):
    """The SQL insight as a model: predict the training no-show rate of the lead bucket.

    This is the benchmark. A machine-learning model has to beat it to be worth
    its extra complexity.
    """

    name = "lead_time_baseline"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LeadTimeBaselineModel":
        target = y.to_numpy()
        in_reference = (X[list(LEAD_BUCKET_COLUMNS)].sum(axis=1) == 0).to_numpy()
        overall = float(target.mean())
        # A bucket with no training rows falls back to the overall rate.
        self.reference_rate_ = float(target[in_reference].mean()) if in_reference.any() else overall
        self.bucket_rates_ = {}
        for column in LEAD_BUCKET_COLUMNS:
            in_bucket = (X[column] == 1).to_numpy()
            self.bucket_rates_[column] = float(target[in_bucket].mean()) if in_bucket.any() else overall
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = np.full(len(X), self.reference_rate_)
        for column, rate in self.bucket_rates_.items():
            scores[(X[column] == 1).to_numpy()] = rate
        return scores


class LogisticRiskModel(ExplainableRiskModel):
    """Logistic regression. The scaler lives inside the model, so it cannot be mismatched.

    Only the continuous columns are standardised. Binary columns pass through
    untouched, so each of their coefficients is a plain log odds ratio.
    """

    name = "logistic_regression"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRiskModel":
        self.feature_columns_ = list(X.columns)
        scaled = [column for column in self.feature_columns_ if column in CONTINUOUS_COLUMNS]
        scaler = ColumnTransformer(
            [("continuous", StandardScaler(), scaled)],
            remainder="passthrough", verbose_feature_names_out=False,
        )
        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=self._seed)
        self.pipeline_ = Pipeline([("scale", scaler), ("classify", classifier)]).fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X[self.feature_columns_])[:, 1]

    @property
    def intercept(self) -> float:
        return float(self.pipeline_.named_steps["classify"].intercept_[0])

    def coefficients(self) -> pd.Series:
        """One coefficient per feature, in the order of the training columns."""
        names = self.pipeline_.named_steps["scale"].get_feature_names_out()
        values = self.pipeline_.named_steps["classify"].coef_[0]
        return pd.Series(values, index=names)[self.feature_columns_]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """coefficient x transformed value. Row sums plus the intercept equal the logit."""
        scaler = self.pipeline_.named_steps["scale"]
        transformed = pd.DataFrame(
            scaler.transform(X[self.feature_columns_]), index=X.index, columns=scaler.get_feature_names_out(),
        )
        return (transformed * self.coefficients())[self.feature_columns_]

    def odds_ratios(self) -> pd.DataFrame:
        """Coefficients as odds ratios, largest effect first."""
        coefficients = self.coefficients()
        table = pd.DataFrame({
            "feature": coefficients.index,
            "coefficient": coefficients.to_numpy(),
            "odds_ratio": np.exp(coefficients.to_numpy()),
            # Standardised columns are read "per one standard deviation".
            "unit": ["per 1 SD" if name in CONTINUOUS_COLUMNS else "yes vs no" for name in coefficients.index],
        })
        order = table["coefficient"].abs().sort_values(ascending=False).index
        return table.loc[order].reset_index(drop=True)


class RandomForestRiskModel(RiskModel):
    """Random forest (bagged trees), kept shallow. A comparison model, not the default."""

    name = "random_forest"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestRiskModel":
        self.feature_columns_ = list(X.columns)
        self.forest_ = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=50, random_state=self._seed, n_jobs=-1,
        ).fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.forest_.predict_proba(X[self.feature_columns_])[:, 1]
