"""ReasonExplainer: model contributions to plain-language reasons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from medireach.features.builder import AGE_GROUP_COLUMNS, FLAG_COLUMNS, HISTORY_COLUMNS, LEAD_BUCKET_COLUMNS
from medireach.ml.models import ExplainableRiskModel

# Phrase shown when a column is "on" and pushes the risk up.
REASON_TEXT: dict[str, str] = {
    "lead_bucket_0": "Booked on the day",
    "lead_bucket_4-7": "Booked 4-7 days ahead",
    "lead_bucket_8-14": "Booked 8-14 days ahead",
    "lead_bucket_15-30": "Booked 15-30 days ahead",
    "lead_bucket_31+": "Booked more than a month ahead",
    "age_group_0-17": "Child (0-17), depends on a carer",
    "age_group_18-34": "Younger adult (18-34)",
    "age_group_50-64": "Age 50-64",
    "age_group_65+": "Age 65 or older",
    "sms_received": "SMS reminder already sent",
    "scholarship": "Enrolled in welfare programme (may need support)",
    "hypertension": "Hypertension",
    "diabetes": "Diabetes",
    "alcoholism": "Alcoholism recorded",
    "has_handicap": "Handicap recorded (may need support)",
    "gender_male": "Male patient",
    "is_weekend": "Weekend appointment",
}
HISTORY = "history"
HISTORY_TEXT_MISSED = "Missed a previous appointment"
HISTORY_TEXT_NEW = "No history yet (uncertain)"
HISTORY_TEXT_OTHER = "Pattern of earlier visits"
NO_REASON = "No strong risk factors"
SEPARATOR = "; "

# A concept is a group of columns that describe one thing. One-hot blocks have
# a single active column; the four history columns overlap heavily, so their
# contributions are added up before deciding whether history is a reason.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "lead_time": LEAD_BUCKET_COLUMNS,
    "age": AGE_GROUP_COLUMNS,
    HISTORY: HISTORY_COLUMNS,
    **{column: (column,) for column in FLAG_COLUMNS},
}


class ReasonExplainer:
    """Explains each score with up to three plain-language reasons.

    For logistic regression a feature's contribution is coefficient x value,
    which is exact: the contributions plus the intercept are the log-odds.
    """

    def __init__(self, model: ExplainableRiskModel, top_n: int = 3, min_contribution: float = 0.05) -> None:
        self._model = model
        self._top_n = top_n
        self._min_contribution = min_contribution

    def concept_contributions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Log-odds contribution per concept (the columns of one concept added together)."""
        contributions = self._model.contributions(features)
        return pd.DataFrame({
            concept: contributions[list(columns)].sum(axis=1) for concept, columns in CONCEPTS.items()
        })

    def explain(self, features: pd.DataFrame) -> pd.Series:
        """One string per row: the strongest reasons first, joined with '; '."""
        strength = self.concept_contributions(features).to_numpy()
        phrases = self._phrases(features).to_numpy()
        # Concepts ordered from largest to smallest contribution, per row.
        strongest = np.argsort(-strength, axis=1, kind="stable")[:, : self._top_n]
        reasons = []
        for row, concepts in enumerate(strongest):
            picked = [phrases[row, c] for c in concepts if strength[row, c] > self._min_contribution]
            reasons.append(SEPARATOR.join(picked) if picked else NO_REASON)
        return pd.Series(reasons, index=features.index, name="reasons")

    @staticmethod
    def _phrases(features: pd.DataFrame) -> pd.DataFrame:
        """The phrase each concept would use for each row (same columns as CONCEPTS)."""
        phrases = {}
        for concept, columns in CONCEPTS.items():
            if concept == HISTORY:
                is_new, missed = features["has_history"] == 0, features["had_prior_no_show"] == 1
                phrases[concept] = np.select([is_new, missed], [HISTORY_TEXT_NEW, HISTORY_TEXT_MISSED], HISTORY_TEXT_OTHER)
            else:
                # For a one-hot block, name the column that is switched on. (If
                # none is on, the contribution is zero and the phrase is unused.)
                phrases[concept] = features[list(columns)].idxmax(axis=1).map(REASON_TEXT).to_numpy()
        return pd.DataFrame(phrases, index=features.index)
