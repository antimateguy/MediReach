"""ArtifactStore: the small result files written by the pipelines and read by the app and docs."""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from medireach.errors import DataAccessError

CLEANING_REPORT = "cleaning_report.json"
HYPOTHESES = "hypotheses.json"
METRICS = "metrics.json"
MODEL_COMPARISON = "model_comparison.csv"
GAINS_TABLE = "gains_table.csv"
CALIBRATION = "calibration.csv"
SEGMENT_METRICS = "segment_metrics.csv"
TEST_SCORES = "test_scores.csv"
ODDS_RATIOS = "odds_ratios.csv"
HISTORY_BLOCK = "history_block.csv"
FEATURE_ABLATION = "feature_ablation.csv"
REPLAY_VALIDATION = "replay_validation.json"
IMPACT_COMPARISON = "impact_comparison.csv"
IMPACT_SENSITIVITY = "impact_sensitivity.csv"
IMPACT_SUMMARY = "impact_summary.json"


class ArtifactStore:
    """Reads and writes result files in one folder. Every number in the docs comes from here."""

    def __init__(self, artifacts_dir: str) -> None:
        self._artifacts_dir = artifacts_dir

    def path(self, name: str) -> str:
        return os.path.join(self._artifacts_dir, name)

    def write_json(self, name: str, content: Any) -> None:
        os.makedirs(self._artifacts_dir, exist_ok=True)
        with open(self.path(name), "w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2)

    def write_csv(self, name: str, frame: pd.DataFrame) -> None:
        os.makedirs(self._artifacts_dir, exist_ok=True)
        # Six decimals keeps the files small and stable between runs.
        frame.to_csv(self.path(name), index=False, float_format="%.6f")

    def read_json(self, name: str) -> Any:
        try:
            with open(self.path(name), "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError) as exc:
            raise DataAccessError(f"read_artifact {name}", f"{exc} (run the pipelines first)") from exc

    def read_csv(self, name: str) -> pd.DataFrame:
        try:
            return pd.read_csv(self.path(name))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataAccessError(f"read_artifact {name}", f"{exc} (run the pipelines first)") from exc
