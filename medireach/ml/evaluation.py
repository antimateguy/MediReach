"""ModelEvaluator: ranking metrics, gains, calibration, segment slices."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ArrayLike = Sequence[float] | np.ndarray | pd.Series


def expected_hits_in_top(y_true: ArrayLike, y_score: ArrayLike, k: int) -> float:
    """No-shows among the k highest-scored rows.

    When several rows tie at the cut-off score, an arbitrary pick among them
    would make the answer depend on row order. Instead the tied group
    contributes its average rate for the places that are left. This matters
    for the lead-time baseline, which only has six distinct scores.
    """
    y, score = np.asarray(y_true, dtype=float), np.asarray(y_score, dtype=float)
    if k <= 0:
        return 0.0
    cutoff = np.sort(score)[::-1][min(k, len(score)) - 1]
    above, tied = score > cutoff, score == cutoff
    places_left = min(k, len(score)) - int(above.sum())
    return float(y[above].sum() + places_left * y[tied].mean())


class ModelEvaluator:
    """Scores a model the way it will be used: as a ranking under limited capacity."""

    def __init__(self, k_shares: Sequence[float] = (0.05, 0.10, 0.20, 0.30), min_segment_rows: int = 200) -> None:
        self._k_shares = tuple(k_shares)
        self._min_segment_rows = min_segment_rows

    def precision_at_k(self, y_true: ArrayLike, y_score: ArrayLike, share: float) -> float:
        """Of the top `share` of rows by score, what fraction were no-shows?"""
        k = self._k(len(y_true), share)
        return expected_hits_in_top(y_true, y_score, k) / k

    def capture_rate_at_k(self, y_true: ArrayLike, y_score: ArrayLike, share: float) -> float:
        """What fraction of all no-shows sit in the top `share` of rows by score?"""
        total = float(np.asarray(y_true, dtype=float).sum())
        hits = expected_hits_in_top(y_true, y_score, self._k(len(y_true), share))
        return hits / total if total > 0 else 0.0

    @staticmethod
    def _k(n: int, share: float) -> int:
        return max(1, int(round(share * n)))

    def summary(self, y_true: ArrayLike, y_score: ArrayLike) -> dict[str, float]:
        """Headline metrics for one model on one period."""
        y, score = np.asarray(y_true, dtype=float), np.asarray(y_score, dtype=float)
        base_rate = float(y.mean())
        metrics = {
            "n": float(len(y)),
            "base_rate": base_rate,
            "roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)),
            "brier": float(brier_score_loss(y, score)),
        }
        for share in self._k_shares:
            label = f"{int(round(100 * share))}"
            metrics[f"precision_at_{label}"] = self.precision_at_k(y, score, share)
            metrics[f"capture_at_{label}"] = self.capture_rate_at_k(y, score, share)
        metrics["lift_top_decile"] = self.precision_at_k(y, score, 0.10) / base_rate
        return metrics

    def gains_table(self, y_true: ArrayLike, y_score: ArrayLike, bins: int = 10) -> pd.DataFrame:
        """Deciles from highest to lowest score: rate, lift, and cumulative share of no-shows."""
        y = np.asarray(y_true, dtype=float)
        n, total, base_rate = len(y), float(y.sum()), float(y.mean())
        cut_points = [int(round(n * step / bins)) for step in range(bins + 1)]
        cumulative_hits = [expected_hits_in_top(y, y_score, k) for k in cut_points]
        rows = []
        for decile in range(1, bins + 1):
            size = cut_points[decile] - cut_points[decile - 1]
            hits = cumulative_hits[decile] - cumulative_hits[decile - 1]
            rows.append({
                "decile": decile, "n": size, "no_shows": hits, "no_show_rate": hits / size,
                "lift": hits / size / base_rate, "cumulative_capture": cumulative_hits[decile] / total,
            })
        return pd.DataFrame(rows)

    def calibration_table(self, y_true: ArrayLike, y_score: ArrayLike, bins: int = 10) -> pd.DataFrame:
        """Ten equal-sized groups by predicted risk: does predicted match observed?"""
        frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "score": np.asarray(y_score, dtype=float)})
        # Ranking first gives equal-sized groups even when many scores are identical.
        frame["bin"] = pd.qcut(frame["score"].rank(method="first"), q=bins, labels=False) + 1
        table = frame.groupby("bin").agg(n=("y", "size"), mean_predicted=("score", "mean"), observed_rate=("y", "mean"))
        table["gap"] = table["mean_predicted"] - table["observed_rate"]
        return table.reset_index()

    def segment_table(self, y_true: ArrayLike, y_score: ArrayLike, segments: pd.DataFrame) -> pd.DataFrame:
        """ROC-AUC and base rate within each value of each segment column (e.g. gender)."""
        frame = segments.reset_index(drop=True).copy()
        frame["_y"], frame["_score"] = np.asarray(y_true, dtype=float), np.asarray(y_score, dtype=float)
        rows = []
        for column in segments.columns:
            for value, group in frame.groupby(column, sort=True):
                # AUC needs both outcomes present and enough rows to be stable.
                if len(group) < self._min_segment_rows or group["_y"].nunique() < 2:
                    continue
                rows.append({
                    "segment": column, "value": str(value), "n": len(group),
                    "base_rate": float(group["_y"].mean()),
                    "roc_auc": float(roc_auc_score(group["_y"], group["_score"])),
                })
        return pd.DataFrame(rows, columns=["segment", "value", "n", "base_rate", "roc_auc"])
