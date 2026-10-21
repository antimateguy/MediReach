"""AnalyticsQueries (reads SQL views) and statistics helpers."""
from __future__ import annotations

import math
from typing import NamedTuple

import pandas as pd
from scipy.stats import norm

from medireach.data.database import SqliteDatabase

VIEW_NAMES: tuple[str, ...] = (
    "v_appointment_base", "v_lead_time", "v_age_group", "v_weekday", "v_clinic",
    "v_sms_by_lead", "v_patient_history", "v_history_effect", "v_weekly_trend",
)
# View names are constants from this file, never user input.
_VIEW_SQL: dict[str, str] = {name: f"SELECT * FROM {name}" for name in VIEW_NAMES}


class RateComparison(NamedTuple):
    """Result of comparing two proportions (group 1 minus group 2)."""

    rate_1: float
    rate_2: float
    difference: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n.

    Preferred over the textbook p +/- z*se interval because it stays inside
    [0, 1] and behaves for small n (for example the 39 Saturday appointments).
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def compare_rates(k1: int, n1: int, k2: int, n2: int) -> RateComparison:
    """Two-proportion z-test: is rate 1 different from rate 2?

    The p-value uses the pooled rate (the null hypothesis says the groups share
    one rate). The 95% interval for the difference uses each group's own rate.
    """
    p1, p2 = k1 / n1, k2 / n2
    difference = p1 - p2
    pooled = (k1 + k2) / (n1 + n2)
    se_null = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = difference / se_null if se_null > 0 else 0.0
    p_value = float(2 * norm.sf(abs(z)))
    se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return RateComparison(p1, p2, difference, difference - 1.96 * se_diff, difference + 1.96 * se_diff, z, p_value)


def with_wilson_ci(frame: pd.DataFrame, k: str = "no_shows", n: str = "n", z: float = 1.96) -> pd.DataFrame:
    """Copy of `frame` with ci_low and ci_high columns for the rate k/n."""
    intervals = [wilson_ci(int(row_k), int(row_n), z) for row_k, row_n in zip(frame[k], frame[n])]
    result = frame.copy()
    result["ci_low"] = [low for low, _ in intervals]
    result["ci_high"] = [high for _, high in intervals]
    return result


class AnalyticsQueries:
    """Read-only access to the analytical views, one method per view."""

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def _view(self, name: str) -> pd.DataFrame:
        return self._database.query(f"select_{name}", _VIEW_SQL[name])

    def appointment_base(self) -> pd.DataFrame:
        return self._view("v_appointment_base")

    def lead_time(self) -> pd.DataFrame:
        return self._view("v_lead_time")

    def age_group(self) -> pd.DataFrame:
        return self._view("v_age_group")

    def weekday(self) -> pd.DataFrame:
        return self._view("v_weekday")

    def clinic(self) -> pd.DataFrame:
        return self._view("v_clinic")

    def sms_by_lead(self) -> pd.DataFrame:
        return self._view("v_sms_by_lead")

    def patient_history(self) -> pd.DataFrame:
        return self._view("v_patient_history")

    def history_effect(self) -> pd.DataFrame:
        return self._view("v_history_effect")

    def weekly_trend(self) -> pd.DataFrame:
        return self._view("v_weekly_trend")

    def overall(self) -> tuple[int, int]:
        """(no_shows, n) across all appointments."""
        sql = "SELECT COALESCE(SUM(no_show), 0) AS no_shows, COUNT(*) AS n FROM appointments"
        row = self._database.query("select_overall_rate", sql).iloc[0]
        return int(row["no_shows"]), int(row["n"])
