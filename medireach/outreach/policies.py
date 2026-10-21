"""OutreachPolicy interface with threshold and capacity policies."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from medireach.domain import Action, Priority

# Each priority maps to exactly one recommended action.
ACTION_FOR: dict[Priority, Action] = {
    Priority.HIGH: Action.CALL,
    Priority.MEDIUM: Action.SMS,
    Priority.LOW: Action.NONE,
}
REQUIRED_COLUMNS: tuple[str, ...] = ("appointment_id", "appointment_date", "risk_score")


class OutreachPolicy(ABC):
    """Turns risk scores into a priority and a recommended action per appointment."""

    name: str

    @abstractmethod
    def assign(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `scored` with day_rank, priority and action added.

        `scored` needs appointment_id, appointment_date and risk_score. The
        input is never modified and no row is added, dropped or reordered.
        """

    @staticmethod
    def _ranked(scored: pd.DataFrame) -> pd.DataFrame:
        """Copy with day_rank: 1 = highest risk that day; ties go to the lower appointment_id."""
        missing = sorted(set(REQUIRED_COLUMNS) - set(scored.columns))
        if missing:
            raise ValueError(f"scored appointments are missing columns: {missing}")
        result = scored.copy()
        order = result.sort_values(
            ["appointment_date", "risk_score", "appointment_id"], ascending=[True, False, True],
        )
        result["day_rank"] = order.groupby("appointment_date").cumcount().add(1).reindex(result.index)
        return result

    @staticmethod
    def _with_actions(ranked: pd.DataFrame, high: pd.Series, medium: pd.Series) -> pd.DataFrame:
        ranked["priority"] = np.select([high, medium], [Priority.HIGH.value, Priority.MEDIUM.value], Priority.LOW.value)
        ranked["action"] = ranked["priority"].map({p.value: a.value for p, a in ACTION_FOR.items()})
        return ranked


class ThresholdPolicy(OutreachPolicy):
    """Fixed probability cut-offs. The naive baseline: the workload changes every day."""

    name = "threshold"

    def __init__(self, high: float = 0.50, medium: float = 0.35) -> None:
        if not 0.0 <= medium <= high <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= medium <= high <= 1")
        self.high, self.medium = high, medium

    def assign(self, scored: pd.DataFrame) -> pd.DataFrame:
        ranked = self._ranked(scored)
        # A score exactly on a cut-off belongs to the higher tier.
        high = ranked["risk_score"] >= self.high
        medium = ~high & (ranked["risk_score"] >= self.medium)
        return self._with_actions(ranked, high, medium)


class FixedCountPolicy(OutreachPolicy):
    """Call the top `calls` of each day and SMS the next `sms`: capacity in people, not percent.

    This is how a manager plans: receptionists on shift x calls each = calls.
    """

    name = "fixed_count"

    def __init__(self, calls: int, sms: int) -> None:
        if calls < 0 or sms < 0:
            raise ValueError("calls and sms must not be negative")
        self.calls, self.sms = calls, sms

    def assign(self, scored: pd.DataFrame) -> pd.DataFrame:
        ranked = self._ranked(scored)
        high = ranked["day_rank"] <= self.calls
        medium = ~high & (ranked["day_rank"] <= self.calls + self.sms)
        return self._with_actions(ranked, high, medium)


class CapacityPolicy(OutreachPolicy):
    """Call the top share of each day's list, SMS the next share. The workload is predictable."""

    name = "capacity"

    def __init__(self, call_share: float = 0.10, sms_share: float = 0.20) -> None:
        if call_share < 0 or sms_share < 0 or call_share + sms_share > 1:
            raise ValueError("shares must be non-negative and sum to at most 1")
        self.call_share, self.sms_share = call_share, sms_share

    @staticmethod
    def _quota(share: float, day_size: pd.Series) -> pd.Series:
        # Round before ceil: 0.1 * 30 is 3.0000000000000004 in floating point,
        # and a plain ceil would turn that into 4 calls instead of 3.
        return (share * day_size).round(9).map(math.ceil)

    def assign(self, scored: pd.DataFrame) -> pd.DataFrame:
        ranked = self._ranked(scored)
        day_size = ranked.groupby("appointment_date")["appointment_id"].transform("size")
        calls = self._quota(self.call_share, day_size)
        texts = self._quota(self.sms_share, day_size)
        high = ranked["day_rank"] <= calls
        medium = ~high & (ranked["day_rank"] <= calls + texts)
        return self._with_actions(ranked, high, medium)
