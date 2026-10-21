"""Recalibrator and ExperimentAnalyzer: honest planning numbers, and what the hold-out shows."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from medireach.data.analytics import RateComparison, compare_rates
from medireach.domain import HOLDOUT_SHARES, REACHED_OUTCOMES, RELEASED_OUTCOMES, Arm, ImpactAssumptions, PlanStatus, TaskKind
from medireach.impact.pilot import PilotDesigner
from medireach.ops.workflow_store import PlanRepository, TaskRepository

_EPS = 1e-6
MIN_ARM_FOR_OBSERVED_RATE = 30


class Recalibrator:
    """Intercept-only recalibration: shift every log-odds by the same amount.

    The shift makes the average prediction on recent days equal the no-show rate
    actually observed on those days. The order of patients, and so the list,
    does not change; only the planning numbers (expected no-shows) do.
    """

    def __init__(self, shift: float = 0.0) -> None:
        self.shift = shift

    @classmethod
    def fit(cls, predicted: np.ndarray, observed: np.ndarray) -> "Recalibrator":
        if len(predicted) == 0:
            return cls(0.0)
        target = float(np.clip(np.mean(observed), _EPS, 1 - _EPS))
        logits = _logit(np.asarray(predicted, dtype=float))
        gap = lambda shift: float(np.mean(_sigmoid(logits + shift))) - target  # noqa: E731
        return cls(float(brentq(gap, -15.0, 15.0)))

    def apply(self, predicted: np.ndarray | pd.Series) -> np.ndarray:
        return _sigmoid(_logit(np.asarray(predicted, dtype=float)) + self.shift)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class ExperimentSummary:
    """What the randomised hold-out shows so far, plus the operational measures."""

    closed_days: int
    source: str                       # "historical", "simulated", "mixed" or "none"
    patients_call: int
    patients_holdout: int
    no_show_call: float | None
    no_show_holdout: float | None
    comparison: RateComparison | None
    relative_reduction: float | None
    reduction_ci: tuple[float, float] | None
    patients_attempted: int
    patients_reached: int
    reach: float | None
    released_slots: int
    outcome_mix: dict[str, int] = field(default_factory=dict)
    overrides: dict[str, int] = field(default_factory=dict)
    required_call: int | None = None
    required_holdout: int | None = None


def relative_reduction(k_call: int, n_call: int, k_holdout: int, n_holdout: int) -> tuple[float, tuple[float, float]] | None:
    """1 - risk ratio (call arm / hold-out), with a 95% CI from the log risk ratio.

    None when the hold-out has no patients or no no-shows (the ratio is undefined).
    When a count is 0 or equals its arm size, 0.5 is added to each cell (the
    Haldane correction) for both the estimate and the interval, so the estimate
    always lies inside its interval.
    """
    if min(n_call, n_holdout) == 0 or k_holdout == 0:
        return None
    a, b, c, d = float(k_call), float(n_call), float(k_holdout), float(n_holdout)
    if a == 0 or a == b or c == d:
        a, b, c, d = a + 0.5, b + 1.0, c + 0.5, d + 1.0
    ratio = (a / b) / (c / d)
    se = math.sqrt(1 / a - 1 / b + 1 / c - 1 / d)
    z = float(norm.ppf(0.975))
    low, high = ratio * math.exp(-z * se), ratio * math.exp(z * se)
    return 1 - ratio, (1 - high, 1 - low)


class ExperimentAnalyzer:
    """Compares the call arm with the hold-out over all closed plans (intention to treat).

    The unit is a patient on a target day, as randomised: no-show if any of
    their appointments that day was missed. Everyone assigned to the call arm
    counts, whether or not the call was answered.
    """

    def __init__(self, plans: PlanRepository, tasks: TaskRepository, pilot: PilotDesigner,
                 assumptions: ImpactAssumptions) -> None:
        self._plans = plans
        self._tasks = tasks
        self._pilot = pilot
        self._assumptions = assumptions

    def _patient_days(self, plan_ids: list[int]) -> pd.DataFrame:
        frames = [self._plans.items(plan_id).assign(plan_id=plan_id) for plan_id in plan_ids]
        items = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if items.empty:
            return pd.DataFrame(columns=["plan_id", "patient_id", "arm", "no_show", "source", "risk_calibrated"])
        outcomes = self._plans.outcomes(plan_ids)
        joined = items.merge(outcomes, on=["plan_id", "appointment_id"], how="inner")
        return joined.groupby(["plan_id", "patient_id"], as_index=False).agg(
            arm=("arm", "first"), no_show=("no_show", "max"), source=("source", "first"),
            risk_calibrated=("risk_calibrated", "max"),
        )

    def summary(self) -> ExperimentSummary:
        plans = self._plans.list_plans([PlanStatus.APPROVED, PlanStatus.CLOSED])
        closed = [plan.plan_id for plan in plans if plan.status == PlanStatus.CLOSED]
        days = self._patient_days(closed)
        call, holdout = days[days["arm"] == Arm.CALL.value], days[days["arm"] == Arm.HOLDOUT.value]
        k_call, n_call = int(call["no_show"].sum()), len(call)
        k_holdout, n_holdout = int(holdout["no_show"].sum()), len(holdout)
        comparison = compare_rates(k_call, n_call, k_holdout, n_holdout) if n_call and n_holdout else None
        reduction = relative_reduction(k_call, n_call, k_holdout, n_holdout)
        attempted, reached, released, mix = self._operations([plan.plan_id for plan in plans])
        sources = set(days["source"]) if not days.empty else set()
        required = self._required(plans, days)
        return ExperimentSummary(
            closed_days=len(closed),
            source="none" if not sources else (sources.pop() if len(sources) == 1 else "mixed"),
            patients_call=n_call, patients_holdout=n_holdout,
            no_show_call=k_call / n_call if n_call else None,
            no_show_holdout=k_holdout / n_holdout if n_holdout else None,
            comparison=comparison,
            relative_reduction=reduction[0] if reduction else None,
            reduction_ci=reduction[1] if reduction else None,
            patients_attempted=attempted, patients_reached=reached,
            reach=reached / attempted if attempted else None,
            released_slots=released, outcome_mix=mix, overrides=self._overrides([plan.plan_id for plan in plans]),
            required_call=required[0] if required else None, required_holdout=required[1] if required else None,
        )

    def _operations(self, plan_ids: list[int]) -> tuple[int, int, int, dict[str, int]]:
        """Patients with a call attempt, patients reached, released slots, outcome counts."""
        tasks = self._tasks.for_plans(plan_ids)
        done = tasks[tasks["outcome"].notna()]
        calls = done[done["kind"] == TaskKind.CALL.value]
        reached_values = {outcome.value for outcome in REACHED_OUTCOMES}
        per_patient = calls.assign(reached=calls["outcome"].isin(reached_values)).groupby(["plan_id", "patient_id"])["reached"].max()
        released = int(done["outcome"].isin({outcome.value for outcome in RELEASED_OUTCOMES}).sum())
        mix = {str(key): int(value) for key, value in done["outcome"].value_counts().items()}
        return len(per_patient), int(per_patient.sum()), released, mix

    def _overrides(self, plan_ids: list[int]) -> dict[str, int]:
        reasons = [self._plans.items(plan_id)["override_reason"].dropna() for plan_id in plan_ids]
        counts = pd.concat(reasons).value_counts() if reasons and any(len(r) for r in reasons) else pd.Series(dtype=int)
        return {str(key): int(value) for key, value in counts.items()}

    def _required(self, plans: list, days: pd.DataFrame) -> tuple[int, int] | None:
        """Patients needed per arm for 80% power, at the current hold-out share."""
        approved = [plan for plan in plans if plan.holdout_share]
        if not approved:
            return None
        holdout = days[days["arm"] == Arm.HOLDOUT.value]
        if len(holdout) >= MIN_ARM_FOR_OBSERVED_RATE:
            p_control = float(holdout["no_show"].mean())
        else:
            p_control = float(days["risk_calibrated"].mean()) if not days.empty else 0.35
        share = approved[-1].holdout_share or HOLDOUT_SHARES[approved[-1].holdout_mode]
        reduction = self._assumptions.call_reach_rate * self._assumptions.call_effectiveness
        if not 0 < p_control < 1 or not 0 < reduction < 1:
            return None
        return self._pilot.sample_sizes(p_control, reduction, ratio=(1 - share) / share)

