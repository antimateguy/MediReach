"""PilotDesigner: sample size and plan for a randomised pilot."""
from __future__ import annotations

import math

from scipy.stats import norm

from medireach.domain import PilotPlan

GUARDRAILS: tuple[str, ...] = (
    "Randomise only within the HIGH list; everyone else is handled as usual.",
    "Never withhold care: the control arm keeps every existing reminder.",
    "If the clinic requires it, give the control arm a delayed call instead of no call.",
    "Fix the sample size and the primary metric before starting; do not stop early on a good week.",
    "Compare arms as randomised (intention to treat), whether or not the call was answered.",
)


class PilotDesigner:
    """Sizes the experiment that would measure what a call really does.

    The backtest can only assume an effectiveness. A randomised pilot measures
    it: split the HIGH list at random into call and no-call, compare no-show rates.
    """

    @staticmethod
    def sample_size_per_arm(p_control: float, relative_reduction: float,
                            alpha: float = 0.05, power: float = 0.8) -> int:
        """Patients needed in each arm to detect the reduction (two-sided two-proportion test)."""
        if not 0 < p_control < 1 or not 0 < relative_reduction < 1:
            raise ValueError("p_control and relative_reduction must be strictly between 0 and 1")
        p_treated = p_control * (1 - relative_reduction)
        pooled = (p_control + p_treated) / 2
        z_alpha = norm.ppf(1 - alpha / 2)
        z_power = norm.ppf(power)
        # Under "no difference" both arms share the pooled rate; under the
        # alternative each arm has its own. The formula covers both.
        null_spread = z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        alternative_spread = z_power * math.sqrt(p_control * (1 - p_control) + p_treated * (1 - p_treated))
        return math.ceil((null_spread + alternative_spread) ** 2 / (p_control - p_treated) ** 2)

    @staticmethod
    def sample_sizes(p_control: float, relative_reduction: float, ratio: float = 1.0,
                     alpha: float = 0.05, power: float = 0.8) -> tuple[int, int]:
        """(called, held out) patients needed when `ratio` called patients are kept per held-out one.

        With ratio 1 this equals sample_size_per_arm. A 10% hold-out is ratio 9:
        far fewer held-out patients, but more patients in total.
        """
        if not 0 < p_control < 1 or not 0 < relative_reduction < 1 or ratio <= 0:
            raise ValueError("p_control and relative_reduction must be in (0, 1) and ratio positive")
        p_treated = p_control * (1 - relative_reduction)
        pooled = (p_control + ratio * p_treated) / (1 + ratio)
        z_alpha, z_power = norm.ppf(1 - alpha / 2), norm.ppf(power)
        null_spread = z_alpha * math.sqrt(pooled * (1 - pooled) * (1 + 1 / ratio))
        alternative_spread = z_power * math.sqrt(p_treated * (1 - p_treated) / ratio + p_control * (1 - p_control))
        control = math.ceil((null_spread + alternative_spread) ** 2 / (p_control - p_treated) ** 2)
        return math.ceil(ratio * control), control

    def plan(self, p_control: float, relative_reduction: float, daily_high_volume: float,
             alpha: float = 0.05, power: float = 0.8) -> PilotPlan:
        """A full plan: arm size, total sample and how many clinic days it takes."""
        if daily_high_volume <= 0:
            raise ValueError("daily_high_volume must be positive")
        per_arm = self.sample_size_per_arm(p_control, relative_reduction, alpha, power)
        total = 2 * per_arm
        return PilotPlan(
            p_control=p_control, relative_reduction=relative_reduction, alpha=alpha, power=power,
            sample_size_per_arm=per_arm, total_sample=total, daily_high_volume=daily_high_volume,
            estimated_days=math.ceil(total / daily_high_volume), guardrails=GUARDRAILS,
        )
