"""ImpactSimulator: assumption-driven benefit estimate."""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from medireach.domain import Action, ImpactAssumptions, ImpactResult
from medireach.outreach.policies import CapacityPolicy, OutreachPolicy

RANDOM = "Random (no targeting)"
ORACLE = "Oracle (perfect knowledge)"
_ID_COLUMNS = ["appointment_id", "appointment_date", "no_show"]


class ImpactSimulator:
    """Estimates what an outreach policy is worth on a period where outcomes are known.

    The data is observational, so the simulator does not know how well a call
    works. It counts how many real no-shows each strategy would have contacted,
    then applies explicit, editable assumptions about reach and effectiveness.
    """

    def evaluate(self, scored_with_outcomes: pd.DataFrame, policy: OutreachPolicy,
                 assumptions: ImpactAssumptions, strategy: str = "model") -> ImpactResult:
        """Apply `policy` to rows with risk_score and no_show, and value the result."""
        assigned = policy.assign(scored_with_outcomes)
        called = assigned["action"] == Action.CALL.value
        texted = assigned["action"] == Action.SMS.value
        return self._result(
            strategy, assumptions, n=len(assigned), calls=int(called.sum()), texts=int(texted.sum()),
            no_shows=int(assigned["no_show"].sum()),
            no_shows_called=float(assigned.loc[called, "no_show"].sum()),
            no_shows_texted=float(assigned.loc[texted, "no_show"].sum()),
        )

    @staticmethod
    def _result(strategy: str, assumptions: ImpactAssumptions, n: int, calls: int, texts: int,
                no_shows: int, no_shows_called: float, no_shows_texted: float) -> ImpactResult:
        """The money arithmetic, shared by every strategy."""
        # A no-show is prevented only if the contact reaches the patient AND works.
        prevented = (no_shows_called * assumptions.call_reach_rate * assumptions.call_effectiveness
                     + no_shows_texted * assumptions.sms_reach_rate * assumptions.sms_effectiveness)
        cost = calls * assumptions.cost_per_call + texts * assumptions.cost_per_sms
        benefit = prevented * assumptions.value_per_kept_visit
        contacted, found = calls + texts, no_shows_called + no_shows_texted
        precision = found / contacted if contacted else 0.0
        base_rate = no_shows / n if n else 0.0
        return ImpactResult(
            strategy=strategy, n_appointments=n, n_calls=calls, n_sms=texts, no_shows_total=no_shows,
            no_shows_called=no_shows_called, no_shows_sms=no_shows_texted, prevented=prevented,
            cost=cost, benefit=benefit, net=benefit - cost,
            cost_per_prevented_no_show=cost / prevented if prevented > 0 else float("nan"),
            precision_in_contacted=precision,
            capture_rate=found / no_shows if no_shows else 0.0,
            lift_vs_random=precision / base_rate if base_rate else 0.0,
        )

    def compare(self, scores: pd.DataFrame, score_columns: Mapping[str, str], policy: OutreachPolicy,
                assumptions: ImpactAssumptions) -> pd.DataFrame:
        """One row per strategy, all with the same policy: Random, each score column, Oracle.

        `score_columns` maps a display name to a column of `scores`. Random and
        Oracle use the same number of calls and SMS per day as the first score
        column, so every row spends the same budget under a capacity policy.
        """
        results = []
        for strategy, column in score_columns.items():
            scored = scores[_ID_COLUMNS].assign(risk_score=scores[column])
            results.append(self.evaluate(scored, policy, assumptions, strategy))
        reference = scores[_ID_COLUMNS].assign(risk_score=scores[next(iter(score_columns.values()))])
        budget = policy.assign(reference)
        ordered = [self._random(budget, assumptions), *results, self._oracle(budget, assumptions)]
        return pd.DataFrame([result.to_dict() for result in ordered])

    def _daily_budget(self, budget: pd.DataFrame) -> pd.DataFrame:
        """Per day: appointments, no-shows, and how many calls and SMS the policy spent."""
        return budget.groupby("appointment_date").agg(
            n=("no_show", "size"), no_shows=("no_show", "sum"),
            calls=("action", lambda actions: int((actions == Action.CALL.value).sum())),
            texts=("action", lambda actions: int((actions == Action.SMS.value).sum())),
        )

    def _random(self, budget: pd.DataFrame, assumptions: ImpactAssumptions) -> ImpactResult:
        """Expected result of contacting the same number of patients picked at random.

        Computed, not simulated: a random pick of k patients on a day contains
        k x (that day's no-show rate) no-shows on average.
        """
        days = self._daily_budget(budget)
        rate = days["no_shows"] / days["n"]
        return self._result(
            RANDOM, assumptions, n=int(days["n"].sum()), calls=int(days["calls"].sum()),
            texts=int(days["texts"].sum()), no_shows=int(days["no_shows"].sum()),
            no_shows_called=float((days["calls"] * rate).sum()),
            no_shows_texted=float((days["texts"] * rate).sum()),
        )

    def _oracle(self, budget: pd.DataFrame, assumptions: ImpactAssumptions) -> ImpactResult:
        """Upper bound: every call goes to a real no-show first, then every SMS."""
        days = self._daily_budget(budget)
        called = np.minimum(days["calls"], days["no_shows"])
        texted = np.minimum(days["texts"], days["no_shows"] - called)
        return self._result(
            ORACLE, assumptions, n=int(days["n"].sum()), calls=int(days["calls"].sum()),
            texts=int(days["texts"].sum()), no_shows=int(days["no_shows"].sum()),
            no_shows_called=float(called.sum()), no_shows_texted=float(texted.sum()),
        )

    def break_even_effectiveness(self, scored_with_outcomes: pd.DataFrame, policy: OutreachPolicy,
                                 assumptions: ImpactAssumptions) -> float | None:
        """Call effectiveness at which net benefit is zero, SMS assumptions unchanged.

        Returns 0.0 if the SMS tier alone already covers the cost, and None if
        even a call that always works (effectiveness 1) could not pay for itself.
        """
        no_calls_work = self.evaluate(scored_with_outcomes, policy, replace(assumptions, call_effectiveness=0.0))
        if no_calls_work.net >= 0:
            return 0.0
        # Net benefit rises in a straight line with call effectiveness.
        gain_per_unit = no_calls_work.no_shows_called * assumptions.call_reach_rate * assumptions.value_per_kept_visit
        if gain_per_unit <= 0:
            return None
        needed = -no_calls_work.net / gain_per_unit
        return needed if needed <= 1.0 else None

    def sensitivity(self, scored_with_outcomes: pd.DataFrame, assumptions: ImpactAssumptions,
                    call_effectiveness_values: Sequence[float], call_share_values: Sequence[float],
                    sms_share: float) -> pd.DataFrame:
        """Net benefit over a grid of call effectiveness x call share (capacity policy)."""
        rows = []
        for call_share in call_share_values:
            policy = CapacityPolicy(call_share, sms_share)
            for effectiveness in call_effectiveness_values:
                result = self.evaluate(scored_with_outcomes, policy, replace(assumptions, call_effectiveness=effectiveness))
                rows.append({"call_share": call_share, "call_effectiveness": effectiveness,
                             "n_calls": result.n_calls, "prevented": result.prevented, "net": result.net})
        return pd.DataFrame(rows)
