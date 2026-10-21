"""HypothesisTests: turns the analytical views into tested, worded findings (H1-H6)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import chi2_contingency, norm

from medireach.data.analytics import AnalyticsQueries, RateComparison, compare_rates, with_wilson_ci

SIGNIFICANCE = 0.05
# With 110,000 rows almost any gap is "statistically significant". A gap must
# also be at least two percentage points before we call it worth acting on.
PRACTICAL_GAP = 0.02
LOW_N = 100

SUPPORTED = "Supported"
PARTLY = "Partly supported"
NOT_SUPPORTED = "Not supported"


@dataclass(frozen=True)
class HypothesisResult:
    """One hypothesis: the verdict, a one-line stat, and the evidence table."""

    hypothesis_id: str
    statement: str
    view: str
    verdict: str
    summary: str
    table: pd.DataFrame
    details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id, "statement": self.statement, "view": self.view,
            "verdict": self.verdict, "summary": self.summary, "details": self.details,
            "table": self.table.to_dict(orient="records"),
        }


def _totals(frame: pd.DataFrame, mask: pd.Series) -> tuple[int, int]:
    return int(frame.loc[mask, "no_shows"].sum()), int(frame.loc[mask, "n"].sum())


def _verdict(comparison: RateComparison, expected_sign: int) -> str:
    """Supported = significant, in the expected direction, and big enough to matter."""
    right_direction = comparison.difference * expected_sign > 0
    if comparison.p_value >= SIGNIFICANCE or not right_direction:
        return NOT_SUPPORTED
    return SUPPORTED if abs(comparison.difference) >= PRACTICAL_GAP else PARTLY


def _p_text(p_value: float) -> str:
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def _gap_text(comparison: RateComparison) -> str:
    points = [100 * value for value in (comparison.difference, comparison.ci_low, comparison.ci_high)]
    return f"{points[0]:+.1f} pts, 95% CI {points[1]:+.1f} to {points[2]:+.1f}, {_p_text(comparison.p_value)}"


def _comparison_details(prefix: str, comparison: RateComparison) -> dict[str, float]:
    return {f"{prefix}_{name}": float(value) for name, value in comparison._asdict().items()}


def _chi_square_p(frame: pd.DataFrame) -> float:
    """Chi-square test of independence: does the no-show rate differ across the rows?"""
    table = [frame["no_shows"].tolist(), (frame["n"] - frame["no_shows"]).tolist()]
    return float(chi2_contingency(table)[1])


class HypothesisTests:
    """Runs the six hypothesis checks. Reads views only; no SQL of its own."""

    def __init__(self, analytics: AnalyticsQueries) -> None:
        self._analytics = analytics

    def run_all(self) -> list[HypothesisResult]:
        return [self.h1_lead_time(), self.h2_history(), self.h3_sms(),
                self.h4_age(), self.h5_weekday(), self.h6_clinic()]

    def h1_lead_time(self) -> HypothesisResult:
        table = with_wilson_ci(self._analytics.lead_time())
        long_lead = table["lead_bucket"].isin(["8-14", "15-30", "31+"])
        same_day = table["lead_bucket"] == "0"
        # Same-day bookings are a separate, low-risk case, so the main test
        # compares 8+ days against 1-7 days.
        main = compare_rates(*_totals(table, long_lead), *_totals(table, ~long_lead & ~same_day))
        with_same_day = compare_rates(*_totals(table, long_lead), *_totals(table, ~long_lead))
        same_day_rate = float(table.loc[same_day, "no_show_rate"].iloc[0])
        summary = (f"Booked 8+ days ahead: {main.rate_1:.1%} no-show vs {main.rate_2:.1%} for 1-7 days "
                   f"({_gap_text(main)}). Same-day bookings: {same_day_rate:.1%}.")
        details = {**_comparison_details("long_vs_1_7", main),
                   **_comparison_details("long_vs_0_7", with_same_day), "same_day_rate": same_day_rate}
        statement = "Longer lead time raises no-show risk; same-day bookings are a low-risk special case."
        return HypothesisResult("H1", statement, "v_lead_time", _verdict(main, +1), summary, table, details)

    def h2_history(self) -> HypothesisResult:
        table = with_wilson_ci(self._analytics.history_effect())
        none_before = table["prior_no_show_bucket"] == "0"
        comparison = compare_rates(*_totals(table, ~none_before), *_totals(table, none_before))
        summary = (f"Patients with a prior no-show: {comparison.rate_1:.1%} no-show vs {comparison.rate_2:.1%} "
                   f"for patients with history but no prior no-show ({_gap_text(comparison)}).")
        statement = "Patients with a prior no-show are more likely to no-show again."
        return HypothesisResult("H2", statement, "v_history_effect", _verdict(comparison, +1), summary,
                                table, _comparison_details("prior_vs_none", comparison))

    def h3_sms(self) -> HypothesisResult:
        cells = self._analytics.sms_by_lead()
        naive = compare_rates(*_totals(cells, cells["sms_received"] == 1), *_totals(cells, cells["sms_received"] == 0))
        table = self._sms_strata(cells)
        weights = table["n"] / table["n"].sum()
        adjusted = float((weights * table["difference"]).sum())
        half_width = 1.96 * math.sqrt(float((weights ** 2 * table["se"] ** 2).sum()))
        lower_in = int((table["difference"] < 0).sum())
        significant = adjusted + half_width < 0
        summary = (f"Naive: SMS patients no-show more ({naive.rate_1:.1%} vs {naive.rate_2:.1%}). Within the same "
                   f"lead bucket SMS patients no-show less in {lower_in} of {len(table)} buckets; size-weighted gap "
                   f"{100 * adjusted:+.1f} pts (95% CI {100 * (adjusted - half_width):+.1f} to "
                   f"{100 * (adjusted + half_width):+.1f}).")
        details = {**_comparison_details("naive", naive), "adjusted_difference": adjusted,
                   "adjusted_ci_low": adjusted - half_width, "adjusted_ci_high": adjusted + half_width,
                   "buckets_compared": float(len(table)), "buckets_sms_lower": float(lower_in)}
        # The claim is causal and the data is observational, so the best
        # possible verdict here is "partly": an association, not proof.
        verdict = PARTLY if significant else NOT_SUPPORTED
        statement = "SMS reminders reduce no-shows (naive comparison is confounded by lead time)."
        return HypothesisResult("H3", statement, "v_sms_by_lead", verdict, summary, table.drop(columns="se"), details)

    @staticmethod
    def _sms_strata(cells: pd.DataFrame) -> pd.DataFrame:
        """One row per lead bucket that has both SMS and no-SMS appointments."""
        rows = []
        for (bucket, sort_key), group in cells.groupby(["lead_bucket", "sort_key"], sort=False):
            sms, no_sms = group[group["sms_received"] == 1], group[group["sms_received"] == 0]
            if sms.empty or no_sms.empty:
                continue
            result = compare_rates(int(sms["no_shows"].iloc[0]), int(sms["n"].iloc[0]),
                                   int(no_sms["no_shows"].iloc[0]), int(no_sms["n"].iloc[0]))
            rows.append({
                "lead_bucket": bucket, "sort_key": sort_key, "n": int(group["n"].sum()),
                "n_sms": int(sms["n"].iloc[0]), "rate_sms": result.rate_1,
                "n_no_sms": int(no_sms["n"].iloc[0]), "rate_no_sms": result.rate_2,
                "difference": result.difference, "ci_low": result.ci_low, "ci_high": result.ci_high,
                # The CI is difference +/- 1.96 * se, so this recovers the standard error.
                "p_value": result.p_value, "se": (result.ci_high - result.ci_low) / (2 * 1.96),
            })
        return pd.DataFrame(rows).sort_values("sort_key").reset_index(drop=True)

    def h4_age(self) -> HypothesisResult:
        table = with_wilson_ci(self._analytics.age_group())
        young, older = table["age_group"] == "18-34", table["age_group"] == "65+"
        young_test = compare_rates(*_totals(table, young), *_totals(table, ~young))
        older_test = compare_rates(*_totals(table, older), *_totals(table, ~older))
        verdicts = {_verdict(young_test, +1), _verdict(older_test, -1)}
        verdict = SUPPORTED if verdicts == {SUPPORTED} else NOT_SUPPORTED if verdicts == {NOT_SUPPORTED} else PARTLY
        summary = (f"Ages 18-34: {young_test.rate_1:.1%} vs {young_test.rate_2:.1%} for everyone else "
                   f"({_gap_text(young_test)}). Ages 65+: {older_test.rate_1:.1%} vs {older_test.rate_2:.1%} "
                   f"({_gap_text(older_test)}).")
        details = {**_comparison_details("young_vs_rest", young_test), **_comparison_details("older_vs_rest", older_test)}
        statement = "Age groups differ: young adults are higher risk, older adults lower."
        return HypothesisResult("H4", statement, "v_age_group", verdict, summary, table, details)

    def h5_weekday(self) -> HypothesisResult:
        table = with_wilson_ci(self._analytics.weekday())
        table["low_n"] = table["n"] < LOW_N
        usable = table[~table["low_n"]]
        return self._spread_result("H5", "Day of the week matters.", "v_weekday", table, usable,
                                   "weekday", "weekdays")

    def h6_clinic(self) -> HypothesisResult:
        table = self._analytics.clinic()
        k = len(table)
        # Bonferroni: with k clinics tested at once, each one must clear
        # a stricter bar (alpha / k) to keep the overall false-alarm rate at 5%.
        strict = with_wilson_ci(table, z=float(norm.ppf(1 - SIGNIFICANCE / (2 * k))))
        table = with_wilson_ci(table)
        overall = table["overall_rate"]
        table["differs"] = (table["ci_low"] > overall) | (table["ci_high"] < overall)
        table["differs_strict"] = (strict["ci_low"] > overall) | (strict["ci_high"] < overall)
        result = self._spread_result("H6", "Clinic locations differ beyond chance.", "v_clinic",
                                     table, table, "clinic", "clinic locations with n >= 100")
        extra = {"clinics": float(k), "differ_at_95": float(table["differs"].sum()),
                 "expected_by_chance": SIGNIFICANCE * k, "differ_after_bonferroni": float(table["differs_strict"].sum())}
        summary = (f"{result.summary} {int(extra['differ_at_95'])} of {k} have a 95% CI that excludes the overall "
                   f"rate (about {extra['expected_by_chance']:.0f} expected by chance); "
                   f"{int(extra['differ_after_bonferroni'])} remain after a Bonferroni correction.")
        return HypothesisResult(result.hypothesis_id, result.statement, result.view, result.verdict,
                                summary, table, {**result.details, **extra})

    @staticmethod
    def _spread_result(hypothesis_id: str, statement: str, view: str, table: pd.DataFrame,
                       usable: pd.DataFrame, label: str, noun: str) -> HypothesisResult:
        """Shared logic for 'do these groups differ?': chi-square test plus the size of the spread."""
        p_value = _chi_square_p(usable)
        low, high = usable.loc[usable["no_show_rate"].idxmin()], usable.loc[usable["no_show_rate"].idxmax()]
        spread = float(high["no_show_rate"] - low["no_show_rate"])
        if p_value >= SIGNIFICANCE:
            verdict = NOT_SUPPORTED
        else:
            verdict = SUPPORTED if spread >= PRACTICAL_GAP else PARTLY
        summary = (f"Across {len(usable)} {noun}, rates run from {low['no_show_rate']:.1%} ({low[label]}) to "
                   f"{high['no_show_rate']:.1%} ({high[label]}): a spread of {100 * spread:.1f} pts "
                   f"(chi-square {_p_text(p_value)}).")
        details = {"chi_square_p": p_value, "spread": spread, "groups_tested": float(len(usable))}
        return HypothesisResult(hypothesis_id, statement, view, verdict, summary, table, details)
