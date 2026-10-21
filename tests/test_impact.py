"""Phase 7: impact simulator and pilot design."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from medireach.domain import ImpactAssumptions
from medireach.impact.pilot import PilotDesigner
from medireach.impact.simulator import ORACLE, RANDOM, ImpactSimulator
from medireach.outreach.policies import CapacityPolicy, ThresholdPolicy

POLICY = CapacityPolicy(call_share=0.10, sms_share=0.20)
ASSUMPTIONS = ImpactAssumptions()


@pytest.fixture(scope="module")
def scores() -> pd.DataFrame:
    """Five days of 200 appointments with a good score, a weak score, and the outcome."""
    rng = np.random.default_rng(4)
    n = 1000
    risk = rng.uniform(0.02, 0.6, n)
    no_show = (rng.random(n) < risk).astype(int)
    return pd.DataFrame({
        "appointment_id": np.arange(n),
        "appointment_date": np.repeat(pd.bdate_range("2016-06-06", periods=5), n // 5),
        "no_show": no_show,
        "good": risk,
        "weak": 0.8 * rng.uniform(0.02, 0.6, n) + 0.2 * risk,
    })


def _scored(scores: pd.DataFrame, column: str = "good") -> pd.DataFrame:
    return scores[["appointment_id", "appointment_date", "no_show"]].assign(risk_score=scores[column])


@pytest.fixture(scope="module")
def simulator() -> ImpactSimulator:
    return ImpactSimulator()


# ------------------------------------------------------------------- evaluate

def test_result_arithmetic_by_hand(simulator: ImpactSimulator) -> None:
    # 10 appointments on one day: 1 call (a no-show), 2 SMS (one no-show), 3 no-shows in total.
    scored = pd.DataFrame({
        "appointment_id": range(10), "appointment_date": pd.Timestamp("2016-06-06"),
        "risk_score": np.linspace(0.9, 0.1, 10), "no_show": [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    })
    result = simulator.evaluate(scored, POLICY, ASSUMPTIONS, "by hand")
    assert (result.n_calls, result.n_sms, result.no_shows_total) == (1, 2, 3)
    assert (result.no_shows_called, result.no_shows_sms) == (1, 1)
    assert result.prevented == pytest.approx(1 * 0.6 * 0.3 + 1 * 0.9 * 0.1)
    assert result.cost == pytest.approx(1 * 40 + 2 * 1)
    assert result.benefit == pytest.approx(0.27 * 500)
    assert result.net == pytest.approx(135 - 42)
    assert result.cost_per_prevented_no_show == pytest.approx(42 / 0.27)
    assert result.precision_in_contacted == pytest.approx(2 / 3)
    assert result.capture_rate == pytest.approx(2 / 3)
    assert result.lift_vs_random == pytest.approx((2 / 3) / 0.3)


def test_zero_effectiveness_prevents_nothing(simulator: ImpactSimulator, scores: pd.DataFrame) -> None:
    useless = replace(ASSUMPTIONS, call_effectiveness=0.0, sms_effectiveness=0.0)
    result = simulator.evaluate(_scored(scores), POLICY, useless)
    assert result.prevented == 0 and result.benefit == 0
    assert result.net == -result.cost < 0
    assert np.isnan(result.cost_per_prevented_no_show)


def test_net_benefit_rises_with_effectiveness(simulator: ImpactSimulator, scores: pd.DataFrame) -> None:
    nets = [simulator.evaluate(_scored(scores), POLICY, replace(ASSUMPTIONS, call_effectiveness=e)).net
            for e in (0.0, 0.1, 0.3, 0.6, 1.0)]
    assert nets == sorted(nets) and nets[0] < nets[-1]


def test_evaluate_does_not_modify_its_input(simulator: ImpactSimulator, scores: pd.DataFrame) -> None:
    scored = _scored(scores)
    before = scored.copy(deep=True)
    simulator.evaluate(scored, POLICY, ASSUMPTIONS)
    pd.testing.assert_frame_equal(scored, before)


# -------------------------------------------------------------------- compare

def test_oracle_beats_model_beats_random(simulator: ImpactSimulator, scores: pd.DataFrame) -> None:
    table = simulator.compare(scores, {"Weak model": "weak", "Good model": "good"}, POLICY, ASSUMPTIONS)
    assert list(table["strategy"]) == [RANDOM, "Weak model", "Good model", ORACLE]
    by_name = table.set_index("strategy")
    for metric in ("capture_rate", "precision_in_contacted", "prevented", "net"):
        ordered = by_name.loc[[RANDOM, "Weak model", "Good model", ORACLE], metric]
        assert ordered.is_monotonic_increasing, metric
    assert by_name.loc[RANDOM, "lift_vs_random"] == pytest.approx(1.0)
    # The oracle wastes a contact only on a day with fewer no-shows than contacts (60 a day here).
    daily_no_shows = scores.groupby("appointment_date")["no_show"].sum()
    assert by_name.loc[ORACLE, "precision_in_contacted"] == pytest.approx(daily_no_shows.clip(upper=60).sum() / 300)


def test_every_strategy_spends_the_same_budget(simulator: ImpactSimulator, scores: pd.DataFrame) -> None:
    table = simulator.compare(scores, {"Weak model": "weak", "Good model": "good"}, POLICY, ASSUMPTIONS)
    assert table["n_calls"].nunique() == 1 and table["n_sms"].nunique() == 1
    assert table["cost"].nunique() == 1
    assert (table["n_calls"].iloc[0], table["n_sms"].iloc[0]) == (100, 200)


def test_random_is_the_expected_value_of_a_random_pick(simulator: ImpactSimulator, scores: pd.DataFrame) -> None:
    table = simulator.compare(scores, {"Good model": "good"}, POLICY, ASSUMPTIONS).set_index("strategy")
    daily_rate = scores.groupby("appointment_date")["no_show"].mean()
    assert table.loc[RANDOM, "no_shows_called"] == pytest.approx((20 * daily_rate).sum())
    assert table.loc[RANDOM, "no_shows_sms"] == pytest.approx((40 * daily_rate).sum())


def test_oracle_cannot_contact_more_no_shows_than_exist(simulator: ImpactSimulator) -> None:
    scored = pd.DataFrame({
        "appointment_id": range(10), "appointment_date": pd.Timestamp("2016-06-06"),
        "no_show": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], "model": np.linspace(0.1, 0.9, 10),
    })
    table = simulator.compare(scored, {"Model": "model"}, CapacityPolicy(0.3, 0.3), ASSUMPTIONS).set_index("strategy")
    assert table.loc[ORACLE, "no_shows_called"] == 1 and table.loc[ORACLE, "no_shows_sms"] == 0
    assert table.loc[ORACLE, "capture_rate"] == 1.0


# ---------------------------------------------- break-even and sensitivity

def test_break_even_effectiveness_gives_zero_net(simulator: ImpactSimulator, scores: pd.DataFrame) -> None:
    # Calls priced high enough that the SMS tier alone cannot pay for them.
    costly = replace(ASSUMPTIONS, cost_per_call=200.0)
    break_even = simulator.break_even_effectiveness(_scored(scores), POLICY, costly)
    assert break_even is not None and 0 < break_even < 1
    at_break_even = simulator.evaluate(_scored(scores), POLICY, replace(costly, call_effectiveness=break_even))
    assert at_break_even.net == pytest.approx(0.0, abs=1e-6)


def test_break_even_edge_cases(simulator: ImpactSimulator, scores: pd.DataFrame) -> None:
    free = replace(ASSUMPTIONS, cost_per_call=0.0, cost_per_sms=0.0)
    assert simulator.break_even_effectiveness(_scored(scores), POLICY, free) == 0.0
    ruinous = replace(ASSUMPTIONS, cost_per_call=100_000.0)
    assert simulator.break_even_effectiveness(_scored(scores), POLICY, ruinous) is None
    nobody_called = ThresholdPolicy(high=1.0, medium=1.0)
    assert simulator.break_even_effectiveness(_scored(scores), nobody_called, ASSUMPTIONS) == 0.0


def test_sensitivity_grid(simulator: ImpactSimulator, scores: pd.DataFrame) -> None:
    grid = simulator.sensitivity(_scored(scores), ASSUMPTIONS, (0.1, 0.3, 0.5), (0.05, 0.10, 0.20), sms_share=0.20)
    assert len(grid) == 9 and set(grid.columns) == {"call_share", "call_effectiveness", "n_calls", "prevented", "net"}
    for _, row in grid.groupby("call_share"):
        assert row.sort_values("call_effectiveness")["net"].is_monotonic_increasing
    assert list(grid.groupby("call_share")["n_calls"].first()) == [50, 100, 200]
    same = grid[(grid["call_share"] == 0.10) & (grid["call_effectiveness"] == 0.3)]["net"].iloc[0]
    assert same == pytest.approx(simulator.evaluate(_scored(scores), POLICY, ASSUMPTIONS).net)


# ---------------------------------------------------------------------- pilot

def test_sample_size_matches_the_textbook_value() -> None:
    per_arm = PilotDesigner.sample_size_per_arm(0.20, 0.25)
    assert 850 <= per_arm <= 960
    assert per_arm == 906


def test_sample_size_moves_the_right_way() -> None:
    size = PilotDesigner.sample_size_per_arm
    assert size(0.20, 0.10) > size(0.20, 0.25) > size(0.20, 0.50)   # smaller effects need more patients
    assert size(0.20, 0.25, power=0.9) > size(0.20, 0.25, power=0.8)
    assert size(0.20, 0.25, alpha=0.01) > size(0.20, 0.25, alpha=0.05)
    assert size(0.40, 0.25) < size(0.20, 0.25)                      # a higher base rate is easier to move


def test_plan_turns_the_sample_into_clinic_days() -> None:
    plan = PilotDesigner().plan(p_control=0.20, relative_reduction=0.25, daily_high_volume=400)
    assert plan.sample_size_per_arm == 906 and plan.total_sample == 1812
    assert plan.estimated_days == 5
    assert plan.primary_metric == "no-show rate" and len(plan.guardrails) >= 3


def test_pilot_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError):
        PilotDesigner.sample_size_per_arm(0.0, 0.25)
    with pytest.raises(ValueError):
        PilotDesigner.sample_size_per_arm(0.2, 1.0)
    with pytest.raises(ValueError):
        PilotDesigner().plan(0.2, 0.25, daily_high_volume=0)
