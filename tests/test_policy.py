"""Phase 6: outreach policies. Both policies must pass the same contract tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medireach.domain import Action, Priority
from medireach.outreach.policies import CapacityPolicy, FixedCountPolicy, OutreachPolicy, ThresholdPolicy

DAY_SIZES = {"2016-06-01": 1, "2016-06-02": 3, "2016-06-03": 10, "2016-06-06": 30, "2016-06-07": 25}


def _scored(seed: int = 0) -> pd.DataFrame:
    """Five days of very different sizes, with an unhelpful index and row order."""
    rng = np.random.default_rng(seed)
    dates = [pd.Timestamp(day) for day, size in DAY_SIZES.items() for _ in range(size)]
    frame = pd.DataFrame({
        "appointment_id": 1000 + np.arange(len(dates)),
        "patient_id": 1,
        "appointment_date": dates,
        "risk_score": rng.uniform(0.01, 0.8, len(dates)).round(3),
    })
    return frame.sample(frac=1.0, random_state=seed).set_index(pd.Index(range(500, 500 + len(frame))))


# ------------------------------------------------------------------- contract

@pytest.fixture(params=["threshold", "capacity", "fixed_count"])
def policy(request: pytest.FixtureRequest) -> OutreachPolicy:
    return {"threshold": ThresholdPolicy(), "capacity": CapacityPolicy(), "fixed_count": FixedCountPolicy(3, 5)}[request.param]


def test_contract_adds_three_columns_and_keeps_rows_as_they_were(policy: OutreachPolicy) -> None:
    scored = _scored()
    result = policy.assign(scored)
    assert list(result.columns) == [*scored.columns, "day_rank", "priority", "action"]
    assert result.index.equals(scored.index)
    pd.testing.assert_frame_equal(result[scored.columns], scored)
    assert isinstance(policy.name, str) and policy.name


def test_contract_does_not_modify_its_input(policy: OutreachPolicy) -> None:
    scored = _scored()
    before = scored.copy(deep=True)
    policy.assign(scored)
    pd.testing.assert_frame_equal(scored, before)


def test_contract_values_are_the_enum_strings_and_each_priority_has_one_action(policy: OutreachPolicy) -> None:
    result = policy.assign(_scored())
    assert set(result["priority"]) <= {p.value for p in Priority}
    assert set(result["action"]) <= {a.value for a in Action}
    pairs = set(zip(result["priority"], result["action"]))
    assert pairs <= {("HIGH", "Call"), ("MEDIUM", "SMS"), ("LOW", "None")}


def test_contract_day_rank_orders_each_day_by_score(policy: OutreachPolicy) -> None:
    result = policy.assign(_scored())
    for _, day in result.groupby("appointment_date"):
        ordered = day.sort_values("day_rank")
        assert list(ordered["day_rank"]) == list(range(1, len(day) + 1))
        assert ordered["risk_score"].is_monotonic_decreasing


def test_contract_a_higher_priority_never_has_a_lower_score_that_day(policy: OutreachPolicy) -> None:
    result = policy.assign(_scored())
    for _, day in result.groupby("appointment_date"):
        lowest = day.groupby("priority")["risk_score"].min()
        highest = day.groupby("priority")["risk_score"].max()
        for upper, lower in (("HIGH", "MEDIUM"), ("MEDIUM", "LOW"), ("HIGH", "LOW")):
            if upper in lowest and lower in highest:
                assert lowest[upper] >= highest[lower]


def test_contract_result_does_not_depend_on_input_order(policy: OutreachPolicy) -> None:
    scored = _scored()
    first = policy.assign(scored).set_index("appointment_id").sort_index()
    second = policy.assign(scored.sample(frac=1.0, random_state=9)).set_index("appointment_id").sort_index()
    pd.testing.assert_frame_equal(first, second)


def test_contract_missing_columns_are_rejected(policy: OutreachPolicy) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        policy.assign(_scored().drop(columns=["risk_score"]))


# ------------------------------------------------------------------- capacity

def test_capacity_counts_are_exact_for_every_day_size() -> None:
    result = CapacityPolicy(call_share=0.10, sms_share=0.20).assign(_scored())
    counts = result.groupby(["appointment_date", "priority"]).size().unstack(fill_value=0)
    # day size -> (calls, SMS): ceil(10%) and ceil(20%), limited by the rows that are left.
    expected = {1: (1, 0), 3: (1, 1), 10: (1, 2), 30: (3, 6), 25: (3, 5)}
    for day, size in DAY_SIZES.items():
        row = counts.loc[pd.Timestamp(day)]
        assert (row.get("HIGH", 0), row.get("MEDIUM", 0)) == expected[size]
        assert row.sum() == size


def test_capacity_calls_go_to_the_highest_scores_of_the_day() -> None:
    result = CapacityPolicy(0.10, 0.20).assign(_scored())
    big_day = result[result["appointment_date"] == pd.Timestamp("2016-06-06")].sort_values("risk_score", ascending=False)
    assert list(big_day["priority"].head(3)) == ["HIGH"] * 3
    assert list(big_day["priority"].iloc[3:9]) == ["MEDIUM"] * 6
    assert (big_day["priority"].iloc[9:] == "LOW").all()


def test_capacity_breaks_ties_by_appointment_id() -> None:
    scored = pd.DataFrame({
        "appointment_id": [14, 11, 13, 12, 15, 16, 17, 18, 19, 20],
        "appointment_date": pd.Timestamp("2016-06-06"),
        "risk_score": 0.4,
    })
    result = CapacityPolicy(0.10, 0.20).assign(scored).set_index("appointment_id")
    assert result.loc[11, "priority"] == "HIGH" and result.loc[11, "day_rank"] == 1
    assert list(result.loc[[12, 13], "priority"]) == ["MEDIUM", "MEDIUM"]
    assert (result.loc[[14, 15, 16, 17, 18, 19, 20], "priority"] == "LOW").all()


def test_capacity_shares_of_zero_and_one() -> None:
    scored = _scored()
    nobody = CapacityPolicy(call_share=0.0, sms_share=0.0).assign(scored)
    assert (nobody["priority"] == "LOW").all()
    everybody = CapacityPolicy(call_share=1.0, sms_share=0.0).assign(scored)
    assert (everybody["priority"] == "HIGH").all()


def test_capacity_rejects_impossible_shares() -> None:
    with pytest.raises(ValueError):
        CapacityPolicy(call_share=0.7, sms_share=0.5)
    with pytest.raises(ValueError):
        CapacityPolicy(call_share=-0.1, sms_share=0.2)


# ------------------------------------------------------------------ threshold

def test_threshold_boundaries_are_inclusive_on_the_lower_edge() -> None:
    scored = pd.DataFrame({
        "appointment_id": [1, 2, 3, 4, 5, 6],
        "appointment_date": pd.Timestamp("2016-06-06"),
        "risk_score": [0.50, 0.4999, 0.35, 0.3499, 0.99, 0.0],
    })
    result = ThresholdPolicy(high=0.50, medium=0.35).assign(scored).set_index("appointment_id")
    assert list(result["priority"]) == ["HIGH", "MEDIUM", "MEDIUM", "LOW", "HIGH", "LOW"]
    assert list(result["action"]) == ["Call", "SMS", "SMS", "None", "Call", "None"]


def test_threshold_workload_varies_while_capacity_workload_does_not() -> None:
    # The reason capacity is the default: a fixed cut-off gives an unpredictable number of calls.
    low_risk_day = pd.DataFrame({
        "appointment_id": range(100), "appointment_date": pd.Timestamp("2016-06-06"),
        "risk_score": np.linspace(0.01, 0.30, 100),
    })
    high_risk_day = low_risk_day.assign(risk_score=np.linspace(0.40, 0.90, 100))
    calls = lambda policy, day: int((policy.assign(day)["action"] == "Call").sum())
    assert calls(ThresholdPolicy(), low_risk_day) == 0 and calls(ThresholdPolicy(), high_risk_day) > 50
    assert calls(CapacityPolicy(), low_risk_day) == calls(CapacityPolicy(), high_risk_day) == 10


def test_threshold_rejects_cutoffs_in_the_wrong_order() -> None:
    with pytest.raises(ValueError):
        ThresholdPolicy(high=0.3, medium=0.5)


def test_fixed_count_calls_and_texts_exact_numbers_per_day() -> None:
    result = FixedCountPolicy(calls=3, sms=5).assign(_scored())
    counts = result.groupby(["appointment_date", "priority"]).size().unstack(fill_value=0)
    for day, size in DAY_SIZES.items():
        row = counts.loc[pd.Timestamp(day)]
        assert row.get("HIGH", 0) == min(3, size)
        assert row.get("MEDIUM", 0) == min(5, max(0, size - 3))
    with pytest.raises(ValueError):
        FixedCountPolicy(calls=-1, sms=0)
