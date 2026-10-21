"""Phase 4: the chronological train / validation / test split."""
from __future__ import annotations

import pandas as pd
import pytest

from medireach.features.builder import time_split


def _meta(rows_per_day: list[int]) -> pd.DataFrame:
    days = pd.bdate_range("2016-05-02", periods=len(rows_per_day))
    dates = [day for day, count in zip(days, rows_per_day) for _ in range(count)]
    return pd.DataFrame({"appointment_date": dates})


def test_split_is_chronological_and_never_splits_a_date() -> None:
    meta = _meta([10] * 20)
    split = time_split(meta)
    dates = meta["appointment_date"]
    assert dates[split.train].max() < dates[split.validation].min()
    assert dates[split.validation].max() < dates[split.test].min()
    parts = [set(dates[mask]) for mask in split]
    assert not (parts[0] & parts[1]) and not (parts[1] & parts[2]) and not (parts[0] & parts[2])


def test_every_row_lands_in_exactly_one_part() -> None:
    meta = _meta([7, 13, 9, 11, 10, 8, 12, 10, 9, 11])
    split = time_split(meta)
    assert ((split.train.astype(int) + split.validation.astype(int) + split.test.astype(int)) == 1).all()
    assert split.train.index.equals(meta.index)


def test_fractions_are_respected_when_days_are_even() -> None:
    split = time_split(_meta([10] * 20))
    assert (split.train.mean(), split.validation.mean(), split.test.mean()) == pytest.approx((0.6, 0.2, 0.2))


def test_fractions_are_approximate_when_days_are_uneven(cleaned) -> None:
    _, appointments, _ = cleaned
    split = time_split(appointments)
    assert split.train.mean() == pytest.approx(0.6, abs=0.1)
    assert split.validation.mean() == pytest.approx(0.2, abs=0.1)
    assert split.test.mean() == pytest.approx(0.2, abs=0.1)


def test_custom_fractions_and_row_order_do_not_matter() -> None:
    meta = _meta([10] * 10).sample(frac=1.0, random_state=1)
    split = time_split(meta, fractions=(0.5, 0.3, 0.2))
    assert (split.train.sum(), split.validation.sum(), split.test.sum()) == (50, 30, 20)
    assert meta["appointment_date"][split.train].max() < meta["appointment_date"][split.test].min()


def test_too_few_dates_is_an_error() -> None:
    with pytest.raises(ValueError, match="distinct dates"):
        time_split(_meta([10, 10]))
