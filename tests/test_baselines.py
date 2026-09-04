"""Tests for the B1 baseline forecasters, in isolation from the harness."""

from __future__ import annotations

import pandas as pd

from src.forecast.baselines import ETS, SeasonalNaive


def _train(id_: str, sales: list[int]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(sales), freq="D")
    return pd.DataFrame({"id": id_, "date": dates, "sales": sales})


def _test_keys(id_: str, start: str, n: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"id": id_, "date": dates})


def test_seasonal_naive_repeats_last_season():
    train = _train("A", [1, 2, 3, 4, 5, 6, 7])
    keys = _test_keys("A", "2020-01-08", 10)
    out = SeasonalNaive(season=7).forecast(train, keys)
    assert list(out) == [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]  # last 7 cycled across the horizon


def test_seasonal_naive_uses_only_the_last_season():
    # 14 days; only the last 7 (values 8..14) should drive the forecast.
    train = _train("A", list(range(1, 15)))
    keys = _test_keys("A", "2020-01-15", 3)
    out = SeasonalNaive(season=7).forecast(train, keys)
    assert list(out) == [8, 9, 10]


def test_seasonal_naive_unseen_series_predicts_zero():
    train = _train("A", [1, 2, 3])
    keys = _test_keys("B", "2020-01-08", 3)  # id not in train
    out = SeasonalNaive().forecast(train, keys)
    assert list(out) == [0.0, 0.0, 0.0]


def test_ets_short_series_falls_back_to_last_value():
    # len 3 < 2*season -> fall back to last observed value, repeated.
    train = _train("A", [0, 0, 5])
    keys = _test_keys("A", "2020-01-04", 3)
    out = ETS(season=7).forecast(train, keys)
    assert list(out) == [5.0, 5.0, 5.0]


def test_ets_all_zero_series_forecasts_zero():
    train = _train("A", [0] * 20)
    keys = _test_keys("A", "2020-01-21", 3)
    out = ETS(season=7).forecast(train, keys)
    assert list(out) == [0.0, 0.0, 0.0]
