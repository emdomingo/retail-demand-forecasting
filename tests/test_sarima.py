"""Tests for the B3 SARIMA comparison model, in isolation from the harness. We test the
robustness contract (fallbacks, non-negativity, alignment) rather than fitted values — a
per-series SARIMA fit is a stochastic optimisation and asserting exact numbers would be brittle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast.sarima import SARIMA


def _train(id_: str, sales: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(sales), freq="D")
    return pd.DataFrame({"id": id_, "date": dates, "sales": sales})


def _test_keys(id_: str, start: str, n: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"id": id_, "date": dates})


def test_short_series_falls_back_to_last_value():
    # len 5 < 2*season(7) -> fall back to last observed value, repeated.
    train = _train("A", [0, 0, 0, 2, 5])
    keys = _test_keys("A", "2020-01-06", 3)
    out = SARIMA(season=7).forecast(train, keys)
    assert list(out) == [5.0, 5.0, 5.0]


def test_all_zero_series_forecasts_zero():
    train = _train("A", [0.0] * 30)
    keys = _test_keys("A", "2020-01-31", 4)
    out = SARIMA(season=7).forecast(train, keys)
    assert list(out) == [0.0, 0.0, 0.0, 0.0]


def test_unseen_series_predicts_zero():
    train = _train("A", list(range(30)))
    keys = _test_keys("B", "2020-01-31", 3)  # id absent from train
    out = SARIMA(season=7).forecast(train, keys)
    assert list(out) == [0.0, 0.0, 0.0]


def test_forecast_is_non_negative_and_aligned():
    # A real seasonal-ish series that will actually fit; check the output contract, not values.
    rng = np.random.default_rng(0)
    weekly = np.array([10, 12, 9, 11, 20, 25, 8], dtype=float)
    sales = np.tile(weekly, 8) + rng.normal(0, 1.0, 56)
    train = _train("A", sales.clip(0).tolist())
    keys = _test_keys("A", train["date"].max().strftime("%Y-%m-%d"), 14)
    keys["date"] = keys["date"] + pd.Timedelta(days=1)  # start strictly after train
    out = SARIMA(season=7).forecast(train, keys)
    assert out.index.equals(keys.index)  # aligned to test_keys.index
    assert (out >= 0).all()  # sales can't be negative
    assert np.isfinite(out).all()


def test_name_encodes_the_order():
    assert SARIMA().name == "sarima_111_100_7"
    assert SARIMA(order=(2, 0, 1), seasonal_order=(1, 1, 0), season=7).name == "sarima_201_110_7"
