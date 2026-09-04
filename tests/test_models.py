"""Tests for the global LightGBM forecaster (B2), in isolation from the harness. The
load-bearing properties are honesty (the model never sees test-window actuals) and the
forecaster contract (a Series aligned to test_keys.index, non-negative)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast.models import EXOG_CAT, EXOG_NUM, LightGBMForecaster


def _panel(n_series: int = 4, days: int = 60, seed: int = 0) -> pd.DataFrame:
    """A small multi-series panel with the columns B2 needs: two departments, a weekly sales
    rhythm plus noise, and the known-future exogenous columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    rows = []
    for s in range(n_series):
        base = 5 + 2 * (s % 3)
        for d in dates:
            wk = 1 + (d.dayofweek >= 5)  # weekend bump
            sales = max(0, int(rng.poisson(base * wk)))
            rows.append(
                {
                    "id": f"S{s}",
                    "dept_id": f"D{s % 2}",
                    "cat_id": "C0",
                    "date": d,
                    "sales": sales,
                    "sell_price": 1.0 + 0.1 * (s % 4),
                    "price_change_pct": 0.0,
                    "snap": int(d.day <= 10),
                    "is_event": 0,
                    "event_type_1": None,
                }
            )
    return pd.DataFrame(rows)


def _split(df: pd.DataFrame, horizon: int):
    """Last `horizon` days are the test window; the rest is train."""
    cutoff = df["date"].sort_values().unique()[-horizon - 1]
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    known = ["id", "date", *EXOG_NUM, *EXOG_CAT]
    return train, test, test[known]


def _small_model() -> LightGBMForecaster:
    # Tiny data needs a low leaf floor or the trees never split (they'd predict the global mean).
    return LightGBMForecaster(
        num_boost_round=30, params={"min_data_in_leaf": 5, "min_data_in_bin": 1}
    )


def test_forecast_aligns_to_test_index_and_is_nonnegative():
    df = _panel()
    train, test, keys = _split(df, horizon=14)
    out = _small_model().forecast(train, keys)
    assert list(out.index) == list(keys.index)  # aligned to the keys we were handed
    assert len(out) == len(test)
    assert not out.isna().any()
    assert (out >= 0).all()  # sales can't be negative — predictions are clipped


def test_horizon_beyond_lag_window_still_forecasts_every_row():
    # horizon 28 > lag_7: later days must be filled by recursion (own predictions feeding lags),
    # not left null. A missing row here would mean the recursion silently stopped.
    df = _panel(days=90)
    train, _, keys = _split(df, horizon=28)
    out = _small_model().forecast(train, keys)
    assert len(out) == len(keys)
    assert not out.isna().any()


def test_ignores_a_sales_column_on_test_keys():
    # The honesty guarantee: even if test rows carry actuals, the model must not read them.
    # Passing garbage 'sales' on the test keys must not change a single prediction.
    df = _panel()
    train, test, keys = _split(df, horizon=14)
    clean = _small_model().forecast(train, keys)

    leaky = keys.copy()
    leaky["sales"] = 999  # actuals the model must never touch
    peeked = _small_model().forecast(train, leaky)
    pd.testing.assert_series_equal(clean, peeked)


def test_constant_series_predicts_near_that_constant():
    # A flat series carries no seasonality or trend — predictions should sit near the level.
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {
            "id": "A",
            "dept_id": "D0",
            "cat_id": "C0",
            "date": dates,
            "sales": 10,
            "sell_price": 1.0,
            "price_change_pct": 0.0,
            "snap": 0,
            "is_event": 0,
            "event_type_1": None,
        }
    )
    train, _, keys = _split(df, horizon=7)
    out = _small_model().forecast(train, keys)
    assert np.allclose(out.to_numpy(), 10.0, atol=3.0)


def test_version_flows_into_name():
    assert LightGBMForecaster(version="v1").name == "lightgbm_global_v1"
    assert LightGBMForecaster(version="v2").name == "lightgbm_global_v2"
