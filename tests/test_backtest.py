"""Tests for the rolling-origin harness (B1). The load-bearing property is no leakage:
train is always strictly on/before the origin, test strictly after."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast.backtest import (
    BacktestConfig,
    evaluate_origin,
    rolling_origins,
    run_backtest,
    train_test_split,
)
from src.forecast.baselines import SeasonalNaive


def _panel(n_series: int = 3, days: int = 60, period: int = 7) -> pd.DataFrame:
    """A small contiguous panel; sales are exactly period-7 so a seasonal-naive forecast
    should reproduce the future perfectly (→ zero error)."""
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    pattern = np.arange(1, period + 1)  # 1..7 repeating
    rows = []
    for s in range(n_series):
        for i, d in enumerate(dates):
            rows.append((f"S{s}", d, int(pattern[i % period])))
    return pd.DataFrame(rows, columns=["id", "date", "sales"])


def test_rolling_origins_count_and_chronology():
    dates = np.sort(_panel(days=60)["date"].unique())
    cfg = BacktestConfig(horizon=7, step=7, n_origins=3)
    origins = rolling_origins(dates, cfg)
    assert len(origins) == 3
    assert origins == sorted(origins)  # chronological
    # latest origin leaves exactly `horizon` days after it
    assert origins[-1] == dates[len(dates) - 1 - cfg.horizon]


def test_rolling_origins_drops_when_history_too_short():
    dates = np.sort(_panel(days=20)["date"].unique())
    cfg = BacktestConfig(horizon=7, step=7, n_origins=5)
    origins = rolling_origins(dates, cfg)
    assert 1 <= len(origins) < 5  # some origins dropped for lack of training history


def test_split_has_no_leakage():
    df = _panel(days=60)
    dates = np.sort(df["date"].unique())
    cfg = BacktestConfig(horizon=7, step=7, n_origins=3)
    for origin in rolling_origins(dates, cfg):
        train, test = train_test_split(df, origin, cfg, dates)
        assert train["date"].max() <= origin  # nothing after the origin trains
        assert test["date"].min() > origin  # test strictly in the future
        assert test.groupby("id")["date"].nunique().eq(cfg.horizon).all()


def test_split_is_expanding():
    df = _panel(days=60)
    dates = np.sort(df["date"].unique())
    cfg = BacktestConfig(horizon=7, step=7, n_origins=3)
    origins = rolling_origins(dates, cfg)
    sizes = [len(train_test_split(df, o, cfg, dates)[0]) for o in origins]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]  # train grows with later origins


def test_evaluate_origin_perfect_periodic_forecast_scores_zero():
    df = _panel(days=60)
    dates = np.sort(df["date"].unique())
    cfg = BacktestConfig(horizon=7, season=7)
    origin = rolling_origins(dates, cfg)[-1]
    train, test = train_test_split(df, origin, cfg, dates)
    m = evaluate_origin(train, test, SeasonalNaive(season=7))
    assert m["rmsse"] == 0.0
    assert m["wmape"] == 0.0
    assert m["n_series"] == 3


class _ColumnSpy:
    """A fake forecaster that records the columns it was handed for the test rows, then
    predicts zeros. Lets us assert the harness's known_future contract directly."""

    name = "column_spy"

    def __init__(self):
        self.seen_cols = None

    def forecast(self, train, test_keys):
        self.seen_cols = list(test_keys.columns)
        return pd.Series(0.0, index=test_keys.index)


def test_known_future_columns_are_passed_but_sales_is_withheld():
    df = _panel(days=60)
    df["sell_price"] = 1.0  # a known-future exog column to request
    dates = np.sort(df["date"].unique())
    cfg = BacktestConfig(horizon=7, season=7, known_future=["sell_price"])
    origin = rolling_origins(dates, cfg)[-1]
    train, test = train_test_split(df, origin, cfg, dates)

    spy = _ColumnSpy()
    evaluate_origin(train, test, spy, cfg.known_future)
    assert spy.seen_cols == ["id", "date", "sell_price"]
    assert "sales" not in spy.seen_cols  # actuals never reach the model


def test_known_future_defaults_to_id_date_only():
    # Backward compatibility: with no known_future, the model sees exactly (id, date) — the
    # strict contract the baselines were written against.
    df = _panel(days=60)
    dates = np.sort(df["date"].unique())
    cfg = BacktestConfig(horizon=7, season=7)
    origin = rolling_origins(dates, cfg)[-1]
    train, test = train_test_split(df, origin, cfg, dates)

    spy = _ColumnSpy()
    evaluate_origin(train, test, spy, cfg.known_future)
    assert spy.seen_cols == ["id", "date"]


def test_run_backtest_returns_per_origin_frame(tmp_path):
    # Point MLflow at a throwaway SQLite store so tests don't touch the repo's mlruns.db.
    cfg = BacktestConfig(
        horizon=7, step=7, n_origins=3, tracking_uri=f"sqlite:///{tmp_path}/mlruns.db"
    )
    df = _panel(days=60)
    res = run_backtest(df, SeasonalNaive(), cfg)
    assert len(res) == 3
    assert {"origin", "rmsse", "wmape", "n_series"}.issubset(res.columns)
