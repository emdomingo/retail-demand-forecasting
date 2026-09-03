"""Tests for A2 time features. The load-bearing property is leakage-safety: no feature
at day t may reflect sales[t]. Everything else (lag values, warm-up nulls) is secondary."""

from __future__ import annotations

from src.ingest.features import add_time_features
from src.ingest.load import ID_COLS


def _series(spark, sales_by_day):
    """One dense series, d_int = 1..len(sales), with the given daily sales."""
    ids = ("X_1_CA_1_evaluation", "X_1", "X_1", "X", "CA_1", "CA")
    rows = [(*ids, f"d_{i}", i, s) for i, s in enumerate(sales_by_day, start=1)]
    return spark.createDataFrame(rows, [*ID_COLS, "d", "d_int", "sales"])


def _by_day(df, col):
    return {r.d_int: r[col] for r in df.collect()}


def test_no_same_day_leakage_in_rolling_mean(spark):
    # Huge spike on the last day. If any feature there reflects it, the frame leaked.
    sales = [0] * 9 + [1000]
    feats = add_time_features(_series(spark, sales))
    last = {r.d_int: r for r in feats.collect()}[10]
    assert last.rmean_7 == 0.0  # avg of days 3..9, all zero — NOT the day-10 spike
    assert last.lag_7 == 0  # day 3, not day 10


def test_lag_reads_k_days_back(spark):
    sales = list(range(1, 41))  # sales == d_int, so lag_k at t must equal t - k
    feats = add_time_features(_series(spark, sales))
    lag7 = _by_day(feats, "lag_7")
    lag28 = _by_day(feats, "lag_28")
    assert lag7[10] == 3 and lag7[40] == 33
    assert lag28[40] == 12


def test_lag_warmup_is_null(spark):
    feats = add_time_features(_series(spark, list(range(1, 41))))
    lag7 = _by_day(feats, "lag_7")
    assert lag7[7] is None  # first 7 days have no 7-day-ago value
    assert lag7[8] == 1


def test_rolling_mean_excludes_today_and_averages_prior_window(spark):
    sales = list(range(1, 41))  # sales == d_int
    feats = add_time_features(_series(spark, sales))
    rmean7 = _by_day(feats, "rmean_7")
    # at t=10, the 7 prior days are d_int 3..9 -> mean(3..9) = 6.0 (day 10 excluded)
    assert rmean7[10] == 6.0


def test_feature_columns_added(spark):
    feats = add_time_features(_series(spark, [1, 2, 3]))
    assert {"lag_7", "lag_28", "rmean_7", "rmean_28"}.issubset(set(feats.columns))
