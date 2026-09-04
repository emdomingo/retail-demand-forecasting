"""Tests for the B1 accuracy metrics — pinned against hand-computed values."""

from __future__ import annotations

import numpy as np

from src.forecast.metrics import naive_scale, rmsse, wmape


def test_wmape_basic():
    # sum|err| = |10-8| + 0 + 0 = 2 ; sum|actual| = 15 -> 2/15
    assert wmape([10, 0, 5], [8, 0, 5]) == 2 / 15


def test_wmape_perfect_is_zero():
    assert wmape([3, 4, 5], [3, 4, 5]) == 0.0


def test_wmape_all_zero_actuals_is_nan():
    assert np.isnan(wmape([0, 0], [1, 2]))


def test_naive_scale_mean_squared_diff():
    # diffs of [1,2,3,4] = [1,1,1] -> mean square = 1
    assert naive_scale([1, 2, 3, 4]) == 1.0


def test_naive_scale_trims_leading_zeros():
    # from first sale: [2,4] -> diff [2] -> mean square 4
    assert naive_scale([0, 0, 2, 4]) == 4.0


def test_naive_scale_flat_series_is_nan():
    assert np.isnan(naive_scale([3, 3, 3]))


def test_rmsse_known_value():
    # scale from train [1,2,3,4] = 1 ; mse of preds = ((5-5)^2 + (6-7)^2)/2 = 0.5
    # rmsse = sqrt(0.5 / 1)
    assert rmsse([5, 6], [5, 7], [1, 2, 3, 4]) == np.sqrt(0.5)


def test_rmsse_perfect_forecast_is_zero():
    assert rmsse([5, 6], [5, 6], [1, 2, 3, 4]) == 0.0


def test_rmsse_undefined_scale_is_nan():
    # flat training series -> scale 0 -> nan (can't scale)
    assert np.isnan(rmsse([5, 6], [5, 7], [3, 3, 3]))
