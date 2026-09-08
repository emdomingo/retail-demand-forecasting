"""Tests for the D2 asymmetric-cost decision layer. The load-bearing properties: the critical
ratio is the textbook newsvendor formula, orders are read from the predictive-quantile grid and
move the right way with cost asymmetry, and the realised-cost accounting is correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecast.decision import (
    critical_ratio,
    newsvendor_orders,
    policy_costs,
    realised_cost,
)


def _grid() -> pd.DataFrame:
    """A monotone-in-q offset grid for one horizon: offset rises from negative (low q) through 0
    (median) to positive (high q), like a real scaled-residual quantile function."""
    qs = np.round(np.arange(0.1, 1.0, 0.1), 1)
    return pd.DataFrame({"h": 1, "q": qs, "offset": (qs - 0.5) * 4})  # -1.6 .. +1.6


def test_critical_ratio_is_the_newsvendor_formula():
    assert critical_ratio(1, 1) == 0.5
    assert critical_ratio(3, 1) == 0.75
    assert critical_ratio(1, 3) == 0.25
    with pytest.raises(ValueError):
        critical_ratio(0, 1)


def test_orders_read_the_nearest_grid_quantile_and_scale():
    fc = pd.DataFrame({"h": [1, 1], "yhat": [10.0, 10.0], "scale": [1.0, 2.0]})
    # q*=0.7 -> offset (0.7-0.5)*4 = 0.8; order = yhat + scale*offset.
    orders = newsvendor_orders(fc, _grid(), q_star=0.7)
    assert orders.iloc[0] == pytest.approx(10.8)
    assert orders.iloc[1] == pytest.approx(11.6)  # 2x scale widens the safety buffer


def test_orders_rise_with_the_service_level():
    fc = pd.DataFrame({"h": [1], "yhat": [10.0], "scale": [1.0]})
    lo = newsvendor_orders(fc, _grid(), 0.2).iloc[0]
    mid = newsvendor_orders(fc, _grid(), 0.5).iloc[0]
    hi = newsvendor_orders(fc, _grid(), 0.9).iloc[0]
    assert lo < mid < hi
    assert mid == pytest.approx(10.0)  # median offset is 0 -> order to the point forecast


def test_orders_are_clipped_non_negative():
    fc = pd.DataFrame({"h": [1], "yhat": [0.5], "scale": [1.0]})
    assert newsvendor_orders(fc, _grid(), 0.1).iloc[0] >= 0.0  # yhat + 1*(-1.6) would be < 0


def test_realised_cost_charges_each_side_correctly():
    actual = np.array([10.0, 10.0])
    order = np.array([7.0, 12.0])  # 3 short, then 2 long
    cost = realised_cost(actual, order, cu=4.0, co=1.0)
    assert cost[0] == pytest.approx(12.0)  # 3 units short * 4
    assert cost[1] == pytest.approx(2.0)  # 2 units long * 1


def test_policy_costs_lift_fill_rate_as_stockouts_get_costlier():
    rng = np.random.default_rng(0)
    n = 400
    fc = pd.DataFrame(
        {
            "h": 1,
            "sales": rng.poisson(10, n).astype(float),
            "yhat": np.full(n, 10.0),
            "scale": np.full(n, np.sqrt(10.0)),
        }
    )
    qs = np.round(np.arange(0.05, 1.0, 0.05), 2)
    # A monotone, roughly symmetric offset grid (a stand-in quantile function).
    grid = pd.DataFrame({"h": 1, "q": qs, "offset": (qs - 0.5) * 6})

    cheap = policy_costs(fc, grid, cu=1.0, co=1.0)
    dear = policy_costs(fc, grid, cu=9.0, co=1.0)
    nv_cheap = cheap[cheap.policy == "newsvendor"].iloc[0]
    nv_dear = dear[dear.policy == "newsvendor"].iloc[0]
    assert nv_dear.q_star > nv_cheap.q_star
    assert nv_dear.mean_order > nv_cheap.mean_order  # order more when short is costly
    assert nv_dear.fill_rate > nv_cheap.fill_rate  # -> higher service level
