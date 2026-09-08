"""Asymmetric-cost decision layer (D2) — turning the interval into an order quantity.

A point forecast answers "how much will sell?"; a *planner* has to answer "how much do I order?"
— and those differ whenever the cost of being short differs from the cost of being long. In
retail they almost always do: a stockout loses a sale (and maybe the customer), while an overstock
just ties up cash and shelf space. Under-forecasting usually hurts more (CLAUDE.md guardrail).

The classical answer is the **newsvendor model**. With per-unit underage cost ``Cu`` (short) and
overage cost ``Co`` (long), expected cost is minimised by ordering to the **critical-ratio
quantile** of demand:

    q* = Cu / (Cu + Co)          # only the *ratio* matters for where to order
    order = F⁻¹(q*)             # the q*-quantile of the predictive demand distribution

So the order point is a *quantile of the forecast*, and B4's conformal machinery already gives us
calibrated predictive quantiles (per horizon, scale-normalised) — the interval isn't decoration,
it's the input to the order. Ordering to the point forecast (``yhat``, ≈ the 50th percentile) is
the newsvendor solution only when Cu = Co; the moment stockouts cost more, q* > 0.5 and the
optimal order sits *above* the point forecast, inside the upper half of the band.

This module reads the persisted predictive-quantile grid (no conformal recompute) and evaluates
policies on the **held-out origin's actuals** — a realised, not assumed, cost comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def critical_ratio(cu: float, co: float) -> float:
    """Newsvendor critical ratio q* = Cu/(Cu+Co) — the service level to order to. Cu = cost per
    unit short (stockout), Co = cost per unit long (overstock). Cu=Co → 0.5 (order to the median);
    Cu>Co → >0.5 (order above the point forecast)."""
    if cu <= 0 or co <= 0:
        raise ValueError("costs must be positive")
    return cu / (cu + co)


def newsvendor_orders(
    forecast: pd.DataFrame, quantiles: pd.DataFrame, q_star: float
) -> pd.Series:
    """Order-up-to quantity per forecast row at service level ``q_star``: ``yhat + scale ×
    offset_h(q*)``, using the nearest grid quantile per horizon. Clipped at 0 (can't order
    negative). Index-aligned to ``forecast``.

    ``forecast`` needs ``h``, ``yhat``, ``scale``; ``quantiles`` has columns ``h``, ``q``,
    ``offset`` (the persisted per-horizon scaled-residual quantile grid)."""
    grid_q = np.sort(quantiles["q"].unique())
    nearest = float(grid_q[np.argmin(np.abs(grid_q - q_star))])
    offsets = (
        quantiles[quantiles["q"] == nearest].set_index("h")["offset"]
    )
    off = forecast["h"].map(offsets)
    # Any horizon missing from the grid falls back to the point forecast (offset 0).
    off = off.fillna(0.0).to_numpy()
    order = forecast["yhat"].to_numpy() + forecast["scale"].to_numpy() * off
    return pd.Series(np.clip(order, 0.0, None), index=forecast.index)


def realised_cost(actual: np.ndarray, order: np.ndarray, cu: float, co: float) -> np.ndarray:
    """Per-row realised newsvendor cost: Cu·max(demand−order,0) + Co·max(order−demand,0)."""
    actual = np.asarray(actual, dtype=float)
    order = np.asarray(order, dtype=float)
    short = np.clip(actual - order, 0.0, None)
    over = np.clip(order - actual, 0.0, None)
    return cu * short + co * over


def policy_costs(
    forecast: pd.DataFrame, quantiles: pd.DataFrame, cu: float, co: float
) -> pd.DataFrame:
    """Compare ordering policies on the held-out origin under the (Cu, Co) cost structure.

    Two policies on the same actuals:
    - **point** — order to ``yhat`` (what you'd do ignoring asymmetry);
    - **newsvendor** — order to the q* = Cu/(Cu+Co) predictive quantile (interval-aware).

    Returns one row per policy with total realised cost, achieved **fill rate** (units sold ÷
    units demanded), and mean order — so the panel can show that the interval-aware policy trades a
    little overstock for far fewer stockouts, and costs less whenever Cu ≠ Co."""
    q_star = critical_ratio(cu, co)
    actual = forecast["sales"].to_numpy(dtype=float)
    demand_total = actual.sum()

    policies = {
        "point": forecast["yhat"].clip(lower=0).to_numpy(),
        "newsvendor": newsvendor_orders(forecast, quantiles, q_star).to_numpy(),
    }
    rows = []
    for name, order in policies.items():
        cost = realised_cost(actual, order, cu, co)
        met = np.minimum(actual, order).sum()  # units of demand actually satisfied
        rows.append(
            {
                "policy": name,
                "total_cost": float(cost.sum()),
                "fill_rate": float(met / demand_total) if demand_total else np.nan,
                "mean_order": float(np.mean(order)),
                "q_star": q_star,
            }
        )
    return pd.DataFrame(rows)
