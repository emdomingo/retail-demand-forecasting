"""Forecast accuracy metrics (B1): RMSSE and WMAPE.

RMSSE is the M5 error metric — a *scaled* error. The forecast's mean squared error is
divided by the mean squared error of a one-step naive forecast on the **training** series,
so a score < 1 beats naive-on-history and the metric is comparable across series of wildly
different volume. (M5's leaderboard uses WRMSSE, the same thing weighted by each series'
dollar sales; we report the unweighted mean RMSSE and name WRMSSE as an extension.)

WMAPE (weighted MAPE) = sum|error| / sum|actual| — a volume-weighted percentage error that,
unlike per-point MAPE, doesn't explode on the many zero-demand days in retail data.
"""

from __future__ import annotations

import numpy as np


def naive_scale(y_train: np.ndarray, trim_leading_zeros: bool = True) -> float:
    """Mean squared 1-step difference of the training series — the RMSSE denominator.

    M5 measures the scale from the item's first actual sale, so leading (pre-launch,
    structural) zeros don't deflate it. Returns nan if the series is too short or flat
    (no variation → an undefined scale).
    """
    y = np.asarray(y_train, dtype=float)
    if trim_leading_zeros:
        nonzero = np.flatnonzero(y > 0)
        if nonzero.size == 0:
            return np.nan
        y = y[nonzero[0]:]
    if y.size < 2:
        return np.nan
    scale = float(np.mean(np.diff(y) ** 2))
    # A flat series has zero naive error -> RMSSE would divide by zero; treat as undefined.
    return scale if scale > 0 else np.nan


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Root Mean Squared Scaled Error for one series over its forecast horizon."""
    scale = naive_scale(y_train)
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mse = np.mean((yt - yp) ** 2)
    return float(np.sqrt(mse / scale))


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE = sum|actual - forecast| / sum|actual|. nan if the actuals sum to 0."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(yt))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(yt - yp)) / denom)
