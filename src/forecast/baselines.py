"""Baseline forecasters (B1). Every model in this project is scored against these — a win
is only meaningful relative to seasonal-naive (CLAUDE.md).

Model shape (structural, no base class needed): a forecaster has a ``name`` and a
``forecast(train, test_keys) -> pd.Series`` that returns predictions aligned to
``test_keys.index``. ``train`` has at least (id, date, sales); ``test_keys`` has the
(id, date) rows to predict. This one shape serves per-series classical models here and the
global LightGBM later — the harness never needs to know which kind it's driving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalNaive:
    """Forecast = the last observed season, repeated. With season=7, tomorrow's forecast is
    'what sold on the same weekday last week' — a strong, honest baseline for weekly retail
    demand and the one everything else must beat."""

    def __init__(self, season: int = 7):
        self.season = season
        self.name = f"seasonal_naive_{season}"

    def forecast(self, train: pd.DataFrame, test_keys: pd.DataFrame) -> pd.Series:
        preds = pd.Series(0.0, index=test_keys.index)
        last_season = {
            id_: g.sort_values("date")["sales"].to_numpy()[-self.season:]
            for id_, g in train.groupby("id", sort=False)
        }
        for id_, g in test_keys.sort_values("date").groupby("id", sort=False):
            season_vals = last_season.get(id_)
            if season_vals is None or season_vals.size == 0:
                continue  # unseen series -> leave at 0
            h = np.arange(len(g))
            preds.loc[g.index] = season_vals[h % season_vals.size]
        return preds


class ETS:
    """Exponential smoothing (Holt-Winters) fit per series. A classical statistical baseline;
    on intermittent daily demand it's often mediocre, which is exactly why it's a baseline.
    Fits that fail (too-short or degenerate series) fall back to the last value, and negative
    forecasts are clipped to 0 (sales can't be negative)."""

    def __init__(self, season: int = 7):
        self.season = season
        self.name = f"ets_{season}"

    def _forecast_one(self, y: np.ndarray, h: int) -> np.ndarray:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        # Seasonal ETS needs at least two full seasons; otherwise fall back to last value.
        if y.size < 2 * self.season or np.all(y == 0):
            return np.repeat(y[-1] if y.size else 0.0, h)
        try:
            fit = ExponentialSmoothing(
                y, trend=None, seasonal="add", seasonal_periods=self.season
            ).fit()
            fc = np.asarray(fit.forecast(h), dtype=float)
        except Exception:
            fc = np.repeat(y[-1], h)
        return np.clip(fc, 0.0, None)

    def forecast(self, train: pd.DataFrame, test_keys: pd.DataFrame) -> pd.Series:
        preds = pd.Series(0.0, index=test_keys.index)
        train_by_id = {
            id_: g.sort_values("date")["sales"].to_numpy(dtype=float)
            for id_, g in train.groupby("id", sort=False)
        }
        for id_, g in test_keys.sort_values("date").groupby("id", sort=False):
            y = train_by_id.get(id_, np.array([], dtype=float))
            preds.loc[g.index] = self._forecast_one(y, len(g))
        return preds
