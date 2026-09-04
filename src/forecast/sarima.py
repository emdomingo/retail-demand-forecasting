"""SARIMA comparison model (B3) — the classical, per-series counterpart to the global
LightGBM (B2). It plugs into the B1 harness through the same shape the baselines use: a
``name`` and ``forecast(train, test_keys) -> pd.Series`` aligned to ``test_keys.index``
(see baselines.py). SPEC frames B3 as the classical comparison and the slack-absorber; the
job here is an honest head-to-head, not to win.

Design choices that must survive interview scrutiny:

1. **Per-series, like ETS — the deliberate contrast to B2's global model.** SARIMA fits one
   state-space model per series on that series' own history. It is the "each series is its own
   little universe" philosophy; LightGBM is the "pool signal across all series" philosophy.
   Running both on the same harness/sample is the whole point of the comparison. Per-series
   fitting is slow, so we keep B2's fixed 200-series sample.

2. **A fixed, motivated order — NOT a per-series auto-search.** pmdarima's stepwise search is
   tempting but (a) carries numpy-2 compatibility friction and a heavy dependency, and (b) more
   importantly, searching an order per series across ~30k intermittent, often-short series is
   neither reproducible nor defensible — it overfits the *order itself* to noise. A single
   well-argued order is the honest scoped choice, consistent with "scope the model, state it
   explicitly." Order: **SARIMA(1,1,1)(1,0,0)_7**. Reasoning: AR(1)+MA(1) on a first difference
   (d=1) captures short-run level dynamics; one seasonal AR term (P=1, m=7) captures the weekly
   cycle — the structure ETS's additive season also targets, so the comparison is fair; **no
   seasonal differencing (D=0)** because differencing over m=7 on zero-heavy short series loses
   a week and routinely destabilises the fit for little gain.

3. **Robust by construction, because classical fits fail on intermittent demand.** Too-short or
   all-zero series skip the fit and fall back to the last value (matches ETS's contract). The
   optimiser runs with stationarity/invertibility unenforced and a bounded iteration count (it
   converges more often and can't hang), and any convergence/linalg failure also falls back.
   Forecasts are clipped to 0 — sales can't be negative.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class SARIMA:
    """Per-series seasonal ARIMA via statsmodels SARIMAX. Fixed order (see module docstring);
    fits one model per series on each origin's training window, so it is genuinely rolling."""

    def __init__(
        self,
        season: int = 7,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int] = (1, 0, 0),
        maxiter: int = 50,
    ):
        self.season = season
        self.order = order
        self.seasonal_order = seasonal_order
        self.maxiter = maxiter
        p, d, q = order
        P, D, Q = seasonal_order
        # Order encoded into the name so the MLflow run is self-describing and distinct from ETS.
        self.name = f"sarima_{p}{d}{q}_{P}{D}{Q}_{season}"

    def _forecast_one(self, y: np.ndarray, h: int) -> np.ndarray:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # A seasonal fit needs at least two full seasons of history; a degenerate all-zero series
        # has nothing to model. Both fall back to the last value, repeated (ETS's contract).
        if y.size < 2 * self.season or np.all(y == 0):
            return np.repeat(y[-1] if y.size else 0.0, h)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # convergence/frequency chatter, not signal
                model = SARIMAX(
                    y,
                    order=self.order,
                    seasonal_order=(*self.seasonal_order, self.season),
                    # Unenforced constraints => the optimiser converges more often on messy
                    # retail series; we clip the output to 0 anyway, so we don't need a provably
                    # stationary/invertible parameterisation.
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False, maxiter=self.maxiter)
            fc = np.asarray(fit.forecast(h), dtype=float)
            if not np.all(np.isfinite(fc)):  # diverged fit can emit inf/nan
                raise ValueError("non-finite forecast")
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


def main() -> None:
    from src.forecast.backtest import BacktestConfig, run_backtest
    from src.query.slices import read_store_slice

    store = "CA_3"
    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "sales"]
    df = read_store_slice(store, columns=cols)

    # Same fixed 200-series sample and seed as the baseline and LightGBM runs — a like-for-like
    # head-to-head so the RMSSE/WMAPE numbers are directly comparable across model families.
    rng = np.random.default_rng(0)
    sample_ids = rng.choice(df["id"].unique(), size=200, replace=False)
    sample = df[df["id"].isin(sample_ids)]
    cfg = BacktestConfig(extra_params={"scope": f"{store}_sample200"})

    model = SARIMA()
    print(f"Backtesting {model.name} on {len(sample_ids)} series from {store} (per-series fits)...")
    res = run_backtest(sample, model, cfg)
    print(f"\n{model.name}:")
    print(res.to_string(index=False))
    print(f"  MEAN  rmsse={res['rmsse'].mean():.4f}  wmape={res['wmape'].mean():.4f}")
    print("\n  bar to beat (B1/B2): lightgbm_v2 rmsse=0.727 | ets_7 rmsse=0.735 | snaive=0.96")


if __name__ == "__main__":
    main()
