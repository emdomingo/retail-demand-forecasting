"""Global LightGBM forecaster (B2) — one gradient-boosted model fit across *all* series in the
slice, the approach that won M5. It plugs into the B1 harness through the same shape the
baselines use: a ``name`` and ``forecast(train, test_keys) -> pd.Series`` aligned to
``test_keys.index`` (see baselines.py).

Three design choices that must survive interview scrutiny:

1. **Recursive multi-step.** The horizon is 28 days but a lag_7 for day 10 depends on the sale
   on day 3 — inside the forecast window, unknown to a real planner. So we forecast day by day,
   feed each prediction back in as if it were the actual, and recompute lags/means. Errors
   compound across the horizon; that is honest, and it is exactly the uncertainty B4's intervals
   must capture. (The alternative, reading the store's precomputed lag_7 for test rows, would
   feed the model future actuals — leakage.)

2. **AR features are rebuilt here from raw sales**, not read from the feature store. Training and
   the recursive test path call the *same* builder, so they can never silently disagree, and the
   test path is provably a function of (past actuals + own past predictions) only.

3. **Series identity via lags/means + dept/cat, not a 3049-way item_id categorical.** The lag and
   rolling-mean features already carry each series' level; a per-item categorical would explode
   tree size and invite overfit. dept_id/cat_id give the model the coarse structure that pools
   well across series. Calendar (wday/month/year) is derived from the date in both paths so the
   convention is identical; price/snap/event come in as known-future exogenous columns the
   harness passes for test rows (BacktestConfig.known_future).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

LAGS = (7, 28)
ROLL_WINDOWS = (7, 28)

# Static per-series attributes (looked up from train for test rows) and time-varying exogenous
# columns (handed in per test row via known_future). Kept explicit so the feature contract is
# auditable.
STATIC_CATS = ["dept_id", "cat_id"]
EXOG_NUM = ["sell_price", "price_change_pct", "snap", "is_event"]
EXOG_CAT = ["event_type_1"]
CAL_NUM = ["wday", "month", "year"]

_DEFAULT_PARAMS = {
    "objective": "tweedie",  # intermittent, non-negative retail demand (many zeros)
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}


class LightGBMForecaster:
    """Global recursive LightGBM. Refit at every origin on that origin's train window (the
    harness calls ``forecast`` once per origin), so it is genuinely rolling — no future leaks."""

    def __init__(
        self,
        lags: tuple[int, ...] = LAGS,
        roll_windows: tuple[int, ...] = ROLL_WINDOWS,
        num_boost_round: int = 300,
        params: dict | None = None,
        version: str = "v1",
    ):
        self.lags = lags
        self.roll_windows = roll_windows
        self.num_boost_round = num_boost_round
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        # Version flows into the MLflow run name so the v1 (thin) and v2 (enriched) feature sets
        # are two comparable rows, not one run silently overwritten.
        self.version = version
        self.name = f"lightgbm_global_{version}"
        self._cat_features = [*STATIC_CATS, *EXOG_CAT]

    # --- feature construction -------------------------------------------------------------

    def _calendar(self, dates: pd.Series) -> pd.DataFrame:
        """Calendar features straight from the date — known for any future day, leak-free, and
        derived identically for train and test so the two paths agree."""
        dt = pd.DatetimeIndex(dates)
        return pd.DataFrame(
            {"wday": dt.dayofweek, "month": dt.month, "year": dt.year},
            index=dates.index,
        )

    def _ar_train(self, train: pd.DataFrame) -> pd.DataFrame:
        """Lag and trailing-mean features for the training rows, per series, chronological.
        Rolling means use ``shift(1)`` so 'today' is never in its own average (matches A2)."""
        g = train.groupby("id", sort=False)["sales"]
        feats = {}
        for k in self.lags:
            feats[f"lag_{k}"] = g.shift(k)
        for w in self.roll_windows:
            feats[f"rmean_{w}"] = g.shift(1).rolling(w).mean()
        return pd.DataFrame(feats, index=train.index)

    def _feature_columns(self) -> list[str]:
        ar = [f"lag_{k}" for k in self.lags] + [f"rmean_{w}" for w in self.roll_windows]
        return ar + CAL_NUM + EXOG_NUM + STATIC_CATS + EXOG_CAT

    def _as_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Give the categorical columns a pandas ``category`` dtype so LightGBM treats them
        natively. Categories are frozen at fit time and reused at predict time (via
        ``self._cat_dtypes``) so train and test share one encoding."""
        for col in self._cat_features:
            X[col] = X[col].astype(self._cat_dtypes[col])
        return X

    # --- fit / forecast -------------------------------------------------------------------

    def _fit(self, train: pd.DataFrame) -> None:
        train = train.sort_values(["id", "date"])
        ar = self._ar_train(train)
        cal = self._calendar(train["date"])
        X = pd.concat([ar, cal, train[EXOG_NUM + STATIC_CATS + EXOG_CAT]], axis=1)
        X = X[self._feature_columns()]
        y = train["sales"].to_numpy(dtype=float)

        # Freeze categorical encodings on the training data, then apply.
        self._cat_dtypes = {
            col: pd.CategoricalDtype(categories=X[col].dropna().unique())
            for col in self._cat_features
        }
        X = self._as_categoricals(X)

        dset = lgb.Dataset(X, label=y, categorical_feature=self._cat_features, free_raw_data=False)
        self._model = lgb.train(self.params, dset, num_boost_round=self.num_boost_round)

        # Static per-series attributes, looked up for test rows the harness only gives id/date.
        self._series_attrs = train.groupby("id", sort=False)[STATIC_CATS].first().to_dict("index")
        # Per-series sales history (chronological) seeds the recursion.
        self._history = {
            id_: g.sort_values("date")["sales"].to_numpy(dtype=float)
            for id_, g in train.groupby("id", sort=False)
        }

    def _ar_row(self, hist: np.ndarray) -> dict:
        """AR features for the next step given a series' history so far (actuals then own
        predictions). Short histories fall back to NaN, which LightGBM handles natively."""
        feats = {}
        for k in self.lags:
            feats[f"lag_{k}"] = hist[-k] if hist.size >= k else np.nan
        for w in self.roll_windows:
            feats[f"rmean_{w}"] = hist[-w:].mean() if hist.size >= 1 else np.nan
        return feats

    def forecast(self, train: pd.DataFrame, test_keys: pd.DataFrame) -> pd.Series:
        self._fit(train)
        preds = pd.Series(0.0, index=test_keys.index)
        history = {id_: h.copy() for id_, h in self._history.items()}

        # Lockstep over the horizon: for each future date, build one row per series, predict the
        # whole batch, then append predictions to each history before stepping to the next date.
        for _, day_rows in test_keys.sort_values("date").groupby("date", sort=True):
            rows = day_rows.reset_index()  # keep original index in 'index'
            batch = []
            for r in rows.itertuples(index=False):
                id_ = r.id
                hist = history.get(id_, np.array([], dtype=float))
                feats = self._ar_row(hist)
                feats.update(self._series_attrs.get(id_, {c: np.nan for c in STATIC_CATS}))
                for col in EXOG_NUM + EXOG_CAT:
                    feats[col] = getattr(r, col, np.nan)
                batch.append(feats)

            X = pd.DataFrame(batch, index=rows["index"])
            cal = self._calendar(rows.set_index("index")["date"])
            X = pd.concat([X, cal], axis=1)[self._feature_columns()]
            X = self._as_categoricals(X)

            yhat = np.clip(self._model.predict(X), 0.0, None)
            for idx, id_, y in zip(rows["index"], rows["id"], yhat, strict=True):
                preds.loc[idx] = y
                history[id_] = np.append(history.get(id_, np.array([], dtype=float)), y)
        return preds


# Exogenous columns a planner legitimately knows for the future — handed to the model for test
# rows by the harness. sales and precomputed AR lags are deliberately NOT here (they'd leak).
KNOWN_FUTURE = ["sell_price", "price_change_pct", "snap", "is_event", "event_type_1"]


def main() -> None:
    from src.forecast.backtest import BacktestConfig, run_backtest
    from src.query.slices import read_store_slice

    store = "CA_3"
    cols = [
        "id",
        "dept_id",
        "cat_id",
        "date",
        "sales",
        *KNOWN_FUTURE,
    ]
    df = read_store_slice(store, columns=cols)

    # Same fixed 200-series sample and seed as the baseline run (backtest.main) — a like-for-like
    # head-to-head. The global model itself scales to all series unchanged; the sample is a
    # runtime choice, kept identical so the RMSSE numbers are directly comparable.
    rng = np.random.default_rng(0)
    sample_ids = rng.choice(df["id"].unique(), size=200, replace=False)
    sample = df[df["id"].isin(sample_ids)]
    cfg = BacktestConfig(known_future=KNOWN_FUTURE, extra_params={"scope": f"{store}_sample200"})

    print(f"Backtesting LightGBM on {len(sample_ids)} series from {store}...")
    res = run_backtest(sample, LightGBMForecaster(), cfg)
    print(f"\n{LightGBMForecaster().name}:")
    print(res.to_string(index=False))
    print(f"  MEAN  rmsse={res['rmsse'].mean():.4f}  wmape={res['wmape'].mean():.4f}")
    print("  bar to beat (B1): ets_7 rmsse=0.735  |  seasonal_naive rmsse=0.96")


if __name__ == "__main__":
    main()
