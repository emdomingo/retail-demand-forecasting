"""Rolling-origin backtesting harness (B1) — built before any model, because it's the
anti-leakage scaffold everything scores against (CLAUDE.md).

Rolling-origin evaluation mimics how a planner actually lives: stand at a past cutoff
(the *origin*), train only on what was known then, forecast the next `horizon` days, score
against what actually happened. Repeat at several origins. Train is always strictly on or
before the origin; test is strictly after — no future ever leaks into a fit.

Every run is logged to MLflow (params + per-origin and mean metrics) so the "compare model
families and defend the choice" comparison is reproducible, not a hand-typed table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.forecast.metrics import rmsse, wmape

# MLflow 3 blocks the legacy file store for tracking and recommends a SQL backend; SQLite is
# the zero-infra local choice. View runs with: uv run mlflow ui --backend-store-uri <this>.
_REPO = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = f"sqlite:///{_REPO / 'mlruns.db'}"


@dataclass
class BacktestConfig:
    horizon: int = 28          # days forecast per origin (M5's horizon)
    n_origins: int = 4         # number of rolling cutoffs
    step: int = 28             # days between successive origins
    season: int = 7            # weekly seasonality (for baselines / scale)
    experiment: str = "retail-demand-forecasting"
    tracking_uri: str | None = None  # defaults to the repo-local SQLite store
    extra_params: dict = field(default_factory=dict)  # e.g. {"scope": "CA_3"}


def rolling_origins(unique_dates: np.ndarray, cfg: BacktestConfig) -> list:
    """Origin dates, chronological. The latest origin leaves exactly `horizon` days of
    actuals after it; earlier origins step back by `cfg.step`. Origins that would leave no
    training history are dropped."""
    n = len(unique_dates)
    origins = []
    for i in range(cfg.n_origins):
        idx = n - 1 - cfg.horizon - i * cfg.step
        if idx < 1:  # need at least one training day before the origin
            break
        origins.append(unique_dates[idx])
    return list(reversed(origins))


def train_test_split(
    df: pd.DataFrame, origin, cfg: BacktestConfig, unique_dates: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window split at `origin`: train = all rows on/before origin, test = the next
    `horizon` days. Positional slicing on the sorted unique dates, so it's robust to any gaps."""
    i = int(np.searchsorted(unique_dates, origin))
    test_dates = set(unique_dates[i + 1 : i + 1 + cfg.horizon])
    train = df[df["date"] <= origin]
    test = df[df["date"].isin(test_dates)]
    return train, test


def evaluate_origin(train: pd.DataFrame, test: pd.DataFrame, model) -> dict:
    """Fit/forecast for one origin and score. RMSSE is per-series (scaled by that series'
    own training history) then averaged; WMAPE is pooled across all test rows."""
    yhat = model.forecast(train, test[["id", "date"]])
    scored = test[["id", "date", "sales"]].copy()
    scored["yhat"] = yhat

    train_sales = {id_: g["sales"].to_numpy() for id_, g in train.groupby("id", sort=False)}
    per_series_rmsse = [
        rmsse(g["sales"].to_numpy(), g["yhat"].to_numpy(), train_sales.get(id_, np.array([])))
        for id_, g in scored.groupby("id", sort=False)
    ]
    return {
        "rmsse": float(np.nanmean(per_series_rmsse)),
        "wmape": wmape(scored["sales"].to_numpy(), scored["yhat"].to_numpy()),
        "n_series": scored["id"].nunique(),
    }


def run_backtest(df: pd.DataFrame, model, cfg: BacktestConfig | None = None) -> pd.DataFrame:
    """Run the rolling-origin backtest for one model and log it to MLflow.

    Returns a per-origin metrics frame; the MLflow run also records the mean across origins.
    """
    cfg = cfg or BacktestConfig()
    unique_dates = np.sort(df["date"].unique())
    origins = rolling_origins(unique_dates, cfg)
    if not origins:
        raise ValueError("No valid origins — series too short for this horizon/step/n_origins.")

    rows = []
    mlflow.set_tracking_uri(cfg.tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=model.name):
        mlflow.log_params(
            {
                "model": model.name,
                "horizon": cfg.horizon,
                "n_origins": len(origins),
                "step": cfg.step,
                "season": cfg.season,
                "n_series": df["id"].nunique(),
                **cfg.extra_params,
            }
        )
        for step, origin in enumerate(origins):
            train, test = train_test_split(df, origin, cfg, unique_dates)
            m = evaluate_origin(train, test, model)
            m["origin"] = pd.Timestamp(origin).date().isoformat()
            rows.append(m)
            mlflow.log_metric("rmsse", m["rmsse"], step=step)
            mlflow.log_metric("wmape", m["wmape"], step=step)

        result = pd.DataFrame(rows)
        mlflow.log_metric("rmsse_mean", float(result["rmsse"].mean()))
        mlflow.log_metric("wmape_mean", float(result["wmape"].mean()))
    return result


def main() -> None:
    from src.forecast.baselines import ETS, SeasonalNaive
    from src.query.slices import read_store_slice

    store = "CA_3"
    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "sales"]
    df = read_store_slice(store, columns=cols)

    # Fair head-to-head on a fixed sample (ETS fits per series and is slow); the harness and
    # the seasonal-naive baseline scale to all series unchanged — this is a runtime choice.
    rng = np.random.default_rng(0)
    sample_ids = rng.choice(df["id"].unique(), size=200, replace=False)
    sample = df[df["id"].isin(sample_ids)]
    cfg = BacktestConfig(extra_params={"scope": f"{store}_sample200"})

    print(f"Backtesting on {len(sample_ids)} series from {store}, horizon {cfg.horizon}...")
    for model in (SeasonalNaive(), ETS()):
        res = run_backtest(sample, model, cfg)
        print(f"\n{model.name}:")
        print(res.to_string(index=False))
        print(f"  MEAN  rmsse={res['rmsse'].mean():.4f}  wmape={res['wmape'].mean():.4f}")


if __name__ == "__main__":
    main()
