"""Persist the banded forecast (D1 wiring) — the bridge from the modelling half to the dashboard.

The dashboard is a *viewer*, not a model runner (CLAUDE.md: Spark→Parquet→DuckDB→models→
intervals→Streamlit; the dashboard reads a slice). So the expensive part — refit the model at
every rolling origin, calibrate conformal, band the held-out latest origin — runs here, offline,
and lands as a small Parquet artifact plus a JSON metadata sidecar. Streamlit then just reads
them; nothing in the app path imports LightGBM or touches the harness.

What lands: one row per (series, horizon day) of the **held-out latest origin** — the same
origin B4 reports coverage on — with the actual (``sales``), the point forecast (``yhat``), the
calibrated band (``lower``/``upper``/``width``), and the display keys (``dept_id``/``cat_id``).
The sidecar carries the header facts a planner needs to read the panel honestly: which model,
which origin, the target vs the *empirical* coverage on this origin, and the series count.

Scope note: we persist the **same fixed 200-series sample** (seed 0) used across B1–B4, so the
coverage in the sidecar is the number those runs report — the dashboard shows exactly the
backtest origin the intervals were evaluated on, not a fresh, unevaluated forecast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# NOTE: the model/harness/query imports (LightGBM, MLflow, Spark-backed slices) are deliberately
# *not* at module top. They're needed only by `build_forecast_artifact` (the offline build), and
# are imported lazily inside it — so importing this module for `load_forecast` (what the deployed
# dashboard does) stays lightweight and pulls in none of pyspark/lightgbm/mlflow. Tests lock this.

_REPO = Path(__file__).resolve().parents[2]
# The committed presentation bundle the dashboard reads (small model output, NOT the dataset —
# raw CSVs and the 59M-row feature store stay gitignored under data/). Committed so the read-only
# app deploys to $0 hosting without a Kaggle pull, a Spark build, or the heavy modelling deps.
FORECAST_DIR = _REPO / "dashboard_data"

SAMPLE_SIZE = 200  # the fixed B1–B4 sample; keeps the persisted coverage == the reported coverage
SAMPLE_SEED = 0
CONTEXT_DAYS = 56  # days of pre-origin actuals bundled for the chart's context line (2x horizon)
# Predictive-quantile grid the D2 newsvendor reads (order to the q* = Cu/(Cu+Co) quantile).
QUANTILE_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


@dataclass
class ForecastArtifact:
    """The persisted forecast for one store: the banded held-out-origin frame, the per-horizon
    predictive-quantile grid (for the D2 asymmetric-cost newsvendor), and the header metadata the
    dashboard renders above the panel."""

    forecast: pd.DataFrame
    meta: dict
    quantiles: pd.DataFrame | None = None
    context: pd.DataFrame | None = None  # pre-origin actuals for the chart's context line

    def parquet_path(self) -> Path:
        return FORECAST_DIR / f"forecast_{self.meta['store']}.parquet"

    def meta_path(self) -> Path:
        return FORECAST_DIR / f"forecast_{self.meta['store']}.json"

    def quantiles_path(self) -> Path:
        return FORECAST_DIR / f"quantiles_{self.meta['store']}.parquet"

    def context_path(self) -> Path:
        return FORECAST_DIR / f"context_{self.meta['store']}.parquet"


def build_forecast_artifact(
    store: str = "CA_3",
    alpha: float = 0.1,
    mode: str = "asymmetric",
    sample_size: int | None = SAMPLE_SIZE,
    df: pd.DataFrame | None = None,
) -> ForecastArtifact:
    """Refit v2 across the rolling origins, calibrate conformal, band the latest origin, and
    return it enriched with display keys. ``sample_size=None`` uses the whole store (slower; the
    sidecar coverage then no longer matches the documented 200-series B4 number). Pass ``df`` to
    band a caller-supplied panel instead of reading the store (used by the tests)."""
    # Heavy modelling/query deps are imported here, not at module top, so `load_forecast` stays
    # lightweight for the deployed reader (no pyspark/lightgbm/mlflow at import time).
    from src.forecast.backtest import BacktestConfig
    from src.forecast.intervals import (
        SplitConformal,
        calibration_quantile_grid,
        calibration_test_split,
        coverage_report,
        origin_forecasts,
    )
    from src.forecast.models import KNOWN_FUTURE, v2

    if df is None:
        from src.query.slices import read_store_slice

        cols = ["id", "dept_id", "cat_id", "date", "sales", *KNOWN_FUTURE]
        df = read_store_slice(store, columns=cols)

    if sample_size is not None and df["id"].nunique() > sample_size:
        rng = np.random.default_rng(SAMPLE_SEED)
        ids = rng.choice(df["id"].unique(), size=sample_size, replace=False)
        df = df[df["id"].isin(ids)]

    cfg = BacktestConfig(known_future=KNOWN_FUTURE, extra_params={"scope": f"{store}_sample200"})
    ff = origin_forecasts(df, v2(), cfg)
    # One split feeds both artifacts: cal → band + quantile grid, test → the held-out origin.
    cal, test, test_origin = calibration_test_split(ff)
    banded = SplitConformal(alpha=alpha, mode=mode).fit(cal).apply(test)
    quantiles = calibration_quantile_grid(cal, QUANTILE_GRID)
    summary, _ = coverage_report(banded)

    # Attach the display keys (one static row per series) so the dashboard needn't re-join.
    keys = df[["id", "dept_id", "cat_id"]].drop_duplicates("id")
    forecast = banded.merge(keys, on="id", how="left")
    forecast["origin"] = test_origin
    forecast = forecast[
        ["id", "dept_id", "cat_id", "origin", "date", "h",
         "sales", "yhat", "lower", "upper", "width", "scale"]
    ].sort_values(["id", "date"], ignore_index=True)

    # Bundle the last CONTEXT_DAYS of pre-origin actuals per series, so the deployed dashboard can
    # draw the chart's context line without reading the (gitignored, absent-on-host) feature store.
    pre = df[df["date"] <= test_origin][["id", "date", "sales"]]
    context = (
        pre.sort_values(["id", "date"])
        .groupby("id", sort=False)
        .tail(CONTEXT_DAYS)
        .reset_index(drop=True)
    )

    meta = {
        "store": store,
        "model": v2().name,
        "mode": mode,
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "empirical_coverage": summary["coverage"],
        "mean_width": summary["mean_width"],
        "test_origin": test_origin.date().isoformat(),
        "horizon": int(forecast["h"].max()),
        "n_series": int(forecast["id"].nunique()),
        "n_rows": int(len(forecast)),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return ForecastArtifact(
        forecast=forecast, meta=meta, quantiles=quantiles, context=context
    )


def persist_forecast(artifact: ForecastArtifact) -> ForecastArtifact:
    """Write the artifact's four files (forecast + quantile grid + context Parquet, meta JSON) into
    the committed dashboard_data/ bundle."""
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    artifact.forecast.to_parquet(artifact.parquet_path(), index=False)
    artifact.meta_path().write_text(json.dumps(artifact.meta, indent=2))
    if artifact.quantiles is not None:
        artifact.quantiles.to_parquet(artifact.quantiles_path(), index=False)
    if artifact.context is not None:
        artifact.context.to_parquet(artifact.context_path(), index=False)
    return artifact


def load_forecast(store: str = "CA_3") -> ForecastArtifact:
    """Read a persisted forecast artifact back (the dashboard's entry point)."""
    stub = ForecastArtifact(forecast=pd.DataFrame(), meta={"store": store})
    pq, mj, qp, cp = (
        stub.parquet_path(), stub.meta_path(), stub.quantiles_path(), stub.context_path()
    )
    if not all(p.exists() for p in (pq, mj, qp, cp)):
        raise FileNotFoundError(
            f"No persisted forecast for {store} at {pq}. "
            "Build it first: uv run python -m src.forecast.persist"
        )
    return ForecastArtifact(
        forecast=pd.read_parquet(pq),
        meta=json.loads(mj.read_text()),
        quantiles=pd.read_parquet(qp),
        context=pd.read_parquet(cp),
    )


def main() -> None:
    store = "CA_3"
    print(f"Building + persisting forecast artifact for {store} (v2, 90% conformal)...")
    art = persist_forecast(build_forecast_artifact(store))
    m = art.meta
    print(f"  wrote {art.parquet_path().relative_to(_REPO)}  "
          f"({m['n_rows']:,} rows, {m['n_series']} series)")
    print(f"  test origin {m['test_origin']}  horizon {m['horizon']}d")
    print(f"  coverage {m['empirical_coverage']:.3f} (target {m['target_coverage']:.2f})  "
          f"mean width {m['mean_width']:.2f}")
    print(f"  wrote {art.quantiles_path().relative_to(_REPO)}  "
          f"({len(art.quantiles):,} rows = {art.quantiles['q'].nunique()} quantiles x "
          f"{art.quantiles['h'].nunique()} horizons)")


if __name__ == "__main__":
    main()
