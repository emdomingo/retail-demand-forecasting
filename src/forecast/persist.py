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

from src.forecast.backtest import BacktestConfig
from src.forecast.intervals import calibrate_and_band, coverage_report, origin_forecasts
from src.forecast.models import KNOWN_FUTURE, v2
from src.query.slices import read_store_slice

_REPO = Path(__file__).resolve().parents[2]
FORECAST_DIR = _REPO / "data" / "processed" / "forecast"

SAMPLE_SIZE = 200  # the fixed B1–B4 sample; keeps the persisted coverage == the reported coverage
SAMPLE_SEED = 0


@dataclass
class ForecastArtifact:
    """The persisted forecast for one store: the banded held-out-origin frame plus the header
    metadata the dashboard renders above the panel."""

    forecast: pd.DataFrame
    meta: dict

    def parquet_path(self) -> Path:
        return FORECAST_DIR / f"forecast_{self.meta['store']}.parquet"

    def meta_path(self) -> Path:
        return FORECAST_DIR / f"forecast_{self.meta['store']}.json"


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
    if df is None:
        cols = ["id", "dept_id", "cat_id", "date", "sales", *KNOWN_FUTURE]
        df = read_store_slice(store, columns=cols)

    if sample_size is not None and df["id"].nunique() > sample_size:
        rng = np.random.default_rng(SAMPLE_SEED)
        ids = rng.choice(df["id"].unique(), size=sample_size, replace=False)
        df = df[df["id"].isin(ids)]

    cfg = BacktestConfig(known_future=KNOWN_FUTURE, extra_params={"scope": f"{store}_sample200"})
    ff = origin_forecasts(df, v2(), cfg)
    banded, test_origin = calibrate_and_band(ff, alpha=alpha, mode=mode)
    summary, _ = coverage_report(banded)

    # Attach the display keys (one static row per series) so the dashboard needn't re-join.
    keys = df[["id", "dept_id", "cat_id"]].drop_duplicates("id")
    forecast = banded.merge(keys, on="id", how="left")
    forecast["origin"] = test_origin
    forecast = forecast[
        ["id", "dept_id", "cat_id", "origin", "date", "h",
         "sales", "yhat", "lower", "upper", "width", "scale"]
    ].sort_values(["id", "date"], ignore_index=True)

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
    return ForecastArtifact(forecast=forecast, meta=meta)


def persist_forecast(artifact: ForecastArtifact) -> ForecastArtifact:
    """Write the artifact's Parquet + JSON sidecar under data/processed/forecast/ (gitignored)."""
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    artifact.forecast.to_parquet(artifact.parquet_path(), index=False)
    artifact.meta_path().write_text(json.dumps(artifact.meta, indent=2))
    return artifact


def load_forecast(store: str = "CA_3") -> ForecastArtifact:
    """Read a persisted forecast artifact back (the dashboard's entry point)."""
    stub = ForecastArtifact(forecast=pd.DataFrame(), meta={"store": store})
    pq, mj = stub.parquet_path(), stub.meta_path()
    if not pq.exists() or not mj.exists():
        raise FileNotFoundError(
            f"No persisted forecast for {store} at {pq}. "
            "Build it first: uv run python -m src.forecast.persist"
        )
    return ForecastArtifact(forecast=pd.read_parquet(pq), meta=json.loads(mj.read_text()))


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


if __name__ == "__main__":
    main()
