"""Tests for the D1 forecast-persistence layer. The load-bearing properties: the artifact
round-trips to disk unchanged, the schema the dashboard reads is stable, the sidecar coverage is
*the same number* B4 reports on the banded frame (not a re-labelled one), and a missing artifact
fails loudly rather than showing a stale/blank panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecast import persist
from src.forecast.intervals import coverage_report
from src.forecast.persist import (
    ForecastArtifact,
    build_forecast_artifact,
    load_forecast,
    persist_forecast,
)

FORECAST_COLUMNS = [
    "id", "dept_id", "cat_id", "origin", "date", "h",
    "sales", "yhat", "lower", "upper", "width", "scale",
]


@pytest.fixture(autouse=True)
def _tmp_forecast_dir(tmp_path, monkeypatch):
    """Redirect the artifact directory to a tmp path so tests never touch data/processed/."""
    monkeypatch.setattr(persist, "FORECAST_DIR", tmp_path)
    return tmp_path


def _panel(n_series: int = 5, days: int = 160, seed: int = 0) -> pd.DataFrame:
    """A small multi-series panel with the columns the v2 model + known-future contract need —
    enough days for 4 rolling origins (needs > ~114 days) so calibrate_and_band has >=2 origins."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=days, freq="D")
    rows = []
    for s in range(n_series):
        base = 5 + 3 * (s % 3)
        for d in dates:
            wk = 1 + (d.dayofweek >= 5)
            rows.append(
                {
                    "id": f"S{s}",
                    "dept_id": f"D{s % 2}",
                    "cat_id": "C0",
                    "date": d,
                    "sales": max(0, int(rng.poisson(base * wk))),
                    "sell_price": 3.0,
                    "price_change_pct": 0.0,
                    "snap": int(d.day <= 10),
                    "is_event": 0,
                    "event_type_1": "none",
                }
            )
    return pd.DataFrame(rows)


def _artifact() -> ForecastArtifact:
    return build_forecast_artifact(store="TEST", df=_panel(), sample_size=None)


def test_artifact_has_the_dashboard_schema():
    art = _artifact()
    assert list(art.forecast.columns) == FORECAST_COLUMNS
    assert art.forecast["h"].min() == 1
    # One row per (series, horizon day); horizon defaults to 28.
    assert art.meta["horizon"] == 28
    assert len(art.forecast) == art.meta["n_series"] * art.meta["horizon"]


def test_sidecar_coverage_matches_the_banded_frame():
    # The number shown in the header must BE coverage_report's number, not a re-derived one.
    art = _artifact()
    summary, _ = coverage_report(art.forecast)
    assert art.meta["empirical_coverage"] == pytest.approx(summary["coverage"])
    assert art.meta["mean_width"] == pytest.approx(summary["mean_width"])
    assert art.meta["target_coverage"] == pytest.approx(1 - art.meta["alpha"])


def test_quantile_grid_is_built_and_monotone():
    # The D2 newsvendor reads this grid; per horizon the offset must rise with q (a valid
    # quantile function), and every horizon must be present.
    art = _artifact()
    q = art.quantiles
    assert set(q.columns) == {"h", "q", "offset"}
    assert set(q["h"].unique()) == set(range(1, art.meta["horizon"] + 1))
    for _, g in q.groupby("h"):
        offs = g.sort_values("q")["offset"].to_numpy()
        assert (np.diff(offs) >= -1e-9).all()  # non-decreasing in q


def test_persist_load_roundtrips(_tmp_forecast_dir):
    art = _artifact()
    persist_forecast(art)
    assert art.parquet_path().exists() and art.meta_path().exists()
    assert art.quantiles_path().exists()

    loaded = load_forecast("TEST")
    pd.testing.assert_frame_equal(loaded.forecast, art.forecast)
    pd.testing.assert_frame_equal(loaded.quantiles, art.quantiles)
    assert loaded.meta == art.meta


def test_load_requires_the_quantile_grid(_tmp_forecast_dir):
    # A forecast without its quantile grid is incomplete for D2 -> fail loudly, not silently.
    art = _artifact()
    persist_forecast(art)
    art.quantiles_path().unlink()
    with pytest.raises(FileNotFoundError, match="Build it first"):
        load_forecast("TEST")


def test_load_missing_artifact_raises_with_guidance():
    with pytest.raises(FileNotFoundError, match="Build it first"):
        load_forecast("NOPE")


def test_bands_are_ordered_and_non_negative():
    # The invariants the chart relies on: lower <= upper, lower >= 0 (sales can't be negative).
    fc = _artifact().forecast
    assert (fc["upper"] >= fc["lower"]).all()
    assert (fc["lower"] >= 0).all()
