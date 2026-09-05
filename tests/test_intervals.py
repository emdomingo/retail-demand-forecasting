"""Tests for the B4 conformal prediction intervals. We test the calibration contract and that
coverage lands near target on a *controlled* synthetic residual distribution — the interval, not
the point forecast, is the deliverable, so its coverage is what must be right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast.intervals import (
    SplitConformal,
    coverage_report,
    series_scale,
)


def test_series_scale_uses_naive_scale_root():
    # A clean linear ramp: 1-step diffs are all 1 -> naive_scale=1 -> scale=sqrt(1)=1.
    y = np.arange(1, 21, dtype=float)
    assert series_scale(y) == 1.0


def test_series_scale_floors_a_flat_series():
    # Flat series: naive_scale undefined and std 0 -> falls back to the floor, never 0.
    assert series_scale(np.full(10, 5.0)) == 1.0


def test_bands_are_ordered_and_non_negative():
    cal = pd.DataFrame(
        {"h": [1] * 100, "resid": np.linspace(-5, 5, 100), "scale": np.ones(100)}
    )
    conf = SplitConformal(alpha=0.1, mode="asymmetric").fit(cal)
    pred = pd.DataFrame({"h": [1, 1, 1], "yhat": [0.5, 10.0, 100.0], "scale": [1.0, 1.0, 1.0]})
    out = conf.apply(pred)
    assert (out["upper"] >= out["lower"]).all()
    assert (out["lower"] >= 0).all()  # sales can't be negative
    assert (out["width"] >= 0).all()


def test_width_scales_with_series_scale():
    cal = pd.DataFrame({"h": [1] * 200, "resid": np.linspace(-3, 3, 200), "scale": np.ones(200)})
    conf = SplitConformal(alpha=0.1).fit(cal)
    pred = pd.DataFrame({"h": [1, 1], "yhat": [50.0, 50.0], "scale": [1.0, 4.0]})
    out = conf.apply(pred)
    # Same yhat, 4x scale -> ~4x band width (offsets are de-normalised by the series scale).
    assert out["width"].iloc[1] > 3.5 * out["width"].iloc[0]


def test_per_horizon_offsets_differ_when_residuals_grow():
    # Residuals tight at h=1, wide at h=28 -> the h=28 band must be wider.
    rng = np.random.default_rng(0)
    cal = pd.concat(
        [
            pd.DataFrame({"h": 1, "resid": rng.normal(0, 1, 500), "scale": 1.0}),
            pd.DataFrame({"h": 28, "resid": rng.normal(0, 5, 500), "scale": 1.0}),
        ]
    )
    conf = SplitConformal(alpha=0.1).fit(cal)
    pred = pd.DataFrame({"h": [1, 28], "yhat": [10.0, 10.0], "scale": [1.0, 1.0]})
    out = conf.apply(pred)
    assert out["width"].iloc[1] > 3 * out["width"].iloc[0]


def test_symmetric_coverage_hits_target_on_synthetic_residuals():
    # Calibrate on one draw, evaluate coverage on an independent draw from the same distribution.
    rng = np.random.default_rng(42)
    cal = pd.DataFrame({"h": 1, "resid": rng.normal(0, 2.0, 5000), "scale": 1.0})
    conf = SplitConformal(alpha=0.1, mode="symmetric").fit(cal)

    n = 5000
    yhat = np.full(n, 20.0)
    y = yhat + rng.normal(0, 2.0, n)  # true actuals share the residual distribution
    test = pd.DataFrame({"h": 1, "sales": y, "yhat": yhat, "scale": 1.0})
    banded = conf.apply(test)
    summary, _ = coverage_report(banded)
    assert 0.88 <= summary["coverage"] <= 0.93  # ~90% target, small-sample slack


def test_coverage_report_counts_correctly():
    banded = pd.DataFrame(
        {
            "h": [1, 1, 2, 2],
            "sales": [5.0, 50.0, 5.0, 5.0],  # row 2 (50) falls outside its band
            "lower": [0.0, 0.0, 0.0, 0.0],
            "upper": [10.0, 10.0, 10.0, 10.0],
        }
    )
    summary, per_h = coverage_report(banded)
    assert summary["coverage"] == 0.75  # 3 of 4 covered
    assert summary["n"] == 4
    assert per_h.loc[per_h["h"] == 1, "coverage"].iloc[0] == 0.5  # 1 of 2 at h=1
    assert per_h.loc[per_h["h"] == 2, "coverage"].iloc[0] == 1.0
