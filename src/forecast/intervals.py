"""Prediction intervals (B4) — the deliverable. A point forecast is a guess; a *calibrated
interval* is what a planner can actually staff and stock against, so this is the point of the
whole forecasting half (CLAUDE.md: "the interval is the deliverable, not the point estimate").

The method is **split conformal prediction**, adapted for our setting in two ways that matter and
must survive scrutiny:

1. **Per-horizon calibration.** B2's forecast is recursive, so error *compounds* across the
   28-day horizon — day 28 is far shakier than day 1. A single residual pool would hand every
   horizon the same width and be simultaneously too wide early and too narrow late. So we
   calibrate a **separate** residual quantile for each horizon step h = 1..28. The band widens
   with h exactly as the recursion's uncertainty does.

2. **Scaled (normalised) residuals.** Retail series span orders of magnitude in volume. Pooling
   raw residuals across series gives a band sized for the average series — far too wide for a
   slow item, too narrow for a fast one. So we normalise each residual by that series' scale
   (`sqrt(naive_scale)`, the RMSSE denominator's root — the series' typical one-step move) before
   pooling, calibrate in scaled space, then multiply the offset back by each series' own scale.
   High-volume series get wide bands, low-volume series get tight ones, from one calibration.

**The exchangeability caveat (named, per the guardrails).** Split conformal's coverage guarantee
assumes calibration and test residuals are *exchangeable* — which time series violate (temporal
dependence, drift). We mitigate honestly: calibrate on **earlier** origins and evaluate on a
**later** one (never the reverse — no leakage), and report *empirical* coverage on that held-out
future origin rather than asserting the theoretical guarantee. **EnbPI** (ensemble batch
prediction intervals) is the time-series-correct upgrade — a bootstrap ensemble with leave-one-out
residuals updated online as actuals arrive, needing no held-out calibration set — named here, not
built. **Quantile regression** (LightGBM `objective="quantile"` at α/2 and 1−α/2) is the
adaptive-width alternative; it's noted rather than built because under *recursive* forecasting its
quantiles are muddy — a quantile model fed its own point predictions as lags no longer emits a
true predictive quantile. Conformal wraps the existing point model cleanly and sidesteps that.

Coverage here is **marginal** (averaged over series), not conditional per series — the scaling
narrows that gap but doesn't close it; stated, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd

from src.forecast.backtest import (
    DEFAULT_TRACKING_URI,
    BacktestConfig,
    rolling_origins,
    train_test_split,
)
from src.forecast.metrics import naive_scale


def series_scale(y_train: np.ndarray, floor: float = 1.0) -> float:
    """Per-series interval scale = sqrt of the RMSSE naive scale (the series' typical one-step
    move, in sales units). Falls back to the training std, then a floor, so a flat or tiny series
    never yields a zero/undefined scale that would collapse the band."""
    s2 = naive_scale(y_train)
    if np.isfinite(s2) and s2 > 0:
        return float(np.sqrt(s2))
    y = np.asarray(y_train, dtype=float)
    std = float(np.std(y)) if y.size else 0.0
    return max(std, floor)


def origin_forecasts(
    df: pd.DataFrame, model, cfg: BacktestConfig | None = None
) -> pd.DataFrame:
    """Run the rolling-origin backtest and return the *raw* per-row forecasts (not just summary
    metrics) — the residual source conformal calibrates on. One row per (origin, series, test
    day) with: origin, id, date, ``h`` (1..horizon), sales, yhat, ``scale`` (from that origin's
    train), and ``resid`` = sales − yhat. Refits the model once per origin, so it's leakage-safe
    the same way the harness is."""
    cfg = cfg or BacktestConfig()
    unique_dates = np.sort(df["date"].unique())
    origins = rolling_origins(unique_dates, cfg)
    if not origins:
        raise ValueError("No valid origins — series too short for this horizon/step/n_origins.")

    frames = []
    for origin in origins:
        train, test = train_test_split(df, origin, cfg, unique_dates)
        test_cols = ["id", "date", *cfg.known_future]
        yhat = model.forecast(train, test[test_cols])

        scored = test[["id", "date", "sales"]].copy()
        scored["yhat"] = np.asarray(yhat)
        # Horizon step = position of the date within this origin's sorted test dates (1..horizon).
        test_dates = np.sort(scored["date"].unique())
        h_of = {d: i + 1 for i, d in enumerate(test_dates)}
        scored["h"] = scored["date"].map(h_of).astype(int)
        scored["origin"] = pd.Timestamp(origin)

        scale = {
            id_: series_scale(g.sort_values("date")["sales"].to_numpy(dtype=float))
            for id_, g in train.groupby("id", sort=False)
        }
        scored["scale"] = scored["id"].map(scale).fillna(1.0)
        scored["resid"] = scored["sales"] - scored["yhat"]
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


@dataclass
class SplitConformal:
    """Per-horizon, scale-normalised split conformal calibrator.

    ``alpha`` is the target miscoverage (0.1 → a 90% interval). ``mode``:
    - ``"asymmetric"`` (default): calibrate the two tails independently (α/2 each) on *signed*
      scaled residuals, so a right-skewed demand residual produces a band that's wider above than
      below — the honest shape, and the hook for the asymmetric-cost story.
    - ``"symmetric"``: one offset from the |scaled residual| (1−α) quantile.

    Quantiles use ``method="higher"``/``"lower"`` (conservative rounding) — the finite-sample
    conformal adjustment for small calibration sets, erring toward *slightly wider* bands so
    coverage isn't lost to interpolation."""

    alpha: float = 0.1
    mode: str = "asymmetric"

    def _tail_offsets(self, z: np.ndarray) -> tuple[float, float]:
        if self.mode == "symmetric":
            q = float(np.quantile(np.abs(z), 1 - self.alpha, method="higher"))
            return -q, q
        lo = float(np.quantile(z, self.alpha / 2, method="lower"))
        hi = float(np.quantile(z, 1 - self.alpha / 2, method="higher"))
        return lo, hi

    def fit(self, cal: pd.DataFrame) -> SplitConformal:
        """Learn (lower, upper) offsets in *scaled* residual space, one pair per horizon step.
        ``cal`` needs columns ``h``, ``resid``, ``scale``."""
        z_all = (cal["resid"] / cal["scale"]).to_numpy()
        self.offsets_: dict[int, tuple[float, float]] = {}
        for h, g in cal.groupby("h"):
            self.offsets_[int(h)] = self._tail_offsets((g["resid"] / g["scale"]).to_numpy())
        # Pooled fallback for any horizon unseen in calibration.
        self._pooled = self._tail_offsets(z_all)
        return self

    def apply(self, pred: pd.DataFrame) -> pd.DataFrame:
        """Attach ``lower``/``upper``/``width`` to ``pred`` (needs ``h``, ``yhat``, ``scale``).
        Offsets are de-normalised by each row's series scale; lower is clipped at 0 (sales ≥ 0)."""
        lo = pred["h"].map(lambda h: self.offsets_.get(int(h), self._pooled)[0]).to_numpy()
        hi = pred["h"].map(lambda h: self.offsets_.get(int(h), self._pooled)[1]).to_numpy()
        out = pred.copy()
        out["lower"] = np.clip(pred["yhat"].to_numpy() + pred["scale"].to_numpy() * lo, 0.0, None)
        out["upper"] = pred["yhat"].to_numpy() + pred["scale"].to_numpy() * hi
        out["upper"] = np.maximum(out["upper"].to_numpy(), out["lower"].to_numpy())
        out["width"] = out["upper"] - out["lower"]
        return out


def coverage_report(banded: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Empirical coverage and mean width, overall and per horizon step. ``banded`` needs
    ``sales``, ``lower``, ``upper``, ``h``. Coverage = fraction of actuals inside [lower, upper]."""
    b = banded.copy()
    b["covered"] = (b["sales"] >= b["lower"]) & (b["sales"] <= b["upper"])
    b["width"] = b["upper"] - b["lower"]
    summary = {
        "coverage": float(b["covered"].mean()),
        "mean_width": float(b["width"].mean()),
        "n": int(len(b)),
    }
    per_h = (
        b.groupby("h")
        .agg(coverage=("covered", "mean"), mean_width=("width", "mean"), n=("covered", "size"))
        .reset_index()
    )
    return summary, per_h


def calibration_test_split(
    ff: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """The time-ordered split every conformal consumer shares: every origin *before* the latest
    calibrates, the held-out latest origin is the test set. Returns (cal, test, test_origin).
    Calibrate on the past, test on the future, never the reverse — no leakage."""
    origins = np.sort(ff["origin"].unique())
    if len(origins) < 2:
        raise ValueError("Need >=2 origins: earlier ones calibrate, the latest one tests.")
    to = origins[-1]
    return ff[ff["origin"] < to], ff[ff["origin"] == to], pd.Timestamp(to)


def calibrate_and_band(
    ff: pd.DataFrame, alpha: float = 0.1, mode: str = "asymmetric"
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Calibrate conformal on the earlier origins and band the held-out latest origin. Returns the
    banded test-origin frame (with ``lower``/``upper``/``width``) and the test origin."""
    cal, test, test_origin = calibration_test_split(ff)
    banded = SplitConformal(alpha=alpha, mode=mode).fit(cal).apply(test)
    return banded, test_origin


def calibration_quantile_grid(
    cal: pd.DataFrame, quantiles: np.ndarray
) -> pd.DataFrame:
    """Per-horizon quantiles of the *scaled* calibration residuals — the conformal predictive
    quantile function the decision layer (D2 newsvendor) reads. One row per (h, q) with ``offset``
    = the q-quantile of ``resid/scale`` at horizon h; a predictive quantile for a test row is then
    ``yhat + scale * offset``. Same scale-normalisation as the band, so a single grid serves every
    series. Kept as a compact grid (not a live calibrator) so the dashboard never recomputes
    conformal — it just indexes the nearest q."""
    rows = []
    for h, g in cal.groupby("h"):
        z = (g["resid"] / g["scale"]).to_numpy()
        for q in quantiles:
            rows.append({"h": int(h), "q": float(q), "offset": float(np.quantile(z, q))})
    return pd.DataFrame(rows)


def evaluate_conformal(
    df: pd.DataFrame,
    model,
    cfg: BacktestConfig | None = None,
    alpha: float = 0.1,
    mode: str = "asymmetric",
) -> tuple[dict, pd.DataFrame]:
    """End-to-end: generate rolling-origin forecasts, calibrate conformal on all but the last
    origin, evaluate coverage on the held-out **last** (latest) origin, and log to MLflow.
    Time-ordered split — calibrate on the past, test on the future, never the reverse."""
    cfg = cfg or BacktestConfig()
    ff = origin_forecasts(df, model, cfg)
    banded, test_origin = calibrate_and_band(ff, alpha=alpha, mode=mode)
    summary, per_h = coverage_report(banded)

    mlflow.set_tracking_uri(cfg.tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=f"conformal_{mode}_a{alpha}"):
        mlflow.log_params(
            {
                "method": "split_conformal",
                "mode": mode,
                "alpha": alpha,
                "target_coverage": 1 - alpha,
                "base_model": getattr(model, "name", "unknown"),
                "cal_origins": int(ff["origin"].nunique()) - 1,
                "test_origin": test_origin.date().isoformat(),
                "n_series": int(df["id"].nunique()),
                **cfg.extra_params,
            }
        )
        mlflow.log_metric("coverage", summary["coverage"])
        mlflow.log_metric("mean_width", summary["mean_width"])
        for row in per_h.itertuples(index=False):
            mlflow.log_metric("coverage_by_h", float(row.coverage), step=int(row.h))
            mlflow.log_metric("width_by_h", float(row.mean_width), step=int(row.h))
    return summary, per_h


def main() -> None:
    from src.forecast.models import KNOWN_FUTURE, v2
    from src.query.slices import read_store_slice

    store = "CA_3"
    cols = ["id", "dept_id", "cat_id", "date", "sales", *KNOWN_FUTURE]
    df = read_store_slice(store, columns=cols)

    # Same fixed 200-series sample/seed as B1-B3 — the intervals wrap the model that won B2.
    rng = np.random.default_rng(0)
    sample_ids = rng.choice(df["id"].unique(), size=200, replace=False)
    sample = df[df["id"].isin(sample_ids)]
    cfg = BacktestConfig(known_future=KNOWN_FUTURE, extra_params={"scope": f"{store}_sample200"})

    print("Conformal intervals on lightgbm_global_v2 (target coverage 90%)...")
    for mode in ("asymmetric", "symmetric"):
        summary, per_h = evaluate_conformal(sample, v2(), cfg, alpha=0.1, mode=mode)
        print(f"\n{mode}:  coverage={summary['coverage']:.3f}  "
              f"mean_width={summary['mean_width']:.2f}  (target 0.900, n={summary['n']})")
        early = per_h[per_h["h"] <= 7]["coverage"].mean()
        late = per_h[per_h["h"] >= 22]["coverage"].mean()
        w_early = per_h[per_h["h"] <= 7]["mean_width"].mean()
        w_late = per_h[per_h["h"] >= 22]["mean_width"].mean()
        print(f"  horizon check — days 1-7:  coverage={early:.3f}  width={w_early:.2f}")
        print(f"                  days 22-28: coverage={late:.3f}  width={w_late:.2f}"
              "   (width should grow with the horizon)")


if __name__ == "__main__":
    main()
