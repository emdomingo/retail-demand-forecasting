"""Streamlit dashboard (D1) — the planner's viewing surface.

D1 is the forecast half made visible: for a chosen series, the recent actuals, then the 28-day
held-out-origin forecast (B2's LightGBM v2) wrapped in B4's calibrated conformal band, against
what actually happened. The interval is the deliverable, so the band — not the point line — is
the visual centre of the panel.

This app is a *pure reader* (CLAUDE.md: the dashboard reads a slice, it doesn't run models). The
banded forecast is pre-computed and persisted by ``src.forecast.persist``; here we only load that
Parquet + its metadata sidecar, pull the pre-origin actuals for context via the DuckDB query
layer, and draw. Nothing in this path imports LightGBM or the backtest harness.

Run: ``uv run streamlit run src/dashboard/app.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` executes this file as a script, so the repo root isn't on sys.path the way it
# is under pytest (pythonpath=.) or `python -m`. Put it there before the `src.` imports resolve,
# so `uv run streamlit run src/dashboard/app.py` works from a clean checkout with no PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.forecast.persist import load_forecast  # noqa: E402
from src.query.slices import read_store_slice  # noqa: E402

CONTEXT_DAYS = 56  # actual history shown before the origin (two seasonal cycles of 28)
DEFAULT_STORE = "CA_3"


@st.cache_data(show_spinner=False)
def _load_artifact(store: str):
    """Persisted banded forecast + metadata for one store (cached across reruns)."""
    art = load_forecast(store)
    return art.forecast, art.meta


@st.cache_data(show_spinner=False)
def _load_history(store: str, ids: tuple[str, ...]) -> pd.DataFrame:
    """Pre-origin actuals for the selected series, via the DuckDB query layer. Reads
    the whole store slice once (cached) and filters — the ORDER BY date discipline
    lives in the query layer."""
    df = read_store_slice(store, columns=["id", "date", "sales"])
    return df[df["id"].isin(ids)]


def _forecast_chart(hist: pd.DataFrame, fc: pd.DataFrame, origin: pd.Timestamp) -> alt.Chart:
    """Actual line (context + horizon) + conformal band + point forecast, with an origin rule.
    The shaded band is the star: everything else is drawn thin so the interval reads first."""
    band = (
        alt.Chart(fc)
        .mark_area(opacity=0.25, color="#4c78a8")
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("lower:Q", title="units/day"),
            y2="upper:Q",
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("lower:Q", format=".1f"),
                alt.Tooltip("upper:Q", format=".1f"),
            ],
        )
    )
    yhat = (
        alt.Chart(fc)
        .mark_line(color="#4c78a8", strokeDash=[4, 3])
        .encode(x="date:T", y="yhat:Q")
    )
    # Actual in a bold orange, not near-black: it must read on BOTH the light and the dark
    # Streamlit theme, and against the blue band it needs a contrasting hue (black-on-dark was
    # invisible outside the band region — the context history vanished into the background).
    actual = (
        alt.Chart(hist)
        .mark_line(color="#f58518", strokeWidth=2)
        .encode(
            x="date:T",
            y="sales:Q",
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("sales:Q", title="actual", format=".0f")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"origin": [origin]}))
        .mark_rule(color="#c44", strokeDash=[2, 2])
        .encode(x="origin:T")
    )
    return (band + yhat + actual + rule).properties(height=380).interactive()


def main() -> None:
    st.set_page_config(page_title="Retail Demand Forecast", layout="wide")
    st.title("Retail demand forecast — point + calibrated interval")

    store = DEFAULT_STORE
    try:
        fc_all, meta = _load_artifact(store)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    origin = pd.Timestamp(meta["test_origin"])

    # Header: the facts a planner needs to read the panel honestly.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", meta["model"])
    c2.metric(
        "Interval coverage",
        f"{meta['empirical_coverage']:.1%}",
        f"target {meta['target_coverage']:.0%}",
        delta_color="off",
    )
    c3.metric("Mean band width", f"{meta['mean_width']:.2f} units")
    c4.metric("Held-out origin", meta["test_origin"])
    st.caption(
        f"{meta['mode'].title()} split-conformal band on the held-out latest backtest origin "
        f"({meta['n_series']} series from {store}). Coverage is marginal over series, not "
        "per-series conditional — a named B4 limitation."
    )

    # Series picker, ordered by forecast-window volume so the default panel is meaningful.
    vol = fc_all.groupby("id")["sales"].mean().sort_values(ascending=False)
    order = vol.index.tolist()
    labels = {
        i: f"{i.replace('_evaluation', '')}  (~{vol[i]:.0f}/day)" for i in order
    }
    series = st.selectbox("Series", order, format_func=lambda i: labels[i])

    fc = fc_all[fc_all["id"] == series].sort_values("date")
    hist_all = _load_history(store, (series,))
    ctx = hist_all[(hist_all["date"] <= origin)].tail(CONTEXT_DAYS)
    # Bridge the context line into the horizon so the actual line is continuous.
    hist = pd.concat(
        [ctx[["date", "sales"]], fc[["date", "sales"]]], ignore_index=True
    ).drop_duplicates("date")

    st.altair_chart(_forecast_chart(hist, fc, origin), width="stretch")
    st.caption(
        "🟧 actual  ·  ▫ shaded = 90% conformal interval  ·  ┄ dashed = point forecast  "
        "·  ┊ red = forecast start (held-out origin). Left of the red line is history the "
        "model trained on; right of it is the 28-day forecast vs what actually happened."
    )

    # Per-series read: how did the band actually do on this one series over the 28 days?
    covered = ((fc["sales"] >= fc["lower"]) & (fc["sales"] <= fc["upper"])).mean()
    d1, d2, d3 = st.columns(3)
    d1.metric(
        "This series — coverage", f"{covered:.0%}",
        f"{len(fc)} horizon days", delta_color="off",
    )
    d1.caption(
        "A single series is noisy; the header coverage is the calibrated "
        "number over all series."
    )
    d2.metric("Mean band width", f"{fc['width'].mean():.2f} units")
    d3.metric("Dept / category", f"{fc['dept_id'].iloc[0]} · {fc['cat_id'].iloc[0]}")


if __name__ == "__main__":
    main()
