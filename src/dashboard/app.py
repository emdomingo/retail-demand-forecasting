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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# These four imports pull in NO modelling deps (no lightgbm/mlflow/statsmodels/pyspark) — the
# heavy build-time code in the persist modules is lazy-imported. The dashboard is a genuine pure
# reader over the committed dashboard_data/ bundle, which is what lets it deploy to $0 hosting.
from src.causal.persist import load_causal  # noqa: E402
from src.forecast.decision import critical_ratio, policy_costs  # noqa: E402
from src.forecast.persist import load_forecast  # noqa: E402
from src.query.segments import segment_error, volume_tier_error  # noqa: E402

DEFAULT_STORE = "CA_3"


@st.cache_data(show_spinner=False)
def _load_artifact(store: str):
    """Persisted forecast + quantile grid + pre-origin context + metadata (cached across reruns).
    All of it comes from the committed bundle — no feature store, no model refit."""
    art = load_forecast(store)
    return art.forecast, art.quantiles, art.context, art.meta


@st.cache_data(show_spinner=False)
def _load_causal():
    """Persisted causal results (C2/C2b DiD + C3 SNAP), cached across reruns."""
    return load_causal()


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
        fc_all, quantiles, context, meta = _load_artifact(store)
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
    # Pre-origin context comes from the bundle (already the per-series tail up to the origin) — no
    # feature-store read, so this works on the deployed host where the feature store is absent.
    ctx = context[context["id"] == series]
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

    _segment_panel(fc_all)
    _cost_panel(fc_all, quantiles)
    _causal_panel()


def _segment_panel(fc_all: pd.DataFrame) -> None:
    """D2a — segment-level error. One headline number hides the items that matter for planning;
    this breaks WMAPE/RMSSE down by category, department, or volume tier, and pairs each segment's
    error with its *share of units* so a big-but-accurate segment and a tiny-but-wild one read
    differently."""
    st.divider()
    st.subheader("Where the error lands — segment breakdown")
    dim = st.radio(
        "Break down by", ["Volume tier", "Category", "Department"], horizontal=True
    )
    if dim == "Volume tier":
        seg = volume_tier_error(fc_all)
    else:
        seg = segment_error(fc_all, "cat_id" if dim == "Category" else "dept_id")

    st.altair_chart(_segment_chart(seg), width="stretch")
    st.caption(
        "Bar length = WMAPE (% error); bar shade = share of total units. The pattern to read: "
        "the segments with the worst % error are usually the smallest slice of volume, while the "
        "high-volume segments a planner cares most about forecast best — which is exactly what a "
        "single headline WMAPE hides."
    )
    show = seg.assign(
        units=seg["units"].round().astype(int),
        unit_share=(seg["unit_share"] * 100).round(1),
        wmape=(seg["wmape"] * 100).round(1),
        rmsse=seg["rmsse"].round(3),
    ).rename(columns={"unit_share": "unit_share_%", "wmape": "wmape_%"})
    st.dataframe(show, hide_index=True, width="stretch")


def _cost_panel(fc_all: pd.DataFrame, quantiles: pd.DataFrame) -> None:
    """D2b — the asymmetric-cost visual. A point forecast ignores that a stockout usually costs
    more than an overstock; the newsvendor model orders to the q* = Cu/(Cu+Co) quantile of the
    *forecast distribution* — which is exactly what the conformal interval gives us. This shows the
    realised cost of that interval-aware policy vs ordering to the point forecast, on the held-out
    origin's actuals, across all series in the store."""
    st.divider()
    st.subheader("What the interval is worth — asymmetric ordering cost")
    ratio = st.slider(
        "How much costlier is a stockout than an overstock? (Cu : Co)",
        min_value=1, max_value=10, value=4,
        help="Under-forecasting usually hurts more. This ratio sets the newsvendor "
             "service level q* = Cu/(Cu+Co).",
    )
    costs = policy_costs(fc_all, quantiles, cu=float(ratio), co=1.0)
    point = costs[costs["policy"] == "point"].iloc[0]
    nv = costs[costs["policy"] == "newsvendor"].iloc[0]
    saving = (point["total_cost"] - nv["total_cost"]) / point["total_cost"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Service level q*", f"{critical_ratio(ratio, 1.0):.0%}")
    m2.metric("Cost saved vs point forecast", f"{saving:.0%}", delta_color="off")
    m3.metric(
        "Fill rate — point → newsvendor",
        f"{point['fill_rate']:.0%} → {nv['fill_rate']:.0%}", delta_color="off",
    )
    m4.metric(
        "Avg order lift", f"+{(nv['mean_order'] / point['mean_order'] - 1):.0%}",
        delta_color="off",
    )
    st.altair_chart(_cost_chart(costs), width="stretch")
    st.caption(
        "Cost is in normalised units (overstock = 1 per unit, stockout = the ratio). Ordering to "
        "the point forecast (≈ the median) is the newsvendor optimum only when the two costs are "
        "equal; the moment stockouts cost more, q* > 50% and the optimal order sits inside the "
        "upper half of the conformal band — the interval turned into a decision, not decoration."
    )


def _segment_chart(seg: pd.DataFrame) -> alt.Chart:
    """Horizontal WMAPE bars per segment, shaded by unit share."""
    return (
        alt.Chart(seg)
        .mark_bar()
        .encode(
            x=alt.X("wmape:Q", title="WMAPE (% error)", axis=alt.Axis(format="%")),
            y=alt.Y("segment:N", sort="-x", title=None),
            color=alt.Color(
                "unit_share:Q", title="unit share",
                scale=alt.Scale(scheme="blues"), legend=alt.Legend(format="%"),
            ),
            tooltip=[
                alt.Tooltip("segment:N"),
                alt.Tooltip("wmape:Q", title="WMAPE", format=".1%"),
                alt.Tooltip("rmsse:Q", title="RMSSE", format=".3f"),
                alt.Tooltip("unit_share:Q", title="unit share", format=".1%"),
                alt.Tooltip("n_series:Q", title="series"),
            ],
        )
        .properties(height=alt.Step(34))
    )


def _cost_chart(costs: pd.DataFrame) -> alt.Chart:
    """Total realised cost by policy — the headline of the asymmetric-cost panel."""
    color = alt.Color(
        "policy:N",
        scale=alt.Scale(domain=["point", "newsvendor"], range=["#bbb", "#4c78a8"]),
        legend=None,
    )
    return (
        alt.Chart(costs)
        .mark_bar()
        .encode(
            x=alt.X("policy:N", title=None, sort=["point", "newsvendor"]),
            y=alt.Y("total_cost:Q", title="total realised cost (normalised)"),
            color=color,
            tooltip=[
                alt.Tooltip("policy:N"),
                alt.Tooltip("total_cost:Q", title="total cost", format=",.0f"),
                alt.Tooltip("fill_rate:Q", title="fill rate", format=".1%"),
            ],
        )
        .properties(height=260)
    )


def _causal_panel() -> None:
    """D3 — the causal layer: *why* demand moved. When actuals diverge from the forecast, the
    planner's real question is the cause; this surfaces the two estimated interventions — the price
    cut (C2 DiD + C2b five-store replication) and SNAP (C3 cross-state counterfactual) — each as an
    effect with its confidence interval and its falsification evidence. Reads the persisted causal
    JSON; the app runs no statsmodels."""
    st.divider()
    st.subheader("Why demand moved — intervention effects")
    try:
        causal = _load_causal()
    except FileNotFoundError as e:
        st.info(str(e))
        return

    pc_tab, snap_tab = st.tabs(["💲 Price cut (DiD)", "🍎 SNAP benefits"])
    with pc_tab:
        _price_cut_tab(causal["price_cut"])
    with snap_tab:
        _snap_tab(causal["snap"])


def _price_cut_tab(pc: dict) -> None:
    st.markdown(
        f"**{pc['treated']} @ {pc['store']}** — price cut {pc['price']} on {pc['cut']}. "
        "How much of the post-cut jump did the cut *cause*?"
    )
    a, b, c = st.columns(3)
    lo, hi = pc["ci"]["low"], pc["ci"]["high"]
    a.metric("DiD demand lift", f"{pc['lift']:+.1%}", f"95% CI [{lo:+.0%}, {hi:+.0%}]",
             delta_color="off")
    b.metric("Disciplined down from", f"{pc['naive_lift']:+.1%}",
             "naive pre/post jump", delta_color="off")
    c.metric("Implied elasticity", f"{pc['elasticity']:.1f}", "≈ Δlift / Δprice", delta_color="off")

    st.altair_chart(_event_study_chart(pc["event_study"]), width="stretch")
    st.caption(
        "Event study: demand gap vs matched controls, by week relative to the cut (vs the week "
        "before). **Leads flat around zero = parallel pre-trends** — the DiD identifying "
        "assumption, shown not asserted. The jump at week 0 and after is the effect."
    )
    placebo = pc["placebo"]
    verdict = "✅ passes" if placebo["covers_zero"] else "⚠️ fails"
    st.caption(
        f"Placebo (fake cut in the pre-period): {placebo['lift']:+.1%} — CI covers zero, "
        f"so no spurious pre-cut divergence ({verdict})."
    )

    st.markdown("**Chain-wide replication (C2b)** — the same cut in five stores neutralises the "
                "thin single-store pre-period. Five agreeing estimates beat one.")
    rep = pc["replication"]
    st.altair_chart(_forest_chart(rep), width="stretch")
    rlo, rhi = rep["pooled_re"]["ci"]["low"], rep["pooled_re"]["ci"]["high"]
    st.caption(
        f"{rep['n_positive']}/{rep['k']} positive, {rep['n_sig']}/{rep['k']} significant. "
        f"Heterogeneity I² = {rep['i_squared']:.0%} (Q p = {rep['q_pvalue']:.2f}), so quote the "
        f"**random-effects** pool: **{rep['pooled_re']['lift']:+.1%}** [{rlo:+.0%}, {rhi:+.0%}] — "
        "wider than the fixed-effect CI, honestly reflecting the between-store spread."
    )


def _snap_tab(sn: dict) -> None:
    st.markdown(
        f"**{sn['category']} demand @ {sn['store']} ({sn['state']})** — the average lift on a SNAP "
        "benefit day. SNAP has no clean pre-period and is store-wide, so DiD can't be used; the "
        "counterfactual is **cross-state** (TX/WI stores are on different SNAP schedules)."
    )
    main = sn["main"]
    lo, hi = main["ci"]["low"], main["ci"]["high"]
    naive = sn["ladder"][0]
    a, b = st.columns(2)
    a.metric("SNAP-day lift (cross-state)", f"{main['lift']:+.1%}",
             f"95% CI [{lo:+.1%}, {hi:+.1%}]", delta_color="off")
    b.metric("Disciplined down from", f"{naive['lift']:+.1%}",
             "naive (no controls)", delta_color="off")

    st.altair_chart(_ladder_chart(sn["ladder"]), width="stretch")
    st.caption(
        "The estimate ladder: as the counterfactual improves (naive → +calendar → +cross-state "
        "controls), the lift disciplines down and the fit R² climbs — the same lesson as the price "
        "cut. The cross-state estimate is the one to quote."
    )
    p_ok = all(p["covers_zero"] for p in sn["placebos"])
    clean = sn["clean_day"]
    st.caption(
        f"Falsification: both cross-state placebos {'✅ cover zero' if p_ok else '⚠️ do not'} "
        f"(CA's schedule is not a fake treatment on TX/WI). Clean-day estimator (CA-only SNAP days "
        f"vs no-SNAP days) corroborates at {clean['lift']:+.1%}."
    )


def _pct_frame(records: list[dict]) -> pd.DataFrame:
    """Convert log-point coef/CI records to % lift (exp−1) for display."""
    df = pd.DataFrame(records)
    for src, dst in [("coef", "lift"), ("ci_low", "lo"), ("ci_high", "hi")]:
        df[dst] = np.expm1(df[src])
    return df


def _event_study_chart(records: list[dict]) -> alt.Chart:
    df = _pct_frame(records)
    base = alt.Chart(df)
    err = base.mark_rule(color="#4c78a8").encode(
        x=alt.X("rel_week:Q", title="weeks relative to cut"),
        y=alt.Y("lo:Q", title="demand gap vs control", axis=alt.Axis(format="%")),
        y2="hi:Q",
    )
    pts = base.mark_point(color="#4c78a8", filled=True, size=45).encode(
        x="rel_week:Q", y="lift:Q",
        tooltip=[alt.Tooltip("rel_week:Q", title="rel week"),
                 alt.Tooltip("lift:Q", title="gap", format="+.1%")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#999").encode(y="y:Q")
    cut = alt.Chart(pd.DataFrame({"x": [-0.5]})).mark_rule(
        color="#c44", strokeDash=[4, 3]).encode(x="x:Q")
    return (zero + cut + err + pts).properties(height=300)


def _forest_chart(rep: dict) -> alt.Chart:
    rows = [
        {"label": f"{s['store']} (cut {s['cut']})", "lift": s["lift"],
         "lo": s["ci"]["low"], "hi": s["ci"]["high"], "kind": "store"}
        for s in rep["stores"]
    ]
    rows.append({
        "label": "Pooled (random-effects)", "lift": rep["pooled_re"]["lift"],
        "lo": rep["pooled_re"]["ci"]["low"], "hi": rep["pooled_re"]["ci"]["high"],
        "kind": "pooled",
    })
    df = pd.DataFrame(rows)
    order = df["label"].tolist()
    color = alt.Color("kind:N", scale=alt.Scale(
        domain=["store", "pooled"], range=["#4c78a8", "#c44"]), legend=None)
    base = alt.Chart(df)
    err = base.mark_rule().encode(
        y=alt.Y("label:N", sort=order, title=None),
        x=alt.X("lo:Q", title="price-cut demand lift", axis=alt.Axis(format="%")),
        x2="hi:Q", color=color,
    )
    pts = base.mark_point(filled=True, size=90).encode(
        y=alt.Y("label:N", sort=order), x="lift:Q", color=color,
        tooltip=[alt.Tooltip("label:N"), alt.Tooltip("lift:Q", format="+.1%")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#999").encode(x="x:Q")
    return (zero + err + pts).properties(height=alt.Step(34))


def _ladder_chart(ladder: list[dict]) -> alt.Chart:
    df = pd.DataFrame(
        [{"label": r["label"], "lift": r["lift"], "lo": r["ci"]["low"],
          "hi": r["ci"]["high"], "r2": r["r_squared"]} for r in ladder]
    )
    order = df["label"].tolist()  # naive -> calendar -> cross_state
    is_main = alt.condition(
        alt.datum.label == "cross_state", alt.value("#c44"), alt.value("#4c78a8"))
    base = alt.Chart(df)
    err = base.mark_rule().encode(
        y=alt.Y("label:N", sort=order, title=None),
        x=alt.X("lo:Q", title="SNAP-day lift", axis=alt.Axis(format="%")), x2="hi:Q", color=is_main,
    )
    pts = base.mark_point(filled=True, size=90).encode(
        y=alt.Y("label:N", sort=order), x="lift:Q", color=is_main,
        tooltip=[alt.Tooltip("label:N"), alt.Tooltip("lift:Q", format="+.1%"),
                 alt.Tooltip("r2:Q", title="R²", format=".2f")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#999").encode(x="x:Q")
    return (zero + err + pts).properties(height=alt.Step(40))


if __name__ == "__main__":
    main()
