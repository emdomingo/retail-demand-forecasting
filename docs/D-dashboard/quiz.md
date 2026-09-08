# Feature D — Dashboard & narrative — Quiz

Append-only across D1–D4. Self-quiz: recall, predict-the-decision, spot-the-flaw.

---

## D1 — Streamlit skeleton + forecast-vs-actual + interval panel

**Recall**
1. What two files make up the persisted forecast artifact, and what does each hold?
2. Which forecast origin does the dashboard display, and why that one specifically?
3. In the chart, which mark is drawn first/largest, and why?
4. How does the actual line stay continuous across the context window and the horizon when the
   two halves come from different sources?

**Predict-the-decision**
5. We persist the forecast to disk rather than computing it when the page loads. Give two distinct
   reasons that's the right call for this project.
6. The persisted artifact uses the fixed 200-series sample (seed 0), not the whole store. What's
   the honest-labelling reason for that, beyond speed?
7. Why does the app read pre-origin actuals through `read_store_slice` (DuckDB) instead of just
   using the `sales` column already in the artifact?
8. The header shows one coverage number; a second metric shows coverage for the selected series.
   Why show both, and which is "the calibrated number"?
9. We used Altair for the chart instead of adding Plotly or matplotlib. Why did that add no
   dependency?

**Spot-the-flaw**
10. A teammate changes the dashboard to refit LightGBM on page load so it always shows "the latest
    forecast." Two things break — what are they?
11. Someone points at a series where the actual pierces the band on 6 of 28 days and says "the
    interval is broken — that's only 79% coverage." Why is that not necessarily a problem?
12. A refactor drops the `@st.cache_data` decorators "to simplify." What's the user-visible cost,
    given how Streamlit executes a script?
13. The persisted Parquet is committed to git "so the dashboard works on a fresh clone." Why is
    that against the project's conventions, and what's the intended reproduction path?

---

### Answers — D1

1. `forecast_CA_3.parquet` — one row per (series, horizon day) with actual `sales`, point `yhat`,
   band `lower`/`upper`/`width`, `scale`, and display keys `dept_id`/`cat_id` (5,600 rows). And
   `forecast_CA_3.json` — the header metadata (model, mode, target vs empirical coverage, mean
   width, test origin, series count, timestamp).
2. The **held-out latest origin** — the exact origin B4 calibrated against and measured coverage
   on. Showing it means the on-screen coverage (91.2%) *is* the backtest's reported number; a
   freshly recomputed forecast would be unevaluated and break that link.
3. The **conformal band** (`mark_area` between lower/upper). The interval is the deliverable, so
   it's the filled region and the point line + actuals are thin marks on top — the band reads first.
4. The pre-origin context actuals (from the feature store) and the horizon actuals (from the
   artifact's `sales`) are concatenated and `drop_duplicates("date")` — one unbroken black line.
5. Any two of: (a) the app stays a **pure reader** — nothing in the Streamlit path imports
   LightGBM or the harness, matching the Spark→…→Streamlit spine where the dashboard is a viewer;
   (b) it's **fast + deployable** — a Parquet read is milliseconds and needs no ML runtime, so D4's
   free hosting works; (c) it shows the **evaluated** forecast, keeping the coverage number honest.
6. The 200-series sample is what B1–B4 were run and documented on, so its coverage is exactly the
   number those runs report. Forecasting the whole store would produce a *different*, unevaluated
   coverage — the sidecar number would no longer be the backtest's number. It's about keeping the
   displayed metric truthful, not just about compute.
7. The artifact only holds the 28-day horizon; the **context** history before the origin lives in
   the feature store. Reading it via `read_store_slice` also keeps the `ORDER BY date` discipline
   in one place (the query layer owns it) rather than re-implementing ordering in the app.
8. The **header** number is coverage marginal over all series — the calibrated guarantee. The
   per-series number is noisy (small n=28) and can wander well off 90% on any one series. Showing
   both surfaces the B4 limitation (coverage is marginal, not per-series conditional) honestly in
   the UI instead of implying the guarantee holds series-by-series.
9. Altair ships **inside** Streamlit, so it's already an installed dependency — using it adds
   nothing to the environment, unlike pulling in Plotly or matplotlib.
10. (a) The app stops being a pure reader — LightGBM + the harness must now be in the deployed
    runtime, which breaks the lightweight free-hosting story; (b) it would show an *unevaluated*
    forecast whose interval coverage isn't the number the backtest reports — the honesty link is gone.
11. Coverage is **marginal over series**, not conditional per series. Any single series is a tiny,
    noisy sample (28 days) and will scatter around the target; the calibrated ~90% holds *on
    average across series*. One series at 79% is expected variance, not a broken interval — that's
    exactly why the caption flags the single-series number as noisy.
12. Streamlit **re-runs the entire script top-to-bottom on every interaction** (every selectbox
    change, every zoom). Without caching, each rerun re-reads the Parquet and re-queries the whole
    store slice from disk — visibly sluggish. `@st.cache_data` memoises those loads across reruns.
13. `data/processed/` is **gitignored** — the repo is reproducible from the pipeline, not from
    committed data (same rule as the feature store and MLflow db). The intended path is to
    rebuild: `uv run python -m src.forecast.persist` regenerates the artifact from the feature
    store, and `load_forecast` raises a clear "build it first" error if it's missing.

---

## D2 — Segment-level error panel + asymmetric-cost visual

**Recall**
1. What two metrics does the segment panel show, and how is each aggregated within a segment?
2. How is per-series RMSSE reconstructed from the persisted artifact without re-reading training
   history?
3. State the newsvendor critical-ratio formula and what `Cu` and `Co` mean.
4. What second file does D2 add to the persisted artifact, and what does it hold?

**Predict-the-decision**
5. Why does the segment panel show each segment's *unit share* next to its error, instead of just
   the error?
6. Ordering to the point forecast is the newsvendor optimum only in one special case — which, and
   why does the optimal order move above the forecast when stockouts cost more?
7. We persist a static quantile *grid* rather than shipping the `SplitConformal` calibrator into
   the app. Give the architectural reason.
8. The quantile grid is per-horizon (28 quantile functions, not one). Why does that matter for the
   order quantity?
9. The cost panel aggregates across all series, but the segment panel disaggregates. Why the
   different altitude for each?

**Spot-the-flaw**
10. A colleague reports "our model is good — blended WMAPE is 0.6" and orders every item to `yhat`.
    Point to the two things D2 shows that make that a risky planning decision.
11. Someone sets the cost panel to `Cu:Co = 1:1`, sees the newsvendor policy order *less* than the
    point forecast and still save money, and calls it a bug. Are they right? Explain.
12. A teammate "simplifies" by reading the newsvendor order from the 90% band's upper edge for
    every cost ratio. What breaks, and for which ratios?
13. The cost numbers are printed as raw dollars in a screenshot. Why is that an overclaim given how
    `policy_costs` is defined?

---

### Answers — D2

1. **WMAPE** — pooled `Σ|sales−yhat| / Σ|sales|` over all the segment's rows. **RMSSE** —
   computed per series (each scaled by its own history) then **averaged** within the segment. Same
   aggregation rules as B1 (WMAPE pools, RMSSE is per-series-then-mean).
2. The persisted `scale` is `√(naive_scale)` (the RMSSE denominator's root). So per-series
   RMSSE = `RMSE_series / scale` = `sqrt(mean(resid²)) / scale` — computable straight from the
   artifact's `sales`, `yhat`, `scale`. The SQL does it in an inner per-series query.
3. `q* = Cu / (Cu + Co)`. `Cu` = underage cost (per unit **short** / stockout), `Co` = overage
   cost (per unit **long** / overstock). Order to the `q*`-quantile of demand.
4. `quantiles_<store>.parquet` — the per-horizon predictive-quantile function (99 quantiles × 28
   horizons) of the scaled calibration residuals: columns `h`, `q`, `offset`. An order is
   `yhat + scale · offset_h(q*)`.
5. Because the error percentage alone is misleading for planning: a segment can have huge WMAPE and
   be a trivial share of volume (the slow-mover long tail), or modest WMAPE and dominate volume.
   Pairing error with unit share is exactly what a single headline number hides — the guardrail.
6. Only when **`Cu = Co`** (`q* = 0.5`, order to the median). When stockouts cost more, `q* > 0.5`,
   so you order to a higher quantile of the forecast — a safety buffer above the point forecast,
   inside the upper half of the band — because the expected cost of being short outweighs the cost
   of the extra stock.
7. It keeps the dashboard a **pure reader** — no `SplitConformal` (and thus no model machinery) in
   the app path — and makes the slider a lookup instead of a conformal recompute. Same
   persist-don't-compute principle as D1.
8. Recursive forecast error compounds across the horizon, so day 28 needs a wider safety buffer
   than day 1 for the *same* service level. A single pooled quantile would under-buffer late
   horizons and over-buffer early ones; per-horizon offsets size the order correctly per day.
9. A store's **total ordering cost** is the planning-relevant number, so the cost panel pools all
   series. The segment panel's entire job is to *disaggregate* — to show where error concentrates
   — so it must break down, not pool. Different questions, different altitude, on purpose.
10. (a) The **volume-tier breakdown**: the blended 0.6 hides that low-volume items have ~176% WMAPE
    and high-volume items ~48% — the number describes no actual item. (b) The **asymmetric-cost
    panel**: ordering to `yhat` ignores that stockouts usually cost more, leaving fill rate low and
    realised cost well above the newsvendor optimum.
11. **Not a bug.** At `1:1` the newsvendor orders to the *median*; retail demand is right-skewed, so
    `yhat` (≈ the mean) sits above the median and slightly over-orders. Ordering to the median trims
    that overstock, so cost falls a little and fill rate drops — the correct behaviour when the two
    costs are equal.
12. The 90% asymmetric band's upper edge is fixed at the **0.95 quantile**. Reading orders off it
    ignores the slider entirely: it would over-order for any `q* < 0.95` (most ratios) and
    *under-order* for `Cu:Co` steeper than 19:1. The order must track `q* = Cu/(Cu+Co)`, which is
    why we persist the whole grid, not just the band edges.
13. `policy_costs` sets `Co = 1` and `Cu =` the ratio, so cost is in **normalised units**, not
    currency — only the *ratio* affects the order, and the absolute scale is arbitrary. Printing it
    as dollars claims a precision the model doesn't have; the caption says "normalised units."
