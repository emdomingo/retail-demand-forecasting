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
