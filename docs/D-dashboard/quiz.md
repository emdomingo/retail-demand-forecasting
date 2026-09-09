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

---

## D3 — Intervention-effect panel (the causal layer, surfaced)

**Recall**
1. Which two interventions does D3 show, and what estimator does each use?
2. What single file does D3 read, and why doesn't the app import `statsmodels`?
3. In the event-study chart, what does "leads flat around zero" demonstrate?
4. Which pooled estimate does the replication caption quote — fixed- or random-effects — and why?

**Predict-the-decision**
5. The price-cut card shows both the DiD lift (+42%) and the naive jump (+56%). Why show the naive
   number at all instead of just the headline?
6. SNAP is shown with a cross-state counterfactual, not a DiD. What two properties of SNAP rule DiD
   out?
7. Event-study and forest coefficients are stored in log points but displayed as %. Why convert,
   and what is preserved?
8. D3 rebuilds the event study / forest / ladder in Altair even though Feature C already saved PNGs
   of them. What does rebuilding buy?

**Spot-the-flaw**
9. A reviewer says "the dashboard should recompute the DiD live so it's always current." Why is
   that the wrong call here (give two reasons)?
10. Someone reads the SNAP ladder and reports the naive +16.8% as the SNAP effect. What did they
    miss, and which bar should they quote?
11. A teammate quotes the fixed-effect replication pool because its CI is tighter. Why is that the
    less honest choice given I² ≈ 57%?
12. The elasticity card shows −2.5. A colleague computes it as `lift / price` = 0.42 / 2.98. What's
    wrong with that, and what is the right denominator?

---

### Answers — D3

1. The **price cut** — C2 Difference-in-Differences (TWFE on log-units, clustered by item) plus the
   C2b five-store replication with a meta-analytic pool — and **SNAP** — the C3 cross-state OLS
   counterfactual (HAC/Newey-West SEs).
2. `data/processed/causal/causal.json`, written offline by `src/causal/persist.py`. The estimators
   need `statsmodels` + DuckDB passes that don't belong in the Streamlit runtime, so — same as
   D1/D2 — the app is a pure reader and imports no modelling code.
3. That the treated and control series moved *together before* the cut — the **parallel-trends**
   identifying assumption of DiD, shown rather than asserted. A pre-trend would invalidate the
   causal reading.
4. **Random-effects.** I² ≈ 57% (with Q's p ≈ 0.06) says the stores estimate genuinely different
   effects, so the fixed-effect CI (which assumes one shared effect) is too narrow; the
   random-effects pool +56.9% [+34.6%, +82.9%] widens it to reflect the between-store spread.
5. Because the gap *is* the method's value: the raw pre/post jump (+56%) overstates the cut's effect
   by a third because it ignores what controls were doing over the same weeks. Showing both makes
   the discipline visible — the DiD isn't a smaller number pulled from nowhere, it's the naive
   number corrected for the counterfactual.
6. SNAP is a **recurring monthly pulse with no clean pre-period** (it's present across the whole
   window) and it's **store-wide** (every item in the store is "treated," so there's no within-store
   control group). DiD needs both a pre-period and untreated units; SNAP has neither — hence
   cross-state controls (states on different SNAP schedules).
7. A planner reads "demand gap," not "log-units gap"; `exp(coef)−1` turns each coefficient into a %
   lift. The **shape and ordering** (flat leads, the jump at 0, relative magnitudes) are preserved —
   the conversion is monotonic.
8. Interactivity and consistency: the Altair versions are hover-able, theme-aware, and match the
   D1/D2 chart style, all rebuilt from the same persisted numbers. The PNGs are for the docs; the
   dashboard shouldn't ship static images where it renders everything else live.
9. (a) It would drag `statsmodels` + the feature-store passes into the deployed app, breaking the
   pure-reader / free-hosting story; (b) the estimates are over a **frozen historical dataset** —
   nothing to be "current" about — so live recompute adds cost and risk for zero freshness.
10. They quoted the **uncontrolled** estimate — naive OLS with no counterfactual, which conflates
    SNAP with the early-month calendar wave and cross-state demand. The **cross-state** bar (+10.7%)
    is the one to quote; the ladder exists precisely to show that discipline.
11. With real between-store heterogeneity (I² ≈ 57%), the fixed-effect model's assumption of a
    single shared effect is false, so its narrow CI understates the true uncertainty. Picking it
    *because* it's tighter is choosing the more confident-looking number over the more honest one.
12. `lift / price` mixes a percentage with a dollar level. Elasticity is `%Δquantity / %Δprice`, so
    the denominator is the **percentage** price change (−16.8%), not the $2.98 level: 0.421 /
    (−0.168) ≈ −2.5.

---

## D4 — Deploy to free hosting

**Recall**
1. Where do the deployed app's data files live, and how big is the bundle?
2. What minimal dependency set does the host install, and why not the full `pyproject.toml`?
3. What two latent problems did D4 have to fix before the app could deploy?

**Predict-the-decision**
4. Why commit the artifacts (option A) rather than fetch them from a GitHub Release or an external
   bucket, given "never commit data"?
5. The heavy imports moved *inside* the build functions. What property does that give the app, and
   how is it enforced rather than just claimed?
6. Why bundle the pre-origin context into the artifact instead of reading it live like D1 did?

**Spot-the-flaw**
7. A teammate points at the D1 quiz answer ("the parquet is gitignored; rebuild it") and says the
   committed `dashboard_data/` violates the project's rules. How do you answer?
8. Someone deploys and the app installs pyspark + lightgbm + statsmodels on the host "to be safe."
   What's wrong with that, and what makes it unnecessary?
9. A reviewer says "just point the host at the feature store so the context line is always fresh."
   Why can't the host do that?

---

### Answers — D4

1. In a committed top-level `dashboard_data/` directory (separate from the gitignored `data/`) —
   five files (forecast, quantile grid, context, forecast meta, causal JSON), ~250 KB total.
2. `streamlit, altair, pandas, numpy, duckdb, pyarrow` via `requirements.txt`. The full env has
   pyspark (needs Java), lightgbm, mlflow, statsmodels — all build-time only; installing them on
   the host is slow, heavy, and needless because the app imports none of them.
3. (a) The app wasn't actually a pure reader — importing the persist modules pulled in pyspark/
   lightgbm/mlflow/statsmodels transitively; (b) the context line read the gitignored feature
   store live, which is absent on the host.
4. For ~250 KB of tiny derived output, a Release-asset download or an external bucket + secret is
   over-engineering — extra steps, a service, a credential. Committing is simplest and still
   honest: it's **model output**, not the dataset (you can't rebuild M5 from it), so it doesn't
   violate the *spirit* of "never commit data" — that rule is about the 350 MB of raw CSVs and the
   59M-row feature store, which stay gitignored.
5. It makes the app a genuine **pure reader** — importing it pulls in no modelling deps, so it runs
   on a minimal host. Enforced by `test_dashboard_imports.py`, which imports the app in a clean
   subprocess and asserts none of pyspark/lightgbm/mlflow/statsmodels ended up in `sys.modules`.
6. Because the feature store is gitignored and **absent on the host** — a live read would have no
   history to draw. Bundling the last `CONTEXT_DAYS` of actuals per series makes the app
   self-contained: it reads only `dashboard_data/`, no feature store, no request-time DuckDB slice.
7. That answer described the **pre-D4** state and still holds for the *dataset*. D4 deliberately
   introduced a separate, tiny, committed **presentation bundle** (`dashboard_data/`, ~250 KB of
   model output) so the read-only app can deploy at $0. The dataset (`data/`) is still gitignored
   and reproducible from Kaggle; nothing about that changed. It's "ship the predictions, not the
   training data."
8. It installs a heavy, Java-dependent stack (pyspark) the app never uses — slow cold starts,
   wasted resources, possible build-limit failures. Unnecessary because the pure-reader refactor
   means nothing the app imports touches Spark/LightGBM/statsmodels; the minimal `requirements.txt`
   is sufficient and correct.
9. The feature store is the gitignored 59M-row Parquet built by the Spark pipeline; it isn't in the
   repo the host clones, and the host has neither the raw M5 data, a Kaggle token, nor Spark/Java to
   rebuild it. The context therefore has to travel *with* the app, in the committed bundle.
