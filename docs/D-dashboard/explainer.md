# Feature D — Dashboard & narrative — Explainer

Append-only across D1–D4. The planner's viewing surface: the forecast, its honest interval, the
segment-level error, and the causal "why" — turned from Parquet + MLflow numbers into something a
supply-chain planner would actually read. Streamlit, $0 hosting.

---

## D1 — Streamlit skeleton + forecast-vs-actual + interval panel

**What D1 is:** the first visible surface. For a chosen series it draws the recent actuals, then
the 28-day held-out-origin forecast (B2's LightGBM v2) wrapped in B4's calibrated conformal band,
against what actually happened. The **band is the star** — the interval is the deliverable, so it
is drawn as the filled region and everything else (point line, actuals) is thin on top of it.

### The decision that shapes everything: persist, don't compute (option 1)

The dashboard could either (a) import the harness and run a forecast when a page loads, or
(b) read a **pre-computed** forecast off disk. We chose (b), and it's the architecturally
important call:

- **The app stays a pure reader.** Nothing in the Streamlit path imports LightGBM or the backtest
  harness. This matches the project's spine (`Spark → Parquet → DuckDB → models → intervals →
  Streamlit`): the dashboard is the *last* stage, a viewer over a slice, not a model runner.
- **It's fast and deployable.** A recursive 28-day refit across 4 rolling origins is seconds of
  compute and needs LightGBM in the runtime; a Parquet read is milliseconds and needs neither.
  D4's free hosting (Streamlit Community Cloud) would choke on the former.
- **It shows the *evaluated* forecast, not a fresh one.** We persist the **held-out latest
  origin** — the exact origin B4 measured coverage on — so the number on screen (91.2% coverage,
  5.10 mean width) *is* the number the backtest reports. A dashboard that recomputed a new
  forecast would show an unevaluated one and quietly break that link.

### The two-file artifact (see `src/forecast/persist.py`)

`persist.py` runs the expensive part once, offline, and lands two files under
`data/processed/forecast/` (gitignored, like the rest of the feature store):

- **`forecast_CA_3.parquet`** — one row per (series, horizon day): the actual `sales`, the point
  `yhat`, the band `lower`/`upper`/`width`, the `scale`, and the display keys `dept_id`/`cat_id`.
  5,600 rows = 200 series × 28 days.
- **`forecast_CA_3.json`** — the header facts a planner needs to read the panel honestly: model,
  mode, target vs **empirical** coverage, mean width, the test origin, series count, timestamp.

The build reuses the B4 code path exactly — `origin_forecasts` → `calibrate_and_band` (extracted
from `evaluate_conformal` so the evaluator and the persister share one split) → `coverage_report`.
No forecasting logic is re-implemented for the dashboard; it consumes the same functions the
backtest scores.

**Scope choice:** we persist the **same fixed 200-series sample** (seed 0) used across B1–B4. That
keeps the persisted coverage identical to the documented B4 number. Passing `sample_size=None`
would forecast the whole store, but the sidecar coverage would then no longer be the number the
backtest reports — an honest-labelling reason, not a compute one.

### The app (see `src/dashboard/app.py`)

Three moving parts, all thin:

1. **Load + cache.** `_load_artifact` reads the Parquet + JSON once; `_load_history` pulls the
   pre-origin actuals through the **DuckDB query layer** (`read_store_slice`, which owns the
   `ORDER BY date` discipline). Both are wrapped in `@st.cache_data` so a rerun (Streamlit
   re-executes the whole script top-to-bottom on every interaction) doesn't re-hit disk.
2. **The chart** (`_forecast_chart`, Altair). Four layered marks: a shaded `mark_area` between
   `lower`/`upper` (the band), a dashed `yhat` line, a solid black `actual` line spanning both the
   context window and the horizon, and a red `mark_rule` at the origin marking where forecasting
   begins. Altair ships *inside* Streamlit, so this adds no dependency. Layering is `a + b + c`;
   `.interactive()` enables pan/zoom.
3. **The reads.** Header metrics (model, coverage vs target, mean width, origin) frame the panel;
   a per-series coverage metric below it shows how the band did on *this* series over its 28 days
   — with the caption that a single series is noisy and the header number is the calibrated one.
   That single-vs-marginal distinction is the same B4 limitation (coverage is marginal over
   series, not conditional per series), surfaced in the UI rather than buried.

### The continuous-actual trick

The actual line must run unbroken from the context window into the horizon, but the two come from
different sources (feature store vs the artifact's `sales` column). We concat the pre-origin
context actuals with the horizon actuals and `drop_duplicates("date")` — one continuous black line
the band and point line sit under.

### What's tested

`persist.py` is covered by `tests/test_persist.py`: the artifact schema and required columns, the
round-trip (`build → persist → load` returns the same frame + meta), the sidecar coverage matching
`coverage_report` on the same banded frame, and `load_forecast` raising a clear error when nothing
is persisted yet. The Streamlit app itself is glue over tested functions — the chart spec is
validated by building it and calling `.to_dict()` (Altair raises on a bad spec), and the full
load→filter→chart path is exercised as a smoke check; there's no value in mocking the Streamlit
runtime.

---

## D2 — Segment-level error panel + asymmetric-cost visual

**What D2 is:** two panels that answer the two questions a planner asks *after* seeing the
forecast. First, *where does the error actually land?* — because one headline WMAPE hides the
items that matter. Second, *what is the interval worth?* — turning the band into an order quantity
under an asymmetric cost. Both read the same persisted artifact; the dashboard stays a pure reader.

### D2a — segment error rollups (see `src/query/segments.py`)

The guardrail: *aggregate accuracy hides the spiky high-value items.* So we break the held-out
origin's error down three ways — by **category**, **department**, and **volume tier** (quartiles of
mean demand) — and, crucially, report each segment's **share of units** next to its error.

**DuckDB owns the aggregation** (CLAUDE.md), reading the persisted forecast Parquet directly.
Two metrics, each rolled up the way B1 defines it:

- **WMAPE** — pooled `Σ|sales−yhat| / Σ|sales|` over the segment. Rolls up cleanly.
- **RMSSE** — per-series then averaged. The trick: the interval `scale` we persisted is
  `√(naive_scale)`, so per-series RMSSE reconstructs as `RMSE_series / scale` — no need to
  re-read training history. The SQL computes it in an inner per-series query, then averages.

The finding is the whole point of the panel. On CA_3's 200-series sample:

| Volume tier | WMAPE | unit share |
|---|---|---|
| Q1 (low) | ~176% | 3% |
| Q4 (high) | ~48% | 71% |

The slow movers have wild percentage error but are a rounding error in volume; the fast items that
dominate planning forecast *best*. A single blended WMAPE (~0.6) reports neither honestly. RMSSE
stays flatter across tiers (~0.6–0.76) — a reminder the two metrics answer different questions.

### D2b — the asymmetric-cost visual (see `src/forecast/decision.py`)

This is where the interval stops being a picture and becomes a decision. The model is the classic
**newsvendor**: with underage cost `Cu` (stockout) and overage cost `Co` (overstock), expected
cost is minimised by ordering to the **critical-ratio quantile**:

```
q* = Cu / (Cu + Co)        # only the ratio matters for where to order
order = yhat + scale · offset_h(q*)   # the q*-quantile from the conformal grid
```

So the order point *is* a quantile of the forecast — and B4's conformal machinery already produces
calibrated, per-horizon, scale-normalised quantiles. Ordering to the point forecast (`yhat`, ≈ the
median) is the newsvendor optimum **only when `Cu = Co`**; the moment stockouts cost more, `q* >
0.5` and the optimal order sits in the *upper half of the band*.

**The predictive-quantile grid.** To let the slider pick any `q*`, `persist.py` writes a second
artifact — `quantiles_CA_3.parquet`, the per-horizon quantile function (99 quantiles × 28 horizons)
of the scaled calibration residuals (`calibration_quantile_grid`). The dashboard indexes the
nearest `q`; it never recomputes conformal. Per-horizon matters: later horizons need a bigger
safety buffer, the same reason the band flares.

**Realised, not assumed.** `policy_costs` evaluates two policies — order-to-`yhat` vs
order-to-`q*` — against the held-out origin's *actual* sales, and reports total cost, **fill rate**
(units met ÷ demanded), and mean order. On the sample, as `Cu:Co` climbs 1→9 the interval-aware
policy lifts fill rate ~68%→92% and cuts realised cost up to ~40%. Even at `1:1` it saves a little
by ordering to the median rather than the right-skew-inflated mean. Cost is normalised (`Co = 1`)
because only the ratio drives the order; that's stated in the caption, not hidden.

### Design decisions worth defending

- **Grid, not a live calibrator.** We persist a static quantile table rather than shipping the
  `SplitConformal` object into the app. Keeps the reader-pure principle and makes the slider a
  lookup, not a model call.
- **Store-level cost, per-series error.** The cost panel aggregates across all series (a store's
  total ordering cost is the planning-relevant number); the segment panel disaggregates (that's
  its entire job). Different altitudes on purpose.
- **Two metrics, shown together.** WMAPE for volume-weighted % error, RMSSE for scale-free
  per-series accuracy. Neither alone tells the segment story.

### What's tested

`tests/test_decision.py`: the critical ratio is the textbook formula, orders read the nearest grid
quantile and scale correctly, orders rise with `q*`, are clipped ≥ 0, the realised-cost accounting
charges each side right, and the policy comparison lifts fill rate as stockouts get costlier.
`tests/test_segments.py`: WMAPE pools per segment, RMSSE uses the persisted scale, unit shares sum
to one, and the volume tiers partition the series. `tests/test_persist.py` gains the quantile-grid
schema + monotonicity checks and that `load_forecast` now requires the grid file.

---

## D3 — Intervention-effect panel (the causal layer, surfaced)

**What D3 is:** the dashboard's answer to *why did demand move?* When actuals diverge from the
forecast, the planner's real question is the cause — and Feature C already estimated two: the price
cut (C2 DiD + C2b five-store replication) and SNAP (C3 cross-state counterfactual). D3 surfaces
each as an **effect with its confidence interval** plus the falsification evidence that makes it
credible. It is the on-ramp from the forecast half to the causal half.

### Persist-then-read, again (see `src/causal/persist.py`)

The causal estimators need `statsmodels` and several DuckDB passes over the feature store — none of
which belong in the Streamlit runtime. So the same pattern as D1/D2: `causal/persist.py` runs
C2/C2b/C3 once, offline (~1 min), and lands a single small JSON, `data/processed/causal/causal.json`
(gitignored). The app imports no `statsmodels` and refits nothing; it reads fields.

The artifact is assembled straight from the existing causal functions — `did.estimate_did`,
`did.event_study`, `did.placebo_test`, `replication.replicate` + `pool`, `snap.estimate_ladder`,
`snap.placebo`, `snap.clean_day_estimate` — so the numbers on screen are exactly the module
outputs, not a reimplementation. Two small derived fields: the **naive jump** (`_naive_jump`, the
treated item's raw pre/post ratio, the uncontrolled number DiD disciplines *down* from) and the
**elasticity** (lift ÷ the −16.8% price change).

### The two tabs

**Price cut (DiD).** Three cards — the DiD lift +42.1% [+12.5%, +79.4%], the naive +55.9% it was
disciplined down from (the gap is the method's value), and the implied elasticity ≈ −2.5. Then the
**event-study chart**: the treated-vs-control demand gap by week relative to the cut. The story is
in the shape — *leads flat around zero* is the parallel-trends assumption shown, not asserted; the
jump at week 0 is the effect. Below it, the **replication forest**: each of the five chain-wide cuts
with its CI, plus the pooled diamond. The caption quotes the **random-effects** pool +56.9%
[+34.6%, +82.9%] (not the too-narrow fixed-effect CI) because I² ≈ 57% says the stores genuinely
differ — the honest number to report.

**SNAP.** Two cards — the cross-state lift +10.7% [+9.1%, +12.3%] and the naive +16.8% it improves
on. Then the **estimate ladder** (naive → +calendar → +cross-state): the lift disciplines down and
R² climbs as the counterfactual improves, the same lesson as the price cut, with the cross-state bar
highlighted as the one to quote. The caption carries the falsification: both cross-state placebos
cover zero, and the clean-day estimator corroborates.

### Design choices

- **Log points → % for display.** Event-study and forest coefficients are in log points; the panel
  converts each to a % lift (`exp−1`) so a planner reads "demand gap," not "log-units." Shape and
  ordering are preserved.
- **Rebuilt in Altair, not the committed PNGs.** Feature C saved static matplotlib figures for the
  docs; the dashboard rebuilds the event study, forest, and ladder as interactive Altair from the
  persisted numbers — consistent with D1/D2, hover-able, and theme-aware.
- **Falsifications shown, not hidden.** The placebo (price cut) and the two SNAP placebos + clean-day
  check are surfaced in captions. The credibility *is* the deliverable here, not just the point
  estimate — same spirit as "the interval is the deliverable" on the forecast side.

### What's tested

`tests/test_causal_persist.py`: the persist→load round-trip, the loud FileNotFoundError when
nothing is built, and the pure assembly helpers (`_naive_jump` = the uncontrolled treated ratio,
`_snap_row` reads effect + covers-zero). The estimators themselves are already covered by
`test_did.py`, `test_replication.py`, and `test_snap.py`, so D3's tests stay on the new plumbing
rather than re-estimating (a ~1-min refit) in the suite.

---

## D4 — Deploy to free hosting

**What D4 is:** making the dashboard a real, public, $0-hosted link — and the work that
*actually* required, which was more than "click deploy." Streamlit Community Cloud clones the
GitHub repo and runs the app; that exposed two things the earlier subfeatures had glossed.

### The two problems D4 had to fix

1. **The app wasn't a pure reader — it just looked like one.** D1–D3 said "the app imports no
   LightGBM/statsmodels." In truth, importing `forecast.persist` / `causal.persist` pulled in
   **pyspark, lightgbm, mlflow, and statsmodels** transitively (through their module-top imports of
   the models, the harness, and the causal estimators). On a $0 host that means installing Spark
   (which needs Java), LightGBM, etc. — slow, heavy, and pointless for a reader.

   **Fix:** the heavy, build-only imports moved *inside* the build functions (lazy import). Now
   importing the module for `load_forecast` / `load_causal` pulls in nothing but pandas/duckdb.
   A subprocess guard test (`test_dashboard_imports.py`) asserts importing the app loads none of
   the four heavy modules — so the claim is enforced, not just asserted.

2. **The app read the gitignored feature store live.** The D1 context line (the 56 days of actuals
   before the origin) came from `read_store_slice`, i.e. the 59M-row feature store — which is
   gitignored and *absent on the host*. The chart would have had no history to draw.

   **Fix:** `persist.py` now bundles the pre-origin context (`context_<store>.parquet`, the last
   `CONTEXT_DAYS` of actuals per series) into the artifact, and the app reads that. No feature
   store, no DuckDB slice at request time.

### The committed bundle (the decision)

The deploy needs data, and the host can't rebuild it (no Kaggle pull, no Spark, no modelling deps).
So the artifacts are **committed** — in a top-level `dashboard_data/` directory, deliberately
*separate* from the gitignored `data/`. Five files, **~250 KB total**: the forecast, the quantile
grid, the context, the forecast meta, and the causal JSON.

This bends the letter of "never commit data," so it's worth stating the distinction precisely:
`data/` holds the **dataset** — the 350 MB of raw M5 CSVs and the 59M-row feature store, both
large and reproducible from Kaggle, correctly gitignored. `dashboard_data/` holds **model output**
— tiny, derived, and the deliverable's display inputs. You cannot reconstruct the dataset from it,
and the pipeline is still fully reproducible; committing it is the standard "ship the predictions,
not the training data" pattern. (This supersedes the D1 framing, which described the pre-D4 state
where the artifacts were gitignored and rebuilt locally.)

Alternatives weighed and rejected for ~250 KB: a GitHub Release asset fetched at startup (repo
stays literally data-free, but adds a download step and a release chore), and an external bucket +
secret (most production-like, but a third-party service and a credential to maintain). For this
size, committing is the simplest honest option.

### The host dependency set (`requirements.txt`)

The deployed app declares a **minimal** dependency set — `streamlit, altair, pandas, numpy,
duckdb, pyarrow` — not the full `pyproject.toml` modelling env. The pure-reader refactor is what
makes that possible: nothing the app imports needs Spark/LightGBM/statsmodels. Smaller install,
faster cold start, no Java on the host. Local dev still uses `uv`; `requirements.txt` is for the
host only.

### Deploying (the manual step)

The final click is the repo owner's — it needs their GitHub + Streamlit Community Cloud account,
so it can't be automated from here. The app is **deploy-ready**; the steps are:

1. Push `main` (with `dashboard_data/`, `requirements.txt`, `.streamlit/config.toml`) to GitHub.
2. On share.streamlit.io → *New app* → pick the repo, branch `main`, main file
   `src/dashboard/app.py`.
3. Deploy. Community Cloud installs `requirements.txt` and runs the app; it reads `dashboard_data/`
   straight from the clone. The public URL then goes in the E1 README.

### What's tested

`test_dashboard_imports.py` (the pure-reader guard, in a clean subprocess) + `test_persist.py`
gains the context-tail schema check and that `load_forecast` now requires *every* bundle file. The
render itself is still exercised end-to-end by the Streamlit `AppTest` smoke check.
