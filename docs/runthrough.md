# Subfeature runthrough — decisions log

One short entry per subfeature: the decision made at each gate + a one-line why.
Built during the A0–E2 runthrough (map, not build). Plan-changing decisions are
folded back into [SPEC.md](../SPEC.md). Provisional entries reflect the recommended
call; they firm up as the user says stream-ahead / pause.

Legend: **[settled]** confirmed this session · **[rec]** recommended, awaiting confirm.

---

## Feature A — Ingestion & feature engineering (Spark)

### A0 — Scaffold + uv env + local Spark + Kaggle pull
- **Where A0 runs:** **[settled]** do it on the **Mac**, not Windows — Spark on Windows needs winutils + HADOOP_HOME (fails in opaque ways); macOS is `brew install openjdk@17` + `uv add pyspark`. Doing it on Windows would debug a Windows-only problem you're about to delete.
- **Java version:** pin **Java 17** (LTS; PySpark 3.5 unhappy on 21+ in some configs). uv.lock carries Python deps, *not* Java — JDK is a separate manual install on the Mac.

### A1 — Spark session + load CSVs + wide→long melt (~59M rows)
- **Melt engine:** **[settled]** Spark (`stack`/`explode`), not pandas — this is the load-bearing distributed-compute justification. pandas would work but is the "Spark as decoration" trap.
- **Local perf:** set `spark.sql.shuffle.partitions` ~8–16 (not default 200); stay lazy, no `.collect()` of the long frame. DoD: row count = 30490 × n_days, correct dtypes.

### A2 — Time features (7/28-day lags, rolling means)
- **Leakage discipline:** **[settled]** rolling means use `rowsBetween(-N, -1)` (excludes today), lags use `F.lag(sales, k)`; window = `partitionBy(series).orderBy(date)`. No alternative — just correct vs silently-broken.
- **DoD includes a leakage unit test** in `tests/`: assert lag_7 == own value 7 rows earlier, and no feature at row t uses any value at ≥ t. This test is the defensibility artefact.

### A3 — Exogenous features (calendar/event/SNAP flags, price deltas)
- **Event encoding (open question → resolved):** **[settled]** `event_type` (4 dummies) + a handful of named high-impact events (Super Bowl, Thanksgiving, pre-Christmas) + a **separate Christmas-closure structural-zero flag** — NOT all ~30 `event_name` one-hots. Why: one store = tiny per-event sample; 30 sparse dummies add variance and read as no-judgment. Christmas flag is a structural zero (store closed), grouped with pre-launch-zero masking, not with demand events.
- **Join safety:** assert row count unchanged before/after joins (catch many-to-one fan-out).

### A4 — Persist partitioned Parquet feature store + downcast + read slice via DuckDB
- **Partition key:** **[settled]** partition by `store_id` then `dept_id` — the only consumer reads one store+dept, so the partition IS the query optimisation (DuckDB pushdown skips ~29k series). Don't add date partitioning (small-file problem).
- **What to persist:** **[settled]** full population feature store (the amortised-compute claim), downcast float64→float32 / int64→int32-16 before write (~halves size).
- **ORDER BY discipline:** lives on the READ side — every DuckDB time-series pull ends `ORDER BY date` (Parquet/DuckDB don't guarantee row order; unordered rows scramble lags — same failure class as A2).
- **Time-budget:** most likely Day-1 over-runner (59M-row partitioned write + first DuckDB-over-Parquet wiring). Finish carefully; a broken feature store poisons everything downstream.

---

## Feature B — Forecasting core (the heart)

### B1 — Rolling-origin backtest harness + metrics + baselines + MLflow
- **MLflow footprint (open question → resolved):** **[settled]** log runs only (params/metrics/family tag per fold); NO registry, NO staged promotion. A results DataFrame would be simpler on pure function; MLflow kept as a portfolio signal + run-comparison UI, at the logging tier only. Matches SPEC "partially in scope."
- **Harness shape:** **[settled]** expanding (anchored) window; horizon h=28; origin step 28 (non-overlapping → ~independent fold errors); refit every origin (cut to every-other if time-tight). ~8–12 folds.
- **Metric denominator:** **[settled/teach]** RMSSE scale = in-sample one-step seasonal-naive MSE computed on the TRAIN window per series, then aggregated; WMAPE/WRMSSE weight by dollar volume. Test must pin scale to train window (leakage guard).
- **ETS reframe:** **[settled]** keep ETS in B1 (fast via statsforecast) — it's a strong classical benchmark AND the de-risk for B3, making SARIMA genuinely droppable.
- **DoD:** harness wraps any fit/predict; seasonal-naive + ETS scored; test proves no train row post-dates its fold origin. **Protect this subfeature; resist config-framework gold-plating.**

### B2 — LightGBM global model (point forecast)
- **Global model:** **[settled]** one LightGBM across all FOODS_3/CA_3 series (series id categorical) — pooled cross-series learning, the reason the Spark feature store exists.
- **Recursive vs direct multi-step:** **[settled]** DIRECT — forecast all 28 days using only origin-known features (lags ≥ 28), drop lag_7 for h>7. Structurally leak-proof; gives up a little short-horizon sharpness vs recursive (M5-winner style, but error-compounding + leakage-risky). Cleaner story for the scrutiny bar.
- **Time-budget:** low-med; trap is hyperparameter fiddling — fix a config, let the backtest judge.

### B3 — SARIMA comparison (classical; slack-absorber)
- **What SARIMA forecasts:** **[settled]** Option A — a representative SAMPLE of series (top-N dollar volume), scored on the same harness, reported as indicative. NOT the aggregate FOODS_3 total (Option B forecasts a different target → RMSSE not comparable to disaggregated models; apples/oranges footgun).
- **Time-budget:** highest-risk in B (auto_arima order search × many series). Designated slack-absorber: cut sample size or drop to E1 ("classical baseline, not run") — ETS in B1 covers the classical slot.

### B4 — Prediction intervals (conformal-led; THE deliverable)
- **Method:** **[settled]** build split conformal (simple, distribution-free marginal coverage guarantee) + EnbPI (time-series-correct, no exchangeability assumption); name the exchangeability caveat explicitly (time-series residuals are autocorrelated). CQR (adaptive width) named-not-built unless time allows.
- **Why not just quantile regression:** it estimates quantiles with NO coverage guarantee; conformal wraps any point model WITH a finite-sample guarantee — and "honest uncertainty" is the whole pitch.
- **DoD:** 90% intervals over the harness with MEASURED empirical coverage + average width; split-conformal vs EnbPI compared. B4 takes priority over B3 if time forces a choice.

---

## Feature C — Causal layer

### C1 — Intervention + control-group selection
- **[settled in EDA §3]** treated = `FOODS_3_697 @ CA_3`, cut ~2011-08-06 ($3.58→$2.98, −17%); controls = top-15 same-store/same-dept, pre-cut corr 0.52, price-stable ±8wk; rejection trail kept (FOODS_3_822 stockout, FOODS_2_227 near-dead).
- **Residual flag (carry, don't re-decide):** pre-period thin (~26 weeks) — this is the causal layer's real weakness and the reason C2b replication exists (replication answers thin single-series inference).

### C2 — Difference-in-Differences on the price cut
- **Levels vs logs:** **[settled]** log(sales) DiD — interaction coef reads as ~% change (elasticity story); tames variance. Stay weekly (0% zero-weeks → no log(0)); if ever daily, log1p + justify.
- **Inference with ONE treated unit (the trap):** **[settled]** standard clustered SEs are untrustworthy with a single treated unit → use **placebo/permutation inference** (treat each control as pseudo-treated, place the real effect in that distribution) as the actual p-value. Five-store replication (C2b) is the other half of the answer.
- **Event-study (lead/lag) plot** = the parallel-trends *evidence* (flat pre-cut leads), distinct from the effect estimate. Build it — most persuasive causal visual.
- **Large elasticity:** **[settled]** +60–80% off −17% ⇒ elasticity ≈ −4/−5 (large). Report with uncertainty + a confound sanity check (promo/display that week?); do NOT assert the number.

### C2b — Scale beyond one case (five-store replication)
- **Simultaneous, not staggered:** **[settled/teach]** same item + same week in 5 stores = *simultaneous* adoption → clean **pooled/stacked DiD**; the TWFE-bias literature (Goodman-Bacon, negative weights) does NOT apply here.
- **Committed:** five per-store DiD + pooled estimate + verdict (does it replicate?).
- **Stretch (staggered many-cut panel, 632 candidates):** **[settled]** named-not-built unless time is generous; IF built, must use a heterogeneity-robust estimator (**Callaway–Sant'Anna**), not naive TWFE.

### C3 — CausalImpact on SNAP  **(reframed — folded into SPEC)**
- **The problem:** **[settled]** classic CausalImpact assumes a single persistent onset + clean pre-period; SNAP is a recurring monthly pulse (no pre-period) and store-wide (no in-store control) → naive CausalImpact violates its own assumptions.
- **The reframe:** **[settled]** drive the counterfactual with **cross-state control series** (SNAP days differ by state → a TX/WI store is unaffected on CA's SNAP days = valid predictor); estimand = **average SNAP-day lift**, not single-onset effect. Keep CausalImpact/BSTS (or synthetic control) for the "second named method" story.
- **Fallback if squeezed:** SNAP-day coefficient from LightGBM feature / regression indicator — legitimate magnitude.
- **Time-budget:** highest *conceptual* risk in C (identification, not code) — budget thinking-time or downgrade to fallback consciously.

---

## Feature D — Dashboard & narrative (Streamlit)

### D1 — Streamlit skeleton + forecast-vs-actual + interval panel
- **Live compute vs pre-computed (governs all of D):** **[settled]** dashboard READS pre-computed artefacts (forecasts/intervals/backtest errors/causal estimates written by B & C as lightweight Parquet/CSV); never trains live. This is what makes D4 free-hosting feasible.

### D2 — Segment-level error panel + asymmetric-cost visual (DuckDB rollups)
- **[settled]** sound — direct expression of the "aggregates hide spiky items" principle; DuckDB does hierarchical rollups (second consumer justifying the query layer). Nudge: asymmetric-cost visual shows *signed* error / cost curve, not just magnitude.

### D-optional — Rolling-backtest-error-over-time panel (open question → resolved)
- **[settled] IN.** Near-zero new infra (reuses B1 per-origin errors) and it's the narrative hinge forecast→causal ("error degraded here → why?"). Highest value-per-effort in D.

### D3 — Intervention-effect panel (effect + CI)
- **[settled]** sound — read-out of Feature C artefacts (price-cut lift + CI, five-store replication, SNAP estimate).

### D4 — Deploy to free hosting (Streamlit Community Cloud, $0)
- **Data-size constraint:** **[settled/teach]** the 59M-row feature store CANNOT go to the free host (gitignored). Deploy ships only the small CA_3/FOODS_3 result artefacts (~MB), committed and read by relative path. Watch: don't gitignore the dashboard's own result artefacts the way you (correctly) gitignore the feature store.
- **Time-budget:** low IF pre-compute respected from D1; a panel needing live data at D4 = a D1 design failure surfacing late.

---

## Feature E — README & positioning (the output; no explainer/quiz)

### E1 — README
- **[settled]** sound — assembly of existing SPEC content. Leads with backtesting rigour, scope-control (Spark-honesty line), M5 provenance, asymmetric cost, "Production considerations (out of scope)" section. Discipline: say every scope caveat out loud (Spark not needed for modelling; SARIMA on a sample; SNAP via cross-state controls + recurring-treatment caveat) — each honest caveat scores a point.

### E2 — CV bullets mapped to JD language (DS vs ML-eng)
- **[settled]** sound — DS framing (causal inference, rolling-origin backtesting, calibrated intervals) vs ML-eng framing (Spark feature store, DuckDB query layer, MLflow tracking, deployed surface).

---

## Open questions — all resolved
- **MLflow footprint** → log runs, no registry (B1).
- **Rolling-backtest-error panel** → IN (D-optional).
- **Events encoding** → `event_type` + a few named + Christmas-closure flag (A3).
- **A0 on Mac / clean Spark** → confirmed (A0).
