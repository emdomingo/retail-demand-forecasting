# Retail Demand Forecasting — SPEC

## Anchor

> A system that gives supply-chain planners a defensible inbound-volume forecast with honest uncertainty bounds, plus a causal layer to explain accuracy degradation when actuals diverge from plan — because in planning, the interval and the "why" matter as much as the point estimate.

Every subfeature below serves that sentence. If a piece of work doesn't feed it, it's scope creep.

---

## Scope decisions (locked this session)

- **Data:** M5 Forecasting – Accuracy (Kaggle). Public Walmart benchmark. Described honestly as such.
- **Modelling scope:** train/serve on **one store** (or one category). The *pipeline* is population-scale; the *model* is scoped for tractability. Stated explicitly in the README.
- **Spark = feature store, not theatre.** Feature engineering runs in Spark across all ~30,490 series and persists a partitioned Parquet feature table. Justification: build features once across the whole population; models subscribe to the slice they need (amortised, not wasted compute). We do *not* claim the modelling required Spark.
- **DuckDB = the query layer.** All downstream slice-reads and hierarchical/segment aggregations over the Parquet feature store go through DuckDB (SQL, in-process, Parquet-native pushdown). It owns this job — it does not merely supplement pandas. Serves two consumers: the models and the dashboard rollups. Reinforces the existing SQL narrative; chosen over Polars for that reason. **Gotcha: DuckDB does not guarantee row order without `ORDER BY` — every time-series pull must `ORDER BY date`, or lags and models are silently scrambled. This is leakage-adjacent; treat it with the same care as the A2 shift discipline.**
- **Causal:** price cut → **DiD** (clean unit-level control group); SNAP → **CausalImpact** (modelled counterfactual, no untreated twins needed). Two methods × two interventions = a genuine robustness pairing.
- **Intervals:** **conformal prediction** leads (distribution-free coverage guarantee, cheap, model-agnostic). Name the exchangeability caveat for time series and **EnbPI** as the time-series-correct upgrade. Quantile regression noted as the adaptive-width alternative.
- **Backtesting:** rolling-origin only. No future leakage, ever. Every model scored on the same harness against a seasonal-naive baseline.
- **Deliverable surface:** Streamlit dashboard (a planner's viewing surface), not an API. $0 infra throughout.

---

## Feature → subfeature tree

Granularity: **medium** — a subfeature is one coherent concept worth its own quiz, ~one code session. Each subfeature = code → append explainer → append quiz → self-quiz → decide stream-ahead/pause. No gating; the quiz is a checkpoint, never a blocker.

### Feature A — Ingestion & feature engineering *(Spark)*
- **A0** Repo scaffold + venv + local Spark install + Kaggle data pull
- **A1** Spark session + load 3 CSVs + wide→long melt (~59M rows) *(the melt is the reshape; the feature store is the Spark justification)*
- **A2** Time features in Spark — 7/28-day lags, rolling means *(gate: leakage-safe shifts)*
- **A3** Exogenous features in Spark — calendar/event flags, SNAP flags, price-change deltas + joins
- **A4** Persist partitioned Parquet feature store + downcasting + read back working slice via DuckDB *(gate: what to persist; every series pull `ORDER BY date`)*

### Feature B — Forecasting core *(the heart — intentionally 4)*
- **B1** Rolling-origin backtesting harness + metrics (RMSSE/WMAPE) + seasonal-naive & ETS baselines + **MLflow run tracking** *(built first; the anti-leakage scaffold everything scores against. MLflow logs every backtest run — params, metrics, model family — so the "compare families and defend a choice" comparison is reproducible, not a hand-typed table)*
- **B2** LightGBM global model — point forecast *(what won M5)*
- **B3** SARIMA comparison *(classical; the slack-absorber if B runs over)*
- **B4** Prediction intervals — conformal (EnbPI caveat noted); quantile as alternative *(the interval is the deliverable)*

### Feature C — Causal layer
- **C1** Intervention + control-group selection — EDA + parallel-trends check *(price cut identification; how to pick a clean control in hierarchical data)*. **Settled in EDA (`notebooks/01-eda.ipynb` §3):** treated = **`FOODS_3_697` @ `CA_3`**, price cut ~**2011-08-06** ($3.58 → $2.98, −17%); density 42.2 units/wk, 0% zero-weeks. Controls = same-store/same-dept, matched on pre-cut weekly-sales correlation (top-15, mean pre-cut corr **0.52**), price-stable within ±8 weeks of the cut. The rejection trail is kept deliberately (FOODS_3_822 = stockout at the cut; FOODS_2_227 = 96% zero-weeks) — the honest iteration is part of the narrative.
- **C2** Difference-in-Differences on the price cut — single-intervention estimate **+ event-study (lead/lag) pre-trend evidence + effect CI + a placebo/pre-period test**. *(The eyeballed ~+60–80% lift implies elasticity ≈ −4/−5 — large; validate, don't assert.)*
- **C2b** Scale beyond the one test case — **the "more items" plan, not a one-off.** The chosen cut is **chain-wide** (same item, same week, in CA_3/CA_1/TX_1/TX_2/CA_4): replicate the DiD across all five stores and report whether the effect holds — five agreeing estimates beat one and neutralise the thin (~26-week) pre-period. *Optional stretch:* a **panel / staggered-adoption DiD (two-way fixed effects)** across many cut events from the 632-candidate shortlist, reporting an *average* price-cut effect — the generalisable version, built only if time allows, else named in E1.
- **C3** CausalImpact (Bayesian structural TS) on SNAP — robustness cross-check. **Reframed at runthrough:** classic CausalImpact assumes a *single, persistent onset with a clean pre-period*; SNAP is a **recurring monthly pulse present across the whole window** (no pre-period) and **store-wide** (no in-store control). So drive the counterfactual with **cross-state control series** — SNAP disbursement days differ by state, so a TX/WI store is unaffected on CA's SNAP days and is a valid predictor — and frame the estimand as the **average SNAP-day lift**, not a single-onset effect (CausalImpact/BSTS or synthetic control; keep CausalImpact for the "second named method" story). **Fallback if squeezed:** the SNAP-day coefficient from LightGBM's SNAP feature / a regression indicator — a legitimate magnitude if the full counterfactual doesn't fit the budget. The recurring-treatment caveat is written down either way.

### Feature D — Dashboard & narrative *(Streamlit)*
- **D1** Streamlit skeleton + forecast-vs-actual + interval panel
- **D2** Segment-level error panel + asymmetric-cost visual — hierarchical/segment rollups via DuckDB *(aggregate accuracy hides spiky high-value items)*
- **D3** Intervention-effect panel — effect + CI
- **D4** Deploy to free hosting *(thin)*

> *Optional, thematically-tight (not a committed subfeature — decide at runthrough):* a rolling-backtest-error-over-time panel. Visualises accuracy degradation across forecast origins using machinery B1 already built, and acts as the on-ramp from the forecast half to the causal half ("error degraded here — why?"). Zero new infrastructure.

### Feature E — README & positioning *(no explainer/quiz — this IS the output)*
- **E1** README — leads with backtesting rigour, scope-control statement, data provenance, honest baseline reporting, asymmetric-cost note, **+ the "Production considerations (out of scope)" section below**
- **E2** CV bullets mapped to JD language per role type (DS vs ML-eng)
- **E3** Polish pass — read the README top-to-bottom as a screener would: leads with rigour (not a chart), links the artefacts (SPEC, `runthrough.md`, `study-plan.md`, the dashboard), screenshots the key panels, prunes anything that doesn't serve the anchor, and re-checks every claim survives scrutiny (esp. the Spark-honesty framing and the DiD "large effect → validated" caveat).

---

## Sequencing

| Day | Focus | Subfeatures |
|---|---|---|
| Day 0 | Design decisions & specs | (this session) |
| Day 1 | Spark ingestion + feature store | A0–A4 |
| Day 2 | Baseline + backtest harness + LightGBM *(protect this)* | B1–B2 |
| Day 3 | SARIMA + intervals | B3–B4 |
| Day 4 | Causal analysis | C1–C3 |
| Day 5 | Dashboard + README | D1–D4, E1–E2 |

*If Day 2 runs over, protect the backtesting harness (B1) and let SARIMA (B3) absorb the slack.*

---

## Principles (hold throughout)

- The interval is the deliverable, not the point estimate.
- Name the asymmetric cost (under-forecast usually hurts more) even without fully modelling it.
- "Why did it miss" is the actual job — causal is the other half, not a bolt-on.
- Backtest like a planner lives in the past. Rolling-origin only.
- Beat the naive baseline or explain why not. An honest small win beats a suspicious large one.
- Segment-level matters — aggregate accuracy hides the items that matter for planning.
- Spark is lean but honest: population-scale feature store, scoped model, no overclaiming.
- Every claim survives interview scrutiny.

---

## Production considerations (out of scope) — an E1 README section

Deliberately *named, not built.* Scoping judgment reads as maturity; half-built plumbing over a static dataset reads as the opposite. This section is the answer to "how would you productionise this?" — write it in the README so the whole system surface is visible without spending days on it.

What a real deployment adds, and why each is out of scope here:

- **Real ingestion layer** — orchestrated batch/stream pulls (Airflow/Dagster/Prefect), incremental loads, data contracts. *Out of scope:* M5 is a frozen historical dataset; a live feed would be theatre.
- **Data validation** — schema + distribution checks (Great Expectations / Pandera) on every load.
- **Monitoring, three flavours:**
  - *Input/feature drift* — PSI, KS tests (evidently).
  - *Accuracy degradation* — live RMSSE/WMAPE vs actuals. **Note: the causal layer (Feature C) is the analytical core of this — root-cause for "why did accuracy degrade." We have the hard part; only the automated detection is out of scope.**
  - *Interval coverage* — is the 90% conformal band actually covering 90% live?
- **Retraining pipeline** — drift-triggered or scheduled, champion/challenger. *Out of scope:* drift is simulated on static data; the trigger closes no gap.
- **Serving** — batch/online API. *Deliberately a dashboard, not an API* — the churn API already owns the deployment story.
- **Model registry / experiment tracking** — **partially in scope: MLflow lands in B1** (serves the model-comparison narrative directly). A full registry with staged promotion is out of scope.
- **Lineage / governance** — data + feature + model versioning.

The honest one-liner for the README: *the pipeline is population-scale and the model comparison is tracked; the operational MLOps surface (live ingestion, drift-triggered retraining) is named here rather than built, because faking it over a static benchmark would undercut the "every claim survives scrutiny" bar.*
