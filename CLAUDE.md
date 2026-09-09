# CLAUDE.md — Working agreement

Context for every coding session on this project. Read alongside [SPEC.md](SPEC.md).

**Division of labour:** CLAUDE.md is operational — how we work, where things go, how to run them — and is loaded every session, so keep it lean. SPEC.md is the detailed plan (the feature tree and rationale). The decisions list below is a summary with pointers to SPEC; don't grow it into a second copy of SPEC. This is a **living file** — update the layout, data, and commands sections as they materialise.

## What this project is

A portfolio-grade retail demand-forecasting system: defensible inbound-volume forecast + honest uncertainty intervals + a causal layer explaining divergence. Modelled on the Amazon SCOT Data Scientist role. Built to close three evidenced profile gaps: **time-series forecasting, causal inference, distributed compute (Spark)**.

## How we work (the learning loop)

This is a learn-by-building project. The workflow is fixed:

- **Two living artefacts per _feature_** (not per subfeature — avoids repo explosion): an **explainer** and a **quiz**.
  - Location: `docs/<feature>/explainer.md` and `docs/<feature>/quiz.md` (e.g. `docs/A-ingestion/`).
  - Feature E (README & positioning) gets **no** explainer/quiz — it is itself the output.
- **Per-subfeature loop:** code the subfeature → **append** to the explainer → **append** to the quiz → user self-quizzes → user decides stream-ahead or pause.
- **No gating.** The quiz is a checkpoint, never a blocker. The user decides per subfeature whether to pause.
- **Definition of done (per subfeature):** code works + explainer appended + quiz appended + committed to git.

### Artefact conventions
- **Explainer = concise prose.** Explain how this part works and how it fits the whole. Offer a text/visual diagram only if asked. Prioritise the three learning goals: systems thinking (how parts compose), syntax on unfamiliar libraries, and decision gates (forks, tradeoffs, reversible vs irreversible).
- **Quiz = a mix** of recall, predict-the-decision, and spot-the-flaw questions. Plain markdown, self-quiz format.
- Both files are **append-only** across a feature's subfeatures; complete by the feature's end.

## Repository layout

```
src/            importable pipeline + modelling code (.py modules)
  ingest/       Spark melt + feature engineering (A)
  query/        DuckDB slice-reads + hierarchical/segment aggregations
  forecast/     backtest harness, models, intervals (B)
  causal/       DiD + CausalImpact (C)
  dashboard/    Streamlit app (D)
tests/          tests for the harness and pipeline logic
notebooks/      exploratory EDA + learning scratch (not the deliverable)
data/
  raw/          M5 CSVs — gitignored
  processed/    partitioned Parquet feature store + slices — gitignored
docs/           per-feature explainer.md + quiz.md (e.g. docs/A-ingestion/)
SPEC.md         the detailed plan
CLAUDE.md       this file
```

## Data & secrets

- M5 data is public but large. Raw CSVs in `data/raw/`, feature store in `data/processed/` — **both gitignored**. Never commit data; the repo is reproducible from the Kaggle CLI, not from committed files.
- Kaggle credentials (`kaggle.json` / `KAGGLE_*`) are **never committed** — keep the token in `~/.kaggle/` and gitignore it.
- Re-fetch: `kaggle competitions download -c m5-forecasting-accuracy` (documented in A0).

## Code vs notebooks

Both a learning exercise and a portfolio piece — the repo must read as production-shaped, not a pile of notebooks.

- **Notebooks** (`notebooks/`): EDA, learning, one-off exploration. Not the deliverable.
- **Modules** (`src/`): anything reused or defended — especially the **backtest harness (B1)**, the pipeline, and the models — lives in importable `.py` modules **with tests** in `tests/`. If an interviewer would scrutinise it, it's a module, not a cell.

## Commands

*Living section — fill in each as it's built.*

```
# env: create/sync  — uv sync            (recreates the locked env from pyproject.toml + uv.lock)
# env: add a dep    — uv add <pkg>       (edits pyproject.toml, resolves, installs; --dev for tooling)
# run anything      — uv run <cmd>       (e.g. uv run python …, uv run jupyter lab)
# fetch data        — uv run python -m src.ingest.fetch_data   (A0; idempotent, --force to re-pull)
#                     needs KAGGLE_API_TOKEN in .env (gitignored) + accepted M5 rules (else 403)
# lint / format     — uv run ruff check . / uv run ruff format .
# run tests         — uv run pytest       (pythonpath=. set in pyproject; testpaths=tests)
# run backtest (B1) — uv run python -m src.forecast.backtest  (baselines on CA_3 sample, MLflow)
# run LightGBM (B2) — uv run python -m src.forecast.models  (global recursive LGBM, same sample)
# run SARIMA (B3)   — uv run python -m src.forecast.sarima  (per-series SARIMA, same sample; ~2m20s)
# run intervals (B4)— uv run python -m src.forecast.intervals  (conformal coverage eval; ~40s)
# run DiD (C2)      — uv run python -m src.causal.did  (price-cut DiD + event study + placebo; ~9s)
# run replication   — uv run python -m src.causal.replication  (C2b: 5-store DiD + meta-analysis; ~30s)
# run SNAP (C3)     — uv run python -m src.causal.snap  (cross-state regression counterfactual; ~15s)
# mlflow ui         — uv run mlflow ui --backend-store-uri sqlite:///mlruns.db  (view runs)
# melt demo (A1)    — uv run python -m src.ingest.load   (loads 3 CSVs, melts to ~59M long rows)
# features demo (A2)— uv run python -m src.ingest.features  (lags + rolling means; leakage-safe)
# exog demo (A3)    — uv run python -m src.ingest.exogenous  (calendar/SNAP/price joins)
# build store (A4)  — uv run python -m src.ingest.feature_store  (full pipeline -> Parquet, ~2-3 min)
# read slice (A4)   — uv run python -m src.query.slices  (DuckDB slice-read demo)
#   in code: from src.query.slices import read_store_slice; df = read_store_slice("CA_3")
# build forecast (D1)— uv run python -m src.forecast.persist  (bands held-out origin + quantile grid)
# segment rollup(D2)— uv run python -m src.query.segments  (DuckDB WMAPE/RMSSE by cat/dept/tier)
# build causal (D3)— uv run python -m src.causal.persist  (runs C2+C2b+C3 -> causal.json, ~1min)
# launch dashboard  — uv run streamlit run src/dashboard/app.py  (D1-D3; reads persisted artifacts)

# Env facts (A0): Python 3.11 (uv-managed), Java 17 (Homebrew, for the Spark JVM),
#   PySpark 4.2, pandas pinned <3.0 (PySpark 4.2 interop). Verify Spark: it starts a
#   local[*] SparkSession and runs a job — the true A0 gate, not just `import pyspark`.
# Feature store (A4): data/processed/feature_store/ (gitignored), Parquet partitioned by
#   store_id (10 dirs), 59.18M rows, grain = one row per (id, d_int). Downcast: sales/lags/
#   d_int int16, flags int8, means/prices float32, wm_yr_wk stays int32. Read via DuckDB
#   (src/query/slices.py) — always ORDER BY date. Rebuild needs 8g driver heap (get_spark).
# Backtest (B1): rolling-origin harness in src/forecast/ (metrics.py, baselines.py, backtest.py).
#   Model protocol = forecast(train, test_keys)->yhat. MLflow -> sqlite:///mlruns.db (gitignored).
#   Baseline bar on CA_3 200-series sample (4 origins, h=28): seasonal_naive RMSSE 0.96 / WMAPE
#   0.82; ETS RMSSE 0.735 / WMAPE 0.71. B2 LightGBM must beat ~0.735 RMSSE (or explain honestly).
# LightGBM (B2, src/forecast/models.py): global recursive model, one fit across all series.
#   Recursive multi-step (own preds feed lags); AR features rebuilt in-model (not from store);
#   Tweedie objective; series identity via lags+dept/cat, not item_id. Widened the harness:
#   BacktestConfig.known_future passes calendar/price to test rows (never sales/precomputed lags).
#   v1 (lags 7,28 / rmean 7,28): RMSSE 0.732 / WMAPE 0.680 — edges ETS, wins WMAPE.
#   v2 (lags 7,14,28 / rmean 7,28,56): RMSSE 0.727 / WMAPE 0.674 — real gain, clears ETS on both.
#   Ablation finding: naive enrich (add lag_1 + early stopping) REGRESSED to 0.735. lag_1 is
#   recursion-toxic (yesterday = own prediction for 27/28 days -> error compounds); one-shot
#   early stopping tunes a one-step regime the recursive test doesn't share. So v2 omits both.
#   version= flows into MLflow run name (lightgbm_global_v1/_v2) so runs compare, not overwrite.
# SARIMA (B3, src/forecast/sarima.py): classical per-series comparison (contrast to B2's global).
#   Fixed order SARIMA(1,1,1)(1,0,0)_7 — NOT pmdarima auto_arima (per-series order search overfits
#   the order to noise + numpy-2 friction). D=0 on purpose (seasonal diff over m=7 destabilises
#   zero-heavy short series). Robust: <2-season/all-zero -> last-value fallback (matches ETS),
#   stationarity/invertibility unenforced (converges more; clipped to >=0), maxiter=50, any fail
#   falls back. Result: RMSSE 0.734 / WMAPE 0.696 — a TIE with ets (0.735); both classical
#   per-series methods plateau ~0.735 while global LightGBM v2 (0.727) wins. The plateau IS the
#   finding (pooling beats isolation; the specific per-series method barely matters). ~2m20s.
# Intervals (B4, src/forecast/intervals.py): THE deliverable. Split conformal wrapping v2, adapted
#   two ways: (1) per-horizon calibration (h=1..28) since recursive error compounds; (2) scale-
#   normalised residuals (resid/sqrt(naive_scale)) so bands track each series' volatility. Signed
#   tails => asymmetric (default) vs symmetric mode; conservative finite-sample quantiles (err wide).
#   Time-ordered split: calibrate on earlier origins, TEST coverage on held-out latest origin (no
#   leakage). Result @ target 90%: asymmetric coverage 0.912 / width 5.10; symmetric 0.905 / 4.49 —
#   calibrated. Coverage is MARGINAL (over series), not per-series conditional (named limitation).
#   EnbPI (online TS-correct upgrade) + quantile regression (muddy under recursion) NAMED not built.
#   Exchangeability caveat named. MLflow logs coverage + per-h coverage/width.
# DiD (C2, src/causal/did.py): the causal half — how much of the post-cut jump the price cut
#   CAUSED. Treated FOODS_3_697 @ CA_3, cut 2011-08-08 ($3.58->$2.98, -16.8%). Reads the same
#   feature store via DuckDB, aggregates to WEEKLY (M5 prices are weekly; kills day-of-week noise),
#   matched top-15 controls (pre-cut corr, price-stable +/-8w, demand bar). Estimator = TWFE OLS
#   log_units ~ C(item)+C(week)+treated:post, cluster SE by item. Balanced +/-26w window (thin ~27w
#   pre-period). Result: +42.1% lift (95% CI [+12.5%, +79.4%]), elasticity ~ -2.5 — DISCIPLINED
#   DOWN from the naive +56% (raw jump overstated by a third; that gap IS the method's value).
#   Evidence: event-study leads flat (slope ~-0.02/wk, no pre-trend) + placebo covers 0 (pass,
#   though noisy -27.7% point). Few-cluster SE understates uncertainty -> wild bootstrap NAMED not
#   built. Single-store estimate is noisy BY DESIGN -> C2b replicates chain-wide across 5 stores.
#   Event-study plot -> docs/C-causal/figures/event_study.png (committed).
# Replication (C2b, src/causal/replication.py): same DiD re-run across the 5 stores that made the
#   identical chain-wide cut (CA_3/CA_1/TX_1/TX_2/CA_4). Per-store cut DETECTED (chain rolled it
#   out a week apart: CA_3 08-08, rest 08-15 — hardcoding one date would bias 4 toward zero) +
#   per-store matched controls. Pool = inverse-variance meta-analysis, fixed AND random effects
#   (DerSimonian-Laird). Result: 5/5 positive, 5/5 significant; 4 stores +41-50%, TX_1 outlier
#   ~+100% (thinnest store). Heterogeneity Q=9.2 (p=0.06), I^2=57% => quote the RANDOM-effects
#   pool +56.9% [+34.6%, +82.9%], NOT the too-narrow fixed +61.1% [+46%, +78%]. Replication IS the
#   credibility (neutralises C2's thin pre-period). Staggered-adoption panel TWFE across the 632
#   cuts = NAMED not built (Goodman-Bacon/de Chaisemartin bias needs modern estimators). Forest
#   plot -> docs/C-causal/figures/replication_forest.png. ~30s.
# SNAP (C3, src/causal/snap.py): second causal method on the second intervention. SNAP breaks DiD
#   (recurring monthly pulse => no clean pre-period; store-wide => no in-store control), so the
#   counterfactual is CROSS-STATE: SNAP schedules differ by state, so on a CA SNAP day TX/WI stores
#   are live controls for "CA without its SNAP boost". OLS on log daily FOODS units @ CA_3, HAC
#   (Newey-West, 14 lags) SEs. This IS the CausalImpact/BSTS idea in transparent OLS form (BSTS =
#   named Bayesian sibling, not built -> avoids tfcausalimpact/TF friction). Estimate ladder: naive
#   +16.8% -> calendar +16.9% -> cross-state +10.7% [+9.1,+12.3] (R^2 0.12->0.93; disciplined down
#   ~a third, same lesson as C2). Include tx_snap/wi_snap as covariates (CA/TX/WI SNAP windows
#   overlap early-month: 384/640 CA days coincide) to net out control-state SNAP. Falsifications:
#   placebos (CA schedule as fake treatment on TX/WI) both cover 0 (PASS); clean-day estimator (128
#   CA-only SNAP days vs no-SNAP days) +11.8% corroborates. Caveats: average (not one-off) lift,
#   anticipation/pantry-loading, asymptotic HAC SE. Figure -> docs/C-causal/figures/
#   snap_counterfactual.png. ~15s. Closure (Christmas, 0 units) days dropped (log(0)).
# Dashboard D1 (src/dashboard/app.py + src/forecast/persist.py): the planner's viewing surface,
#   forecast-vs-actual + conformal interval. DECISION: persist, don't compute (option 1) — the app
#   is a PURE READER (imports no LightGBM/harness), so free hosting (D4) works and the shown
#   coverage IS the backtest's. persist.py refits v2 across the 4 rolling origins, calibrates
#   conformal (via intervals.calibration_test_split, shared with evaluate_conformal so the split is
#   one code path), bands the HELD-OUT LATEST origin, writes three files to data/processed/forecast/
#   (gitignored): forecast_CA_3.parquet (5,600 rows = 200 series x 28d; sales/yhat/lower/upper/
#   width/scale + dept/cat) + forecast_CA_3.json (model, mode, target vs EMPIRICAL coverage 0.912,
#   mean width 5.10, origin 2016-04-24, n_series) + quantiles_CA_3.parquet (D2 grid, below).
#   Persists the SAME
#   fixed 200-series sample (seed 0) as B1-B4 so the sidecar coverage == the documented B4 number
#   (sample_size=None would forecast the whole store but then the coverage no longer matches).
#   App: @st.cache_data over the Parquet read + a DuckDB read_store_slice for pre-origin context
#   (56d); Altair chart (ships inside Streamlit, no new dep) = shaded band (the star) + dashed
#   yhat + solid actual (context concatenated with horizon, drop_duplicates -> one continuous
#   line) + red origin rule. Header shows marginal coverage; a per-series coverage metric is
#   flagged as noisy (n=28) -> surfaces B4's marginal-vs-conditional caveat in the UI. Tests:
#   test_persist.py (schema, sidecar==coverage_report, round-trip, missing-file error) via a df=
#   seam on build_forecast_artifact (no feature store needed).
# Dashboard D2 (src/query/segments.py + src/forecast/decision.py + app.py panels): two panels over
#   the SAME persisted artifact (still a pure reader). D2a SEGMENT ERROR (segments.py, DuckDB owns
#   the rollup): WMAPE (pooled) + RMSSE (per-series-then-mean, reconstructed as RMSE/scale since
#   persisted scale = sqrt(naive_scale)) + UNIT SHARE, grouped by cat_id/dept_id/volume-tier
#   (pd.qcut quartiles). THE finding = the guardrail: Q1 low-vol WMAPE ~176% but 3% of units vs Q4
#   high-vol WMAPE ~48% & 71% of units -> a single blended WMAPE (~0.6) describes no real item.
#   D2b ASYMMETRIC COST (decision.py = NEWSVENDOR): critical ratio q*=Cu/(Cu+Co); order = yhat +
#   scale*offset_h(q*) read from the persisted per-horizon QUANTILE GRID (quantiles_CA_3.parquet,
#   99 q x 28 h, from intervals.calibration_quantile_grid on the cal residuals). Order to yhat is
#   optimal ONLY at Cu=Co (q*=0.5); stockouts costlier -> q*>0.5 -> order into the upper half of the
#   band. policy_costs evaluates point vs newsvendor on the held-out ACTUALS: as Cu:Co 1->9 fill
#   rate ~68%->92%, realised cost saved up to ~40% (even 1:1 saves ~2% by ordering to the median
#   not the right-skew-inflated mean). Cost is NORMALISED (Co=1; only the ratio drives the order) ->
#   caption says so, never $. Grid not a live calibrator (reader-pure); per-horizon because
#   recursive error compounds. Tests: test_decision.py (critical ratio, grid lookup+scale, orders
#   rise w/ q*, clipped>=0, cost accounting, fill rises w/ Cu:Co) + test_segments.py (WMAPE pool,
#   RMSSE via scale, unit_share sums to 1, tiers partition) + test_persist.py grid schema/monotone.
# Dashboard D3 (src/causal/persist.py + app.py _causal_panel): the causal layer surfaced — WHY
#   demand moved. Same persist-then-read: causal/persist.py runs C2 (did) + C2b (replication) + C3
#   (snap) ONCE offline (~1min) -> data/processed/causal/causal.json (gitignored); app imports NO
#   statsmodels. Numbers are the module outputs verbatim (did.estimate_did/event_study/placebo_test,
#   replication.replicate+pool, snap.estimate_ladder/placebo/clean_day) + 2 derived: _naive_jump
#   (treated raw pre/post ratio = the uncontrolled number DiD disciplines down from) & elasticity
#   (lift / -0.168 price change). Panel = 2 st.tabs. PRICE CUT tab: cards DiD +42.1% [+12.5,+79.4]
#   / naive +55.9% / elasticity -2.5; Altair event-study (leads flat=parallel trends), placebo
#   covers 0; replication forest (5 stores + pooled), caption quotes RANDOM-effects +56.9%
#   [+34.6,+82.9] (NOT fixed, I^2~57%). SNAP tab: cards cross-state +10.7% [+9.1,+12.3] / naive
#   +16.8%; Altair ladder (naive->calendar->cross_state, cross_state highlighted); placebos cover
#   0 + clean-day corroborates. Charts REBUILT in Altair (not the committed C-figures PNGs) from
#   the JSON, log points -> % (expm1) for display. Tests: test_causal_persist.py (round-trip,
#   missing-file error, _naive_jump, _snap_row) — estimators covered by test_did/replication/snap.
```

## Architectural decisions already made (see SPEC.md for rationale)

- **Spark is a feature store**, not decoration. Engineer features across all ~30,490 series in Spark, persist partitioned Parquet, model reads back one slice. Frame as "population-scale pipeline, scoped model" — never claim modelling needed Spark.
- **DuckDB is the query layer.** All slice-reads and hierarchical/segment aggregations over the Parquet feature store go through DuckDB — it owns the job, not a lonely `SELECT` beside pandas. Feeds both the models and the D2 dashboard rollups. **Every time-series pull must `ORDER BY date`** — DuckDB doesn't guarantee row order, and unordered rows silently scramble lags/models (leakage-adjacent; treat like the A2 shift discipline).
- **Scope the model to one store** (or one category). State it explicitly.
- **Backtesting: rolling-origin only.** No future leakage. Build the harness (B1) before any model; score everything against seasonal-naive.
- **Intervals: conformal-led** (coverage guarantee), EnbPI as the time-series-correct variant, quantile regression as the adaptive alternative.
- **Causal: price-cut DiD (C2/C2b) + SNAP cross-state regression counterfactual (C3).** SNAP has no within-store control group and no clean pre-period, so it cannot use DiD; instead cross-state controls (TX/WI, on different SNAP schedules) drive an OLS counterfactual for the average SNAP-day lift. CausalImpact/BSTS is the named Bayesian sibling of that same idea, not built (avoids the tfcausalimpact/TensorFlow dependency friction). See SPEC C3.
- **MLflow in B1**, tracking every backtest run — it's the one MLOps tool that earns its place, because it makes the "compare families and defend a choice" comparison reproducible. The rest of the production surface (real ingestion, drift monitoring, retraining) is **named in the E1 README, not built** — scoping judgment reads as maturity, and faking a live feed over static M5 data would undercut the scrutiny bar. See the "Production considerations" section in SPEC.md.

## Guardrails

- The interval is the deliverable, not the point estimate.
- Name the asymmetric cost (under-forecast usually hurts more).
- Beat the naive baseline or explain why not honestly.
- Segment-level error matters — aggregates hide the spiky high-value items.
- Every claim must survive interview scrutiny. No overclaiming, especially on Spark.
- **Push back on the merits.** The architecture is not set in stone — argue specifics when something doesn't make sense, not because "the brief said so."

## Stack

`PySpark (local) → Parquet feature store → DuckDB (query/aggregation) → LightGBM / SARIMA / ETS on rolling-origin backtesting (MLflow-tracked) → conformal intervals → DiD / CausalImpact → Streamlit (free hosting)`. $0 infra throughout. Forecast and causal are parallel branches off the query layer, not sequential. The backtest harness *wraps* the models (a fit/predict loop), it is not a pipeline stage.

## Environment

- **Package/env manager: uv.** One tool for the Python version, the venv, and deps. Env is defined by `pyproject.toml` + committed `uv.lock`; `uv sync` reproduces it exactly on any machine — this is what makes the Windows→macOS move deterministic. Use `uv add` to add deps, `uv run` to execute in the env (no manual activation). Set up in A0. uv manages Python only — **Java for Spark is installed separately** (see below).
- **Target platform: macOS, zsh.** The project is intended to run here (chosen over Windows specifically to remove the Spark-on-Windows friction). Exploration/scaffolding may start on Windows; the `uv.lock` is what carries the env across.
- Local Spark on macOS is straightforward: Java via Homebrew, no `winutils`/`HADOOP_HOME`, Unix paths throughout. A0 is still its own subfeature but should be quick. (On Windows, Spark additionally needs `winutils.exe` + `HADOOP_HOME` — the friction the macOS move avoids.)
