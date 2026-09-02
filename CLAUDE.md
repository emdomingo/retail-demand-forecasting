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
# environment      — (venv / conda activation; set at A0)
# fetch data       — kaggle competitions download -c m5-forecasting-accuracy   (A0)
# run tests        — (set at B1)
# run backtest      — (set at B1)
# launch dashboard — (set at D1)
```

## Architectural decisions already made (see SPEC.md for rationale)

- **Spark is a feature store**, not decoration. Engineer features across all ~30,490 series in Spark, persist partitioned Parquet, model reads back one slice. Frame as "population-scale pipeline, scoped model" — never claim modelling needed Spark.
- **DuckDB is the query layer.** All slice-reads and hierarchical/segment aggregations over the Parquet feature store go through DuckDB — it owns the job, not a lonely `SELECT` beside pandas. Feeds both the models and the D2 dashboard rollups. **Every time-series pull must `ORDER BY date`** — DuckDB doesn't guarantee row order, and unordered rows silently scramble lags/models (leakage-adjacent; treat like the A2 shift discipline).
- **Scope the model to one store** (or one category). State it explicitly.
- **Backtesting: rolling-origin only.** No future leakage. Build the harness (B1) before any model; score everything against seasonal-naive.
- **Intervals: conformal-led** (coverage guarantee), EnbPI as the time-series-correct variant, quantile regression as the adaptive alternative.
- **Causal: price-cut DiD + SNAP CausalImpact.** SNAP has no within-store control group, so it cannot use DiD — it goes to CausalImpact's modelled counterfactual.
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

- **macOS, zsh.** The whole project runs here (chosen over Windows specifically to remove the Spark-on-Windows friction).
- Local Spark on macOS is straightforward: Java via Homebrew, no `winutils`/`HADOOP_HOME`, Unix paths throughout. A0 is still its own subfeature but should be quick.
