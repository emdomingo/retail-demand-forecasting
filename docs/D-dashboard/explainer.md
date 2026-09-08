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
