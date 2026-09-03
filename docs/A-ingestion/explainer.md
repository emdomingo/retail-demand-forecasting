# Feature A — Ingestion & feature engineering — Explainer

Append-only across A0–A4. Each section explains how that subfeature works and how it
fits the whole pipeline.

---

## A0 — Repo scaffold + env + local Spark + Kaggle pull

**What A0 is:** the foundation the rest of the project stands on. No modelling, no
features — just a reproducible environment and a reproducible path to the data. The bar
is: *any machine can recreate this exactly*, because the repo commits the recipe, never
the data.

### How the pieces fit

```
uv (Python + venv + deps)  ->  pyproject.toml + uv.lock   (committed: the recipe)
Java 17 (Homebrew, system) ->  runs the Spark JVM         (installed separately from uv)
kaggle CLI (a uv dep)      ->  data/raw/*.csv             (gitignored: the payload)
```

Two deliberate splits:

1. **uv manages Python; Homebrew manages Java.** uv installs and pins the *Python*
   interpreter (3.11 here) and every Python package. It does **not** manage the JVM.
   PySpark is a Python wrapper over a Spark JVM, and that JVM needs a system Java. So
   the env is reproducible in two layers: `uv sync` gives you identical Python, and
   `brew install openjdk@17` gives you a compatible JVM. This split is why the
   Windows→macOS move is clean — the `uv.lock` carries the Python half deterministically,
   and macOS removes the `winutils`/`HADOOP_HOME` friction Spark has on Windows.

2. **The recipe is committed; the data is not.** `pyproject.toml` + `uv.lock` are the
   env recipe (committed). `data/raw/` and `data/processed/` are gitignored. The bridge
   between them is `src/ingest/fetch_data.py`, which re-derives the payload from Kaggle.

### The env, concretely (uv syntax)

- `uv init --bare --python 3.11` — create `pyproject.toml` only (no sample code), pinning
  the interpreter. uv downloads the interpreter itself; the system Python (3.9) is untouched.
- `uv add <pkg>` — resolve + install + record in `pyproject.toml` and `uv.lock` in one step.
  `uv add --dev <pkg>` puts it in the `dev` dependency-group (tooling, notebooks) so it
  isn't a runtime dependency of the pipeline.
- `uv run <cmd>` — run inside the env with no manual `activate`. This is how *everything*
  runs (`uv run python …`, `uv run pytest`, `uv run jupyter lab`).
- `uv sync` — recreate the exact locked env on a fresh machine.

Runtime deps chosen for Feature A: `pyspark` (feature store), `duckdb` (query layer),
`pandas` + `pyarrow` (interop + Parquet), `kaggle` (the pull). Modelling deps
(LightGBM, statsmodels, MLflow, Streamlit) are added when their features arrive — not
front-loaded — so the dependency graph stays legible and each addition is attributable
to a feature.

### Decision gates hit in A0

- **Python 3.11, not newer.** PySpark 4.2 targets 3.11; pinning `requires-python =
  ">=3.11,<3.12"` keeps resolution deterministic and inside the wheel-supported range.
- **pandas pinned `<3.0` — a real fork, not cosmetic.** Resolution first pulled pandas
  3.0, and PySpark 4.2 warns at runtime: *"PySpark does not yet fully support pandas >=
  3.0.0."* Spark↔pandas conversion (Arrow `toPandas`, pandas UDFs) runs through the whole
  A→D pipeline and the dashboard, so we pinned pandas to the supported `<3.0` line rather
  than risk subtle interop breakage later. This is the CLAUDE.md guardrail in action:
  argue the specifics, don't accept a default that a runtime warning flags.
- **Java 17 satisfies Spark 4.2.** Verified by actually starting a `SparkSession` and
  running a job (`spark.range(5).count()` + a `toPandas()` round-trip) — the true A0 gate
  is not "PySpark imports" but "the JVM starts and executes a job on this machine."

### The Kaggle pull (`src/ingest/fetch_data.py`)

A thin, idempotent wrapper over the Kaggle CLI: skip if the three CSVs already exist,
else download the competition zip into `data/raw/` and extract. Two prerequisites that
bite silently:

1. Credentials — a single `KAGGLE_API_TOKEN` (the newer `KGAT_...` token) kept in the
   gitignored repo-root `.env`. The script calls `load_dotenv()` to lift it into the
   environment, then `_require_credentials()` fails fast with a clear message if it's
   missing — better than an opaque auth error from deep inside the CLI. kaggle 2.2.4
   reads `KAGGLE_API_TOKEN` natively (it also accepts `~/.kaggle/access_token` or OAuth
   via `kaggle auth login`; we chose the `.env` var so the whole local config lives in
   one gitignored file alongside future secrets).
2. **You must accept the competition rules on the website once**, or the API returns
   **403** even with valid credentials. This is the classic first-run trap.

The three files it fetches: `sales_train_evaluation.csv` (the wide daily-sales matrix
A1 melts — see A1 for why *evaluation*), `calendar.csv` (dates, events, SNAP flags — A3
exogenous features), and `sell_prices.csv` (weekly prices — A3 price deltas + the C
causal layer).

---

## A1 — Spark session + load 3 CSVs + wide→long melt (~59M rows)

**What A1 is:** the reshape that turns M5's competition format into a modelling grain.
The sales table ships **wide** — one row per series, one *column* per day
(`d_1 … d_1941`). Every time-series operation (lags, rolling means, joins to calendar)
needs the opposite: **long** — one row per (series, day). A1 does that melt and nothing
more; features come in A2–A3, persistence in A4.

### How it fits

```
get_spark()  ->  load_sales_wide (30,490 x 1,941)  ->  melt_sales  ->  59,181,090 long rows
                 load_calendar (1,969)                                  (id…, d, d_int, sales)
                 load_sell_prices (6.84M)      [loaded now, joined in A3]
```

This is the first place Spark earns its keep: 59M rows is where "just use pandas" starts
to strain, and the melt is a single wide→long transform applied across the whole
population — exactly the amortised, build-once-for-everyone work the feature-store framing
promises. We do the melt in Spark **because** the output is population-scale, not because
the later model needs it.

### The three modules (systems view)

- **`spark.py::get_spark()`** — one session builder for A1–A4. `local[*]`, Arrow enabled
  (fast Spark↔pandas at the model/dashboard edge — never to pull all 59M rows), shuffle
  partitions trimmed 200→16 (200 is a cluster default; wasteful on one machine).
  `getOrCreate()` means repeated calls reuse the live session rather than fighting over one.
- **`load.py::load_*`** — thin CSV readers. Sales is read with `inferSchema=False`: a
  1,947-column frame isn't worth an inference pass, so day columns come in as strings and
  are cast **once** after the melt. Calendar/prices infer normally (few columns).
- **`load.py::melt_sales()`** — the reshape.

### Syntax worth noting

- **`DataFrame.melt(ids, values, variableColumnName, valueColumnName)`** — Spark's native
  unpivot (PySpark ≥3.4). `ids` = the 6 identifier columns kept as-is; `values` = the 1,941
  `d_N` columns collapsed into two output columns: `d` (the former column name, e.g.
  `"d_5"`) and `sales` (its value). Output rows = input rows × len(values).
- **`F.regexp_extract("d", r"^d_(\d+)$", 1)`** — pull the integer out of `"d_5"` → `"5"`,
  then `.cast("int")` → `d_int`.

### Decision gate: `d_int`, the ordering key (leakage-adjacent)

`d` is a **string**, and strings sort lexically: `"d_10" < "d_2" < "d_9"`. Ordering a
series by `d` would silently scramble the timeline — and a scrambled timeline means A2's
`lag(7)` grabs the wrong day, i.e. **leakage / corruption that no error surfaces**. So the
melt emits `d_int`, and the rule (same family as the CLAUDE.md DuckDB `ORDER BY date`
gotcha) is: **every time-ordered op orders by `d_int` (or the real date), never by `d`.**
There's a dedicated test for exactly this (`test_d_int_orders_numerically_not_lexically`).

### Decision gate: evaluation, not validation

Two sales files exist. `validation` stops at `d_1913`; `evaluation` runs to `d_1941` — the
28 extra days are the held-out horizon, revealed after the competition closed. Since the
competition is over, those days are just more history, so we melt **evaluation** (59.18M
rows = 30,490 × 1,941). The `id` values therefore carry an `_evaluation` suffix.

### What's tested

The melt is load-bearing, so its contract is pinned in `tests/test_melt.py` on a tiny
synthetic frame (fast, no data dependency): row count = series × days, exact output
columns, `sales`/`d_int` are ints, values land on the right day, and — the one that
matters — `d_int` orders `d_1, d_2, d_10` numerically, not `d_1, d_10, d_2`.

---

## A2 — Time features in Spark: lags + rolling means *(gate: leakage-safe shifts)*

**What A2 is:** the first *predictive signal*. A demand model's strongest features are the
series' own recent history — what it sold a week ago, four weeks ago, and its recent
average. A2 computes those per series (`lag_7`, `lag_28`, `rmean_7`, `rmean_28`) on the
long frame from A1. It is small in code and large in consequence, because this is where
**leakage** enters a forecasting project if you're careless.

### The gate: a feature for day *t* may see only days strictly before *t*

Leakage = letting a feature peek at information that wouldn't exist at prediction time. In
time series it's insidious: the backtest looks brilliant, the live forecast collapses, and
nothing errors. Two mechanical rules enforce safety here, both expressed through Spark
**window functions**:

1. **Lags** — `F.lag("sales", k).over(order)` with `k ≥ 1`. Day *t* reads day *t−k*.
   Naturally backward-looking.
2. **Rolling means** — a window *frame* that **ends at −1 (yesterday), never 0 (today)**.
   `order.rangeBetween(-w, -1)` averages the `w` days *before* *t*, excluding *t*. The
   off-by-one that ends the frame at `0` (`Window.currentRow`) would fold today's sales
   into today's feature — the classic leak.

### Syntax: Spark window functions (two flavours)

```python
order = Window.partitionBy("id").orderBy("d_int")   # per series, in day order
F.lag("sales", 7).over(order)                        # ranking fn: ignores any frame
F.avg("sales").over(order.rangeBetween(-7, -1))      # aggregate fn: needs a frame
```

- `partitionBy("id")` — features never cross series boundaries (series X can't see Y).
- `orderBy("d_int")` — the A1 integer key; ordering by the `d` string would scramble time.
- **`lag` is a ranking function** — it uses the ordering but *ignores* the frame.
  **`avg` is an aggregate function** — it operates over the frame you give it. That's why
  the lag columns and the rolling-mean columns use the same `order` but only the means add
  `rangeBetween`.

### Decision gate: `rangeBetween` (value-based) vs `rowsBetween` (position-based)

`rowsBetween(-7, -1)` counts *rows*; `rangeBetween(-7, -1)` counts *values of d_int* — i.e.
actual **days**. On dense M5 (every series has all 1,941 days, no gaps) they're identical.
We chose `rangeBetween` anyway: it says "the last 7 *days*", so it stays correct even if a
series ever had a missing day, where `rowsBetween` would silently average 7 *rows* spanning
more than 7 days. Stating the intent in day-units is the defensible choice under scrutiny.

### Warm-up nulls (deliberately not "fixed" here)

The first `k` days of a series have no `lag_k` (→ null); early `rmean_w` averages only the
partial prior window (and is null on the very first day, where the frame is empty). This is
correct, not a bug — those rows genuinely lack history. **Whose job to handle it?** The
backtest harness (B1), which decides warm-up/burn-in. A2's single responsibility is: the
features never peek ahead. Pushing null-handling into A2 would blur that boundary.

### What's tested (`tests/test_features.py`)

The headline test plants a **1000-unit spike on the last day** of an otherwise-zero series
and asserts every feature on that day is 0 / reads a prior day — if the frame leaked, the
spike would show up. The rest pin the mechanics: `lag_k` reads exactly *k* days back, warm-up
is null, and `rmean_7` at *t=10* equals `mean(days 3..9)` with day 10 excluded.
