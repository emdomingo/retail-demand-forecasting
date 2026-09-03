# Feature A — Ingestion & feature engineering — Quiz

Append-only across A0–A4. Self-quiz: recall, predict-the-decision, spot-the-flaw.
Answers are at the end of each subfeature's block.

---

## A0 — Repo scaffold + env + local Spark + Kaggle pull

**Recall**
1. uv manages the Python interpreter and packages. What part of the Spark stack does it
   *not* manage, and how is that part installed instead?
2. What is the difference between `uv add` and `uv add --dev`, and why does `kaggle` go in
   one group and `pytest` in the other?
3. Which two files are the committed "env recipe," and which two directories are the
   gitignored "payload"? What bridges them?

**Predict-the-decision**
4. Resolution offered pandas 3.0.5. We pinned `pandas>=2.2,<3.0` instead. What signal
   drove that, and why does it matter for *this* pipeline specifically (name the code
   paths at risk)?
5. `requires-python` is `>=3.11,<3.12` rather than just `>=3.11`. What does the upper
   bound buy us, and what could go wrong without it?
6. The A0 verification ran `spark.range(5).count()` and a `toPandas()` round-trip rather
   than just `import pyspark`. Why is that the right gate for A0?

**Spot-the-flaw**
7. A teammate says: "My `KAGGLE_API_TOKEN` in `.env` is valid, but `fetch_data.py`
   returns 403. The token must be wrong." What's the more likely cause?
8. Someone proposes committing the extracted CSVs "so the repo is self-contained and
   reviewers don't need Kaggle." Give two reasons this violates the project's design.
9. `fetch_data.py` checks `already_present()` before downloading and exits early. What
   property does this give the script, and what flag overrides it?

---

### Answers — A0

1. The **JVM (Java)**. PySpark is a Python wrapper over a Spark JVM; uv only handles
   Python. Java 17 is installed separately via Homebrew (`openjdk@17`). Reproducibility
   is therefore two-layered: `uv sync` + a compatible system Java.
2. `uv add` records a **runtime** dependency (in `[project.dependencies]`); `uv add --dev`
   records a **dev-group** dependency (tooling/notebooks, not needed to run the pipeline).
   `kaggle` is runtime (the pipeline re-fetches data); `pytest`/`ruff`/`jupyterlab` are
   dev-only.
3. Recipe: **`pyproject.toml` + `uv.lock`** (committed). Payload: **`data/raw/` +
   `data/processed/`** (gitignored). Bridge: **`src/ingest/fetch_data.py`**, which
   re-derives the payload from Kaggle. (The `.env` holding `KAGGLE_API_TOKEN` is also
   gitignored — a secret, not part of the committed recipe.)
4. Signal: PySpark 4.2's runtime warning *"does not yet fully support pandas >= 3.0.0."*
   It matters because Spark↔pandas interop runs everywhere — Arrow `toPandas()`, pandas
   UDFs in feature engineering, and the Streamlit dashboard's pandas frames. A silent
   interop bug there would be expensive and hard to trace, so we stay on the supported line.
5. The upper bound keeps dependency resolution **deterministic and inside the
   wheel-supported range** for PySpark 4.2 (which targets 3.11). Without it, a fresh
   `uv sync` on a machine with 3.12/3.13 available could resolve to an interpreter that
   lacks compatible wheels or hits untested Spark/pandas combinations.
6. `import pyspark` only proves the Python package is installed. A0's real risk is the
   **JVM starting and executing a job** on this machine (Java compatibility, native libs).
   Running an actual distributed job plus a pandas round-trip exercises the path the whole
   pipeline depends on.
7. The **competition rules haven't been accepted** on the Kaggle website. The M5 API
   returns 403 until you accept the rules once, even with perfectly valid credentials.
8. (a) The data is **large and public** — committing it bloats the repo for no benefit
   when it's one command to re-fetch. (b) The repo's contract is **reproducible from the
   recipe, not from committed files**; committing data hides whether the fetch path
   actually works and drifts from the "recipe, not payload" design. (Also: `.gitignore`
   deliberately excludes `data/`.)
9. **Idempotence** — running it repeatedly is safe and a no-op once the CSVs exist, so it
   won't re-download hundreds of MB on every invocation. `--force` overrides it to
   re-fetch.

---

## A1 — Spark session + load 3 CSVs + wide→long melt (~59M rows)

**Recall**
1. The sales CSV is "wide." What does that mean concretely, and what shape does the melt
   produce? Give the row-count arithmetic.
2. What does `DataFrame.melt(ids, values, variableColumnName, valueColumnName)` do to each
   input row, and what are the two output columns named here?
3. Why is `get_spark()` a single shared builder rather than each module making its own
   session?

**Predict-the-decision**
4. Sales is loaded with `inferSchema=False` while calendar/prices use `inferSchema=True`.
   Why the split?
5. We melt `evaluation` (d_1..d_1941), not `validation` (d_1..d_1913). What's the reasoning,
   and what visible consequence does it have on the `id` column?
6. `shuffle.partitions` is set to 16, not the Spark default of 200. What's the default for,
   and why is it wrong here?

**Spot-the-flaw**
7. A colleague orders a series by the `d` column before computing a 7-day lag. The code runs,
   no error. What's wrong, what's the downstream symptom, and what's the fix already built in?
8. Someone "optimises" the demo by calling `long_df.toPandas()` to inspect the 59M-row
   frame. Why is that a mistake, and what is Arrow actually for here?
9. The melt test uses a 2-series × 3-day synthetic frame with columns `d_2, d_10, d_1`
   (unsorted). Why include `d_10`, and which test would fail if `d_int` were dropped and
   ordering fell back to the string `d`?

---

### Answers — A1

1. Wide = one row per **series** (item × store), with one *column per day* (`d_1 … d_1941`,
   6 id columns beside them). Melt → **long**: one row per (series, day). Rows =
   30,490 series × 1,941 days = **59,181,090**.
2. It **unpivots**: each input row becomes `len(values)` output rows — one per day column.
   The former column name goes into **`d`** (e.g. `"d_5"`) and its value into **`sales`**.
3. So session config (Arrow, shuffle partitions, master, log level) lives in **one place**
   and A1–A4 stay consistent; `getOrCreate()` also means they share one live JVM rather
   than each spinning up / colliding on a session.
4. The wide sales frame has 1,947 columns — an inference pass over that is wasteful, and the
   day columns are all simple counts, so we read them as strings and **cast once after the
   melt**. Calendar/prices are narrow, so inference is cheap and convenient there.
5. The competition is over, so `evaluation`'s 28 extra days (the once-hidden horizon) are
   just more history to train on — more data, no leakage. Consequence: `id` values carry the
   **`_evaluation`** suffix (e.g. `FOODS_3_090_CA_3_evaluation`).
6. 200 is a sensible **cluster** shuffle width; on a single local machine it creates lots of
   tiny partitions and overhead. 16 matches a laptop's parallelism without over-partitioning.
7. `d` is a **string**, so it sorts lexically: `d_10` lands between `d_1` and `d_2`. The
   timeline is silently scrambled, so `lag(7)` reads the wrong day → **leakage-adjacent
   corruption with no error**. Fix: order by **`d_int`** (the integer index the melt emits) —
   the same discipline as DuckDB's `ORDER BY date` rule.
8. `toPandas()` pulls the **entire 59M-row frame into driver memory** → OOM / defeats the
   whole reason for Spark. Arrow is for **fast, small** Spark↔pandas hand-offs at the
   model/dashboard boundary (a single series, an aggregate), not for materialising the
   population frame.
9. `d_10` is the lexical trap: string-sorting puts it before `d_2`. It's there so
   **`test_d_int_orders_numerically_not_lexically`** is meaningful — that test (expecting
   `d_1, d_2, d_10`) is the one that fails if ordering falls back to the `d` string.

---

## A2 — Time features: lags + rolling means *(gate: leakage-safe shifts)*

**Recall**
1. Name the four features A2 adds and, in one line, why a series' own recent history is the
   strongest signal for demand.
2. What is the difference between a Spark *ranking* window function (`lag`) and an
   *aggregate* one (`avg`) with respect to the window **frame**?
3. Both `partitionBy("id")` and `orderBy("d_int")` appear in the window spec. What does each
   one guarantee?

**Predict-the-decision**
4. The rolling-mean frame is `rangeBetween(-w, -1)`. What would `rangeBetween(-w, 0)` do, and
   why is that the exact definition of leakage here?
5. We used `rangeBetween` (value-based) rather than `rowsBetween` (position-based). On M5 they
   give identical results — so why prefer `rangeBetween`?
6. A2 leaves warm-up nulls in place instead of filling or dropping them. Why, and which
   subfeature owns that decision?

**Spot-the-flaw**
7. A model shows near-perfect backtest accuracy that vanishes in production. The rolling-mean
   window was written `rangeBetween(-7, 0)`. Explain the mechanism precisely.
8. A colleague computes lags with `Window.partitionBy("id").orderBy("d")` (the string column)
   instead of `d_int`. Data's dense so counts look fine — what breaks, and when?
9. Someone drops `partitionBy("id")` "to simplify," leaving just `orderBy("d_int")`. What
   goes wrong at the boundary between two series?

---

### Answers — A2

1. `lag_7`, `lag_28`, `rmean_7`, `rmean_28`. Demand is autocorrelated and strongly weekly —
   what a store-item sold 7/28 days ago and its recent average are highly predictive of
   tomorrow, more so than most exogenous signals.
2. A **ranking** function (`lag`) uses the window's *ordering* but ignores any frame — it
   just steps *k* rows back in order. An **aggregate** function (`avg`) computes over the
   **frame** (the `rangeBetween`/`rowsBetween` you supply); without a frame it would default
   to the whole partition-up-to-current-row, which is why the means specify one explicitly.
3. `partitionBy("id")` keeps features **within a single series** — series X never reads Y.
   `orderBy("d_int")` puts each series in true **day order** so "7 back" means 7 days back.
4. `rangeBetween(-w, 0)` extends the frame to **include today (current row)**, so today's
   own `sales` enters `rmean_w` — the feature would partly *be* the target it's used to
   predict. Ending at `-1` excludes today; that off-by-one is the whole gate.
5. `rangeBetween` is defined in **day units** (values of `d_int`), so it means "the last 7
   *days*" and stays correct if a series ever had a gap. `rowsBetween` counts *rows*, which
   would silently span more than 7 days across a gap. Same result on dense M5, but
   `rangeBetween` **states the intent** and is robust — the defensible choice under scrutiny.
6. Warm-up rows genuinely have no history (no *k*-days-ago value; partial early windows), so
   nulls are **correct**, not a bug. How to handle burn-in (drop/keep) is a modelling choice
   owned by the **backtest harness (B1)**; doing it in A2 would blur the single
   responsibility "features never peek ahead."
7. `-7, 0` includes the **current row**, so `rmean_7` at day *t* averages days *t−7…t* —
   it contains `sales[t]`. The model learns to lean on a feature that encodes the answer;
   in the backtest the answer is present, so accuracy looks superb. Live, `sales[t]` doesn't
   exist yet at prediction time, the feature is computed differently (or is missing), and
   accuracy collapses. Textbook **target leakage** via an off-by-one frame bound.
8. Ordering by the **string** `d` sorts `d_10` before `d_2`, so within each series the day
   sequence is scrambled. `lag(7)` then steps 7 *positions* through a wrong order and reads
   the wrong day — **silent leakage-adjacent corruption**. It breaks as soon as a series has
   ≥10 days (i.e. immediately), even though row counts are unaffected.
9. Without `partitionBy("id")`, the window runs over **all series as one ordered stream**, so
   the first rows of series B read the *tail of series A* as their "prior 7 days" (and B's
   early lags pull A's values). Cross-series contamination at every series boundary.

---

## A3 — Exogenous features: calendar/event/SNAP flags + price deltas

**Recall**
1. Which two raw tables does A3 join in, and what is the join key for each?
2. `snap_CA`, `snap_TX`, `snap_WI` become a single `snap` column. How, and why is one flag
   better than three here?
3. What does `has_price` encode, and what real-world fact about `sell_prices` makes it meaningful?

**Predict-the-decision**
4. A2 obsessed over not using day *t*'s value; A3 happily uses day *t*'s calendar/SNAP/price.
   Why is that not a contradiction?
5. Calendar is joined with `F.broadcast(...)` but prices is not. What determines that choice,
   and what does broadcasting actually avoid?
6. Why must the calendar join happen *before* the price join (what does the price join need
   that only calendar provides)?

**Spot-the-flaw**
7. A colleague fills null `sell_price` with `0` "to avoid nulls in the model." What signal does
   that destroy, and what could the model wrongly infer?
8. A join is written as an *inner* join on prices instead of *left*. Data looks fine in spot
   checks. What silently changes, and which test would catch it?
9. Someone computes `price_change_pct` with `lag("sell_price", 1)` (one day) instead of 7.
   Given how M5 prices behave, what does that feature look like most days, and why is 7 chosen?

---

### Answers — A3

1. `calendar` on **`d`** (day index), and `sell_prices` on **`(store_id, item_id, wm_yr_wk)`**
   (store-item-week). `wm_yr_wk` is supplied by the calendar join.
2. A `F.when(state_id=="CA", snap_CA).when(...TX...).when(...WI...)` chain selects the flag for
   the series' own state; the three raw columns are then dropped. A series lives in exactly one
   state, so the other two SNAP columns are noise — one correct flag models the effect and
   feeds the C3 SNAP analysis.
3. `has_price = sell_price is not null`. `sell_prices` only carries a row once an item is
   **actually stocked** in that store, so a null price means "not sold here yet" — `has_price`
   separates **structural** zeros (pre-launch) from **demand** zeros (stocked, sold none).
4. A2's features are derived from the **target** (past sales), which isn't known at prediction
   time — so they must be shifted. A3's are **exogenous covariates known in advance** (calendar
   fixed, SNAP scheduled, prices posted ahead), so using day *t*'s value is legitimate. The
   shift rule protects target-derived features, not known-ahead ones.
5. **Size.** Calendar (~1,969 rows) is tiny enough to broadcast to every executor, turning the
   join into a map-side lookup with **no shuffle**. Prices (~6.8M) is too big, so it's a normal
   shuffle join. Broadcasting the small side avoids shuffling the *large* side across the network.
6. The price join key includes **`wm_yr_wk`**, which the long sales frame doesn't have — it only
   arrives when calendar is joined on `d`. No calendar join, no week key, no price join.
7. It destroys the **structural-vs-demand-zero** distinction: a real posted price of 0 (there
   are none) becomes indistinguishable from "not stocked." The model could read pre-launch
   periods as genuine zero-demand-at-price-0 and learn nonsense about price sensitivity; it also
   corrupts `price_change_pct`. Keep the null (and use `has_price`).
8. An **inner** join drops every series-day with no matching price row — i.e. all pre-stock
   (structural-zero) days vanish, silently shrinking the frame and biasing it toward stocked
   periods. `test_joins_preserve_row_grain` (row count must stay one-per-series-day) catches it.
9. M5 prices are **weekly-constant**, so a 1-day lag is 0 on ~6 of 7 days and only nonzero at the
   week boundary — a noisy, mostly-dead feature. `lag(7)` compares this week's price to last
   week's (same weekday, previous `wm_yr_wk`), giving a stable week-over-week delta on every row.
