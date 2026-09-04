# Feature B — Forecasting core — Explainer

Append-only across B1–B4. The heart of the project: a defensible point forecast with honest
intervals, chosen by backtesting rather than assertion.

---

## B1 — Rolling-origin backtest harness + metrics + baselines + MLflow

**What B1 is:** the *measuring instrument*, built **before any model**. It answers "is model
X actually better than the honest baseline, and by how much?" in a way that can't be fooled by
leakage and can be reproduced later. Nothing in B2–B4 is trusted until it's scored here.

### Why the harness comes first (the anti-leakage argument)

The single easiest way to lie to yourself in forecasting is to let the model see the future.
A2 guarded against it *within* features; B1 guards against it *across* the train/test split.
The discipline is **rolling-origin evaluation**: stand at a past cutoff (the *origin*), train
only on data up to it, forecast the next `horizon` days, and score against what actually
happened. Move the origin forward and repeat. Train is always **strictly ≤ origin**, test
**strictly after** — so a score is what a planner standing at that origin would really have got.

```
origin₁            origin₂            origin₃  (each leaves `horizon` days of actuals after it)
[==== train ====]→ hzn
[======= train =======]→ hzn
[========== train ==========]→ hzn        ← expanding window; test never overlaps train
```

> 🔁 **Interactive walkthrough** — step through the origins, watch the training window expand and
> the test window march forward, and see the no-leakage invariant at each cutoff:
> [rolling-origin backtest visualiser](https://claude.ai/code/artifact/236a5c4e-6291-4102-b32f-5b7b475d2868)
> (private Artifact; the sliders recompute origins with the exact `rolling_origins` logic).

We use an **expanding window** (train grows with each origin) — a planner accumulates history,
they don't throw it away. Origins are placed from the end backward: the latest leaves exactly
`horizon` days after it, earlier ones step back by `step`. Origins with no training history are
dropped (`rolling_origins`).

### The model protocol (one shape for every family)

The harness is model-agnostic. A "model" is anything with a `name` and:

```python
def forecast(self, train: pd.DataFrame, test_keys: pd.DataFrame) -> pd.Series: ...
```

`train` is the history (id, date, sales, …); `test_keys` is the (id, date) rows to predict;
the return is `yhat` aligned to `test_keys.index`. That one structural contract (no base
class — Python duck-typing / `Protocol`) serves per-series classical models now
(seasonal-naive, ETS) and the global feature-based LightGBM in B2 unchanged. The harness never
knows which kind it's driving — it just splits, calls `forecast`, and scores.

### Metrics: RMSSE and WMAPE (see `metrics.py`)

- **RMSSE** — the M5 metric, a *scaled* error. RMSE of the forecast ÷ RMSE of a one-step
  naive forecast on the **training** series. The scaling makes errors comparable across series
  of wildly different volume, and gives a natural yardstick: **< 1 beats naive-on-history, > 1
  loses to it**. The scale is measured from the item's **first actual sale** so pre-launch
  (structural) zeros don't deflate it; a flat series has an undefined (0) scale → RMSSE is nan
  and excluded from the mean. (M5's leaderboard uses **WRMSSE**, the same thing weighted by
  each series' dollar sales — named as an extension, not built.)
- **WMAPE** — `sum|actual − forecast| / sum|actual|`. A volume-weighted percentage error that,
  unlike per-point MAPE, doesn't divide by zero on the many zero-demand days in retail.

Aggregation: RMSSE is computed **per series** (each scaled by its own history) then averaged;
WMAPE is **pooled** across all test rows in the origin. Both are reported per origin and as a
mean across origins.

### MLflow: every run tracked (see `run_backtest`)

Each `run_backtest(df, model, cfg)` opens one MLflow run and logs the **params** (model,
horizon, n_origins, step, season, n_series, scope) and **metrics** (per-origin `rmsse`/`wmape`
as steps, plus `rmsse_mean`/`wmape_mean`). This is what makes "compare families and defend the
choice" reproducible instead of a hand-typed table — the B2/B3 comparison is just more rows in
the same experiment.

Storage decision: **MLflow 3 blocks the legacy file store for tracking** and steers to a SQL
backend, so we use a repo-local **SQLite** store (`sqlite:///mlruns.db`, gitignored) — zero
infra, current, and defensible. View it with `uv run mlflow ui --backend-store-uri sqlite:///mlruns.db`.

### Baselines: what everything must beat (see `baselines.py`)

- **SeasonalNaive(7)** — forecast = the last observed week, repeated ("same weekday last
  week"). Deliberately strong: weekly seasonality is most of retail demand, so beating it is a
  real bar, not a strawman. This is *the* reference (CLAUDE.md: score everything against it).
- **ETS(7)** — Holt-Winters exponential smoothing, fit per series. A classical statistical
  baseline; on intermittent daily demand it's often mediocre, which is the point. Degenerate
  series (too short, all-zero) fall back to the last value; negative forecasts clip to 0.

### First real numbers (200-series CA_3 sample, 4 origins, horizon 28)

| model | mean RMSSE | mean WMAPE |
|---|---|---|
| seasonal_naive_7 | 0.961 | 0.822 |
| ets_7 | **0.735** | **0.706** |

Both beat the RMSSE = 1 line (they're better than one-step-naive-on-history), and ETS clearly
leads seasonal-naive on this sample. So LightGBM (B2) has a concrete bar: **beat ~0.735 RMSSE**,
or explain honestly why not. (ETS is fit per series and slow, so the demo runs a fixed 200-series
sample; seasonal-naive and the harness scale to all ~3,049 CA_3 series unchanged — a runtime
choice, documented, not a limitation of the harness.)

### What's tested

- **No leakage** (`test_split_has_no_leakage`) — for every origin, `train.date.max() ≤ origin
  < test.date.min()`, and each series' test spans exactly `horizon` days.
- **Expanding window** — train grows with later origins.
- **Origins** — correct count/chronology, and dropped when history is too short.
- **Whole path scores 0** — a perfectly periodic series fed to seasonal-naive yields RMSSE = 0,
  WMAPE = 0 (exercises forecast + scoring together).
- **Metrics** — RMSSE/WMAPE against hand-computed values, incl. the first-sale scale trim and
  the undefined-scale → nan case.
- **Baselines** — seasonal cycling, last-season-only, unseen-series → 0, ETS fallbacks.

---

## B2 (v1) — Global LightGBM point forecast

**What B2 is:** the model the whole project is built to defend. One gradient-boosted tree model
fit across *all* series in the slice at once — a **global** model — learning a single function
`features → sales`. This is the approach that won M5, and it contrasts sharply with the
baselines: seasonal-naive and ETS fit **one model per series**; LightGBM pools every series into
one, so a sparse item borrows strength from the thousands of others that share its calendar and
price dynamics. It plugs into the B1 harness through the exact same `forecast(train, test_keys)`
shape the baselines use — the harness never learns it's driving something different.

### The three decisions that make or break it

**1. Recursive multi-step (the horizon problem).** The horizon is 28 days, but `lag_7` for
day 10 is the sale on day 3 — *inside* the forecast window, unknown to a planner standing at the
origin. Reading the feature store's precomputed `lag_7` for a test row would hand the model a
future actual: leakage, and a beautiful dishonest score. So B2 forecasts **day by day**: predict
day 1, append that prediction to the series' history *as if it were the actual*, recompute lags
and rolling means, predict day 2, and so on. Errors compound across the horizon — which is
honest, and is precisely the uncertainty B4's intervals exist to quantify. Implementation is
**lockstep by horizon day**: for each future date, build one feature row per series, predict the
whole batch, append predictions to each series' history, step to the next date (28 batched
`model.predict` calls per origin, not one per series-day).

**2. AR features are rebuilt in the model, not read from the store.** Training and the recursive
test path call the *same* `_ar_row` / `_ar_train` builder. That guarantees they can't silently
disagree, and it makes the test-time features **provably** a function of (past actuals + the
model's own past predictions) only — the exact honesty property an interviewer will probe. The
store's precomputed lag/rmean columns are used by nobody here.

**3. The harness had to be widened (a real B1 change).** B1 passed the model only `(id, date)`
for test rows — safe, but *too* strict: price, SNAP, and events are **legitimately known in
advance** in M5 (Walmart publishes the future calendar and price files), and price is the single
strongest exogenous demand signal in retail. Withholding it would handicap B2 unfairly. So
`BacktestConfig` gained a `known_future` list; `evaluate_origin` passes those columns
(calendar/price) alongside `(id, date)` — but **never `sales` and never the precomputed AR
lags**, which encode the test-window actuals. Default is empty, so the baselines and every
existing test are unchanged. This is the kind of change that reads as maturity: the harness
enforces "known at forecast time," and we corrected *what* is legitimately known.

### The feature set (v1) and why each is safe

| group | features | why leak-free on test rows |
|---|---|---|
| autoregressive | `lag_7`, `lag_28`, `rmean_7`, `rmean_28` | rebuilt recursively from own history |
| calendar | `wday`, `month`, `year` | derived from the date itself, both paths identically |
| price / promo | `sell_price`, `price_change_pct`, `snap`, `is_event`, `event_type_1` | known-future, passed via `known_future` |
| series identity | `dept_id`, `cat_id` (categorical) | static per series, looked up from train |

**Series identity is carried by the lags/means + dept/cat, not a 3,049-way `item_id`
categorical.** The lag and rolling-mean features already encode each series' level; a per-item
categorical would explode tree size and invite overfit. dept/cat give the coarse structure that
*pools* well. Categoricals use pandas `category` dtype with categories **frozen at fit** and
reused at predict, so train and test share one encoding.

**Objective = Tweedie** (`variance_power=1.1`). Daily item demand is intermittent and
non-negative with many zeros; Tweedie is the standard loss for that regime (a compound
Poisson-Gamma), and it beat plain regression in M5 write-ups. Predictions are clipped to ≥ 0.

### v1 result (same 200-series CA_3 sample, 4 origins, horizon 28)

| model | mean RMSSE | mean WMAPE |
|---|---|---|
| seasonal_naive_7 | 0.961 | 0.822 |
| ets_7 | 0.735 | 0.706 |
| **lightgbm_global_v1** | **0.732** | **0.680** |

The honest read: v1 **edges** ETS on RMSSE (0.732 vs 0.735 — within noise) and wins more
clearly on WMAPE (0.680 vs 0.706). The real headline isn't the RMSSE hair: it's that **one
global model matches 200 individually-fit ETS models and scales to all ~30k series unchanged**,
and it's a *single* model we can hang conformal intervals (B4) on. v2 enriches the feature set
to widen the point-estimate margin — tracked as a separate MLflow run so the gain is measured.

### What's tested (B2)

- **Contract** — `forecast` returns a Series aligned to `test_keys.index`, full length,
  no NaN, non-negative.
- **Recursion covers the whole horizon** — horizon 28 > `lag_7`: every day is filled by the
  feedback loop, none left null.
- **No peeking** (`test_ignores_a_sales_column_on_test_keys`) — attaching a garbage `sales`
  column to the test keys doesn't move a single prediction. The honesty guarantee, as a test.
- **Sanity** — a flat series predicts near its constant level.
- **Harness widening** (`test_backtest.py`) — `known_future` columns reach the model, `sales`
  is withheld, and the default stays `(id, date)` only.

---

## B2 (v2) — Enriched features, disciplined by an ablation

**What v2 is:** the *same* global recursive model with a richer feature set — and a cautionary
tale about how "richer" isn't automatically "better" under recursive forecasting. This is the
most interview-valuable part of B2, because the first attempt **regressed** and the recovery
came from a diagnosis, not a guess.

### The naive v2 that made things worse

The obvious enrichment — add a short lag (`lag_1`), a fortnight lag (`lag_14`), a long window
(`rmean_56`), and let **early stopping** pick the round count — scored **RMSSE 0.735 / WMAPE
0.691**: *worse* than v1 (0.732 / 0.680) and a dead tie with the ETS baseline it was supposed to
beat. More features, less accuracy. That's the moment to stop and diagnose, not to ship.

### The ablation (five configs, same sample)

| config | RMSSE | WMAPE |
|---|---|---|
| v1 — lags (7,28), rmean (7,28), fixed rounds | 0.7322 | 0.6802 |
| **no lag_1 — lags (7,14,28), rmean (7,28,56), fixed** | **0.7267** | **0.6738** |
| with lag_1 — lags (1,7,14,28), rmean (7,28,56), fixed | 0.7345 | 0.6846 |
| no lag_1 + early stopping | 0.7296 | 0.6786 |
| v1 features + rmean_56 only | 0.7284 | 0.6789 |

Two clean findings:

**1. `lag_1` is toxic under recursion.** It's the worst config in the table. The reason is
structural: in recursive multi-step, `lag_1` for horizon day *h* is *yesterday's value* — but
for 27 of the 28 forecast days, "yesterday" is the model's **own prior prediction**, not an
actual. A feature the model leans on heavily (yesterday's demand is the strongest single
predictor) becomes a channel that **compounds its own error** forward. Longer lags (7, 14, 28)
are far more robust because for most of the horizon they still point at *real* history, not
predictions.

**2. One-shot early stopping tunes the wrong regime.** Early stopping picks the round count by
watching error on a held-out tail whose features are built the *training* way — real-history
lags, one-step. But the test path is **recursive**. So it optimises round count for a regime the
real forecast never runs in, and it slightly *hurt* (0.7296 vs 0.7267 fixed). We keep the
early-stopping capability in the class (documented, tested) but v2 doesn't use it — and now we
can *say why*, which is stronger than either blindly using or blindly omitting it.

### The v2 that ships

Informed by the ablation: **lags (7, 14, 28), rmean (7, 28, 56), fixed rounds, no `lag_1`, no
early stopping.**

| model | mean RMSSE | mean WMAPE |
|---|---|---|
| ets_7 | 0.735 | 0.706 |
| lightgbm_global_v1 | 0.732 | 0.680 |
| **lightgbm_global_v2** | **0.727** | **0.674** |

A real, defensible gain over v1 that clears the ETS bar comfortably on *both* metrics — and,
unlike v1's RMSSE hair, not within noise. The MLflow experiment now holds `lightgbm_global_v1`
and `lightgbm_global_v2` as separate runs (version flows into the run name), so the improvement
is a recorded comparison, not a claim.

### The transferable lesson

Feature richness interacts with the *forecasting scheme*. Under recursion, prefer lags that
outrun the horizon's error compounding; be suspicious of the shortest lags; and validate on the
regime you'll actually deploy. "I added features and it got worse, so I ran an ablation and found
`lag_1` was feeding the model its own errors" is exactly the kind of story that reads as real
modelling maturity.

---

## B3 — SARIMA: the classical per-series comparison

**What B3 is:** the *classical* comparison model — seasonal ARIMA, fit one model per series, the
same philosophy as ETS but with an explicit AR/MA/seasonal structure. It exists to complete the
model-family comparison the harness was built for: with B3 the same rolling-origin backtest, on
the same 200-series CA_3 sample, now scores **three distinct philosophies** side by side —
per-series smoothing (ETS), per-series ARIMA (SARIMA), and one global pooled tree model
(LightGBM). SPEC frames B3 as the classical entry and the *slack-absorber*; the job is an honest
head-to-head, not to win. It plugs into the harness through the same `forecast(train, test_keys)`
shape as everything else (see `sarima.py`).

### The three decisions that must survive scrutiny

**1. Per-series — the deliberate contrast to B2's global model.** SARIMA fits one state-space
model on each series' own history. That's the "every series is its own universe" stance; B2 is
"pool signal across all series." Running both on the same harness is the entire point — the
comparison is only meaningful because the *only* thing that changes is the model. Per-series
fitting is slow (see runtime below), so B3 keeps B2's fixed 200-series sample for a like-for-like
number.

**2. A fixed, motivated order — not a per-series auto-search.** `pmdarima`'s stepwise
`auto_arima` is the reflex choice, but it's the wrong one here: (a) it carries numpy-2
compatibility friction and a heavy dependency, and (b) more importantly, searching an order *per
series* across ~30k intermittent, often-short series is neither reproducible nor defensible — it
overfits the **order itself** to noise. A single argued order is the honest scoped choice,
consistent with the project's "scope the model, state it explicitly" rule. The order is
**SARIMA(1,1,1)(1,0,0)₇**:

| term | choice | reasoning |
|---|---|---|
| non-seasonal `(p,d,q)` | `(1,1,1)` | AR(1)+MA(1) on a first difference captures short-run level dynamics |
| seasonal `(P,D,Q)` | `(1,0,0)` | one seasonal AR term captures the **weekly cycle** — the structure ETS's additive season also targets, so the comparison is fair |
| seasonal period `m` | `7` | daily data, weekly seasonality |
| seasonal differencing `D` | **`0`** | differencing over m=7 on zero-heavy short series loses a week and routinely **destabilises** the fit for little gain |

**3. Robust by construction, because classical fits fail on intermittent demand.** Too-short
(`< 2` seasons) or all-zero series skip the fit and fall back to the last value — the same
contract ETS uses, so the two classical models degrade identically on the hard series. The
optimiser runs with `enforce_stationarity=False`/`enforce_invertibility=False` (converges far
more often on messy retail series; we clip to 0 anyway, so we don't need a provably
stationary parameterisation) and a bounded `maxiter=50` (can't hang). Any convergence or
non-finite-forecast failure also falls back. Forecasts clip to ≥ 0.

### Result (same 200-series CA_3 sample, 4 origins, horizon 28)

| model | mean RMSSE | mean WMAPE |
|---|---|---|
| seasonal_naive_7 | 0.961 | 0.822 |
| ets_7 | 0.735 | 0.706 |
| **sarima_111_100_7** | **0.734** | **0.696** |
| lightgbm_global_v2 | **0.727** | **0.674** |

**The honest read.** SARIMA and ETS are a **statistical tie** on RMSSE (0.734 vs 0.735); SARIMA
is marginally better on WMAPE. That is not a disappointing result — it's the *expected* and
interesting one: **two different classical per-series methods plateau at essentially the same
place (~0.735)** on intermittent daily demand, and the **global pooled model (LightGBM v2, 0.727)
sits clearly below both.** The story the comparison tells is exactly the M5 lesson — pooling
signal across series beats fitting each series in isolation, and *which* per-series method you
pick barely matters once you're in that regime. A convergent plateau across two independent
classical models is stronger evidence for that claim than beating a single weak baseline would be.

**Runtime is part of the story too.** The SARIMA backtest took **~2m20s** (800 state-space fits:
200 series × 4 origins, each on a growing window). ETS is comparable; the global LightGBM fits
*once per origin* (4 fits total) and is far faster while scoring better — the pooled approach wins
on accuracy **and** on the compute that matters when you scale from 200 series to 30k.

### What's tested (B3)

We test the **robustness contract**, not fitted values — a per-series SARIMA fit is a stochastic
optimisation, so asserting exact numbers would be brittle.

- **Fallbacks** — short (`< 2` seasons) and all-zero series return the last value / 0, repeated.
- **Unseen series** — an id absent from train predicts 0 (harness contract).
- **Output contract** — on a real seasonal series the fit runs and the output is aligned to
  `test_keys.index`, non-negative, and finite.
- **Name encodes the order** — `sarima_111_100_7`, so the MLflow run is self-describing and
  distinct from ETS.
