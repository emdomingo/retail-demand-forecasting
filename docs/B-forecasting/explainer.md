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
