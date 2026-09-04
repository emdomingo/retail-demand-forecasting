# Feature B — Forecasting core — Quiz

Append-only across B1–B4. Self-quiz: recall, predict-the-decision, spot-the-flaw.

---

## B1 — Rolling-origin harness + metrics + baselines + MLflow

**Recall**
1. Describe rolling-origin evaluation in one or two sentences. What is the "origin"?
2. What is the model protocol the harness relies on, and why does one shape work for both
   seasonal-naive and (later) a global LightGBM?
3. What does an RMSSE of exactly 1 mean? Below 1? Above 1?
4. Why WMAPE instead of plain MAPE for this data?

**Predict-the-decision**
5. The harness was built *before* any model. Why is that the right order for this project?
6. Expanding window vs sliding (fixed-length) window — which did we pick and what's the
   justification?
7. RMSSE's scale is measured from each item's *first actual sale*, not from day 1. What problem
   does that avoid?
8. We store MLflow runs in SQLite rather than the classic `./mlruns` file store. Why?
9. The demo backtests ETS on a 200-series sample but seasonal-naive is described as scaling to
   all series. Is that an inconsistency in the harness? Explain.

**Spot-the-flaw**
10. A colleague evaluates by training on all of 2011–2016, then forecasting a random 28-day
    window from 2014 that lies *inside* the training range. Accuracy looks great. What's wrong?
11. Someone reports a single RMSSE averaged by pooling every series-day into one RMSE and
    scaling once. Why can that be misleading versus the per-series-then-average approach?
12. A model returns a forecast for a brand-new item that has no training history. Seasonal-naive
    outputs 0 for it. Bug or intended, and why?
13. RMSSE comes back as `nan` for a series that sold exactly 3 units every single day of the
    training period. Why, and is dropping it from the mean defensible?

---

### Answers — B1

1. Stand at a past cutoff (the **origin**), train only on data up to it, forecast the next
   `horizon` days, and score against the actuals that followed; slide the origin forward and
   repeat. The origin is the last day the model is allowed to "know."
2. A model has a `name` and `forecast(train, test_keys) -> yhat` aligned to `test_keys`. It's a
   structural contract (duck-typed / `Protocol`), so per-series models loop internally and a
   global model fits once — the harness just calls `forecast` and never needs to know which.
3. RMSSE = 1 → the forecast is as accurate as a one-step naive forecast on the training history.
   **< 1 beats** that naive; **> 1 loses** to it. It's a scaled, cross-series-comparable yardstick.
4. Retail demand has many zero days; per-point MAPE divides by the actual and blows up (∞) on
   zeros. WMAPE sums errors and sums actuals first (`Σ|e|/Σ|y|`), so zeros don't detonate it.
5. It's the anti-leakage scaffold and the yardstick. Building it first means every model (B2–B4)
   is scored on the same honest rolling-origin protocol against the same baseline from day one —
   you never retrofit evaluation to flatter a model you've already built.
6. **Expanding** — a real planner accumulates history rather than discarding it, so training on
   all data up to the origin mirrors practice and uses the most information. (Sliding is for when
   you suspect regime change makes old data harmful; not our case.)
7. Pre-launch **structural zeros** (item not yet stocked). Including them would inflate the
   training series with a long flat run, shrinking the naive scale and making RMSSE look
   artificially good/bad. Measuring from first sale scales by the item's *real* demand variability.
8. **MLflow 3 blocks the legacy file store for tracking** and recommends a SQL backend. SQLite is
   the zero-infra, current, forward-compatible choice; it's gitignored and viewable via `mlflow ui`.
9. No — it's a **runtime** choice, not a harness limit. ETS fits a model per series and is slow, so
   the head-to-head demo uses a fixed sample; the harness and seasonal-naive run over all ~3,049
   series with no code change. The point of B1 (harness + metric + tracked baseline) is unaffected.
10. The test window is **inside** the training range → the model trained on the future relative to
    that window. That's leakage; the "great" accuracy is fitting data it also learned from. Rolling
    origins forbid this by construction (test is always strictly after the origin).
11. Pooling lets **high-volume series dominate** the single number and hides poor accuracy on the
    many low-volume (but often high-margin/spiky) items. Per-series RMSSE then averaging gives each
    series equal voice — and segment-level error is exactly what matters for planning (a guardrail).
12. **Intended.** With no history there's no season to repeat, so 0 is the honest "no signal"
    output rather than a fabricated guess. (A real system would fall back to a cohort/category
    prior; noted, not built.)
13. A perfectly flat series has **zero** one-step naive error, so the RMSSE denominator is 0 and
    the ratio is undefined → `nan`. Dropping it from the mean is defensible: the metric genuinely
    can't score that series, and keeping a nan would poison the aggregate. (Such a series is also
    trivially forecastable and not where accuracy questions live.)
