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

---

## B2 (v1) — Global LightGBM point forecast

**Recall**
1. What does "global model" mean here, and how does it differ structurally from how
   seasonal-naive and ETS are fit?
2. What is recursive multi-step forecasting, and why is it forced on us by a 28-day horizon with
   a `lag_7` feature?
3. Which feature groups does v1 use, and which ones does the *harness* have to supply for the
   test rows (vs. which the model derives itself)?
4. Why is the training objective Tweedie rather than plain squared-error regression?

**Predict-the-decision**
5. B2 required widening the B1 harness (`known_future`). What exactly was widened, what is still
   withheld from the model on test rows, and why is the default backward-compatible?
6. The AR features already exist, precomputed, in the Parquet feature store. Why does B2 rebuild
   them from raw sales inside the model instead of reading those columns?
7. We do **not** feed `item_id` (3,049 values) as a categorical, even though a global model
   *could* memorize per-series levels from it. What's the reasoning, and what carries series
   identity instead?
8. v1 beats ETS on RMSSE by only 0.003 but on WMAPE by 0.026. What's the honest way to present
   this result to an interviewer — and what's the *real* argument for the global model over 200
   ETS fits?

**Spot-the-flaw**
9. A teammate speeds B2 up by dropping the recursion: for all 28 test days they read `lag_7`
   straight from the feature store's precomputed column. Backtest RMSSE drops to 0.42. What
   happened?
10. Someone sets `known_future = ["sell_price", "snap", "sales"]` to "give the model more
    signal." What's the bug?
11. The recursion appends each prediction to history before computing the next day's lags. A
    colleague worries this means a bad day-3 prediction corrupts day-10's features. Are they
    right, and is that a flaw or a feature?
12. B2's calendar features are derived from `date` via `pd.DatetimeIndex(...).dayofweek`, ignoring
    the store's existing `wday` column (a different weekday convention). Is mixing conventions a
    bug here?

---

### Answers — B2 (v1)

1. One model fit across **all** series at once, learning a single `features → sales` function; a
   sparse item borrows strength from thousands of others. Seasonal-naive and ETS fit **one model
   per series** in isolation — no pooling.
2. Forecast the horizon **day by day**, feeding each prediction back in as if it were the actual
   before computing the next day's lags/means. Forced because `lag_7` for test-day 10 is the sale
   on day 3 — inside the forecast window, unknown at the origin — so it must be *predicted*, not
   read.
3. AR (`lag_7/28`, `rmean_7/28`), calendar (`wday/month/year`), price/promo
   (`sell_price`, `price_change_pct`, `snap`, `is_event`, `event_type_1`), and series identity
   (`dept_id`, `cat_id`). The **harness supplies** the price/promo columns for test rows
   (`known_future`); the model **derives** calendar from the date, **rebuilds** AR from history,
   and **looks up** dept/cat from train.
4. Daily item demand is intermittent, non-negative, and zero-heavy. Tweedie (compound
   Poisson-Gamma) is the standard loss for that regime and empirically beat plain regression in
   M5; squared error assumes symmetric Gaussian noise the data doesn't have.
5. `BacktestConfig.known_future` lets `evaluate_origin` pass calendar/price columns alongside
   `(id, date)`. Still withheld: **`sales` and the precomputed AR lags** — they encode the
   test-window actuals. Default is an empty list → model sees only `(id, date)`, so baselines and
   existing tests behave exactly as before.
6. So the training path and the recursive test path use the **same** builder and can't silently
   disagree, and so the test-time features are **provably** a function of (past actuals + own
   predictions) only. Reading the store's columns would risk feeding a `lag_7` computed from a
   future actual — leakage.
7. A 3,049-way categorical explodes tree size and invites overfit for little gain, since the
   **lag and rolling-mean features already encode each series' level**. `dept_id`/`cat_id` give
   the coarse structure that pools across series; identity/level comes from the AR features.
8. State it plainly: RMSSE is a **statistical tie** with ETS (0.732 vs 0.735, within noise); the
   clearer edge is WMAPE. The real argument isn't the point-estimate hair — it's that **one**
   model matches 200 per-series fits, scales to all ~30k series unchanged, and is a single object
   to attach conformal intervals to. Don't overclaim the RMSSE win.
9. **Leakage.** The store's `lag_7` for test-day 10 was computed from the actual sale on day 3,
   which lies inside the test window. The model is being fed the future; 0.42 is a fantasy score,
   not a real one. The recursion exists precisely to avoid this.
10. `sales` is the **actual** being predicted — passing it hands the model the answer. `sell_price`
    and `snap` are legitimately known-future; `sales` is not, and the harness contract deliberately
    withholds it. (In our code the model wouldn't even read it, but requesting it signals the
    misunderstanding.)
11. **Right, and it's a feature, not a flaw.** Compounding error is the honest consequence of
    recursive forecasting — a real planner's later-horizon forecasts genuinely are shakier. That
    growing uncertainty is exactly what B4's intervals must widen to cover; hiding it would be the
    dishonest choice.
12. **No.** Only **consistency** between train and test matters — both paths derive the weekday the
    same way, so whatever integer the model learns for "Saturday-like demand" it applies
    identically at predict time. The absolute convention (0=Mon vs M5's 1=Sat) is irrelevant to a
    tree that just splits on the value.

---

## B2 (v2) — Enriched features, disciplined by an ablation

**Recall**
1. What feature changes distinguish the *shipped* v2 from v1?
2. What was the result of the *first* (naive) v2, and what did it include that the shipped v2
   dropped?
3. How does the MLflow experiment keep v1 and v2 comparable rather than overwriting?

**Predict-the-decision**
4. The naive enrichment regressed. What's the right next move — revert, ship it anyway, or
   something else — and what did we actually do?
5. We keep the `early_stopping_rounds` capability in the class but the shipped v2 doesn't use it.
   Why keep dead-looking code, and how would you justify the omission in an interview?

**Spot-the-flaw**
6. Why is `lag_1` specifically harmful in a *recursive* multi-step forecaster, when it's usually
   the single most predictive feature in one-step forecasting?
7. A teammate defends the early-stopping v2: "the validation RMSSE was great, so the round count
   is well-tuned." What regime mismatch are they missing?
8. Someone concludes from the ablation "longer lags are always better than shorter ones." Is that
   the right generalization? What's the actual principle?

---

### Answers — B2 (v2)

1. Shipped v2 adds `lag_14` and `rmean_56` to v1's set (final: lags 7/14/28, rmean 7/28/56),
   fixed rounds, **no `lag_1`**, **no early stopping**. RMSSE 0.727 / WMAPE 0.674 vs v1's
   0.732 / 0.680.
2. Naive v2 scored **0.735 / 0.691** — worse than v1 and a tie with ETS. It included **`lag_1`**
   and **early stopping**, both of which the shipped v2 drops.
3. `version` flows into the model `name` (`lightgbm_global_v1` / `_v2`), so `run_backtest` opens
   two distinct MLflow runs with the same metrics and params schema — a recorded head-to-head,
   not one run silently overwriting the other.
4. Neither revert nor ship-blind: **diagnose**. We ran an ablation over five configs on the same
   sample, which isolated `lag_1` and early stopping as the regressions, and shipped the
   feature set the evidence supported (0.727) — a genuine improvement, honestly earned.
5. It's a legitimate, tested part of the model's API, and keeping it lets us state *why* it's off
   with evidence ("one-shot early stopping tuned a one-step-lag regime the recursive test doesn't
   share, and measurably hurt: 0.7296 vs 0.7267"). "Omitted with a measured reason" is stronger
   than "never considered" or "used because it's standard."
6. Under recursion, `lag_1` for horizon day *h* is *yesterday's value* — and for 27 of 28 days
   "yesterday" is the model's **own prior prediction**, not an actual. A feature the model relies
   on heavily then becomes a channel that **feeds its own error forward**, compounding it. Longer
   lags (7/14/28) point at real history for most of the horizon, so they're far more robust.
7. The validation tail's features are built the **training** way (real-history lags, one-step),
   but the deployed forecast is **recursive**. A round count that minimizes one-step validation
   error isn't the one that minimizes recursive multi-step error — different regime, different
   optimum. The great validation number is measuring the wrong thing.
8. Not quite — it's not "longer is always better." The principle is **match the feature to the
   forecasting scheme**: under recursion, prefer lags that outrun the horizon's error compounding
   and be wary of the shortest lags (which recycle predictions), and validate on the regime you'll
   actually deploy. A short lag could be fine in a *direct* (per-horizon) model that never feeds
   predictions back.
