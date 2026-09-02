# Study plan — enough intuition to push back

Not a curriculum. Each item is tied to a **decision gate** in [runthrough.md](runthrough.md)
that you accepted on trust. The goal is not mastery — it's enough intuition to say
"wait, why direct and not recursive?" and mean it. Skim for the *idea*; you'll cement
it when we build that subfeature and write its quiz.

**How to read each item:** the intuition to grab · the pushback it buys you · one source · one self-test.
Ordered to match the build (forecasting first, causal second).

---

## If you only do three things
1. **Rolling-origin backtesting & leakage** (unlocks all of B, and A2).
2. **DiD identification — parallel trends + why one treated unit is a problem** (unlocks all of C).
3. **Conformal coverage vs a model's own quantiles** (unlocks B4, the deliverable).

Everything else is depth on top of these three.

---

## 1. Backtesting & leakage — *the load-bearing idea* → A2, B1
- **Grab:** you may only use information a planner would actually have *at the forecast origin*. Rolling-origin = slide that origin through time, train on the past, score the future, repeat. A single train/test split on time series is a lie because it ignores that the model gets re-fit as time passes.
- **Pushback it buys:** whether "expanding vs sliding window," "refit every origin," and "non-overlapping folds" are the right calls — and whether any feature (a rolling mean including today) leaks.
- **Source:** Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (FPP3, free online) — the "Time series cross-validation" section. ~30 min.
- **Self-test:** why does a rolling mean using `rowsBetween(-7, 0)` leak, but `(-7, -1)` not? Why do overlapping test windows make an average RMSSE look better/more stable than it is?

## 2. Metrics & intermittency — *why not MAPE* → B1
- **Grab:** retail demand is *intermittent* (lots of zeros). MAPE divides by actuals → explodes on zeros, undefined at zero. RMSSE/WMAPE **scale the error by the seasonal-naive error**, so a score of 1.0 means "as good as naive," <1 means "better." The scale is computed on the **train** window (leak guard).
- **Pushback it buys:** whether the metric denominator is honest, and why "beat 1.0 or explain why not" is the bar.
- **Source:** the M5 competition guide / Makridakis et al. M5 paper — the WRMSSE definition (skim the formula + the *why*, not the derivation). ~20 min.
- **Self-test:** if RMSSE = 0.85, what does that mean in words? Why is dollar-weighting (WMAPE/WRMSSE) more planner-relevant than a plain average?

## 3. Forecast models — global, and direct vs recursive → B2, B3
- **Grab:** a **global** model trains one learner across all series (pooling signal, series-id as a feature) — this is what won M5. For multi-step, **recursive** feeds its own predictions back as lags (errors compound, leakage-risky); **direct** predicts each horizon using only origin-known features (lags ≥ horizon) — no compounding, structurally clean.
- **Pushback it buys:** the B2 direct-vs-recursive call, and *what* SARIMA (a single-series classical model) should even forecast (a sample, not the aggregate).
- **Source:** FPP3 chapters on ETS and ARIMA (skim for *what they assume*, not the algebra); for global/direct-vs-recursive, any M5-winner writeup (Kaggle "1st place solution" discussion). ~40 min.
- **Self-test:** why does direct multi-step avoid the leakage risk recursive has? Why can't you fairly compare SARIMA on the FOODS_3 *total* against a per-series global model on the same RMSSE?

## 4. Prediction intervals & coverage — *the deliverable* → B4
- **Grab:** a model's own quantile outputs (e.g. LightGBM pinball) are *estimates* — they can be miscalibrated. **Conformal prediction** wraps any point model and gives a **distribution-free guarantee**: a 90% band covers ~90% in finite samples — *if* data is exchangeable. Time series **violates exchangeability** (autocorrelated residuals), so **EnbPI** is the time-series-correct version.
- **Pushback it buys:** why conformal leads over "just do quantile regression," and why you must *measure empirical coverage*, not assume it.
- **Source:** Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (free arXiv) — read §1–2 and the coverage guarantee statement. ~40 min.
- **Self-test:** what does "90% coverage guarantee" actually promise, and what breaks it for time series? How would you *check* your 90% band is really 90%?

## 5. DiD identification — *the core of the causal half* → C1, C2, C2b
- **Grab:** DiD = (treated after − before) − (control after − before), which cancels anything common to both. It only works if **parallel trends** holds (absent the cut, treated and control would have moved together) — you evidence that with an **event-study** (flat pre-cut lead coefficients). With **one treated unit**, standard SEs are untrustworthy → use **placebo/permutation** inference. **Simultaneous** adoption (same week, 5 stores) → clean pooled DiD; **staggered** adoption → naive two-way fixed effects is *biased* (needs Callaway–Sant'Anna).
- **Pushback it buys:** basically all of Feature C — whether the control group is clean, whether the CI is honest, and why the staggered panel is named-not-built.
- **Source:** Nick Huntington-Klein, *The Effect* (free online), DiD chapters — or Cunningham's *Causal Inference: The Mixtape* (free online) DiD chapter. Read the parallel-trends + event-study part carefully. ~1 hr.
- **Self-test:** how does the event-study plot *support* (never proves) parallel trends? Why is one treated unit a problem, and how does the placebo test rescue inference? Why doesn't the TWFE-bias literature apply to our 5-store simultaneous case?

## 6. Modelled counterfactuals — CausalImpact / synthetic control → C3
- **Grab:** when there's no untreated twin, you *model* the counterfactual: fit "what the treated series would have done" from control series that predict it pre-intervention **and are themselves unaffected**. CausalImpact (BSTS) assumes a **single onset + clean pre-period** — which SNAP (recurring, always-on, store-wide) breaks. The fix: **cross-state controls** (SNAP days differ by state), estimand = average SNAP-day lift.
- **Pushback it buys:** why the naive "SNAP → CausalImpact" is a scrutiny trap, and whether the cross-state reframe actually identifies the effect.
- **Source:** Brodersen et al. 2015 (CausalImpact paper) — read the intro + the assumptions figure; skim synthetic control (Abadie) for the same idea. ~30 min.
- **Self-test:** what is CausalImpact's "clean pre-period" assumption, and why does SNAP violate it? Why is a Texas store a *valid* control regressor for California on CA's SNAP days?

## 7. Spark-as-feature-store honesty — *the scoping story* → A (light)
- **Grab:** Spark earns its place by building features **once across ~30,490 series** (a real distributed-shaped job); the *model* then reads back one slice. You never claim modelling needed Spark — that's the overclaim to avoid. DuckDB reads the Parquet slice with predicate pushdown (partitioning by store/dept is the query optimisation).
- **Pushback it buys:** defending "why Spark at all?" in an interview without overclaiming, and why the feature store is partitioned the way it is.
- **Source:** no reading needed — this is a framing you'll internalise writing the A-feature explainer. Just hold the one sentence: *population-scale pipeline, scoped model.*
- **Self-test:** an interviewer says "you could've done this in pandas." What's your honest answer?

---

## Sequencing (matches the build)
- **Before A/B (forecasting):** items 1, 2, 3, 4 — this is most of the study weight.
- **Before C (causal):** items 5, 6.
- **Item 7** absorbs while building A; no separate study time.

Total focused time ≈ 4–5 hours of skimming, spread across the build. You're aiming for
*informed pushback*, not exam-readiness — the per-feature quizzes finish the job.
