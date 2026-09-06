# Feature C — Causal layer — Explainer

Append-only across C1–C3. The *other half* of the project. Feature B answers **what** volume to
expect; Feature C answers **why it moved** — the analytical core of "why did the forecast miss."
Where B is prediction, C is identification: separating a real causal effect from everything that
merely correlates with it.

---

## C1 — Intervention & control-group selection (settled in EDA)

C1 is identification, and it was done in `notebooks/01-eda.ipynb` §3 rather than in a module —
picking the intervention *is* exploratory work, and the honest rejection trail is part of the
story. The one-paragraph result:

- **Treated:** `FOODS_3_697` @ `CA_3`. On **2011-08-08** its price stepped **$3.58 → $2.98**
  (−16.8%) and held. Verified straight off the weekly price path (a clean, datable, sustained
  step — not a one-week blip, not a penny-price artefact).
- **Why this one:** the EDA screened for a *moderate, datable, stable* cut (15–45%, non-penny,
  ≥4 stable weeks each side) on an item that **actually sells** (42.2 units/wk pre-cut, 0%
  zero-weeks). The demand bar matters — the first candidates failed it: `FOODS_3_822` *crashed
  to zero at the cut* (a stockout, the opposite of a price response) and `FOODS_2_227` was 96%
  zero-weeks (a dead item whose price is stable precisely because nobody buys it).
- **Controls:** same-store, same-department (FOODS_3 @ CA_3), **matched** on pre-cut weekly-sales
  correlation (top-15, mean pre-cut corr 0.52), and **price-stable** within ±8 weeks of the cut.

The decision gate that carries into C2: **why matched controls, not "the whole department"?**
Averaging 768 department items produces a line far smoother than one noisy series, so
"parallel-trends" becomes unjudgeable — a divergence could be real or just an averaging artefact.
Matching on pre-cut co-movement keeps controls that carry *comparable noise and seasonality*, so
a post-cut gap is signal.

---

## C2 — Difference-in-Differences on the price cut

**What C2 is:** a single-number causal estimate — *how much of the post-cut sales jump did the
price cut cause?* — plus the evidence that the number is trustworthy (an event study for
parallel trends, a placebo, and a confidence interval). It lives in `src/causal/did.py`.

### The identifying idea (one line)

Subtract from the treated item's before→after change the same change measured on comparable
items that **did not** cut. Whatever common force moved everyone — summer, a holiday, a
store-wide promo — is differenced out; what remains is attributable to the cut. That's the
"difference of differences": (treated_after − treated_before) − (control_after − control_before).

### Why the raw number lies, and DiD fixes it

The naive read is seductive: pre-cut 42.2 units/wk, the 12 weeks after averaged ~66 → **+56%**,
implying an elasticity around −4/−5. But some of that +56% is just *when* the cut happened
(late summer, a rising demand season the controls also rode). DiD strips that out. The
disciplined estimate is **+42.1%** (95% CI [+12.5%, +79.4%]) — a demand lift, still large, but
an implied elasticity ≈ **−2.5**, not −4/−5. **The raw jump overstated the effect by a third.**
That gap *is* the value of the method: the naive number was confounded, and we can say by how
much.

### The model — two-way fixed-effects DiD, on log units

The estimator is one line of statsmodels doing a lot of work:

```python
smf.ols("log_units ~ C(item_id) + C(week) + treated:post", data=panel) \
   .fit(cov_type="cluster", cov_kwds={"groups": panel["item_id"]})
```

Reading each piece as a decision gate:

- **`C(item_id)` — item fixed effects.** A dummy per item absorbs its *level*. The treated item
  and each control sell different absolute volumes; DiD only cares that they *move* together, so
  we let each have its own intercept. (This is why the `treated` main effect isn't a free
  parameter — item FE already contains it.)
- **`C(week)` — week fixed effects.** A dummy per week absorbs everything *common to all items
  that week*: seasonality, holidays, macro demand. This is the "difference out the common force"
  step, done properly instead of by eyeballing one control line. (This absorbs the `post` main
  effect too.)
- **`treated:post` — the estimand.** With both sets of FE soaking up levels and common time
  shocks, the *only* thing this interaction can explain is the treated item behaving differently
  from controls *specifically after the cut*. That coefficient is the DiD effect.
- **`log_units` — the outcome transform.** A price cut acts *multiplicatively* on demand, so we
  want a percentage, and a log-outcome coefficient reads directly as one: effect ≈ `exp(β) − 1`.
  We use `log1p` to tolerate the rare zero week; matched controls are dense sellers so the `+1`
  distortion is negligible (units in the tens). Named honestly, not hidden.
- **`cov_type="cluster"` by item — the standard error.** Observations within one item are
  correlated across weeks (autocorrelation); clustering by item accounts for that so the CI
  isn't fraudulently narrow. **Caveat:** ~16 clusters is *few*, and few-cluster clustering
  under-states uncertainty — the CI is approximate. The rigorous upgrade (named, not built) is a
  **wild-cluster bootstrap**.

### Two design gates worth defending

- **Weekly grain.** M5 prices are weekly, and weekly aggregation collapses the strong
  day-of-week seasonality that the DiD doesn't care about. Daily rows would add noise to a level
  shift for no gain.
- **Balanced ±26-week window.** History runs to 2016, but the cut is in 2011 with only ~27
  pre-weeks. A long post window would let unrelated 2012–16 drift leak into the estimate, so we
  cap post symmetric to pre. The estimand is honestly the **sustained near-term** effect — not
  an all-time average.

### The evidence the number is real (not just a number)

A DiD coefficient is only as good as its **parallel-trends** assumption: absent the cut, treated
and controls would have moved together. You can't prove it, but you can make it credible.

- **Event study** (`event_study`). Instead of one before/after split, estimate a treated-vs-control
  gap *per week* relative to the cut (week −1 omitted as baseline). Two things to look for:
  - *Leads* (pre-cut weeks) should scatter around zero with **no systematic trend** — if the
    treated were already pulling away before the cut, the DiD would just be picking up that
    trend. Ours: individual leads are noisy (single-item weekly data), but the **slope across
    leads is ≈ −0.02/week** — essentially flat, no upward pre-trend faking the effect.
  - *Lags* (post-cut weeks) trace how the effect builds. Ours are **predominantly positive, several
    individually significant**, growing over the horizon — a sustained response, not a one-week
    blip. The plot is saved to `docs/C-causal/figures/event_study.png`.
- **Placebo** (`placebo_test`). Run the *identical* DiD on pre-period data only, with a **fake
  cut** at the pre-period midpoint. There was no real intervention there, so a credible design
  returns a null. Ours: point estimate −27.7% but **95% CI covers 0** → passes. Reported
  honestly — the non-trivial point estimate reflects a genuinely noisy, thin single-series
  pre-period, which is the honest limitation, not a hidden one.

### The honest limitation → why C2b exists

One store, one item, ~27 pre-weeks: the point estimate is positive and plausible, but the
pre-trends aren't pristine and the CI is wide. **The credibility isn't in this single estimate —
it's in replication.** The chosen cut was *chain-wide* (same item, same week, in five stores), so
C2b re-runs the DiD across CA_3/CA_1/TX_1/TX_2/CA_4: five agreeing estimates neutralise the thin
pre-period in a way one estimate never can. That's the difference between "I found an effect" and
"the effect reproduces."

### How C2 fits the whole

C2 reads the **same feature store** the forecasting half reads — through the same DuckDB slice
layer (`read_store_slice`), just aggregated to weeks and reshaped into a panel. No new pipeline:
the causal branch and the forecast branch are parallel consumers of one query layer. The output
(effect + CI + event-study plot) feeds the D3 dashboard panel. And the framing closes the loop
with B: when a forecast misses because a planner cut a price, *this* is the machinery that
quantifies why.

---

## C2b — Chain-wide replication (five stores)

**What C2b is:** the same DiD, re-run in each of the five stores that made the identical cut, and
combined into one answer to the only question that matters for a single noisy estimate — *does it
reproduce?* It lives in `src/causal/replication.py`.

### Why replication is the real evidence

C2's honest weakness is structural: one treated series, ~27 pre-weeks, pre-trends that aren't
pristine. No amount of cleverness on *one* store fixes that. But the cut was chain-wide — the same
item dropped $3.58 → $2.98 in the same fortnight in CA_3, CA_1, TX_1, TX_2, CA_4 — so we get five
*independent* natural experiments. Five estimates that agree are worth far more than one, because
the thing that could bias any single store (a local stockout, a coincident local promo, a noisy
control set) won't line up the same way across five. Agreement is the falsification test one store
can't run on itself.

### One decision that had to be right: detect the cut per store

The chain rolled the cut out a week apart — **CA_3 on 2011-08-08, the other four on 2011-08-15**.
Hardcoding a single date would have put a genuinely *pre-cut* week into the post period for four
stores, dragging those estimates toward zero. So each store detects its own cut from its own price
path (`detect_cut`: first ≥10% drop in the intervention era) and builds its own matched controls.
This is the kind of one-line assumption that silently corrupts a result — checked, not assumed.

### The result

| store | cut | lift | 95% CI |
|-------|-----|-----:|--------|
| CA_3 | 2011-08-08 | +42.1% | [+12.5%, +79.4%] |
| CA_1 | 2011-08-15 | +41.1% | [+11.8%, +77.9%] |
| TX_1 | 2011-08-15 | +99.8% | [+68.3%, +137.2%] |
| TX_2 | 2011-08-15 | +47.9% | [+17.6%, +86.1%] |
| CA_4 | 2011-08-15 | +49.7% | [+12.8%, +98.7%] |

**5/5 positive, 5/5 individually significant.** Four stores cluster tightly at +41–50%; TX_1 (the
thinnest store, 15 units/wk pre-cut) is a genuine outlier at ~+100%. The effect *reproduces* — and
the disciplined C2 headline (+42%) turns out to be at the *low* end of the chain, not a fluke.

### Pooling honestly: fixed vs random effects

Combining the five uses **inverse-variance meta-analysis** — weight each store by its precision
(`1/SE²`), so a tight estimate counts more. But that (fixed-effect) pool assumes every store shares
*one* true effect; TX_1 says otherwise. The heterogeneity statistics quantify it: **Cochran's Q =
9.2 (p = 0.06), I² = 57%** — over half the variance is genuine between-store spread, not sampling
noise. So the fixed-effect CI (+61%, [+46%, +78%]) is *too narrow* — it trusts an agreement that
isn't fully there.

The right response is a **random-effects (DerSimonian–Laird)** pool, which adds the estimated
between-store variance τ² to every weight, widening the interval to reflect the disagreement:
**+56.9%, 95% CI [+34.6%, +82.9%]**. That is the number to quote. Reporting the FE point and hiding
the heterogeneity would be exactly the kind of overclaim the project's guardrails forbid.

### The takeaway

The price cut robustly and significantly lifted demand across the whole chain — a large effect
(≈ +40–50% in most stores, elasticity ≈ −2.5 to −3), reproduced five times. The *magnitude* varies
by store (TX_1 markedly more price-sensitive), which the random-effects CI states plainly rather
than papering over. This is the difference between "I found an effect in my one example" and "the
effect holds up when the design is stress-tested" — the latter is what survives interview scrutiny.

The generalisable next step (a panel / staggered-adoption TWFE across the 632-candidate cut
shortlist, reporting an *average* price-cut effect) is **named, not built** — SPEC C2b's optional
stretch — because staggered TWFE carries its own well-known biases (the Goodman-Bacon / de
Chaisemartin critique) that would need the modern estimators to do right, and the five-store
replication already delivers the "more than one case" evidence honestly.

## C3 — SNAP demand lift (cross-state regression counterfactual)

The second causal method on the second intervention. C2/C2b measured a **price cut** with
Difference-in-Differences. This measures **SNAP** — the days each month when food-assistance
benefits are disbursed — and it deliberately uses a *different* method, because SNAP breaks both
things DiD needs.

### Why DiD cannot be reused here

- **No clean pre-period.** SNAP is not a one-off onset with a before and an after; it is a
  *recurring monthly pulse* present across the entire 2011–16 window. There is no "pre-SNAP" era
  to difference against.
- **No in-store control group.** SNAP eligibility is store-wide — every item in a CA store faces
  the same CA SNAP calendar — so there is no untreated twin *inside* the store the way the price
  cut had non-cutting neighbours.

SPEC C3 reframes the estimand accordingly: not a single-onset effect but the **average SNAP-day
lift**.

### The idea that rescues identification: schedules differ by state

The one fact that makes a counterfactual possible: **SNAP disbursement days differ by state.**
California, Texas and Wisconsin pay on overlapping-but-distinct days. So on a CA SNAP day, the TX
and WI stores are a live control for *what CA demand would look like today without its SNAP boost*
— they share the day-of-week, the holidays, the season, the macro demand wave, but not CA's SNAP
schedule.

That is exactly the CausalImpact / BSTS idea — predict the treated series from control series and
read the effect off the gap — done here as a transparent OLS regression instead of a Bayesian
structural time-series model. CausalImpact is the **named Bayesian sibling** (SPEC's "second named
method"); we build the regression form because it needs no fragile heavy dependency (the
tfcausalimpact/TensorFlow friction we avoided with pmdarima in B3) and every coefficient defends by
hand.

### The model — a cross-state regression counterfactual, on log FOODS units

Outcome = log daily **FOODS** units at CA_3 (SNAP is *food* assistance, so we measure the food
category, not all-categories which would dilute it; log so the coefficient reads as a percentage).
The estimate is built as a **ladder**, each rung adding controls:

| spec | controls | lift | 95% CI | R² |
|------|----------|-----:|--------|---:|
| naive | none | +16.8% | [+12.9%, +20.9%] | 0.035 |
| calendar | day-of-week | +16.9% | [+13.0%, +20.8%] | 0.119 |
| **cross-state** | **+ log TX, log WI demand, TX/WI SNAP flags** | **+10.7%** | **[+9.1%, +12.3%]** | **0.928** |

The cross-state controls take R² from 0.12 to **0.93** and discipline the estimate from +16.8% to
**+10.7%** — the naive gap overstates by ~57%, the same lesson as the C2 price cut (raw +56% →
disciplined +42%). The disciplining *is* the method's value.

Two design gates behind that main row:

1. **Control for the control states' own SNAP.** CA/TX/WI SNAP windows overlap heavily (all pay
   early-month — 384 of CA's 640 SNAP days coincide with a TX and WI SNAP day). On those overlap
   days TX/WI demand is *itself* SNAP-lifted, which would soak up part of CA's effect. Including
   `tx_snap` and `wi_snap` as covariates nets that out so `ca_snap` isolates CA's own lift.
2. **HAC (Newey–West) standard errors.** Daily demand is strongly autocorrelated; plain OLS SEs
   would be far too small. HAC with a two-week lag window is the time-series analogue of C2's
   cluster-robust SEs.

### The evidence the number is real

Two falsifications, mirroring the C2 placebo:

- **Placebos.** Apply CA's SNAP schedule as a *fake* treatment to a control state, under the same
  spec (control state as outcome, the *other* control state as its cross-state predictor — never
  CA, which is contaminated by the fake treatment). Both return nulls: placebo→TX −1.8%
  [−4.2%, +0.7%], placebo→WI +3.0% [−0.3%, +6.3%], both CIs covering zero. So `ca_snap` is not
  picking up a generic early-month calendar wave common to all states — it is CA's own SNAP lift.
- **Clean-day estimator.** Restrict to the ~128 days when *only* CA is on SNAP (TX and WI
  genuinely untreated) versus days when *no* state is. This drops the overlap days entirely, so the
  controls are true controls by construction — no modelling of co-scheduled SNAP needed. It gives
  **+11.8% [+9.7%, +14.0%]**, corroborating the full-sample +10.7% and confirming the overlap-day
  contamination is handled right.

### Honest limitations

- **Average, not one-off.** The recurring-treatment framing means this is an *average* SNAP-day
  lift, not a single-onset effect. Stated as such.
- **Anticipation / pantry-loading.** Households may shop just before or after a SNAP day, smearing
  the effect across adjacent days; a sharp one-day indicator would under- or over-count that.
- **HAC SEs are asymptotic.** As with C2's few-cluster SEs, the interval is approximate.
- **Named, not built:** the fully Bayesian counterfactual (CausalImpact / BSTS) and a
  synthetic-control donor-weighting of the states — both would sharpen the counterfactual without
  changing the story.

### How C3 fits the whole

Feature C now carries **two methods × two interventions**: DiD on the price cut (C2/C2b) and a
cross-state regression counterfactual on SNAP (C3) — a genuine robustness pairing rather than one
trick applied twice. Both tell the same disciplining story (the naive gap overstates; a proper
counterfactual shrinks it and survives a placebo), which is the causal half of the project's anchor:
when actuals diverge from the forecast, *why* — and by how much that a real intervention actually
caused. Figure: `docs/C-causal/figures/snap_counterfactual.png`.
