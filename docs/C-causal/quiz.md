# Feature C — Causal layer — Quiz

Append-only across C1–C3. Self-quiz: recall, predict-the-decision, spot-the-flaw.

---

## C1 — Intervention & control-group selection

**Recall**
1. What is the treated unit, the cut date, and the price step? How large is the cut in %?
2. What three properties made this a *usable* DiD intervention (vs the cuts that were rejected)?
3. Why are the controls "same store, same department" rather than, say, the same item in another
   store?

**Predict-the-decision**
4. Two candidates were rejected before this one: `FOODS_3_822` and `FOODS_2_227`. What was wrong
   with each, and what single EDA lesson do both failures share?
5. Why *matched* controls (top-15 by pre-cut correlation) instead of the whole 768-item
   department? What goes wrong with the department average?

**Spot-the-flaw**
6. A colleague says "the item sold 42/wk before and 66/wk after — that's a 56% lift from the
   price cut, elasticity about −4." What's wrong with that inference?

---

## C2 — Difference-in-Differences

**Recall**
7. State the DiD estimand as a "difference of differences" in words.
8. In `log_units ~ C(item_id) + C(week) + treated:post`, what does each of the three terms
   absorb or estimate? Why are the `treated` and `post` main effects *not* free parameters?
9. Why is the outcome `log_units` and not raw units? How do you read the coefficient as a
   percentage?
10. What is the headline effect and its 95% CI? What implied elasticity does that correspond to,
    and how does it compare to the naive estimate?

**Predict-the-decision**
11. Why weekly grain rather than daily?
12. History runs to 2016 but the analysis uses a ±26-week window. Why cap the post-period instead
    of using all available data?
13. Standard errors are clustered by item. What correlation does that account for, and why is the
    resulting CI still only "approximate"? What's the named upgrade?
14. The single-store estimate is called noisy and its pre-trends "not pristine." What is the
    project's answer to that, and why does it work?

**Spot-the-flaw**
15. In the event study, suppose the *leads* (pre-cut coefficients) sloped steadily upward toward
    the cut. Why would that invalidate the DiD, even if the post-cut coefficients were large and
    positive?
16. The placebo test returned a point estimate of −27.7%. A skeptic says "that's a big number,
    your design is broken." Is it? What actually determines whether the placebo passes, and what
    does the non-zero point estimate honestly reflect?
17. Someone proposes using `lag_1`-style daily sales and *all* 278 weeks of history "to get more
    data and tighter CIs." Name two things that go wrong with that.

**Connect-the-project**
18. C2 and the LightGBM forecaster (B2) read the same feature store. Trace the path C2's data
    takes from Parquet to the regression panel. What does it reuse, and what does it add?
19. How does the causal layer answer the question the forecasting half can't — "why did the
    forecast miss"?

---

## C2b — Chain-wide replication

**Recall**
20. What makes five stores a *replication* rather than just five separate analyses? Why is
    agreement across them stronger evidence than any single estimate?
21. What were the results — how many of the five stores were positive, how many significant, and
    where did the estimates cluster?

**Predict-the-decision**
22. Why does each store detect its *own* cut date instead of sharing one? What specifically goes
    wrong if you hardcode 2011-08-08 for all five?
23. Inverse-variance pooling weights each store by `1/SE²`. What does that achieve, and which
    store therefore pulled the fixed-effect pool hardest?
24. The write-up quotes the *random-effects* pooled CI, not the fixed-effect one. What drove that
    choice, and what does the random-effects model add to the weights?

**Spot-the-flaw**
25. A colleague reports the headline as "+61.1%, 95% CI [+46%, +78%]" (the fixed-effect pool). Why
    is that CI misleadingly narrow here? What statistic tells you so?
26. TX_1 shows ~+100% while the others sit at +41–50%. Someone says "drop TX_1, it's an outlier
    distorting the average." Is that the right move? What does I²=57% actually tell you to do
    instead?
27. Why is the staggered-adoption panel DiD (the "average effect across 632 cuts" version) named
    but *not built*? What bias would a naive TWFE on staggered cuts risk?

## C3 — SNAP demand lift (cross-state counterfactual)

**Recall**
28. Why can't the SNAP effect be measured with Difference-in-Differences? Name both properties of
    SNAP that break DiD's requirements.
29. What single fact about SNAP makes a cross-state control group valid? Why is a Texas store a
    legitimate control for a California store on a CA SNAP day?
30. What was the headline SNAP-day lift, and how far did the cross-state controls discipline it
    down from the naive gap? What happened to R² across the ladder?

**Predict-the-decision**
31. The main spec includes `tx_snap` and `wi_snap` as covariates even though TX/WI demand is
    already in the model. Why are the raw SNAP flags needed on top of the demand series? What
    specifically goes wrong on overlap days if you omit them?
32. Why HAC (Newey–West) standard errors instead of ordinary OLS SEs? What is the C2 analogue of
    this choice?
33. The outcome is FOODS-category demand, not all-category store demand. Why scope to FOODS, and
    what would using total demand cost you?

**Spot-the-flaw**
34. A colleague reports the naive SNAP-day lift (+16.8%) as the causal effect. What is the single
    biggest thing that number fails to account for, and roughly how much does it overstate?
35. The placebo regresses a *control* state's demand on `ca_snap`, but uses the *other* control
    state (not CA) as its cross-state predictor. Why must CA be excluded from the placebo's
    right-hand side? What would including it do to the placebo?
36. The clean-day estimator uses only ~128 CA-only SNAP days and throws away the overlap days —
    far less data. Why is that a *strength* of that particular check rather than a weakness, and
    what does its agreement with the full-sample estimate buy you?
37. Why is CausalImpact / BSTS *named but not built* here, when it is the canonical method for this
    kind of question? What does the OLS regression give up, and what does it gain?
