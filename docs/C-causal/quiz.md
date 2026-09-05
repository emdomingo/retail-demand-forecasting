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
