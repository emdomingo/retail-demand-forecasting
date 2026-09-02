# EDA questions — answer these while exploring

A checklist to work through in a notebook (`notebooks/`). These are chosen so that answering them
**settles the deliberately-open decisions** (model scope; which slice; whether the causal layer is even
feasible on the chosen slice) before the subfeature runthrough. Record findings in
[`data-dictionary.md`](data-dictionary.md) and jot the answer next to each question here.

## 1. Shape & hierarchy — get oriented
- How many series are there, and how does the hierarchy fan out (cats → depts → items → stores → states)?
- How many days of history? What real dates do `d_1` and the last `d_N` map to?
- Is the panel balanced (every series present every day), or do series start/stop?

## 2. The target — what kind of forecasting problem is this really?
- What's the distribution of daily units? How heavy is the **zero-inflation / intermittency**, overall and by category?
- Is intermittency uniform, or concentrated (e.g. HOBBIES spiky, FOODS dense)? → this decides whether a single model family can serve the whole slice.
- Is there visible **weekly and annual seasonality** in an aggregate sales curve? Any obvious trend/level shifts?

## 3. Price — does the DiD causal layer have anything to bite on?
_(This is the decision-driving section: the price-cut DiD needs real, identifiable price drops.)_
- Do prices actually **change over time** within a store-item, or are most flat? What fraction of store-items ever change price?
- When prices drop, are the drops **sharp and datable** (a clean step), or slow drift? DiD needs a datable intervention.
- For a candidate treated item, is there a plausible **control group** — similar items in the same store that did *not* change price at the same time?

## 4. SNAP — feasibility for CausalImpact
- How many days per month is `snap_*` on, and is the schedule fixed within a state?
- Because SNAP fires **store-wide**, there's no within-store control — confirm that (it's why SNAP goes to CausalImpact, not DiD). Does aggregate sales visibly lift on SNAP days?

## 5. Events & calendar
- Which events exist, how frequent, and do any show obvious sales spikes worth modelling as features (vs ignoring)?
- Do `event_name_2` / `snap` overlaps create messy days worth being aware of?

## 6. Synthesis — the scope call (decide with the above in hand)
- **One store vs one category?** (Lean in the plan: one store, for clean same-store price-cut controls.) Does the data support that, or does one category give cleaner series?
- **Which specific slice** has (a) enough non-intermittent series to model honestly *and* (b) at least one clean, datable price-cut intervention with a usable control group?
