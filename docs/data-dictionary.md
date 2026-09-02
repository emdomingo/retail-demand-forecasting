# M5 Data Dictionary

A living reference for the raw M5 tables in `data/raw/`. Column definitions are seeded from the
known M5 schema; **empirical facts (cardinality, ranges, null patterns) are filled in through EDA.**
Anything marked _[confirm]_ is a claim to verify against the data, not a given.

The four raw tables and how they join:

```
sales_train_*  (one row per series: id = item_id × store_id)
   │  columns d_1 … d_N are the daily-sales matrix — MELT these to long (date × series)
   │  join key to calendar:  d_N  →  calendar.d
   │  join key to prices:    (store_id, item_id) + wm_yr_wk
calendar       (one row per day: maps d_N → real date, wm_yr_wk, events, SNAP flags)
sell_prices    (one row per store_id × item_id × wm_yr_wk: the weekly price)
```

---

## `calendar.csv`
One row per day. ~1,969 rows _[confirm]_. Maps the `d_N` day-index to a real date and carries event/SNAP context.

| column | type | meaning | notes / to confirm |
|---|---|---|---|
| `date` | date | calendar date (YYYY-MM-DD) | span: ____ to ____ _[confirm]_ |
| `wm_yr_wk` | int | Walmart retail week id (join key to `sell_prices`) | format YYWW-ish; cardinality ____ |
| `weekday` | str | day name (Saturday…) | |
| `wday` | int | day-of-week index | 1 = ? _[confirm which day is 1]_ |
| `month` | int | 1–12 | |
| `year` | int | | range ____ |
| `d` | str | day index `d_1 … d_N` (join key to `sales_train_*`) | |
| `event_name_1` | str | primary event that day (e.g. SuperBowl) | null when no event; % null ____ |
| `event_type_1` | str | event category (Sporting/Cultural/National/Religious) | |
| `event_name_2` | str | secondary event (rare) | % non-null ____ |
| `event_type_2` | str | secondary event type | |
| `snap_CA` | 0/1 | SNAP benefits purchasable in CA that day | share of 1s ____ |
| `snap_TX` | 0/1 | SNAP purchasable in TX | |
| `snap_WI` | 0/1 | SNAP purchasable in WI | |

## `sales_train_validation.csv` / `sales_train_evaluation.csv`
One row per **series** (item × store). Wide format: identifier columns + one column per day.
- `validation`: days `d_1 … d_1913` _[confirm]_
- `evaluation`: days `d_1 … d_1941` _[confirm]_ (28 extra days = the validation horizon revealed)

| column | type | meaning | notes / to confirm |
|---|---|---|---|
| `id` | str | `{item_id}_{store_id}_{validation\|evaluation}` — the series key | ____ rows total (expect ~30,490) |
| `item_id` | str | product id `{cat}_{dept}_{nnn}` | cardinality ____ (expect ~3,049) |
| `dept_id` | str | department (e.g. FOODS_3) | ____ depts |
| `cat_id` | str | category (HOBBIES / HOUSEHOLD / FOODS) | 3 |
| `store_id` | str | store (e.g. CA_1) | 10 stores _[confirm]_ |
| `state_id` | str | CA / TX / WI | 3 |
| `d_1 … d_N` | int | units sold that day for this series | many zeros — intermittency; check zero-share ____ |

## `sell_prices.csv`
One row per store × item × retail-week. The price is **weekly**, not daily.

| column | type | meaning | notes / to confirm |
|---|---|---|---|
| `store_id` | str | store | |
| `item_id` | str | product | |
| `wm_yr_wk` | int | retail week (join to `calendar.wm_yr_wk`) | |
| `sell_price` | float | price for that item/store/week | a series only exists here once the item is sold; absent weeks = not yet stocked _[confirm]_ |

## `sample_submission.csv`
Submission format only (not modelling input). `id` + `F1 … F28` (the 28-day forecast horizon).

---

## Cross-cutting facts to establish in EDA
_(fill these in — they drive the scope decision and the causal design)_

- **Series count / hierarchy fan-out:** ____ series across 3 cats → __ depts → __ stores → 3 states.
- **Date span & horizon:** history ____ to ____ ; forecast horizon = 28 days.
- **Intermittency:** overall share of zero-sales day-series pairs = ____ %. Worse in HOBBIES? _[confirm]_
- **Price-change availability:** do prices actually move within a store-item over time (needed for the DiD price-cut layer)? ____
- **SNAP cadence:** how many days/month is `snap_*` = 1, and is it fixed per state? ____
