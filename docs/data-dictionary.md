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
One row per day. **1,969 rows** (confirmed). Maps the `d_N` day-index to a real date and carries event/SNAP context.

| column | type | meaning | notes / confirmed |
|---|---|---|---|
| `date` | date | calendar date (YYYY-MM-DD) | span **2011-01-29 → 2016-06-19**; sales history ends `d_1941` = **2016-05-22** (the rest is the 28-day horizon) |
| `wm_yr_wk` | int | Walmart retail week id (join key to `sell_prices`) | format `YYWW` (e.g. 11101); ~282 unique (≈ one per week) |
| `weekday` | str | day name (Saturday…) | |
| `wday` | int | day-of-week index | **1 = Saturday** (confirmed: `d_1` = Sat 2011-01-29) |
| `month` | int | 1–12 | |
| `year` | int | | **2011–2016** |
| `d` | str | day index `d_1 … d_N` (join key to `sales_train_*`) | |
| `event_name_1` | str | primary event that day (e.g. SuperBowl) | **~92% null** (162 / 1,969 days carry an event) |
| `event_type_1` | str | event category (Sporting/Cultural/National/Religious) | Religious 55 / National 52 / Cultural 37 / Sporting 18 |
| `event_name_2` | str | secondary event (rare) | **5 days only** (~0.25%) |
| `event_type_2` | str | secondary event type | |
| `snap_CA` | 0/1 | SNAP benefits purchasable in CA that day | **share of 1s ≈ 0.33** (~10 days/month, fixed schedule, store-wide) |
| `snap_TX` | 0/1 | SNAP purchasable in TX | ≈ 0.33 |
| `snap_WI` | 0/1 | SNAP purchasable in WI | ≈ 0.33 |

## `sales_train_validation.csv` / `sales_train_evaluation.csv`
One row per **series** (item × store). Wide format: identifier columns + one column per day.
- `validation`: days `d_1 … d_1913` (confirmed)
- `evaluation`: days `d_1 … d_1941` (confirmed; 28 extra days = the validation horizon revealed) — **we use `evaluation`**

| column | type | meaning | notes / confirmed |
|---|---|---|---|
| `id` | str | `{item_id}_{store_id}_{validation\|evaluation}` — the series key | **30,490 rows** |
| `item_id` | str | product id `{cat}_{dept}_{nnn}` | **3,049** items |
| `dept_id` | str | department (e.g. FOODS_3) | **7 depts** (FOODS 3, HOBBIES 2, HOUSEHOLD 2) |
| `cat_id` | str | category (HOBBIES / HOUSEHOLD / FOODS) | 3 |
| `store_id` | str | store (e.g. CA_1) | **10 stores** (CA 4, TX 3, WI 3) |
| `state_id` | str | CA / TX / WI | 3 |
| `d_1 … d_N` | int | units sold that day for this series | heavy intermittency — **zero-share overall median 0.73**; by cat (mean/median): FOODS 0.62/0.65, HOUSEHOLD 0.72/0.77, HOBBIES 0.77/0.83. ⚠ inflated by structural (pre-launch) zeros → mask via price-row presence before modelling |

## `sell_prices.csv`
One row per store × item × retail-week. The price is **weekly**, not daily.

| column | type | meaning | notes / to confirm |
|---|---|---|---|
| `store_id` | str | store | |
| `item_id` | str | product | |
| `wm_yr_wk` | int | retail week (join to `calendar.wm_yr_wk`) | |
| `sell_price` | float | price for that item/store/week | **73% of store-items change price ≥ once**; a series only exists here once the item is sold (absent weeks = not yet stocked / discontinued — the signal that disambiguates structural vs demand zeros). ⚠ deep cuts (−99% → ~$0.01) are penny-price artifacts, not promos; clean promo band is **15–45%** |

## `sample_submission.csv`
Submission format only (not modelling input). `id` + `F1 … F28` (the 28-day forecast horizon).

---