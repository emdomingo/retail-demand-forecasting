"""Exogenous features via joins to calendar and sell_prices (A3).

A2's features are derived from the *target* (past sales) and so need the strict shift
discipline. A3's features are **exogenous covariates known at prediction time** — the
calendar, the SNAP schedule, and posted prices are all knowable in advance. So the
leakage rule is different here: these may legitimately use the *current* day's value.
The one shift we still apply (price_change_pct) is week-over-week momentum, not a peek
at the target.

Two joins, two shapes:
  - calendar (~1,969 rows) is tiny -> broadcast it (map-side join, no shuffle). Key: d.
  - sell_prices (~6.8M rows) is not broadcastable -> a normal shuffle join.
    Key: (store_id, item_id, wm_yr_wk); wm_yr_wk arrives from the calendar join.

Both joins are LEFT and one-to-many-safe: calendar is unique per d, prices unique per
(store, item, week), so neither multiplies the one-row-per-series-day grain.

Run as a demo (a window around Christmas 2011, where events + SNAP + price all show):
    uv run python -m src.ingest.exogenous
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.ingest.load import load_calendar, load_sales_wide, load_sell_prices, melt_sales
from src.ingest.spark import get_spark

# Calendar columns we keep; weekday (string dup of wday) and the rare event_2 pair are dropped.
_CAL_COLS = [
    "d", "date", "wm_yr_wk", "wday", "month", "year",
    "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI",
]


def join_calendar(long_df: DataFrame, calendar: DataFrame) -> DataFrame:
    """Attach date/week/season/event context; collapse the 3 SNAP columns to the one
    that matches each series' state."""
    cal = calendar.select(*_CAL_COLS)
    out = long_df.join(F.broadcast(cal), on="d", how="left")
    snap = (
        F.when(F.col("state_id") == "CA", F.col("snap_CA"))
        .when(F.col("state_id") == "TX", F.col("snap_TX"))
        .when(F.col("state_id") == "WI", F.col("snap_WI"))
    )
    return (
        out.withColumn("snap", snap)
        .withColumn("is_event", F.col("event_name_1").isNotNull().cast("int"))
        .drop("snap_CA", "snap_TX", "snap_WI")
    )


def join_prices(df: DataFrame, prices: DataFrame) -> DataFrame:
    """Attach the weekly sell_price, a has_price flag (structural-zero signal), and
    week-over-week price momentum."""
    out = df.join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    # has_price disambiguates structural zeros (item not yet stocked -> no price row)
    # from genuine zero-demand days (data-dictionary): a null price means "not sold here yet".
    out = out.withColumn("has_price", F.col("sell_price").isNotNull().cast("int"))
    # Price is weekly-constant, so lag(7) days = last week's price for this series.
    # Not leakage: prices are posted ahead of time and known at prediction time.
    prev_price = F.lag("sell_price", 7).over(Window.partitionBy("id").orderBy("d_int"))
    return out.withColumn("price_change_pct", (F.col("sell_price") - prev_price) / prev_price)


def add_exogenous_features(
    long_df: DataFrame, calendar: DataFrame, prices: DataFrame
) -> DataFrame:
    return join_prices(join_calendar(long_df, calendar), prices)


def main() -> None:
    spark: SparkSession = get_spark("A3-exogenous")
    long_df = melt_sales(load_sales_wide(spark))
    feats = add_exogenous_features(long_df, load_calendar(spark), load_sell_prices(spark))

    cols = [
        "d_int", "date", "sales", "wday", "snap", "is_event", "event_name_1",
        "sell_price", "has_price", "price_change_pct",
    ]
    print("FOODS_3_090_CA_3 around Christmas 2011 (events + SNAP + price all present):")
    (
        feats.filter(F.col("id") == "FOODS_3_090_CA_3_evaluation")
        .filter(F.col("d_int").between(325, 345))
        .orderBy("d_int")
        .select(*cols)
        .show(25, truncate=False)
    )
    spark.stop()


if __name__ == "__main__":
    main()
