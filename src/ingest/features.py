"""Time-series features on the long sales frame (A2): lags + trailing rolling means.

THE gate for this subfeature is leakage: a feature computed for day ``t`` must depend
only on sales strictly *before* ``t``. Get this wrong and the backtest looks brilliant
and the live forecast collapses. Two rules enforce it here:

  1. Lags use ``F.lag(sales, k)`` with k >= 1 — day ``t`` reads day ``t-k``.
  2. Rolling means use a window frame that **ends at -1 (yesterday), never 0 (today)** —
     ``rangeBetween(-N, -1)`` averages the N days before ``t``, excluding ``t`` itself.

Why ``rangeBetween`` over ``d_int`` and not ``rowsBetween``: rangeBetween is defined in
*day units* (the values of d_int), so "the last 7 days" means 7 calendar days even if a
series had gaps. M5's wide melt is dense (every series has all 1,941 days), so here the
two coincide — but the value-based frame states the intent and is robust if that ever
changes. Ordering is always by ``d_int``, never the ``d`` string (see A1).

Run as a demo (one series, features aligned so leakage-safety is visible):
    uv run python -m src.ingest.features
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.ingest.load import load_sales_wide, melt_sales
from src.ingest.spark import get_spark

LAGS = (7, 28)
ROLL_WINDOWS = (7, 28)


def add_time_features(long_df: DataFrame) -> DataFrame:
    """Append lag_{k} and rmean_{w} columns, partitioned per series, ordered by day.

    Warm-up: lag_k is null for the first k days of a series; rmean_w averages whatever
    prior days exist (a partial window near the start). The backtest harness (B1) owns
    how warm-up rows are handled — A2's job is only that the features never peek ahead.
    """
    order = Window.partitionBy("id").orderBy("d_int")

    out = long_df
    for k in LAGS:
        out = out.withColumn(f"lag_{k}", F.lag("sales", k).over(order))

    for w in ROLL_WINDOWS:
        # rangeBetween(-w, -1): the w days before today, today EXCLUDED -> no leakage.
        frame = order.rangeBetween(-w, -1)
        out = out.withColumn(f"rmean_{w}", F.avg("sales").over(frame))

    return out


def main() -> None:
    spark = get_spark("A2-time-features")
    long_df = melt_sales(load_sales_wide(spark))
    feats = add_time_features(long_df)

    cols = ["d_int", "sales", *[f"lag_{k}" for k in LAGS], *[f"rmean_{w}" for w in ROLL_WINDOWS]]
    print("one series, ordered by d_int (note null warm-up, and rmean never sees 'today'):")
    (
        feats.filter(F.col("id") == "FOODS_3_090_CA_3_evaluation")
        .orderBy("d_int")
        .select(*cols)
        .show(35)
    )
    spark.stop()


if __name__ == "__main__":
    main()
