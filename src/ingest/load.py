"""Load the raw M5 CSVs and reshape the sales matrix wide -> long (A1).

The reshape is the point of A1: `sales_train_evaluation` is one row per series with
1,941 daily columns (`d_1 … d_1941`); melting it yields ~59M rows of (series, day, sales)
— the long, tidy grain every downstream feature (A2 lags, A3 joins) is built on.

We use the **evaluation** phase (d_1..d_1941), not validation (d_1..d_1913): the
competition is over, so the extra 28 days are just more history to model (data-dictionary).

Run as a demo:
    uv run python -m src.ingest.load
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.ingest.spark import get_spark

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Identifier (non-day) columns of the wide sales table — everything else is a d_N column.
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _read_csv(spark: SparkSession, name: str, infer: bool = True) -> DataFrame:
    return spark.read.csv(str(RAW_DIR / name), header=True, inferSchema=infer)


def load_sales_wide(spark: SparkSession, phase: str = "evaluation") -> DataFrame:
    """Load the wide sales matrix. Day columns are kept as strings and cast in the melt."""
    # inferSchema=False: 1,941 day columns, all counts — cast once after the melt instead
    # of paying an inference pass over a 1,947-column frame.
    return _read_csv(spark, f"sales_train_{phase}.csv", infer=False)


def load_calendar(spark: SparkSession) -> DataFrame:
    return _read_csv(spark, "calendar.csv")


def load_sell_prices(spark: SparkSession) -> DataFrame:
    return _read_csv(spark, "sell_prices.csv")


def melt_sales(sales_wide: DataFrame) -> DataFrame:
    """Reshape wide (d_1..d_N as columns) -> long (one row per series-day).

    Output columns: ID_COLS + `d` (e.g. "d_5"), `d_int` (5, the ordering key), `sales` (int).
    `d_int` exists because `d` sorts lexically wrong ("d_10" < "d_2"); every time-ordered
    op downstream must order by `d_int` (or the real date), never by the `d` string.
    """
    day_cols = [c for c in sales_wide.columns if c not in ID_COLS]
    long_df = sales_wide.melt(
        ids=ID_COLS,
        values=day_cols,
        variableColumnName="d",
        valueColumnName="sales",
    )
    return (
        long_df.withColumn("d_int", F.regexp_extract("d", r"^d_(\d+)$", 1).cast("int"))
        .withColumn("sales", F.col("sales").cast("int"))
    )


def main() -> None:
    spark = get_spark("A1-load-melt")
    sales_wide = load_sales_wide(spark)
    calendar = load_calendar(spark)
    prices = load_sell_prices(spark)

    n_series = sales_wide.count()
    n_days = len([c for c in sales_wide.columns if c not in ID_COLS])
    print(f"wide sales: {n_series:,} series x {n_days:,} days")
    print(f"calendar:   {calendar.count():,} rows | sell_prices: {prices.count():,} rows")

    long_df = melt_sales(sales_wide)
    print("\nlong schema:")
    long_df.printSchema()
    n_long = long_df.count()
    print(f"long rows: {n_long:,}  (expected {n_series * n_days:,})")
    print("\nsample (one series, ordered by d_int):")
    (
        long_df.filter(F.col("id") == "FOODS_3_090_CA_3_evaluation")
        .orderBy("d_int")
        .show(5)
    )
    spark.stop()


if __name__ == "__main__":
    main()
