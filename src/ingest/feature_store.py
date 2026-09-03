"""Assemble, downcast, and persist the partitioned Parquet feature store (A4).

This is where A1-A3 become a durable artefact. The full frame (melt -> time features ->
exogenous) is built once across all ~30,490 series, downcast to tight types, and written
as Parquet partitioned by store_id. Downstream (Feature B models, the dashboard) never
re-runs Spark — they read the slice they need back through DuckDB (see src/query).

Two persistence decisions ("what to persist" gate):

  * Downcasting — 59M rows makes column types a real cost. Spark defaults counts to
    bigint (8 bytes) and averages to double (8 bytes); most of our columns need far less
    (sales max is 763 -> int16; flags are 0/1 -> int8; prices fit float32). Casting before
    the write roughly halves the on-disk footprint and the bytes every read must scan.
  * Partition by store_id — 10 balanced partitions (each row belongs to exactly one store).
    The scoped model reads one store via **partition pruning** (DuckDB/Spark skip the other
    9 directories entirely), which is the concrete payoff of "population-scale store,
    scoped model". store_id is not row-unique (the key is (id, d_int)) — clean low-cardinality
    grouping, not uniqueness, is what a partition key wants.

Run:
    uv run python -m src.ingest.feature_store          # build + overwrite the store
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.ingest.exogenous import add_exogenous_features
from src.ingest.features import add_time_features
from src.ingest.load import load_calendar, load_sales_wide, load_sell_prices, melt_sales
from src.ingest.spark import get_spark

FEATURE_STORE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "feature_store"

# Columns kept in the store (drops the redundant `d` string — d_int + date carry it — and
# the rare event_2 pair already dropped in A3). store_id is listed but becomes the partition
# directory, so it lives in the path, not the data files.
STORE_COLUMNS = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "date", "d_int", "wm_yr_wk", "wday", "month", "year",
    "sales", "lag_7", "lag_28", "rmean_7", "rmean_28",
    "event_name_1", "event_type_1", "is_event", "snap",
    "sell_price", "has_price", "price_change_pct",
]

# Tight target types. Anything not listed keeps its natural type (strings, date, wm_yr_wk int).
DOWNCASTS = {
    "sales": "short", "lag_7": "short", "lag_28": "short", "d_int": "short",
    "wday": "byte", "month": "byte", "year": "short",
    "is_event": "byte", "snap": "byte", "has_price": "byte",
    "rmean_7": "float", "rmean_28": "float",
    "sell_price": "float", "price_change_pct": "float",
}


def assemble(spark: SparkSession) -> DataFrame:
    """Run the full A1->A3 pipeline into one long feature frame."""
    long_df = melt_sales(load_sales_wide(spark))
    long_df = add_time_features(long_df)
    return add_exogenous_features(long_df, load_calendar(spark), load_sell_prices(spark))


def downcast(df: DataFrame) -> DataFrame:
    for col, typ in DOWNCASTS.items():
        df = df.withColumn(col, F.col(col).cast(typ))
    return df


def build_feature_store(spark: SparkSession, out_dir: Path = FEATURE_STORE_DIR) -> Path:
    df = downcast(assemble(spark)).select(*STORE_COLUMNS)
    (
        df.write.mode("overwrite")
        .partitionBy("store_id")
        .parquet(str(out_dir))
    )
    return out_dir


def main() -> None:
    # The window sorts + partitioned write over 59M rows need more than the default local
    # heap; 8g on a 16g machine, spread across more shuffle partitions to cut per-task memory.
    spark = get_spark("A4-feature-store", driver_memory="8g", shuffle_partitions=48)
    out = build_feature_store(spark)
    print(f"feature store written -> {out}")
    # Read straight back through Spark just to confirm it round-trips and prune works.
    back = spark.read.parquet(str(out))
    n_stores = back.select("store_id").distinct().count()
    print(f"rows: {back.count():,} | partitions (stores): {n_stores}")
    print("dtypes:", dict(back.dtypes))
    spark.stop()


if __name__ == "__main__":
    main()
