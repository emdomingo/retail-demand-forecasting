"""Tests for A4: the downcast type contract and the partitioned-write -> DuckDB-read
round trip (including the ORDER BY date discipline and partition pruning)."""

from __future__ import annotations

from datetime import date

from src.ingest.feature_store import DOWNCASTS, downcast
from src.query.slices import list_stores, read_store_slice

_SPARK_TO_SQL = {"short": "smallint", "byte": "tinyint", "float": "float"}


def test_downcast_applies_tight_types(spark):
    # Build a frame with every DOWNCASTS column at its pre-cast type: ints as bigint,
    # floats as double (what Spark infers before we downcast).
    int_cols = [c for c, t in DOWNCASTS.items() if t in ("short", "byte")]
    float_cols = [c for c, t in DOWNCASTS.items() if t == "float"]
    schema = ", ".join(
        [f"{c} bigint" for c in int_cols] + [f"{c} double" for c in float_cols]
    )
    row = tuple([1] * len(int_cols) + [1.0] * len(float_cols))
    out = dict(downcast(spark.createDataFrame([row], schema)).dtypes)
    for col, target in DOWNCASTS.items():
        assert out[col] == _SPARK_TO_SQL[target], f"{col} -> {out[col]}"


def _write_store(spark, tmp_path):
    # Rows deliberately out of date order to prove the read re-sorts them.
    rows = [
        ("A_CA_1", "CA_1", date(2011, 1, 3), 5),
        ("A_CA_1", "CA_1", date(2011, 1, 1), 3),
        ("A_CA_1", "CA_1", date(2011, 1, 2), 4),
        ("B_TX_1", "TX_1", date(2011, 1, 1), 9),
    ]
    df = spark.createDataFrame(rows, "id string, store_id string, date date, sales int")
    out = tmp_path / "fs"
    df.write.mode("overwrite").partitionBy("store_id").parquet(str(out))
    return out


def test_list_stores(spark, tmp_path):
    store_dir = _write_store(spark, tmp_path)
    assert set(list_stores(store_dir)) == {"CA_1", "TX_1"}


def test_read_slice_prunes_to_one_store(spark, tmp_path):
    store_dir = _write_store(spark, tmp_path)
    df = read_store_slice("CA_1", columns=["id", "date", "sales"], store_dir=store_dir)
    assert len(df) == 3  # only the 3 CA_1 rows; TX_1 partition pruned


def test_read_slice_orders_by_date(spark, tmp_path):
    store_dir = _write_store(spark, tmp_path)
    df = read_store_slice("CA_1", columns=["id", "date", "sales"], store_dir=store_dir)
    # Written 3,1,2 -> must come back 1,2,3 (sales 3,4,5) because the read ORDER BYs date.
    assert list(df["sales"]) == [3, 4, 5]
    assert df["date"].is_monotonic_increasing
