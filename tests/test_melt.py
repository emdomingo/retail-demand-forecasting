"""Tests for the wide->long melt (A1). The melt is the load-bearing reshape every
downstream feature depends on, so its shape and ordering key are pinned here."""

from __future__ import annotations

from src.ingest.load import ID_COLS, melt_sales


def _wide(spark):
    # 2 series x 3 days. d_10 included to catch lexical-vs-numeric ordering bugs.
    rows = [
        ("FOODS_3_001_CA_1_evaluation", "FOODS_3_001", "FOODS_3", "FOODS", "CA_1", "CA",
         "5", "0", "12"),
        ("HOBBIES_1_002_TX_2_evaluation", "HOBBIES_1_002", "HOBBIES_1", "HOBBIES", "TX_2", "TX",
         "1", "3", "0"),
    ]
    cols = [*ID_COLS, "d_2", "d_10", "d_1"]  # deliberately unsorted day columns
    return spark.createDataFrame(rows, cols)


def test_melt_row_count_is_series_times_days(spark):
    long_df = melt_sales(_wide(spark))
    assert long_df.count() == 2 * 3


def test_melt_output_columns(spark):
    long_df = melt_sales(_wide(spark))
    assert set(long_df.columns) == {*ID_COLS, "d", "d_int", "sales"}


def test_sales_and_d_int_are_integers(spark):
    long_df = melt_sales(_wide(spark))
    dtypes = dict(long_df.dtypes)
    assert dtypes["sales"] == "int"
    assert dtypes["d_int"] == "int"


def test_d_int_orders_numerically_not_lexically(spark):
    # The whole reason d_int exists: ordering by the string "d" would put d_10 before d_2.
    long_df = melt_sales(_wide(spark)).filter("id = 'FOODS_3_001_CA_1_evaluation'")
    ordered = [r.d for r in long_df.orderBy("d_int").collect()]
    assert ordered == ["d_1", "d_2", "d_10"]


def test_values_map_to_correct_day(spark):
    long_df = melt_sales(_wide(spark)).filter("id = 'FOODS_3_001_CA_1_evaluation'")
    by_day = {r.d_int: r.sales for r in long_df.collect()}
    assert by_day == {1: 12, 2: 5, 10: 0}
