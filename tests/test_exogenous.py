"""Tests for A3 exogenous joins. Focus: the SNAP-by-state collapse, the structural-zero
price flag, week-over-week price momentum, and that the joins preserve the grain."""

from __future__ import annotations

from src.ingest.exogenous import add_exogenous_features
from src.ingest.load import ID_COLS


def _sales(spark, rows):
    # rows: (id, item_id, store_id, state_id, d_int, sales) — dept/cat filled generically.
    made = [
        (rid, item, "D", "C", store, state, f"d_{di}", di, s)
        for (rid, item, store, state, di, s) in rows
    ]
    return spark.createDataFrame(made, [*ID_COLS, "d", "d_int", "sales"])


def _calendar(spark, rows):
    # rows: (d_int, wm_yr_wk, event_name_1, snap_CA, snap_TX, snap_WI)
    from datetime import date, timedelta

    base = date(2011, 1, 29)
    made = [
        (f"d_{di}", base + timedelta(days=di - 1), wk, 1, 1, 2011,
         ev, ("Cultural" if ev else None), None, None, ca, tx, wi)
        for (di, wk, ev, ca, tx, wi) in rows
    ]
    # Explicit schema: event columns can be entirely None in a small fixture, which
    # defeats type inference (CANNOT_DETERMINE_TYPE).
    schema = (
        "d string, date date, wm_yr_wk int, wday int, month int, year int, "
        "event_name_1 string, event_type_1 string, event_name_2 string, "
        "event_type_2 string, snap_CA int, snap_TX int, snap_WI int"
    )
    return spark.createDataFrame(made, schema)


def _prices(spark, rows):
    # rows: (store_id, item_id, wm_yr_wk, sell_price). Explicit schema so an empty
    # price table (item never stocked) still constructs.
    schema = "store_id string, item_id string, wm_yr_wk int, sell_price double"
    return spark.createDataFrame(list(rows), schema)


def test_snap_reads_the_series_state(spark):
    # One calendar day where CA=1, TX=0, WI=0. A CA series must see snap=1, a TX series 0.
    sales = _sales(spark, [
        ("A_F_CA_1_evaluation", "F", "CA_1", "CA", 1, 3),
        ("B_F_TX_1_evaluation", "F", "TX_1", "TX", 1, 3),
    ])
    cal = _calendar(spark, [(1, 11101, None, 1, 0, 0)])
    out = add_exogenous_features(sales, cal, _prices(spark, [("CA_1", "F", 11101, 5.0)]))
    snap = {r.state_id: r.snap for r in out.collect()}
    assert snap == {"CA": 1, "TX": 0}


def test_is_event_flag(spark):
    sales = _sales(spark, [
        ("A_F_CA_1_evaluation", "F", "CA_1", "CA", 1, 1),
        ("A_F_CA_1_evaluation", "F", "CA_1", "CA", 2, 1),
    ])
    cal = _calendar(spark, [(1, 11101, "Easter", 0, 0, 0), (2, 11101, None, 0, 0, 0)])
    out = add_exogenous_features(sales, cal, _prices(spark, []))
    ev = {r.d_int: r.is_event for r in out.collect()}
    assert ev == {1: 1, 2: 0}


def test_has_price_marks_structural_zero(spark):
    # Week 11101 has a price row; week 11102 does not (item not stocked that week).
    sales = _sales(spark, [
        ("A_F_CA_1_evaluation", "F", "CA_1", "CA", 1, 0),
        ("A_F_CA_1_evaluation", "F", "CA_1", "CA", 8, 0),
    ])
    cal = _calendar(spark, [(1, 11101, None, 0, 0, 0), (8, 11102, None, 0, 0, 0)])
    out = add_exogenous_features(sales, cal, _prices(spark, [("CA_1", "F", 11101, 5.0)]))
    by_day = {r.d_int: (r.sell_price, r.has_price) for r in out.collect()}
    assert by_day[1] == (5.0, 1)
    assert by_day[8] == (None, 0)


def test_week_over_week_price_change(spark):
    # 14 days: price 10.0 for week 1, 8.0 for week 2. lag(7) at d_int=8 reads day 1.
    sales = _sales(spark, [
        ("A_F_CA_1_evaluation", "F", "CA_1", "CA", di, 1) for di in range(1, 15)
    ])
    cal_rows = [(di, 11101 if di <= 7 else 11102, None, 0, 0, 0) for di in range(1, 15)]
    price_rows = [("CA_1", "F", 11101, 10.0), ("CA_1", "F", 11102, 8.0)]
    out = add_exogenous_features(sales, _calendar(spark, cal_rows), _prices(spark, price_rows))
    pct = {r.d_int: r.price_change_pct for r in out.collect()}
    assert pct[8] == -0.2  # (8 - 10) / 10
    assert pct[7] is None  # no price 7 days before day 7


def test_joins_preserve_row_grain(spark):
    sales = _sales(spark, [
        ("A_F_CA_1_evaluation", "F", "CA_1", "CA", di, 1) for di in range(1, 8)
    ])
    cal = _calendar(spark, [(di, 11101, None, 0, 0, 0) for di in range(1, 8)])
    out = add_exogenous_features(sales, cal, _prices(spark, [("CA_1", "F", 11101, 5.0)]))
    assert out.count() == 7  # left joins must not multiply the one-row-per-series-day grain
