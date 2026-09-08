"""Tests for the D2 segment rollups. The load-bearing properties: WMAPE pools correctly per
segment, RMSSE is reconstructed from the persisted scale (RMSE_series / scale), unit shares sum to
one, and the volume tiers partition the series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.query.segments import segment_error, volume_tier_error


def _forecast() -> pd.DataFrame:
    """Two categories, two series each, constant per-row error so the metrics are hand-checkable.
    FOODS series each sell 10/day with a +2 forecast error; HOBBIES sell 1/day with a +0.5 error.
    """
    rows = []
    for cat, sales, err, scale in [("FOODS", 10.0, 2.0, 5.0), ("HOBBIES", 1.0, 0.5, 1.0)]:
        for s in range(2):
            for _d in range(28):
                rows.append(
                    {
                        "id": f"{cat}_{s}",
                        "dept_id": f"{cat}_1",
                        "cat_id": cat,
                        "sales": sales,
                        "yhat": sales + err,
                        "scale": scale,
                    }
                )
    return pd.DataFrame(rows)


def test_wmape_pools_per_segment():
    seg = segment_error(_forecast(), "cat_id").set_index("segment")
    # FOODS: |err|=2 on sales=10 -> WMAPE 0.2; HOBBIES: 0.5 on 1 -> 0.5.
    assert seg.loc["FOODS", "wmape"] == pytest.approx(0.2)
    assert seg.loc["HOBBIES", "wmape"] == pytest.approx(0.5)


def test_rmsse_uses_the_persisted_scale():
    seg = segment_error(_forecast(), "cat_id").set_index("segment")
    # Constant error 2, scale 5 -> RMSE=2, RMSSE=2/5=0.4.
    assert seg.loc["FOODS", "rmsse"] == pytest.approx(0.4)
    # error 0.5, scale 1 -> 0.5.
    assert seg.loc["HOBBIES", "rmsse"] == pytest.approx(0.5)


def test_unit_share_sums_to_one_and_counts_series():
    seg = segment_error(_forecast(), "cat_id")
    assert seg["unit_share"].sum() == pytest.approx(1.0)
    assert set(seg["n_series"]) == {2}
    # FOODS (10/day x 2 series x 28) dominates HOBBIES (1/day) in unit share.
    s = seg.set_index("segment")["unit_share"]
    assert s["FOODS"] > s["HOBBIES"]


def test_volume_tiers_partition_series():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        base = float(i)  # 40 distinct volume levels -> clean quartiles
        for _d in range(28):
            rows.append(
                {
                    "id": f"S{i}",
                    "dept_id": "D1",
                    "cat_id": "C",
                    "sales": base + rng.normal(0, 0.1),
                    "yhat": base,
                    "scale": 1.0,
                }
            )
    tiers = volume_tier_error(pd.DataFrame(rows), n_tiers=4)
    assert len(tiers) == 4
    assert tiers["n_series"].sum() == 40
    assert tiers["unit_share"].sum() == pytest.approx(1.0)
