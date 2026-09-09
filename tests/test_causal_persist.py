"""Tests for the D3 causal-persistence plumbing. The full build refits C2/C2b/C3 (~1 min), so we
test the round-trip, the loud-failure contract, and the pure assembly helpers hermetically — not a
re-estimation, which the causal modules' own tests already cover.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.causal import persist
from src.causal.persist import _naive_jump, _snap_row, load_causal, persist_causal
from src.causal.snap import SnapResult


@pytest.fixture(autouse=True)
def _tmp_causal_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(persist, "CAUSAL_DIR", tmp_path)
    return tmp_path


def test_persist_load_roundtrips():
    art = {"price_cut": {"lift": 0.421}, "snap": {"main": {"lift": 0.107}}, "generated_at": "x"}
    path = persist_causal(art)
    assert path.exists()
    assert load_causal() == art


def test_load_missing_raises_with_guidance():
    with pytest.raises(FileNotFoundError, match="Build them first"):
        load_causal()


def test_naive_jump_is_the_uncontrolled_treated_ratio():
    panel = pd.DataFrame(
        {
            "treated": [1, 1, 1, 1, 0, 0],
            "post": [0, 0, 1, 1, 0, 1],
            "units": [10.0, 10.0, 15.0, 15.0, 99.0, 99.0],  # controls ignored
        }
    )
    # treated pre mean 10 -> post mean 15 -> +50%.
    assert _naive_jump(panel) == pytest.approx(0.5)


def test_snap_row_reads_effect_and_covers_zero():
    r = SnapResult(
        label="cross_state", coef=0.10, se=0.01,
        ci_low=0.08, ci_high=0.12, r_squared=0.93, n_obs=500,
    )
    row = _snap_row(r)
    assert row["label"] == "cross_state"
    assert row["lift"] == pytest.approx(r.pct_effect)
    assert row["ci"] == {"low": pytest.approx(r.pct_ci[0]), "high": pytest.approx(r.pct_ci[1])}
    assert row["covers_zero"] is False  # CI [0.08, 0.12] excludes 0
