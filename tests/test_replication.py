"""Tests for the C2b chain-wide replication (src/causal/replication.py).

The five-store run itself is an integration path (needs the feature store); what we unit-test is
the logic that makes the replication trustworthy: detecting each store's own cut, and the
meta-analytic pooling (inverse-variance weights, heterogeneity, and the random-effects widening).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal.did import DiDResult, detect_cut
from src.causal.replication import StoreEstimate, pool


def _est(store: str, coef: float, se: float) -> StoreEstimate:
    r = DiDResult(
        coef=coef,
        se=se,
        ci_low=coef - 1.96 * se,
        ci_high=coef + 1.96 * se,
        n_obs=100,
        n_controls=15,
    )
    return StoreEstimate(store=store, cut=pd.Timestamp("2011-08-15"), result=r)


def test_detect_cut_finds_the_step():
    """The cut week is the first >=10% price drop; a flat-then-drop path resolves to that week."""
    weeks = pd.date_range("2011-06-06", periods=20, freq="7D")
    price = np.where(weeks >= weeks[9], 2.98, 3.58)  # step down at week index 9
    weekly = pd.DataFrame({"item_id": "FOODS_3_697", "week": weeks, "price": price})
    assert detect_cut(weekly, treated="FOODS_3_697") == weeks[9]


def test_detect_cut_raises_when_no_cut():
    weeks = pd.date_range("2011-06-06", periods=10, freq="7D")
    weekly = pd.DataFrame({"item_id": "FOODS_3_697", "week": weeks, "price": 3.58})
    try:
        detect_cut(weekly, treated="FOODS_3_697")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_pool_inverse_variance_weighting():
    """A precise estimate pulls the pooled mean harder than a noisy one (hand-checked weights)."""
    estimates = [_est("A", 0.20, 0.05), _est("B", 0.60, 0.20)]
    p = pool(estimates)
    # w_A = 1/0.05^2 = 400, w_B = 1/0.20^2 = 25 -> pooled = (400*.2 + 25*.6)/425
    expected = (400 * 0.20 + 25 * 0.60) / 425
    assert abs(p.coef - expected) < 1e-9
    assert abs(p.se - np.sqrt(1 / 425)) < 1e-9
    assert p.coef < 0.30  # pulled toward the precise store A, not the midpoint 0.40


def test_pool_homogeneous_has_no_heterogeneity():
    """Identical estimates => Q~0, I^2=0, tau^2=0, and random-effects collapses to fixed."""
    estimates = [_est(s, 0.35, 0.10) for s in ("A", "B", "C", "D", "E")]
    p = pool(estimates)
    assert p.i_squared == 0.0
    assert p.tau_squared == 0.0
    assert abs(p.re_coef - p.coef) < 1e-9
    assert abs((p.re_ci_high - p.re_ci_low) - (p.ci_high - p.ci_low)) < 1e-9


def test_pool_heterogeneity_widens_random_effects_ci():
    """Disagreeing estimates => I^2>0, tau^2>0, and the RE interval is wider than the FE one."""
    estimates = [
        _est("A", 0.20, 0.05),
        _est("B", 0.22, 0.05),
        _est("C", 1.00, 0.05),  # a clear outlier, tightly estimated
        _est("D", 0.25, 0.05),
        _est("E", 0.21, 0.05),
    ]
    p = pool(estimates)
    assert p.i_squared > 0.5
    assert p.tau_squared > 0.0
    fe_width = p.ci_high - p.ci_low
    re_width = p.re_ci_high - p.re_ci_low
    assert re_width > fe_width  # random effects reflects the disagreement the FE CI hides
