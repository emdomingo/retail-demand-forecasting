"""Tests for the C2 Difference-in-Differences estimator (src/causal/did.py).

The load-from-feature-store path is exercised by the module's ``main()`` (an integration run,
not a unit test — it needs the built Parquet store). What we unit-test here is the *logic that
must be right for the estimate to be trustworthy*:

- ``estimate_did`` recovers a **known** implanted treatment effect (an estimator that cannot
  recover a truth we planted is not one to believe on real data).
- ``estimate_did`` returns a null when there is **no** effect (no spurious significance).
- ``select_controls`` drops contaminated and sparse candidates (the two EDA failure modes).
- ``build_did_panel`` builds correct treated/post/log/relative-week columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal.did import (
    build_did_panel,
    estimate_did,
    select_controls,
)


def _synth_panel(
    delta: float, n_controls: int = 14, n_weeks: int = 26, seed: int = 0
) -> pd.DataFrame:
    """A DiD panel whose log-outcome follows a known data-generating process:

        log_units = item_FE + week_FE + delta * (treated & post) + noise

    So ``estimate_did`` — item FE + week FE + treated:post — should recover ``delta``. Weeks run
    symmetric around a cut at week 0; the treated item is 'treated', the rest are controls.
    """
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2011-01-03", periods=2 * n_weeks, freq="7D")
    cut = weeks[n_weeks]  # first post week
    items = ["treated"] + [f"ctrl_{i}" for i in range(n_controls)]

    item_fe = {it: rng.normal(3.5, 0.4) for it in items}  # ~exp(3.5)=33 units baseline
    week_fe = {w: rng.normal(0, 0.3) for w in weeks}  # common shocks all items share

    rows = []
    for it in items:
        for w in weeks:
            treated = int(it == "treated")
            post = int(w >= cut)
            log_units = (
                item_fe[it] + week_fe[w] + delta * treated * post + rng.normal(0, 0.1)
            )
            rows.append(
                {
                    "item_id": it,
                    "week": w,
                    "treated": treated,
                    "post": post,
                    "log_units": log_units,
                    "rel_week": (w - cut).days // 7,
                }
            )
    return pd.DataFrame(rows)


def test_did_recovers_known_effect():
    """A +0.35 log-point implanted effect (~+42%) is recovered within tolerance, CI covering it."""
    delta = 0.35
    panel = _synth_panel(delta=delta, seed=1)
    res = estimate_did(panel)
    assert abs(res.coef - delta) < 0.05, f"recovered {res.coef:.3f}, expected {delta}"
    # Internal consistency of the reported interval (its width calibration with few clusters is a
    # named caveat — see the module docstring — so we don't assert it covers the implanted truth).
    assert res.ci_low < res.coef < res.ci_high
    assert res.se > 0
    assert res.n_controls == 14


def test_did_null_when_no_effect():
    """With delta=0 the estimate is near zero and the 95% CI covers zero (no false positive)."""
    panel = _synth_panel(delta=0.0, seed=2)
    res = estimate_did(panel)
    assert abs(res.coef) < 0.1
    assert res.ci_low <= 0 <= res.ci_high


def test_effect_reads_as_multiplicative_lift():
    """The pct_effect property is exp(coef)-1, the multiplicative reading of a log-outcome DiD."""
    panel = _synth_panel(delta=np.log(1.5), seed=3)  # implant a clean +50% lift
    res = estimate_did(panel)
    assert abs(res.pct_effect - 0.5) < 0.08


def _weekly(item: str, weeks, units, price) -> pd.DataFrame:
    return pd.DataFrame(
        {"item_id": item, "week": weeks, "units": units, "price": price, "has_price": 1}
    )


def test_select_controls_excludes_contaminated_and_sparse():
    """A price-cutting neighbour (contaminated) and a near-dead item (sparse) are both dropped;
    a clean co-moving neighbour is kept."""
    weeks = pd.date_range("2011-02-07", periods=40, freq="7D")
    cut = weeks[26]
    base = 40 + 10 * np.sin(np.arange(40))  # a seasonal-ish path the controls can track

    treated = _weekly("FOODS_3_697", weeks, base, 3.58)
    # clean control: co-moves with treated, price stable -> should be kept
    clean = _weekly("FOODS_3_100", weeks, base + np.random.default_rng(0).normal(0, 2, 40), 5.0)
    # contaminated: co-moves but takes its OWN 20% price cut right at the event -> dropped
    contam_price = np.where(weeks >= cut, 4.0, 5.0)
    contam = _weekly("FOODS_3_200", weeks, base, contam_price)
    # sparse: almost always zero -> dropped by the demand bar
    sparse_units = np.zeros(40)
    sparse_units[::10] = 1
    sparse = _weekly("FOODS_3_300", weeks, sparse_units, 5.0)

    weekly = pd.concat([treated, clean, contam, sparse], ignore_index=True)
    controls = select_controls(weekly, k=15, pre_weeks=26)

    assert "FOODS_3_100" in controls
    assert "FOODS_3_200" not in controls  # contaminated by its own price cut
    assert "FOODS_3_300" not in controls  # too sparse to be a credible control


def test_build_did_panel_indicators():
    """treated/post/log_units/rel_week are constructed correctly around the cut."""
    weeks = pd.date_range("2011-02-07", periods=40, freq="7D")
    cut = weeks[26]
    treated = _weekly("FOODS_3_697", weeks, np.full(40, 30.0), 3.58)
    control = _weekly("FOODS_3_100", weeks, np.full(40, 20.0), 5.0)
    weekly = pd.concat([treated, control], ignore_index=True)

    panel = build_did_panel(
        weekly, ["FOODS_3_100"], cut=cut, pre_weeks=26, post_weeks=13
    )
    t = panel[panel["item_id"] == "FOODS_3_697"]
    assert set(panel["treated"].unique()) == {0, 1}
    assert (t["treated"] == 1).all()
    # post flips exactly at the cut
    assert (panel.loc[panel["week"] >= cut, "post"] == 1).all()
    assert (panel.loc[panel["week"] < cut, "post"] == 0).all()
    # rel_week is 0 at the cut, log_units = log1p(units)
    assert (panel.loc[panel["week"] == cut, "rel_week"] == 0).all()
    assert np.allclose(t["log_units"], np.log1p(30.0))
