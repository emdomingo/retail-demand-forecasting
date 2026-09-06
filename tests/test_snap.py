"""Tests for the C3 SNAP cross-state regression counterfactual (src/causal/snap.py).

The load-from-feature-store path (``load_daily_panel``) is exercised by the module's ``main()``
as an integration run — it needs the built Parquet store. What we unit-test here is the *logic
that must be right for the estimate to be trustworthy*:

- ``_fit`` (cross-state spec) recovers a **known** implanted SNAP lift on synthetic data.
- ``_fit`` returns a null when there is **no** SNAP effect (no spurious significance).
- ``clean_day_estimate`` selects the right days (CA-only-SNAP vs no-SNAP-anywhere) and recovers
  the truth on that subset.
- ``placebo`` reads the ``ca_snap`` term off a control-state outcome and returns a null when the
  control state does not follow CA's schedule.
- ``pct_effect`` is the multiplicative reading exp(coef)-1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal.snap import (
    SnapResult,
    _fit,
    clean_day_estimate,
    placebo,
)


def _synth_panel(delta: float, n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """A daily panel whose log CA demand follows a known data-generating process:

        lca = 0.6*ltx + 0.3*lwi + delta*ca_snap + noise

    TX and WI are independent demand series sharing a common wave (so they are genuine controls),
    each on its own SNAP schedule. So the cross-state spot ``_fit`` should recover ``delta`` as the
    ca_snap coefficient. SNAP schedules are distinct blocks so CA-only days exist for the clean-day
    test. No control state's demand depends on ca_snap — the placebo truth is zero.
    """
    rng = np.random.default_rng(seed)
    wave = rng.normal(0, 0.2, n).cumsum() * 0.02  # a slow common demand wave
    ltx = 3.0 + wave + rng.normal(0, 0.1, n)
    lwi = 2.8 + wave + rng.normal(0, 0.1, n)

    day = np.arange(n) % 30
    ca_snap = ((day >= 0) & (day < 10)).astype(int)  # CA pays days 0-9
    tx_snap = ((day >= 5) & (day < 15)).astype(int)  # TX pays 5-14 (overlaps CA 5-9)
    wi_snap = ((day >= 20) & (day < 30)).astype(int)  # WI pays 20-29 (never overlaps CA)

    lca = 0.6 * ltx + 0.3 * lwi + delta * ca_snap + rng.normal(0, 0.05, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2011-01-29", periods=n, freq="D"),
            "lca": lca,
            "ltx": ltx,
            "lwi": lwi,
            "ca_snap": ca_snap,
            "tx_snap": tx_snap,
            "wi_snap": wi_snap,
            "wday": (np.arange(n) % 7) + 1,
        }
    )


def test_cross_state_recovers_known_lift():
    """A +0.10 log-point implanted SNAP effect (~+10.5%) is recovered, CI covering it."""
    delta = 0.10
    panel = _synth_panel(delta=delta, seed=1)
    res = _fit(panel, "lca ~ ca_snap + ltx + lwi + tx_snap + wi_snap + C(wday)", "cross_state")
    assert abs(res.coef - delta) < 0.03, f"recovered {res.coef:.3f}, expected {delta}"
    assert res.ci_low < res.coef < res.ci_high
    assert res.r_squared > 0.8  # cross-state controls explain most of the level


def test_null_when_no_snap_effect():
    """With delta=0 the estimate is near zero and the 95% CI covers zero (no false positive)."""
    panel = _synth_panel(delta=0.0, seed=2)
    res = _fit(panel, "lca ~ ca_snap + ltx + lwi + tx_snap + wi_snap + C(wday)", "cross_state")
    assert abs(res.coef) < 0.03
    assert res.covers_zero


def test_clean_day_selects_and_recovers():
    """The clean-day subset is exactly CA-only-SNAP + no-SNAP-anywhere days, and recovers delta."""
    delta = 0.12
    panel = _synth_panel(delta=delta, seed=3)
    res = clean_day_estimate(panel)

    ca_only = (panel["ca_snap"] == 1) & (panel["tx_snap"] == 0) & (panel["wi_snap"] == 0)
    none_on = (panel["ca_snap"] == 0) & (panel["tx_snap"] == 0) & (panel["wi_snap"] == 0)
    assert res.n_obs == int((ca_only | none_on).sum())
    assert res.n_obs < len(panel)  # overlap days were genuinely dropped
    assert abs(res.coef - delta) < 0.04, f"recovered {res.coef:.3f}, expected {delta}"


def test_placebo_returns_null():
    """CA's schedule as a fake treatment on a control state (which never depends on ca_snap in the
    DGP) returns a null — the falsification the real placebo relies on."""
    panel = _synth_panel(delta=0.15, seed=4)
    res = placebo(panel, "wi")  # WI pays days 20-29, never overlapping CA's 0-9
    assert res.label == "placebo->wi"
    assert abs(res.coef) < 0.05
    assert res.covers_zero


def test_pct_effect_is_multiplicative():
    """pct_effect is exp(coef)-1, the multiplicative reading of a log-outcome regression."""
    r = SnapResult(
        label="x", coef=np.log(1.1), se=0.01, ci_low=np.log(1.05), ci_high=np.log(1.15),
        r_squared=0.9, n_obs=100,
    )
    assert abs(r.pct_effect - 0.1) < 1e-9
    lo, hi = r.pct_ci
    assert abs(lo - 0.05) < 1e-9 and abs(hi - 0.15) < 1e-9
