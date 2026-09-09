"""Persist the causal results (D3 wiring) — the bridge from Feature C to the dashboard.

The forecast half answers "how much?"; the causal half answers "why did it move?" — the price cut
(C2/C2b DiD) and SNAP (C3 cross-state counterfactual). Those estimators need statsmodels and a few
DuckDB passes, which we don't want in the Streamlit runtime, so — exactly as with the forecast
artifact (D1/D2) — the expensive work runs here, offline, and lands as one small JSON the dashboard
reads. The app imports no statsmodels and refits nothing.

What lands (``data/processed/causal/causal.json``): for the **price cut**, the headline DiD lift +
CI, the implied elasticity, the naive (uncontrolled) jump it was disciplined down from, the
event-study coefficients (parallel-trends evidence), the placebo, and the five-store replication
with its meta-analytic pool. For **SNAP**, the naive→calendar→cross-state estimate ladder (the same
"disciplined down as the counterfactual improves" story), the main cross-state lift + CI, the two
placebos, and the clean-day corroboration. Everything the D3 panel shows is a field here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.causal.snap import SnapResult

# The causal estimators (did/replication/snap) pull in statsmodels + DuckDB feature-store passes.
# They're imported lazily inside the build functions, so importing this module for `load_causal`
# (what the deployed dashboard does) stays lightweight and pulls in no statsmodels. Tests lock it.

_REPO = Path(__file__).resolve().parents[2]
# Committed alongside the forecast bundle (see src/forecast/persist.FORECAST_DIR) — small model
# output the read-only dashboard reads, not the dataset. Enables $0 hosting.
CAUSAL_DIR = _REPO / "dashboard_data"

# The price step is a fixed property of the intervention ($3.58 → $2.98), documented in did.py.
PRICE_CHANGE_PCT = -0.168


def _ci(lo: float, hi: float) -> dict:
    return {"low": float(lo), "high": float(hi)}


def _naive_jump(panel: pd.DataFrame) -> float:
    """The treated item's raw pre→post mean-units jump, ignoring controls — the uncontrolled
    number the DiD disciplines down from (the gap is the method's value)."""
    t = panel[panel["treated"] == 1]
    pre = t.loc[t["post"] == 0, "units"].mean()
    post = t.loc[t["post"] == 1, "units"].mean()
    return float(post / pre - 1.0)


def _build_price_cut() -> dict:
    from src.causal import did, replication  # heavy (statsmodels); build-time only

    weekly = did.load_weekly_panel()
    controls = did.select_controls(weekly)
    panel = did.build_did_panel(weekly, controls)
    result = did.estimate_did(panel)
    es = did.event_study(panel)
    placebo = did.placebo_test(weekly, controls)

    lift = result.pct_effect
    lo, hi = result.pct_ci

    # C2b replication across the five chain-wide cuts + meta-analytic pool.
    estimates = replication.replicate()
    pooled = replication.pool(estimates)
    stores = [
        {
            "store": e.store,
            "cut": e.cut.date().isoformat(),
            "lift": e.result.pct_effect,
            "ci": _ci(*e.result.pct_ci),
            "n_controls": e.result.n_controls,
            "significant": bool(e.result.ci_low > 0),  # CI excludes 0 (in log space)
        }
        for e in estimates
    ]

    return {
        "treated": did.TREATED,
        "store": did.STORE,
        "cut": did.CUT.date().isoformat(),
        "price": "$3.58 → $2.98 (−16.8%)",
        "lift": lift,
        "ci": _ci(lo, hi),
        "elasticity": lift / PRICE_CHANGE_PCT,
        "naive_lift": _naive_jump(panel),
        "n_controls": result.n_controls,
        "event_study": es.to_dict(orient="records"),
        "placebo": {
            "lift": placebo.pct_effect,
            "ci": _ci(*placebo.pct_ci),
            "covers_zero": bool(placebo.ci_low <= 0 <= placebo.ci_high),
        },
        "replication": {
            "stores": stores,
            "pooled_fe": {"lift": pooled.pct_effect, "ci": _ci(*pooled.pct_ci)},
            "pooled_re": {"lift": pooled.re_pct_effect, "ci": _ci(*pooled.re_pct_ci)},
            "q": pooled.q,
            "q_pvalue": pooled.q_pvalue,
            "i_squared": pooled.i_squared,
            "n_positive": sum(e.result.pct_effect > 0 for e in estimates),
            "n_sig": sum(e.result.ci_low > 0 for e in estimates),
            "k": pooled.k,
        },
    }


def _snap_row(r: SnapResult) -> dict:
    return {
        "label": r.label,
        "lift": r.pct_effect,
        "ci": _ci(*r.pct_ci),
        "r_squared": r.r_squared,
        "covers_zero": r.covers_zero,
    }


def _build_snap() -> dict:
    from src.causal import snap  # heavy (statsmodels); build-time only

    panel = snap.load_daily_panel()
    ladder = snap.estimate_ladder(panel)
    main = next(r for r in ladder if r.label == snap.MAIN_SPEC)
    placebos = [snap.placebo(panel, c) for c in snap.CONTROL_STATES]
    clean = snap.clean_day_estimate(panel)

    return {
        "state": "CA",
        "store": snap.TREATED_STORE,
        "category": snap.CATEGORY,
        "ladder": [_snap_row(r) for r in ladder],
        "main": {"lift": main.pct_effect, "ci": _ci(*main.pct_ci), "label": main.label},
        "placebos": [_snap_row(p) for p in placebos],
        "clean_day": _snap_row(clean),
    }


def build_causal_artifact() -> dict:
    """Run C2/C2b (price-cut DiD + replication) and C3 (SNAP) and assemble the JSON artifact."""
    return {
        "price_cut": _build_price_cut(),
        "snap": _build_snap(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def persist_causal(artifact: dict | None = None) -> Path:
    """Write the causal artifact to data/processed/causal/causal.json (gitignored)."""
    artifact = artifact or build_causal_artifact()
    CAUSAL_DIR.mkdir(parents=True, exist_ok=True)
    path = CAUSAL_DIR / "causal.json"
    path.write_text(json.dumps(artifact, indent=2))
    return path


def load_causal() -> dict:
    """Read the persisted causal artifact back (the D3 panel's entry point)."""
    path = CAUSAL_DIR / "causal.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No persisted causal results at {path}. "
            "Build them first: uv run python -m src.causal.persist"
        )
    return json.loads(path.read_text())


def main() -> None:
    print("Building + persisting causal artifact (C2 DiD + C2b replication + C3 SNAP)...")
    art = build_causal_artifact()
    path = persist_causal(art)
    pc, sn = art["price_cut"], art["snap"]
    print(f"  wrote {path.relative_to(_REPO)}")
    print(f"  price cut: DiD lift {pc['lift']:+.1%} "
          f"[{pc['ci']['low']:+.1%}, {pc['ci']['high']:+.1%}]  "
          f"(naive {pc['naive_lift']:+.1%}, elasticity {pc['elasticity']:.1f})")
    rep = pc["replication"]
    print(f"  replication: {rep['n_sig']}/{rep['k']} significant, "
          f"random-effect pool {rep['pooled_re']['lift']:+.1%} "
          f"[{rep['pooled_re']['ci']['low']:+.1%}, {rep['pooled_re']['ci']['high']:+.1%}]")
    print(f"  SNAP: cross-state lift {sn['main']['lift']:+.1%} "
          f"[{sn['main']['ci']['low']:+.1%}, {sn['main']['ci']['high']:+.1%}]  "
          f"(naive {sn['ladder'][0]['lift']:+.1%})")


if __name__ == "__main__":
    main()
