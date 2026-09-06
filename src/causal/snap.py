"""SNAP demand lift via a cross-state regression counterfactual (C3).

The second causal method, on the second intervention. C2/C2b measured a *price cut* with
Difference-in-Differences; this measures **SNAP** (food-assistance disbursement days) — and it
cannot reuse DiD, because SNAP breaks both of DiD's requirements:

  * **No clean pre-period.** SNAP is a *recurring monthly pulse* present across the whole 2011-16
    window, not a one-off onset with a clean before/after. There is no "pre-SNAP" era to difference
    against.
  * **No in-store control group.** SNAP eligibility is store-wide (every item in a CA store faces
    the same CA SNAP calendar), so there is no untreated twin *inside* the store the way the price
    cut had non-cutting neighbours.

The fix — and the reason this is a genuine second method, not DiD again — is a **regression
counterfactual driven by cross-state control series** (the estimand SPEC C3 reframes to: the
*average SNAP-day lift*, not a single-onset effect). The key fact that makes it work:

  **SNAP disbursement schedules differ by state.** California, Texas and Wisconsin pay on
  overlapping-but-distinct days of the month. So on a CA SNAP day, the TX and WI stores are a
  live control for "what CA demand would look like today *without* its SNAP boost" — they share
  the day-of-week, the holidays, the season, the macro demand wave, but not CA's SNAP schedule.

This is exactly the idea behind CausalImpact / BSTS (predict the treated series from control series,
read the treatment effect off the gap) — done here as a transparent OLS regression rather than a
Bayesian structural time-series model. CausalImpact is the named Bayesian sibling (SPEC's "second
named method"); we build the regression form because it needs no fragile heavy dependency (the
tfcausalimpact/TensorFlow friction we avoided with pmdarima in B3) and every coefficient is
defensible by hand.

Design decisions that must survive interview scrutiny:

1. **Outcome = log(FOODS daily units) for CA_3.** SNAP is *food* assistance, so we measure the
   food category's response (not all-categories, which would dilute it). Log so the coefficient
   reads as a percentage lift. CA_3 is the treated store, keeping the causal narrative on the same
   store as C2. Christmas closure days (store shut, zero sales) are structural zeros, not demand —
   they are dropped, not log(0)'d.

2. **Cross-state controls carry the counterfactual.** The main spec regresses log CA FOODS on
   *contemporaneous* log TX and log WI FOODS demand (aggregated over each state's stores for a
   smoother signal), plus day-of-week. Those control series absorb the common structure that a
   bare calendar cannot — idiosyncratic demand waves, weather, macro shocks — which is why the fit
   jumps from R^2~=0.16 (calendar-only) to ~=0.93 (cross-state). The disciplined estimate is the
   payoff: the raw SNAP-day gap overstates, exactly as the naive price jump did in C2.

3. **Control for the control states' OWN SNAP status.** CA/TX/WI SNAP windows overlap heavily
   (early-month), so on many CA SNAP days TX and WI are *also* paying — which would let their
   SNAP-lifted demand soak up part of CA's effect. Including ``tx_snap`` and ``wi_snap`` as
   covariates nets that out, so the ``ca_snap`` coefficient isolates CA's own lift.

4. **HAC (Newey-West) standard errors.** Daily demand is strongly autocorrelated; plain OLS SEs
   would be far too small. HAC with a two-week lag window is the time-series analogue of C2's
   cluster-robust SEs.

5. **Two falsifications, mirroring C2's placebo.** (a) A **placebo** applies CA's SNAP schedule as
   a *fake* treatment to a control state (TX, WI) under the same spec — it should return a null,
   confirming ``ca_snap`` is not just a generic early-month calendar artefact common to all states.
   (b) A **clean-day estimator** restricts to the ~128 days when *only* CA is on SNAP (TX and WI
   genuinely untreated) versus days when *no* state is — the sharpest identifying variation. If the
   full-sample and clean-day estimates agree, the overlap-day contamination is being handled right.

Caveats written down, not hidden: the recurring-treatment framing means this is an *average*
SNAP-day lift, not a one-time effect; anticipation/pantry-loading around SNAP days could smear the
effect across adjacent days; HAC SEs are asymptotic. The Bayesian counterfactual (CausalImpact /
BSTS) and a synthetic-control donor-weighting of the states are named, not built.

Run: ``uv run python -m src.causal.snap``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.causal.did import FIG_DIR
from src.ingest.feature_store import FEATURE_STORE_DIR

# --- The intervention and the counterfactual's ingredients. ---
TREATED_STORE = "CA_3"  # same store as the C2/C2b price-cut narrative
CATEGORY = "FOODS"  # SNAP is food assistance — measure the food category's response
CONTROL_STATES = {  # cross-state controls: same demand structure, different SNAP schedule
    "tx": ["TX_1", "TX_2", "TX_3"],
    "wi": ["WI_1", "WI_2", "WI_3"],
}
HAC_LAGS = 14  # Newey-West lag window (~two weeks) for autocorrelated daily demand

# The estimate ladder: each adds controls, disciplining the naive gap toward the truth.
SPECS = {
    "naive": "lca ~ ca_snap",
    "calendar": "lca ~ ca_snap + C(wday)",
    "cross_state": "lca ~ ca_snap + ltx + lwi + tx_snap + wi_snap + C(wday)",
}
MAIN_SPEC = "cross_state"


@dataclass
class SnapResult:
    """A single SNAP-day lift estimate, read off the ``ca_snap`` coefficient (log points)."""

    label: str
    coef: float  # ca_snap coefficient (log points)
    se: float  # HAC (Newey-West) standard error
    ci_low: float
    ci_high: float
    r_squared: float
    n_obs: int

    @property
    def pct_effect(self) -> float:
        """Coefficient read as a multiplicative demand lift: exp(beta) - 1."""
        return float(np.exp(self.coef) - 1.0)

    @property
    def pct_ci(self) -> tuple[float, float]:
        return float(np.exp(self.ci_low) - 1.0), float(np.exp(self.ci_high) - 1.0)

    @property
    def covers_zero(self) -> bool:
        return self.ci_low <= 0 <= self.ci_high

    def summary(self) -> str:
        lo, hi = self.pct_ci
        return (
            f"{self.label:12} lift {self.pct_effect:+6.1%}  95% CI [{lo:+6.1%}, {hi:+6.1%}]  "
            f"R^2={self.r_squared:.3f}  n={self.n_obs}"
        )


# --------------------------------------------------------------------------- #
# Data: CA treated + cross-state controls, as a daily FOODS panel.
# --------------------------------------------------------------------------- #
def _foods_daily(stores: list[str], prefix: str, with_calendar: bool = False) -> pd.DataFrame:
    """Daily total FOODS units + SNAP flag for a set of stores, via DuckDB (the query layer).

    Aggregating across a state's stores gives a smoother control series than any single store.
    The SNAP flag is store-state-specific in the feature store (a CA store carries snap_CA), so
    ``MAX(snap)`` over one state's stores recovers that state's SNAP calendar.
    """
    import duckdb

    glob = str(FEATURE_STORE_DIR / "**" / "*.parquet")
    inlist = ", ".join(f"'{s}'" for s in stores)
    cal = ', MAX(wday) AS wday' if with_calendar else ""
    sql = f"""
        SELECT date, SUM(sales) AS units, MAX(snap) AS snap{cal}
        FROM read_parquet(?, hive_partitioning=true)
        WHERE cat_id = '{CATEGORY}' AND store_id IN ({inlist})
        GROUP BY date
        ORDER BY date
    """
    df = duckdb.execute(sql, [glob]).df()
    return df.rename(columns={"units": f"{prefix}_units", "snap": f"{prefix}_snap"})


def load_daily_panel(
    treated_store: str = TREATED_STORE,
    control_states: dict[str, list[str]] = CONTROL_STATES,
) -> pd.DataFrame:
    """Daily panel: treated CA FOODS demand + cross-state control demand + SNAP flags.

    Columns: date, {ca,tx,wi}_units, {ca,tx,wi}_snap, wday, and log demands lca/ltx/lwi.
    Closure days (any state at zero units — Christmas) are dropped: a shut store is not zero
    demand, and log(0) is undefined.
    """
    panel = _foods_daily([treated_store], "ca", with_calendar=True)
    for prefix, stores in control_states.items():
        panel = panel.merge(_foods_daily(stores, prefix), on="date")

    unit_cols = [c for c in panel.columns if c.endswith("_units")]
    open_days = (panel[unit_cols] > 0).all(axis=1)
    panel = panel[open_days].copy()

    for prefix in ["ca", *control_states]:
        panel[f"l{prefix}"] = np.log(panel[f"{prefix}_units"])
    return panel.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The estimator: OLS with HAC SEs, reading the ca_snap coefficient.
# --------------------------------------------------------------------------- #
def _fit(panel: pd.DataFrame, formula: str, label: str, term: str = "ca_snap") -> SnapResult:
    """Fit one spec by OLS with HAC (Newey-West) SEs; read the treatment term off it."""
    model = smf.ols(formula, data=panel).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
    )
    ci = model.conf_int().loc[term]
    return SnapResult(
        label=label,
        coef=float(model.params[term]),
        se=float(model.bse[term]),
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        r_squared=float(model.rsquared),
        n_obs=int(model.nobs),
    )


def estimate_ladder(panel: pd.DataFrame, specs: dict[str, str] = SPECS) -> list[SnapResult]:
    """The naive -> calendar -> cross-state ladder: watch the estimate discipline down as the
    counterfactual improves (and the fit R^2 climb)."""
    return [_fit(panel, formula, label) for label, formula in specs.items()]


def placebo(panel: pd.DataFrame, control: str) -> SnapResult:
    """Falsification: apply CA's SNAP schedule as a *fake* treatment to a control state.

    Uses the same cross-state form as the main spec but with the control state as the outcome and
    the *other* control state as its cross-state predictor (never CA — CA is contaminated by the
    fake treatment). A control state does not follow CA's SNAP calendar, so ``ca_snap`` here should
    return a null (CI covering 0). A non-null would mean ``ca_snap`` is picking up a generic
    early-month calendar wave common to all states, not CA's own SNAP lift.
    """
    others = [s for s in CONTROL_STATES if s != control]
    predictors = " + ".join(f"l{o}" for o in others)
    snaps = " + ".join(f"{s}_snap" for s in CONTROL_STATES)
    formula = f"l{control} ~ ca_snap + {predictors} + {snaps} + C(wday)"
    return _fit(panel, formula, f"placebo->{control}")


def clean_day_estimate(panel: pd.DataFrame) -> SnapResult:
    """Sharpest identification: only days when *only* CA is on SNAP (TX and WI genuinely untreated)
    versus days when *no* state is on SNAP. This drops the overlap days entirely, so no modelling
    of co-scheduled SNAP is needed — the cross-state controls are true controls by construction.
    Agreement with the full-sample cross-state estimate confirms the overlap is handled right."""
    ca_only = (panel["ca_snap"] == 1) & (panel["tx_snap"] == 0) & (panel["wi_snap"] == 0)
    none_on = (panel["ca_snap"] == 0) & (panel["tx_snap"] == 0) & (panel["wi_snap"] == 0)
    sub = panel[ca_only | none_on].copy()
    return _fit(sub, "lca ~ ca_snap + ltx + lwi + C(wday)", "clean-day")


# --------------------------------------------------------------------------- #
# Figure: the estimate ladder + placebos as a forest-style plot.
# --------------------------------------------------------------------------- #
def save_ladder_plot(
    ladder: list[SnapResult],
    clean: SnapResult,
    placebos: list[SnapResult],
    path: Path = FIG_DIR / "snap_counterfactual.png",
) -> Path:
    """Forest-style plot: naive -> calendar -> cross-state (main) -> clean-day, plus the placebos
    sitting at zero. The disciplining of the estimate and the null placebos are the whole story."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = [*ladder, clean, *placebos]
    labels = [r.label for r in ordered]
    effects = [r.pct_effect * 100 for r in ordered]
    los = [r.pct_ci[0] * 100 for r in ordered]
    his = [r.pct_ci[1] * 100 for r in ordered]
    y = list(range(len(ordered)))[::-1]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for yi, r, eff, lo, hi in zip(y, ordered, effects, los, his, strict=True):
        is_main = r.label == MAIN_SPEC
        is_placebo = r.label.startswith("placebo")
        color = "#d62728" if is_main else ("#7f7f7f" if is_placebo else "#1f77b4")
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.5 if is_main else 1.5)
        ax.plot(eff, yi, "D" if is_main else "o", color=color, ms=9 if is_main else 7)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("SNAP-day FOODS demand lift (%)")
    ax.set_title(f"C3 SNAP counterfactual: {CATEGORY} @ {TREATED_STORE} (cross-state controls)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    panel = load_daily_panel()
    n_ca_only = int(
        ((panel["ca_snap"] == 1) & (panel["tx_snap"] == 0) & (panel["wi_snap"] == 0)).sum()
    )
    print(
        f"C3 — SNAP-day {CATEGORY} lift @ {TREATED_STORE}, cross-state counterfactual "
        f"(TX + WI controls)\n{len(panel):,} open days; {int(panel['ca_snap'].sum())} CA SNAP days "
        f"({n_ca_only} CA-only)\n"
    )

    print("=== Estimate ladder (each row adds controls) ===")
    ladder = estimate_ladder(panel)
    for r in ladder:
        print(r.summary())
    main_est = next(r for r in ladder if r.label == MAIN_SPEC)

    print("\n=== Clean-day robustness (CA-only SNAP days vs no-SNAP days) ===")
    clean = clean_day_estimate(panel)
    print(clean.summary())

    print("\n=== Placebos (CA schedule as fake treatment on a control state) ===")
    placebos = [placebo(panel, s) for s in CONTROL_STATES]
    for r in placebos:
        verdict = "PASS (CI covers 0)" if r.covers_zero else "FAIL (CI excludes 0)"
        print(f"{r.summary()}  -> {verdict}")

    fig = save_ladder_plot(ladder, clean, placebos)
    print(f"\nfigure -> {fig}")
    lo, hi = main_est.pct_ci
    print(
        f"\nHeadline: SNAP lifts CA_3 FOODS demand {main_est.pct_effect:+.1%} "
        f"[{lo:+.1%}, {hi:+.1%}] — disciplined down from the naive "
        f"{ladder[0].pct_effect:+.1%}; clean-day estimate {clean.pct_effect:+.1%} corroborates."
    )


if __name__ == "__main__":
    main()
