"""Difference-in-Differences on the FOODS_3_697 @ CA_3 price cut (C2).

The forecasting half (Feature B) answers *what* volume to expect. This is the other half:
*why* it moved. On 2011-08-08 the price of FOODS_3_697 in store CA_3 stepped $3.58 -> $2.98
(-16.8%) and held. The EDA (C1, notebooks/01-eda.ipynb §3) settled this as the cleanest
datable intervention in the dataset. DiD asks: how much of the post-cut sales jump was *caused*
by the price cut, as opposed to seasonality or a chain-wide demand wave that would have lifted
the item anyway?

The identifying idea, in one line: subtract from the treated item's before->after change the
same change measured on comparable items that did *not* cut. Whatever common force (summer,
a holiday, a store-wide promo) moved everyone is differenced out; what remains is the cut.

Design decisions that must survive interview scrutiny:

1. **Weekly grain.** M5 prices are weekly, and weekly aggregation collapses the strong
   day-of-week seasonality (the B-side seasonal-naive baseline exists precisely because it is
   strong) that the DiD does not care about. Daily rows would only add noise to a level shift.

2. **Outcome = log(units) (via log1p).** A price cut acts multiplicatively on demand, so the
   estimand we want is a *percentage* lift, and the coefficient on log-units is exactly that:
   effect ~= exp(beta) - 1. Matched controls are dense sellers, so weeks at zero are rare and
   the +1 in log1p is a negligible distortion (units in the tens). Named honestly, not hidden.

3. **Two-way fixed effects (item + week).** Item fixed effects absorb the fact that the treated
   item and each control sell different absolute volumes (DiD only cares that they *move*
   together). Week fixed effects absorb everything common to all FOODS_3 in that week —
   seasonality, holidays, macro demand. The coefficient on treated x post is then the classic
   2x2 DiD estimand, net of both. This is TWFE done on a single clean event, not the
   staggered-adoption case where TWFE has known bias (that caveat belongs to the C2b stretch).

4. **Matched controls (top-K by pre-cut correlation), price-stable near the event.** Instead of
   "the whole department" (768 items, whose average is far smoother than one noisy series so
   parallel-trends is unjudgeable — the mistake the EDA caught and fixed), keep the K same-store
   /same-dept items whose *pre-cut* weekly sales co-move most with the treated item. Comparable
   level and seasonality means a post-cut divergence is signal, not an averaging artefact. Any
   item with its own >=15% price cut within +/-8 weeks of the event is dropped — a contaminated
   control biases the estimate.

5. **Balanced +/-26-week window.** History runs to 2016 but the cut is in 2011 with only ~27
   pre-weeks (the thin pre-period SPEC flags, and the reason C2b replicates across five stores).
   A long post window would let unrelated 2012-16 drift leak into the estimate, so we cap post
   symmetric to pre. The estimand is the *sustained near-term* effect, stated as such.

6. **Cluster-robust SE by item, plus event-study and placebo as the real credibility.** With
   ~16 clusters the cluster-robust CI is approximate (wild-cluster bootstrap is the rigorous
   upgrade, named not built). The identifying evidence is not the single SE: it is the
   event-study leads sitting flat at zero (parallel pre-trends) and the placebo cut in the
   pre-period returning a null.

Run: ``uv run python -m src.causal.did``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.query.slices import read_store_slice

# --- The intervention, settled in C1 (EDA §3). ---
STORE = "CA_3"
DEPT = "FOODS_3"
TREATED = "FOODS_3_697"
CUT = pd.Timestamp("2011-08-08")  # first week at the new $2.98 price (verified from the price path)

N_CONTROLS = 15  # top-K matched controls (EDA used 15, mean pre-cut corr 0.52)
PRE_WEEKS = 26  # window each side of the cut; pre is ~27 available, post capped to match
POST_WEEKS = 26
CONTAM_WIN = pd.Timedelta(weeks=8)  # a control must be price-stable within +/-8 weeks of the cut
CONTAM_DROP = -0.15  # a >=15% own price cut in that window contaminates a control
ZERO_SHARE_BAR = 0.35  # demand bar (the EDA sparseness lesson): drop near-dead candidate items

FIG_DIR = Path("docs/C-causal/figures")


@dataclass
class DiDResult:
    """A single 2x2 DiD estimate on log-units."""

    coef: float  # treated x post coefficient (log points)
    se: float  # cluster-robust standard error (clustered by item)
    ci_low: float
    ci_high: float
    n_obs: int
    n_controls: int

    @property
    def pct_effect(self) -> float:
        """Coefficient read as a multiplicative demand lift: exp(beta) - 1."""
        return float(np.exp(self.coef) - 1.0)

    @property
    def pct_ci(self) -> tuple[float, float]:
        return float(np.exp(self.ci_low) - 1.0), float(np.exp(self.ci_high) - 1.0)

    def summary(self) -> str:
        lo, hi = self.pct_ci
        return (
            f"DiD treated x post = {self.coef:+.3f} log pts  (SE {self.se:.3f}, "
            f"95% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}])\n"
            f"  => demand lift {self.pct_effect:+.1%}  (95% CI [{lo:+.1%}, {hi:+.1%}])\n"
            f"  n_obs={self.n_obs}  controls={self.n_controls}"
        )


# --------------------------------------------------------------------------- #
# Data: one store/dept slice collapsed to a weekly item panel.
# --------------------------------------------------------------------------- #
def load_weekly_panel(store: str = STORE, dept: str = DEPT) -> pd.DataFrame:
    """Weekly (item x week) panel for a store/department: units, median price, has_price.

    Reads only the store partition and the needed columns via DuckDB (the query layer),
    ordered by date, then aggregates the daily grain up to ISO weeks (Monday start).
    """
    df = read_store_slice(
        store, columns=["item_id", "dept_id", "date", "sales", "sell_price", "has_price"]
    )
    df = df[df["dept_id"] == dept].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    weekly = (
        df.groupby(["item_id", "week"])
        .agg(
            units=("sales", "sum"),
            price=("sell_price", "median"),
            has_price=("has_price", "max"),
        )
        .reset_index()
    )
    # Only weeks the item was actually on sale (a listed price) are real demand observations;
    # pre-launch weeks with no price are structural zeros, not zero demand.
    return weekly[weekly["has_price"] == 1].copy()


# --------------------------------------------------------------------------- #
# Control selection: matched, price-stable, actually-selling.
# --------------------------------------------------------------------------- #
def select_controls(
    weekly: pd.DataFrame,
    treated: str = TREATED,
    cut: pd.Timestamp = CUT,
    k: int = N_CONTROLS,
    pre_weeks: int = PRE_WEEKS,
) -> list[str]:
    """Top-K same-dept controls: pre-cut co-movement, price-stable near the cut, dense enough.

    Matching on pre-cut correlation is what makes parallel-trends credible — comparable level
    and seasonality, so a post-cut divergence is the treatment, not an averaging artefact.
    """
    pre_start = cut - pd.Timedelta(weeks=pre_weeks)
    pre = weekly[(weekly["week"] >= pre_start) & (weekly["week"] < cut)]

    treated_pre = pre[pre["item_id"] == treated].set_index("week")["units"]

    # Candidates that had their own >=15% price cut within +/-8 weeks of the event are contaminated.
    win = weekly[weekly["week"].between(cut - CONTAM_WIN, cut + CONTAM_WIN)].copy()
    win = win.sort_values(["item_id", "week"])
    win["pc"] = win.groupby("item_id")["price"].pct_change()
    contaminated = set(win.loc[win["pc"] <= CONTAM_DROP, "item_id"])

    # Demand bar over the whole panel — the EDA sparseness lesson (dead items match spuriously).
    zero_share = weekly.groupby("item_id")["units"].apply(lambda s: (s == 0).mean())
    too_sparse = set(zero_share[zero_share > ZERO_SHARE_BAR].index)

    scores: dict[str, float] = {}
    for item, g in pre.groupby("item_id"):
        if item == treated or item in contaminated or item in too_sparse:
            continue
        s = g.set_index("week")["units"].reindex(treated_pre.index)
        if s.notna().sum() < pre_weeks // 2:  # need enough overlapping pre-weeks to judge co-move
            continue
        corr = treated_pre.corr(s)
        if pd.notna(corr):
            scores[item] = corr

    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
    return ranked


# --------------------------------------------------------------------------- #
# Panel assembly: treated + controls, balanced window, DiD indicators.
# --------------------------------------------------------------------------- #
def build_did_panel(
    weekly: pd.DataFrame,
    controls: list[str],
    treated: str = TREATED,
    cut: pd.Timestamp = CUT,
    pre_weeks: int = PRE_WEEKS,
    post_weeks: int = POST_WEEKS,
) -> pd.DataFrame:
    """Balanced (item x week) panel with treated/post/log-outcome and relative-week index."""
    lo, hi = cut - pd.Timedelta(weeks=pre_weeks), cut + pd.Timedelta(weeks=post_weeks)
    keep = [treated, *controls]
    p = weekly[weekly["item_id"].isin(keep) & weekly["week"].between(lo, hi)].copy()
    p["treated"] = (p["item_id"] == treated).astype(int)
    p["post"] = (p["week"] >= cut).astype(int)
    p["log_units"] = np.log1p(p["units"])
    p["rel_week"] = ((p["week"] - cut).dt.days // 7).astype(int)  # 0 = cut week, negatives = leads
    return p


# --------------------------------------------------------------------------- #
# The three estimators: DiD point, event study, placebo.
# --------------------------------------------------------------------------- #
def estimate_did(panel: pd.DataFrame) -> DiDResult:
    """Canonical 2x2 TWFE DiD: log_units ~ item FE + week FE + treated:post, clustered by item."""
    # 'treated:post' is the interaction; the two main effects are absorbed by the fixed effects
    # (treated by item FE, post by week FE) so they are intentionally not free parameters.
    model = smf.ols(
        "log_units ~ C(item_id) + C(week) + treated:post", data=panel
    ).fit(cov_type="cluster", cov_kwds={"groups": panel["item_id"]})

    term = "treated:post"
    coef = model.params[term]
    se = model.bse[term]
    ci = model.conf_int().loc[term]
    return DiDResult(
        coef=float(coef),
        se=float(se),
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        n_obs=int(model.nobs),
        n_controls=panel.loc[panel["treated"] == 0, "item_id"].nunique(),
    )


def event_study(panel: pd.DataFrame, ref_week: int = -1) -> pd.DataFrame:
    """Dynamic DiD: a treated x (relative-week) coefficient per week, ref_week omitted as baseline.

    Leads (rel_week < 0) near zero == parallel pre-trends (the identifying assumption, shown not
    asserted). Lags (rel_week >= 0) trace how the effect builds and persists.
    """
    p = panel.copy()
    # Interact treated with each relative-week dummy; omit ref_week so coefficients read as the
    # treated-vs-control gap relative to that baseline week.
    p["rw"] = p["rel_week"].astype(int)
    weeks = sorted(w for w in p["rw"].unique() if w != ref_week)

    # Column names must be patsy-safe: a hyphen in "evt_-5" parses as subtraction, so encode the
    # sign as m/p (minus/plus) and map back to the signed week when reading coefficients out.
    def col(w: int) -> str:
        return f"evt_{'m' if w < 0 else 'p'}{abs(w)}"

    for w in weeks:
        p[col(w)] = ((p["rw"] == w) & (p["treated"] == 1)).astype(int)
    evt_terms = " + ".join(col(w) for w in weeks)
    model = smf.ols(
        f"log_units ~ C(item_id) + C(week) + {evt_terms}", data=p
    ).fit(cov_type="cluster", cov_kwds={"groups": p["item_id"]})

    # baseline week pinned to 0 (it is the omitted reference the others are measured against)
    rows = [{"rel_week": ref_week, "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0}]
    ci = model.conf_int()
    for w in weeks:
        t = col(w)
        rows.append(
            {
                "rel_week": w,
                "coef": float(model.params[t]),
                "ci_low": float(ci.loc[t, 0]),
                "ci_high": float(ci.loc[t, 1]),
            }
        )
    return pd.DataFrame(rows).sort_values("rel_week").reset_index(drop=True)


def placebo_test(
    weekly: pd.DataFrame,
    controls: list[str],
    treated: str = TREATED,
    cut: pd.Timestamp = CUT,
    pre_weeks: int = PRE_WEEKS,
) -> DiDResult:
    """Fake-cut falsification: run the same DiD on pre-period data only, with a fake cut at the
    pre-period midpoint. A null (CI covering 0) says there was no spurious treated-vs-control
    divergence before the real cut — the DiD is not picking up a pre-existing trend."""
    pre_start = cut - pd.Timedelta(weeks=pre_weeks)
    fake_cut = pre_start + (cut - pre_start) / 2  # midpoint of the pre-period
    pre = weekly[(weekly["week"] >= pre_start) & (weekly["week"] < cut)]
    panel = build_did_panel(
        pre, controls, treated=treated, cut=fake_cut, pre_weeks=pre_weeks, post_weeks=pre_weeks
    )
    return estimate_did(panel)


def save_event_study_plot(es: pd.DataFrame, path: Path = FIG_DIR / "event_study.png") -> Path:
    """Save the event-study coefficient plot (leads flat, lags = the effect)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    yerr = [es["coef"] - es["ci_low"], es["ci_high"] - es["coef"]]
    ax.errorbar(es["rel_week"], es["coef"], yerr=yerr, fmt="o-", capsize=3, color="#1f77b4")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(-0.5, color="red", ls="--", lw=1, label="price cut")
    ax.set_xlabel("weeks relative to cut")
    ax.set_ylabel("treated - control log-units gap")
    ax.set_title(f"Event study: {TREATED} @ {STORE} price cut ($3.58 -> $2.98)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    weekly = load_weekly_panel()
    controls = select_controls(weekly)
    print(f"treated: {TREATED} @ {STORE}  cut {CUT.date()}  ($3.58 -> $2.98, -16.8%)")
    print(f"matched controls (top {len(controls)}): {controls}\n")

    panel = build_did_panel(weekly, controls)
    result = estimate_did(panel)
    print("=== C2 Difference-in-Differences ===")
    print(result.summary())

    print("\n=== Event study (parallel-trends evidence) ===")
    es = event_study(panel)
    leads = es[es["rel_week"] < 0]
    print(
        f"leads (pre-cut) mean |coef| = {leads['coef'].abs().mean():.3f}  "
        f"(near zero => parallel pre-trends)"
    )
    fig_path = save_event_study_plot(es)
    print(f"event-study plot -> {fig_path}")

    print("\n=== Placebo (fake cut in the pre-period) ===")
    placebo = placebo_test(weekly, controls)
    lo, hi = placebo.pct_ci
    covers_0 = placebo.ci_low <= 0 <= placebo.ci_high
    null = "PASS (CI covers 0)" if covers_0 else "FAIL (CI excludes 0)"
    print(f"placebo effect {placebo.pct_effect:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]  -> {null}")


if __name__ == "__main__":
    main()
