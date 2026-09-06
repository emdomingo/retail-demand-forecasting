"""Chain-wide replication of the price-cut DiD (C2b).

A single DiD estimate on one store (C2) is suggestive, not conclusive: one noisy treated series
and a thin ~27-week pre-period. The chosen cut, though, was *chain-wide* — the same item
(FOODS_3_697) dropped $3.58 -> $2.98 in the same fortnight across five stores. That turns one
shaky estimate into a **replication design**: run the identical DiD in each store and ask whether
the effect reproduces. Five agreeing estimates neutralise the thin pre-period in a way one never
can — the difference between "I found an effect" and "the effect holds up."

Design decisions that must survive interview scrutiny:

1. **Per-store cut detection, not a shared date.** The chain rolled the cut out a week apart
   (CA_3 on 2011-08-08, the other four on 2011-08-15). Hardcoding one date would misdate the
   others — a pre-cut week would land in the post period and bias those estimates toward zero.
   Each store finds its own cut from its own price path (``detect_cut``).

2. **Per-store controls.** Each store builds its own matched, price-stable control set from its
   own department. A CA_1 control for a CA_1 treated item — same store, same local demand shocks.

3. **Inverse-variance meta-analysis for the pooled number.** Rather than one giant stacked
   regression, we estimate each store separately and combine with the standard fixed-effect
   meta-analytic weights (w_i = 1/SE_i^2): precise stores count more. This keeps the *store* as
   the visible unit of replication (the forest plot is the deliverable) and side-steps the
   few-cluster problem of the single-store SE — five independent estimates are their own evidence.
   Heterogeneity is reported (Cochran's Q, I^2): do the stores actually agree, or is the pooled
   mean papering over a spread?

Run: ``uv run python -m src.causal.replication``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.causal.did import (
    FIG_DIR,
    TREATED,
    DiDResult,
    build_did_panel,
    detect_cut,
    estimate_did,
    load_weekly_panel,
    select_controls,
)

# The five stores that ran the same FOODS_3_697 cut in the same fortnight (verified from prices).
STORES = ["CA_3", "CA_1", "TX_1", "TX_2", "CA_4"]


@dataclass
class StoreEstimate:
    """One store's DiD result, tagged with the store and its detected cut."""

    store: str
    cut: pd.Timestamp
    result: DiDResult


@dataclass
class PooledEstimate:
    """Meta-analytic pool of per-store DiD coefficients (log points), fixed- and random-effect.

    The fixed-effect pool assumes every store estimates the *same* true effect (differences are
    sampling noise). When stores genuinely differ (high I^2), that CI is too narrow; the
    random-effects pool (DerSimonian-Laird) adds the between-store variance tau^2 to each weight,
    widening the CI to reflect the disagreement honestly. We report both.
    """

    coef: float
    se: float
    ci_low: float
    ci_high: float
    q: float  # Cochran's Q heterogeneity statistic
    q_pvalue: float
    i_squared: float  # share of variance due to between-store heterogeneity (0 = stores agree)
    tau_squared: float  # estimated between-store variance (random-effects)
    re_coef: float  # random-effects pooled coefficient
    re_ci_low: float
    re_ci_high: float
    k: int  # number of stores pooled

    @staticmethod
    def _pct(x: float) -> float:
        return float(np.exp(x) - 1.0)

    @property
    def pct_effect(self) -> float:
        return self._pct(self.coef)

    @property
    def pct_ci(self) -> tuple[float, float]:
        return self._pct(self.ci_low), self._pct(self.ci_high)

    @property
    def re_pct_effect(self) -> float:
        return self._pct(self.re_coef)

    @property
    def re_pct_ci(self) -> tuple[float, float]:
        return self._pct(self.re_ci_low), self._pct(self.re_ci_high)


def replicate(stores: list[str] = STORES) -> list[StoreEstimate]:
    """Run the identical C2 DiD in each store, detecting each store's own cut and controls."""
    out: list[StoreEstimate] = []
    for store in stores:
        weekly = load_weekly_panel(store=store)
        cut = detect_cut(weekly, treated=TREATED)
        controls = select_controls(weekly, cut=cut)
        panel = build_did_panel(weekly, controls, cut=cut)
        out.append(StoreEstimate(store=store, cut=cut, result=estimate_did(panel)))
    return out


def pool(estimates: list[StoreEstimate]) -> PooledEstimate:
    """Fixed-effect inverse-variance meta-analysis of the per-store coefficients.

    Weights each store by its precision (1/SE^2), so a tight estimate pulls harder than a noisy
    one. Reports Cochran's Q and I^2 so a pooled mean can't hide genuine disagreement.
    """
    betas = np.array([e.result.coef for e in estimates])
    ses = np.array([e.result.se for e in estimates])
    weights = 1.0 / ses**2

    beta_pool = float(np.sum(weights * betas) / np.sum(weights))
    se_pool = float(np.sqrt(1.0 / np.sum(weights)))
    ci_low, ci_high = beta_pool - 1.96 * se_pool, beta_pool + 1.96 * se_pool

    k = len(estimates)
    # Cochran's Q ~ chi2(k-1) under homogeneity; I^2 is the share of variance that is real spread.
    q = float(np.sum(weights * (betas - beta_pool) ** 2))
    q_pvalue = float(stats.chi2.sf(q, df=k - 1)) if k > 1 else float("nan")
    i_squared = float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0

    # DerSimonian-Laird random-effects: add the between-store variance tau^2 to each weight so the
    # pooled CI widens when stores genuinely disagree, instead of trusting the too-narrow FE CI.
    c = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    tau_squared = float(max(0.0, (q - (k - 1)) / c)) if c > 0 else 0.0
    re_weights = 1.0 / (ses**2 + tau_squared)
    re_coef = float(np.sum(re_weights * betas) / np.sum(re_weights))
    re_se = float(np.sqrt(1.0 / np.sum(re_weights)))

    return PooledEstimate(
        coef=beta_pool,
        se=se_pool,
        ci_low=ci_low,
        ci_high=ci_high,
        q=q,
        q_pvalue=q_pvalue,
        i_squared=i_squared,
        tau_squared=tau_squared,
        re_coef=re_coef,
        re_ci_low=re_coef - 1.96 * re_se,
        re_ci_high=re_coef + 1.96 * re_se,
        k=k,
    )


def save_forest_plot(
    estimates: list[StoreEstimate],
    pooled: PooledEstimate,
    path: Path = FIG_DIR / "replication_forest.png",
) -> Path:
    """Forest plot: each store's % lift with CI, plus the pooled diamond at the bottom."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{e.store}  (cut {e.cut.date()})" for e in estimates] + ["Pooled (IV)"]
    effects = [e.result.pct_effect * 100 for e in estimates] + [pooled.pct_effect * 100]
    los = [e.result.pct_ci[0] * 100 for e in estimates] + [pooled.pct_ci[0] * 100]
    his = [e.result.pct_ci[1] * 100 for e in estimates] + [pooled.pct_ci[1] * 100]
    y = list(range(len(labels)))[::-1]  # top-to-bottom

    fig, ax = plt.subplots(figsize=(9, 5))
    for yi, eff, lo, hi, lab in zip(y, effects, los, his, labels, strict=True):
        is_pool = lab.startswith("Pooled")
        color = "#d62728" if is_pool else "#1f77b4"
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.5 if is_pool else 1.5)
        ax.plot(eff, yi, "D" if is_pool else "o", color=color, ms=9 if is_pool else 7)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("price-cut demand lift (%)")
    ax.set_title(f"C2b replication: {TREATED} cut ($3.58 -> $2.98) across {len(estimates)} stores")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    print(f"C2b — chain-wide replication of the {TREATED} price cut ($3.58 -> $2.98, -16.8%)\n")
    estimates = replicate()

    print(f"{'store':6} {'cut':12} {'lift':>8} {'95% CI':>22} {'controls':>9}")
    for e in estimates:
        r = e.result
        lo, hi = r.pct_ci
        print(
            f"{e.store:6} {str(e.cut.date()):12} {r.pct_effect:+7.1%} "
            f"  [{lo:+6.1%}, {hi:+6.1%}] {r.n_controls:9d}"
        )

    pooled = pool(estimates)
    plo, phi = pooled.pct_ci
    rlo, rhi = pooled.re_pct_ci
    n_positive = sum(e.result.pct_effect > 0 for e in estimates)
    n_sig = sum(e.result.ci_low > 0 for e in estimates)
    print(f"\n=== Pooled (meta-analysis, k={pooled.k}) ===")
    print(f"fixed-effect  lift {pooled.pct_effect:+.1%}  95% CI [{plo:+.1%}, {phi:+.1%}]")
    print(f"random-effect lift {pooled.re_pct_effect:+.1%}  95% CI [{rlo:+.1%}, {rhi:+.1%}]")
    print(f"agreement: {n_positive}/{pooled.k} positive, {n_sig}/{pooled.k} significant")
    print(
        f"heterogeneity: Q={pooled.q:.2f} (p={pooled.q_pvalue:.2f}), "
        f"I^2={pooled.i_squared:.0%} => quote the random-effect CI"
    )

    fig = save_forest_plot(estimates, pooled)
    print(f"\nforest plot -> {fig}")


if __name__ == "__main__":
    main()
