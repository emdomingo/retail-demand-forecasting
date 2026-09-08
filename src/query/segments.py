"""Segment-level error rollups over the persisted forecast (D2, the query layer).

Aggregate accuracy hides the items that matter for planning: one headline WMAPE can look fine
while a high-volume department quietly drives most of the absolute error (CLAUDE.md guardrail —
segment-level error matters). This module breaks the held-out-origin error down by category,
department, and volume tier, and — crucially — reports each segment's **share of units** alongside
its error, so you can see *where the misses actually land*, not just the percentages.

DuckDB owns the aggregation (CLAUDE.md: hierarchical/segment rollups go through DuckDB). It reads
the persisted forecast Parquet directly (columns id, dept_id, cat_id, sales, yhat, scale), so this
is a pure read over the same artifact the dashboard shows — no model, no feature-store round-trip.

Two metrics, aggregated the way each is defined (matching B1):
- **WMAPE** — pooled Σ|sales−yhat| / Σ|sales| over the segment's rows. Rolls up cleanly.
- **RMSSE** — per-series (each scaled by its own history), then averaged within the segment. The
  interval ``scale`` persisted with the forecast is √(naive_scale), so per-series RMSSE is
  RMSE_series / scale — reconstructable from the artifact without re-reading training history.
"""

from __future__ import annotations

import duckdb
import pandas as pd


def _segment_sql(dim: str) -> str:
    """WMAPE + mean-per-series RMSSE + unit share, grouped by one dimension column. Per-series
    RMSSE is computed in an inner query (RMSE_series / scale), then averaged over the segment;
    WMAPE and unit totals pool the raw rows."""
    return f"""
        WITH per_series AS (
            SELECT
                {dim} AS segment,
                id,
                sqrt(avg(pow(sales - yhat, 2))) / any_value(scale) AS rmsse,
                sum(abs(sales - yhat)) AS abs_err,
                sum(abs(sales)) AS abs_actual,
                sum(sales) AS units
            FROM fc
            GROUP BY {dim}, id
        )
        SELECT
            segment,
            count(*) AS n_series,
            sum(units) AS units,
            avg(rmsse) AS rmsse,
            sum(abs_err) / nullif(sum(abs_actual), 0) AS wmape
        FROM per_series
        GROUP BY segment
        ORDER BY units DESC
    """


def segment_error(forecast: pd.DataFrame, by: str = "cat_id") -> pd.DataFrame:
    """Per-segment error table for one dimension (``cat_id`` or ``dept_id``). Adds ``unit_share``
    — the segment's fraction of total forecast-window units — so a low-error-but-huge segment and a
    high-error-but-tiny one are visually distinguishable (the point of the panel)."""
    if by not in {"cat_id", "dept_id"}:
        raise ValueError("by must be 'cat_id' or 'dept_id'")
    con = duckdb.connect()
    con.register("fc", forecast)
    out = con.execute(_segment_sql(by)).df()
    con.close()
    out["unit_share"] = out["units"] / out["units"].sum()
    return out


def volume_tier_error(forecast: pd.DataFrame, n_tiers: int = 4) -> pd.DataFrame:
    """Error by demand-volume tier. Series are ranked by mean forecast-window units and cut into
    ``n_tiers`` equal-count buckets (quartiles by default), then the same WMAPE/RMSSE/unit-share
    rollup runs per tier. This is the sharpest view of the guardrail: it shows whether the fast,
    high-value items (the ones that dominate planning) are forecast better or worse than the long
    tail of slow movers."""
    per_series_units = forecast.groupby("id")["sales"].mean()
    # qcut can collapse tiers when many series share a volume (ties); fall back to rank-based cut.
    try:
        tier = pd.qcut(per_series_units, n_tiers, labels=False, duplicates="drop")
    except ValueError:
        tier = pd.Series(0, index=per_series_units.index)
    labels = {i: f"Q{i + 1} ({'low' if i == 0 else 'high' if i == tier.max() else 'mid'})"
              for i in range(int(tier.max()) + 1)}
    tier_name = tier.map(labels)
    fc = forecast.merge(
        tier_name.rename("tier").reset_index(), on="id", how="left"
    )
    con = duckdb.connect()
    con.register("fc", fc)
    out = con.execute(_segment_sql("tier")).df()
    con.close()
    out["unit_share"] = out["units"] / out["units"].sum()
    return out.sort_values("segment", ignore_index=True)


def main() -> None:
    from src.forecast.persist import load_forecast

    fc = load_forecast("CA_3").forecast
    print("By category:")
    print(segment_error(fc, "cat_id").to_string(index=False))
    print("\nBy department:")
    print(segment_error(fc, "dept_id").to_string(index=False))
    print("\nBy volume tier:")
    print(volume_tier_error(fc).to_string(index=False))


if __name__ == "__main__":
    main()
