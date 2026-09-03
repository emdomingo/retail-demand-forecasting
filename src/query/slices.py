"""DuckDB slice-reads over the Parquet feature store (A4, the query layer).

DuckDB owns this job (CLAUDE.md): all downstream slice-reads and aggregations over the
feature store go through it — in-process, SQL, Parquet-native with predicate/partition
pushdown. The models (Feature B) and the dashboard rollups (D2) are its two consumers.

THE discipline here: **DuckDB does not guarantee row order without ORDER BY.** Read a
series without ordering and the rows can come back scrambled, which silently corrupts any
lag/rolling logic that assumes chronological order — the same class of bug as the A1 d_int
gate. So every time-series pull in this module orders explicitly (by id, then date).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.ingest.feature_store import FEATURE_STORE_DIR


def _glob(store_dir: Path) -> str:
    # hive_partitioning=true recovers store_id (the partition column) as a real column and
    # lets DuckDB prune partitions when the WHERE clause filters on it.
    return str(store_dir / "**" / "*.parquet")


def read_store_slice(
    store_id: str,
    columns: list[str] | None = None,
    store_dir: Path = FEATURE_STORE_DIR,
) -> pd.DataFrame:
    """Return one store's full history as a pandas frame, ordered by (id, date).

    Reads only the matching partition (pruning) and only the requested columns
    (projection pushdown). The ORDER BY is non-negotiable — see module docstring.
    """
    select = "*" if columns is None else ", ".join(columns)
    sql = f"""
        SELECT {select}
        FROM read_parquet(?, hive_partitioning=true)
        WHERE store_id = ?
        ORDER BY id, date
    """
    return duckdb.execute(sql, [_glob(store_dir), store_id]).df()


def list_stores(store_dir: Path = FEATURE_STORE_DIR) -> list[str]:
    """Distinct store_ids present in the feature store (reads partition metadata only)."""
    sql = "SELECT DISTINCT store_id FROM read_parquet(?, hive_partitioning=true) ORDER BY store_id"
    return [r[0] for r in duckdb.execute(sql, [_glob(store_dir)]).fetchall()]


def main() -> None:
    stores = list_stores()
    print("stores in feature store:", stores)
    df = read_store_slice("CA_3", columns=["id", "date", "sales", "lag_7", "rmean_7", "snap"])
    print(f"\nCA_3 slice: {len(df):,} rows x {df.shape[1]} cols")
    one = df[df["id"] == "FOODS_3_090_CA_3_evaluation"].head(10)
    print("\none series, ordered by date:")
    print(one.to_string(index=False))


if __name__ == "__main__":
    main()
