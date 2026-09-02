"""Spark session builder — the single entry point for a local session (A1–A4).

Local Spark is a feature-store engine here (CLAUDE.md), not a cluster: one JVM,
`local[*]` across all cores. Everything downstream calls `get_spark()` so session
config lives in one place.
"""

from __future__ import annotations

from pyspark.sql import SparkSession


def get_spark(app_name: str = "retail-demand-forecasting") -> SparkSession:
    """Return a local[*] SparkSession, reusing the active one if it exists.

    Arrow is enabled for fast Spark<->pandas conversion (used at the model/dashboard
    boundary, never to pull the full 59M-row frame). Shuffle partitions are trimmed
    from the 200 default to something sane for a single machine.
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
