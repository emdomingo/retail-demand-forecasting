"""Spark session builder — the single entry point for a local session (A1–A4).

Local Spark is a feature-store engine here (CLAUDE.md), not a cluster: one JVM,
`local[*]` across all cores. Everything downstream calls `get_spark()` so session
config lives in one place.
"""

from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark(
    app_name: str = "retail-demand-forecasting",
    driver_memory: str | None = None,
    shuffle_partitions: int = 16,
) -> SparkSession:
    """Return a local[*] SparkSession, reusing the active one if it exists.

    Arrow is enabled for fast Spark<->pandas conversion (used at the model/dashboard
    boundary, never to pull the full 59M-row frame). Shuffle partitions default to a
    laptop-sane 16 (vs Spark's cluster default of 200); heavy jobs bump it.

    ``driver_memory`` (e.g. "8g") raises the local driver heap. In local mode the driver
    *is* the executor, and its heap must be sized before the JVM launches — a builder
    ``.config("spark.driver.memory", ...)`` is read too late. So we set it through
    ``PYSPARK_SUBMIT_ARGS`` before the first ``getOrCreate``. This only takes effect if no
    session/JVM exists yet, which is the case in a fresh ``uv run python -m ...`` process.
    """
    if driver_memory:
        os.environ["PYSPARK_SUBMIT_ARGS"] = f"--driver-memory {driver_memory} pyspark-shell"
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
