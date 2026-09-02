"""Shared pytest fixtures. A session-scoped local Spark keeps the JVM startup cost
(a few seconds) out of every individual test."""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = (
        SparkSession.builder.appName("tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()
