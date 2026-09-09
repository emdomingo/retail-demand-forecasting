"""Guard test for the D4 deploy property: importing the dashboard app must pull in NONE of the
heavy modelling deps (pyspark, lightgbm, mlflow, statsmodels). The app is a pure reader over the
committed dashboard_data/ bundle — that's what lets it deploy to $0 hosting where those libs (and
Java for Spark) aren't installed. Run in a clean subprocess, because the rest of the test suite
imports those libs into this process, which would defeat an in-process sys.modules check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

HEAVY = ("pyspark", "lightgbm", "mlflow", "statsmodels")


def test_dashboard_app_imports_no_heavy_modelling_deps():
    code = (
        "import sys; import importlib;"
        "importlib.import_module('src.dashboard.app');"
        f"loaded=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(','.join(loaded))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    assert loaded == "", f"dashboard app pulled in heavy deps: {loaded}"
