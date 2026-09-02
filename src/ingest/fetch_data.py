"""Fetch the M5 Forecasting - Accuracy dataset from Kaggle into data/raw/.

Reproducibility contract (see CLAUDE.md): the repo carries no data, only the means
to re-fetch it. This wraps the Kaggle CLI so the pull is one command and idempotent.

Prerequisites:
  1. A Kaggle API token in KAGGLE_API_TOKEN. Kept in the gitignored .env at the repo
     root (KAGGLE_API_TOKEN=KGAT_...); this script loads it via python-dotenv. Generate
     it at https://www.kaggle.com/settings/api ("Generate New Token" under "API").
  2. You must accept the competition rules once on the website, otherwise the API
     returns 403 even with valid credentials:
     https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules

Usage:
    uv run python -m src.ingest.fetch_data          # download + extract if missing
    uv run python -m src.ingest.fetch_data --force  # re-download even if present
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

COMPETITION = "m5-forecasting-accuracy"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
# The three CSVs A1 loads; their presence means the pull already succeeded.
EXPECTED = ("sales_train_validation.csv", "calendar.csv", "sell_prices.csv")


def already_present() -> bool:
    return all((RAW_DIR / name).exists() for name in EXPECTED)


def _require_credentials() -> None:
    """Load .env and confirm a Kaggle token is present before shelling out."""
    load_dotenv()  # repo-root .env -> os.environ (no-op if the var is already set)
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit(
            "KAGGLE_API_TOKEN not set. Add it to the repo-root .env "
            "(KAGGLE_API_TOKEN=KGAT_...); generate one at "
            "https://www.kaggle.com/settings/api"
        )


def download() -> Path:
    _require_credentials()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {COMPETITION} -> {RAW_DIR}")
    # kaggle is installed in the venv; call it as a module so it resolves inside `uv run`.
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DIR)],
        check=True,
    )
    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected {zip_path} after download; not found.")
    return zip_path


def extract(zip_path: Path) -> None:
    print(f"Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if CSVs exist")
    args = parser.parse_args(argv)

    if already_present() and not args.force:
        print(f"M5 CSVs already in {RAW_DIR}; nothing to do (use --force to re-fetch).")
        return 0

    zip_path = download()
    extract(zip_path)

    missing = [name for name in EXPECTED if not (RAW_DIR / name).exists()]
    if missing:
        print(f"WARNING: expected CSVs still missing after extract: {missing}", file=sys.stderr)
        return 1
    print(f"Done. CSVs available in {RAW_DIR}: {', '.join(EXPECTED)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
