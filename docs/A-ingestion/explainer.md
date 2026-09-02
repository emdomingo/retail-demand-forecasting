# Feature A — Ingestion & feature engineering — Explainer

Append-only across A0–A4. Each section explains how that subfeature works and how it
fits the whole pipeline.

---

## A0 — Repo scaffold + env + local Spark + Kaggle pull

**What A0 is:** the foundation the rest of the project stands on. No modelling, no
features — just a reproducible environment and a reproducible path to the data. The bar
is: *any machine can recreate this exactly*, because the repo commits the recipe, never
the data.

### How the pieces fit

```
uv (Python + venv + deps)  ->  pyproject.toml + uv.lock   (committed: the recipe)
Java 17 (Homebrew, system) ->  runs the Spark JVM         (installed separately from uv)
kaggle CLI (a uv dep)      ->  data/raw/*.csv             (gitignored: the payload)
```

Two deliberate splits:

1. **uv manages Python; Homebrew manages Java.** uv installs and pins the *Python*
   interpreter (3.11 here) and every Python package. It does **not** manage the JVM.
   PySpark is a Python wrapper over a Spark JVM, and that JVM needs a system Java. So
   the env is reproducible in two layers: `uv sync` gives you identical Python, and
   `brew install openjdk@17` gives you a compatible JVM. This split is why the
   Windows→macOS move is clean — the `uv.lock` carries the Python half deterministically,
   and macOS removes the `winutils`/`HADOOP_HOME` friction Spark has on Windows.

2. **The recipe is committed; the data is not.** `pyproject.toml` + `uv.lock` are the
   env recipe (committed). `data/raw/` and `data/processed/` are gitignored. The bridge
   between them is `src/ingest/fetch_data.py`, which re-derives the payload from Kaggle.

### The env, concretely (uv syntax)

- `uv init --bare --python 3.11` — create `pyproject.toml` only (no sample code), pinning
  the interpreter. uv downloads the interpreter itself; the system Python (3.9) is untouched.
- `uv add <pkg>` — resolve + install + record in `pyproject.toml` and `uv.lock` in one step.
  `uv add --dev <pkg>` puts it in the `dev` dependency-group (tooling, notebooks) so it
  isn't a runtime dependency of the pipeline.
- `uv run <cmd>` — run inside the env with no manual `activate`. This is how *everything*
  runs (`uv run python …`, `uv run pytest`, `uv run jupyter lab`).
- `uv sync` — recreate the exact locked env on a fresh machine.

Runtime deps chosen for Feature A: `pyspark` (feature store), `duckdb` (query layer),
`pandas` + `pyarrow` (interop + Parquet), `kaggle` (the pull). Modelling deps
(LightGBM, statsmodels, MLflow, Streamlit) are added when their features arrive — not
front-loaded — so the dependency graph stays legible and each addition is attributable
to a feature.

### Decision gates hit in A0

- **Python 3.11, not newer.** PySpark 4.2 targets 3.11; pinning `requires-python =
  ">=3.11,<3.12"` keeps resolution deterministic and inside the wheel-supported range.
- **pandas pinned `<3.0` — a real fork, not cosmetic.** Resolution first pulled pandas
  3.0, and PySpark 4.2 warns at runtime: *"PySpark does not yet fully support pandas >=
  3.0.0."* Spark↔pandas conversion (Arrow `toPandas`, pandas UDFs) runs through the whole
  A→D pipeline and the dashboard, so we pinned pandas to the supported `<3.0` line rather
  than risk subtle interop breakage later. This is the CLAUDE.md guardrail in action:
  argue the specifics, don't accept a default that a runtime warning flags.
- **Java 17 satisfies Spark 4.2.** Verified by actually starting a `SparkSession` and
  running a job (`spark.range(5).count()` + a `toPandas()` round-trip) — the true A0 gate
  is not "PySpark imports" but "the JVM starts and executes a job on this machine."

### The Kaggle pull (`src/ingest/fetch_data.py`)

A thin, idempotent wrapper over the Kaggle CLI: skip if the three CSVs already exist,
else download the competition zip into `data/raw/` and extract. Two prerequisites that
bite silently:

1. Credentials — a single `KAGGLE_API_TOKEN` (the newer `KGAT_...` token) kept in the
   gitignored repo-root `.env`. The script calls `load_dotenv()` to lift it into the
   environment, then `_require_credentials()` fails fast with a clear message if it's
   missing — better than an opaque auth error from deep inside the CLI. kaggle 2.2.4
   reads `KAGGLE_API_TOKEN` natively (it also accepts `~/.kaggle/access_token` or OAuth
   via `kaggle auth login`; we chose the `.env` var so the whole local config lives in
   one gitignored file alongside future secrets).
2. **You must accept the competition rules on the website once**, or the API returns
   **403** even with valid credentials. This is the classic first-run trap.

The three files it fetches: `sales_train_validation.csv` (the wide daily-sales matrix
A1 melts), `calendar.csv` (dates, events, SNAP flags — A3 exogenous features), and
`sell_prices.csv` (weekly prices — A3 price deltas + the C causal layer).
