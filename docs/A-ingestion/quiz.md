# Feature A — Ingestion & feature engineering — Quiz

Append-only across A0–A4. Self-quiz: recall, predict-the-decision, spot-the-flaw.
Answers are at the end of each subfeature's block.

---

## A0 — Repo scaffold + env + local Spark + Kaggle pull

**Recall**
1. uv manages the Python interpreter and packages. What part of the Spark stack does it
   *not* manage, and how is that part installed instead?
2. What is the difference between `uv add` and `uv add --dev`, and why does `kaggle` go in
   one group and `pytest` in the other?
3. Which two files are the committed "env recipe," and which two directories are the
   gitignored "payload"? What bridges them?

**Predict-the-decision**
4. Resolution offered pandas 3.0.5. We pinned `pandas>=2.2,<3.0` instead. What signal
   drove that, and why does it matter for *this* pipeline specifically (name the code
   paths at risk)?
5. `requires-python` is `>=3.11,<3.12` rather than just `>=3.11`. What does the upper
   bound buy us, and what could go wrong without it?
6. The A0 verification ran `spark.range(5).count()` and a `toPandas()` round-trip rather
   than just `import pyspark`. Why is that the right gate for A0?

**Spot-the-flaw**
7. A teammate says: "My `KAGGLE_API_TOKEN` in `.env` is valid, but `fetch_data.py`
   returns 403. The token must be wrong." What's the more likely cause?
8. Someone proposes committing the extracted CSVs "so the repo is self-contained and
   reviewers don't need Kaggle." Give two reasons this violates the project's design.
9. `fetch_data.py` checks `already_present()` before downloading and exits early. What
   property does this give the script, and what flag overrides it?

---

### Answers — A0

1. The **JVM (Java)**. PySpark is a Python wrapper over a Spark JVM; uv only handles
   Python. Java 17 is installed separately via Homebrew (`openjdk@17`). Reproducibility
   is therefore two-layered: `uv sync` + a compatible system Java.
2. `uv add` records a **runtime** dependency (in `[project.dependencies]`); `uv add --dev`
   records a **dev-group** dependency (tooling/notebooks, not needed to run the pipeline).
   `kaggle` is runtime (the pipeline re-fetches data); `pytest`/`ruff`/`jupyterlab` are
   dev-only.
3. Recipe: **`pyproject.toml` + `uv.lock`** (committed). Payload: **`data/raw/` +
   `data/processed/`** (gitignored). Bridge: **`src/ingest/fetch_data.py`**, which
   re-derives the payload from Kaggle. (The `.env` holding `KAGGLE_API_TOKEN` is also
   gitignored — a secret, not part of the committed recipe.)
4. Signal: PySpark 4.2's runtime warning *"does not yet fully support pandas >= 3.0.0."*
   It matters because Spark↔pandas interop runs everywhere — Arrow `toPandas()`, pandas
   UDFs in feature engineering, and the Streamlit dashboard's pandas frames. A silent
   interop bug there would be expensive and hard to trace, so we stay on the supported line.
5. The upper bound keeps dependency resolution **deterministic and inside the
   wheel-supported range** for PySpark 4.2 (which targets 3.11). Without it, a fresh
   `uv sync` on a machine with 3.12/3.13 available could resolve to an interpreter that
   lacks compatible wheels or hits untested Spark/pandas combinations.
6. `import pyspark` only proves the Python package is installed. A0's real risk is the
   **JVM starting and executing a job** on this machine (Java compatibility, native libs).
   Running an actual distributed job plus a pandas round-trip exercises the path the whole
   pipeline depends on.
7. The **competition rules haven't been accepted** on the Kaggle website. The M5 API
   returns 403 until you accept the rules once, even with perfectly valid credentials.
8. (a) The data is **large and public** — committing it bloats the repo for no benefit
   when it's one command to re-fetch. (b) The repo's contract is **reproducible from the
   recipe, not from committed files**; committing data hides whether the fetch path
   actually works and drifts from the "recipe, not payload" design. (Also: `.gitignore`
   deliberately excludes `data/`.)
9. **Idempotence** — running it repeatedly is safe and a no-op once the CSVs exist, so it
   won't re-download hundreds of MB on every invocation. `--force` overrides it to
   re-fetch.
