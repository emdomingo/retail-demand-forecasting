# A0–E2 subfeature runthrough — session kickoff prompt

Paste the block below into a fresh session to run the pre-code subfeature walk.

---

```
Resuming the retail demand-forecasting project. Today is the A0–E2 SUBFEATURE RUNTHROUGH:
walk the whole feature tree together BEFORE writing any code. This is the map, not the build.

First read SPEC.md (feature tree + rationale) and notebooks/01-eda.ipynb §6 (the settled scope).
CLAUDE.md and the project memory are already loaded — follow the working agreement in them
(teach-before-deciding, decide in front of artefacts, prose + concise).

ALREADY SETTLED — don't re-litigate:
- Scope: forecasting on FOODS_3 @ store CA_3; DiD intervention FOODS_3_697 @ CA_3
  (chain-wide across 5 stores → replicate as robustness). SNAP → CausalImpact (no within-store control).
- Env: uv (pyproject.toml + uv.lock). Building on WINDOWS for now; Mac move later, uv.lock is the bridge.
- Metric: RMSSE/WMAPE vs seasonal-naive (period 7). Nothing committed to git yet.

FOR EACH SUBFEATURE (A0 → E2), give me:
1. What it is and how it composes into the whole (systems view).
2. The DECISION GATES — the forks, tradeoffs, reversible-vs-irreversible calls — stated explicitly
   so I can decide. Teach me where I can't evaluate the call.
3. Definition of done + where it could blow the time budget.

PUSHBACK IS REQUIRED, NOT OPTIONAL. At each decision gate, give the strongest case for the
alternative BEFORE your recommendation. Flag anything weak, over-scoped, wrong for Windows, or
that doesn't serve the anchor sentence — and argue the specific alternative. Do NOT validate to
be agreeable. But do NOT manufacture objections: if a subfeature is genuinely sound, say "sound,
no notes" and move on. I'd rather have three real disagreements than twelve nitpicks.

ARTEFACT: build docs/runthrough.md as we go — one short entry per subfeature: the decision made
at each gate + a one-line why. Fold anything that changes the plan back into SPEC.md. No code.

Go feature by feature; I'll say stream-ahead or pause per subfeature.

OPEN QUESTIONS to resolve as we go (flagged in SPEC/memory):
- MLflow footprint — keep it "log runs," not a registry?
- The optional rolling-backtest-error dashboard panel (Feature D) — in or out?
- Events: encode a handful of high-impact ones + a Christmas-closure flag, not all 30+?
- A0 runs on the Mac (moving right after this session), so Spark setup should be clean — Homebrew Java,
  no winutils/HADOOP_HOME. Confirm this rather than treat it as open.
- FIRST STEP ON THE MAC (before A0): re-fetch the data via the kaggle CLI
  (`kaggle competitions download -c m5-forecasting-accuracy`) — `data/` and `archive/` are gitignored and
  won't travel via git; copy `archive/` by hand only if you want the briefs. 
```

---

**Why the prompt is shaped this way**
- The *already-settled* block stops a cold session from reopening the scope decision closed on 2026-09-02.
- The *pushback* block forces the counter-case-first habit and explicitly permits "no notes," so pushback is real rather than either absent or manufactured.
- The *artefact* line makes the runthrough produce a durable decision log (`docs/runthrough.md`), with plan-changing calls folded back into `SPEC.md` — the single source of truth for the tree.
- The Windows/A0 question is the one genuinely live risk (Spark-on-Windows friction the plan was written macOS-first to avoid).
