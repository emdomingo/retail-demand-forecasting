# Meta-learning — how to use Claude well & learn well

A living companion to [study-plan.md](study-plan.md). That file is *what* to learn; this is *how* to
work and study so it sticks. Notes are grounded in real moments from this project — keep adding as
patterns show up. A subgoal of the project is getting better at both.

---

## How to use Claude better

1. **Decide in front of artefacts, not in the abstract.** The scope call came from *running the EDA cells
   and reading real numbers* (zero-shares, the candidate screener, the parallel-trend plot) — not a "how
   granular?" debate. When facing a choice, ask for the concrete thing (a table, two side-by-side options,
   a plot), then decide. Abstract questions get weak answers.
2. **Ask for the mechanism, not the verdict.** "Tell me how MAPE works, I'll infer why" produced
   defensible understanding instead of trust. Convert "is X right?" → "explain how X works so I can judge."
3. **Make it show the trap, not just the clean answer.** The penny-price histogram and the rejected
   candidates (FOODS_3_822 stockout, FOODS_2_227 dead item) taught more than a clean first hit. Ask "show
   me why the naive version is wrong," not only "give me the filtered result."
4. **Verify it — it errs confidently.** It predicted a "spike near −1.0"; the data showed a *tail*. Treat
   outputs as drafts to interrogate, especially confident predictions.
5. **Name your uncertainty to get targeted help.** "I don't think this is right, but am I close?" got a
   precise fix (DiD = parallel *trends*, not equal levels) instead of a generic re-lecture.
6. **Keystone habit: refuse to rubber-stamp.** "I don't have the knowledge to have an opinion — explain it"
   is why the session produced understanding, not just code. Protect this one. See [[working-style]].

## How to learn better

1. **Retrieval beats recognition.** Following an explanation live ≠ knowing it — the parallel-trends
   misconception only surfaced under active recall (the quiz). Re-quiz **cold and spaced**: next session,
   before re-reading.
2. **Track your own misconceptions — highest-value targets.** "Parallel trends = trends, not levels" is
   worth more than ten things you got right. Keep a running misconception log; that's the real study list.
3. **Study the rejection trail on purpose.** Ask "why did this fail?" — failed candidates carried more
   signal than the winner, and double as portfolio material.
4. **Anchor every concept to the pushback it buys.** `study-plan.md` already does this; learning bounded by
   "what argument does this let me win?" is motivated and finite — better than open-ended mastery.
5. **Teach-back per subfeature.** Quiz answers *are* teach-backs; the gaps show where understanding is thin.
   Honour the per-feature quiz cadence even when short on time.

## Depth control — managing overwhelm

Overwhelm is itself a rubber-stamp risk: output too dense to process gets accepted by exhaustion — the exact
failure "refuse to rubber-stamp" was meant to prevent. Managing density *serves* judge-ability; it doesn't
oppose it. The goal is the right *amount and shape* to judge, not maximum information. Levers:

1. **Lead with the decision, layer the rest** — headline + one-line why first; the full mechanism on request,
   not pre-loaded. Judge the claim, then dive only where you doubt it.
2. **Triage depth to stakes** — go deep on load-bearing / irreversible calls (metric choice, leakage, DiD
   identification); accept routine, reversible ones fast. Rubber-stamping a low-stakes call is fine.
3. **One gate at a time** — sequence dense threads instead of running them in parallel.
4. **Hold the depth dial** — say "3-line it" / "just the call" to compress, "expand" on the one piece worth
   scrutinising. This is your standing move; it puts density under your control, not Claude's.

---

## Next session
Before reading anything new, do a **cold retrieval pass**: re-answer the self-tests in `study-plan.md` for
the subfeature you're about to build, and note any concept you *couldn't* produce from memory (not just
recognise). Those go in the misconception log and lead the session.
