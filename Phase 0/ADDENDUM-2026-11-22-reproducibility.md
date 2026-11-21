# Phase 0 — Addendum, 2026-11-22

## Runs are not individually reproducible from their seed

**This is a separate file by design.** `PHASE0_RESULTS.md` cites commit hashes as its own
provenance and is deliberately left byte-identical; a guard in `tests/test_phase2.py` fails if any
Phase 0 or Phase 1 file is modified or deleted, while permitting additions like this one. Read this
alongside it.

**Nothing in `PHASE0_RESULTS.md` is retracted.** The `PARTIAL` verdict, every criterion and every
number stand. What does not stand is the *implicit* claim that a given
`(oracle, backbone, setting, seed)` cell can be reproduced run-for-run from its seed.

---

## The mechanism

`set_all_seeds` (`src/phase0_lib.py:330`) seeds `random`, `np.random` and torch — all
**process-global**. Every run calls it (`run_lm_optimizer` at `phase0_lib.py:1002`,
`run_ga_optimizer` at `phase0_lib.py:1187`), and Phase 0's sweep ran **12 runs concurrently in one
interpreter** through `ThreadPoolExecutor` (`src/phase0_lib.py:1572`; `PHASE0_RESULTS.md` records
"Workers | **12**, single process" and "~4.9 GB of 11.99 GB steady at 12 concurrent workers,
stable across all 216 runs"). One thread seeds the global stream, the next reseeds it, and both
draw from whatever state they find. The consumers are on the hot path: `np.random.choice` in
Graph GA's mating pool (`phase0_lib.py:1172`) and in the experience-replay buffer
(`phase0_lib.py:939`).

The thread pool was the right call for its stated purpose — all 12 workers share **one** CUDA
context, so the ~500 MB is paid once rather than per worker, which is what made 12-way concurrency
fit in 12 GB at all. The seeding is what is wrong, not the pool.

## The evidence

Phase 2 found this and demonstrated it directly (`Phase 2/logs/determinism_probe.log`, excerpted
in `Phase 2/PHASE2_RESULTS.md` §3.2 and §12):

| Experiment | Setup | Result |
|---|---|---|
| A | `(median1, reinvent, 1)` twice, **sequentially** | `0.1265119026` both times — **bit-identical** |
| B | the same cell inside a **12-wide pool** | `0.1262594795` — differs by **2.5e-4** |
| C | 2 threads seeding the global RNG and drawing, 200 rounds | **58 of 200** rounds contaminated |

Experiment A is the important one for reading Phase 0: the run path itself is deterministic and
correct. Only concurrency breaks it.

## What this changes, and what it does not

- **Distributional results are sound.** Every seed is an honest independent draw, and every
  bootstrap over seeds remains valid. The `PARTIAL` verdict does not depend on any single run.
- **Run-level provenance is weaker than the pre-registration implies.** A reader re-running one
  cell will not recover its recorded value unless they run single-threaded.
- **This bears on Phase 0's flip finding specifically.** Phase 0 confirmed roughly 4 argmax flips
  per 40 on pure noise and froze the argmax in response; Phase 1 then found the flip did not
  replicate (Criterion A, 0 confirmed on every backbone). Non-determinism between concurrent runs
  is a further reason an individual flip should never have been read as a property of a cell. It
  does not change either verdict.
- **Any write-up must not claim seed-level reproducibility.** This is the one sentence this
  addendum exists to prevent.

`src/phase0_lib.py` is unchanged since commit `8692ea0` (2026-11-19), which predates every sweep in
this project, so this is a property of the run path throughout rather than drift between phases.

## Why it is not fixed

The fix is per-run `np.random.Generator` / `torch.Generator` objects threaded through the
optimizers, replacing the global seeding. That changes `phase0_lib.py` — the library Phases 0, 1
and 2 were all run against — and would invalidate the provenance of every existing result. It is
recorded as a decision for the controller, not taken here. See `Phase 2/PHASE2_RESULTS.md` §9.

---

*Added 2026-11-22 after Phase 2. Every number here comes from `Phase 2/` artifacts and from
`src/phase0_lib.py` at the line numbers cited; none is estimated.*
