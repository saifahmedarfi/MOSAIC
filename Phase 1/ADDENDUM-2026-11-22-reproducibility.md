# Phase 1 — Addendum, 2026-11-22

## Runs are not individually reproducible from their seed

**This is a separate file by design.** `PHASE1_RESULTS.md` is frozen and is deliberately left
byte-identical; a guard in `tests/test_phase2.py` fails if any Phase 0 or Phase 1 file is modified
or deleted, while permitting additions like this one. Read this alongside it.

**Nothing in `PHASE1_RESULTS.md` is retracted.** The `NULL` verdict, all four criteria, every
number and every figure stand. What does not stand is the *implicit* claim that a given
`(oracle, backbone, setting, seed)` cell can be reproduced run-for-run from its seed.

---

## The mechanism

`phase0_lib.set_all_seeds` (`src/phase0_lib.py:330`) is the only seeding entry point:

```python
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
```

Every one of those RNGs is **process-global**. Every run calls it — `run_lm_optimizer` at
`phase0_lib.py:1002`, `run_ga_optimizer` at `phase0_lib.py:1187` — and Phase 1's sweep ran **12
runs concurrently in one interpreter** through `ThreadPoolExecutor` (`PHASE1_RESULTS.md`, Sweep
execution: "**12**, single process, `ThreadPoolExecutor` (one shared CUDA context)"). Thread A
seeds the global stream; thread B reseeds it; both then draw from whatever state they find. The
consumers are on the hot path: `np.random.choice` in Graph GA's mating pool
(`phase0_lib.py:1172`) and in the experience-replay buffer (`phase0_lib.py:939`).

## The evidence

Phase 2 found this and demonstrated it directly (`Phase 2/logs/determinism_probe.log`, excerpted
in `Phase 2/PHASE2_RESULTS.md` §3.2 and §12):

| Experiment | Setup | Result |
|---|---|---|
| A | `(median1, reinvent, 1)` twice, **sequentially** | `0.1265119026` both times — **bit-identical** |
| B | the same cell inside a **12-wide pool** | `0.1262594795` — differs by **2.5e-4** |
| C | 2 threads seeding the global RNG and drawing, 200 rounds | **58 of 200** rounds contaminated |

Phase 2 independently re-ran the **63 cells** it shares with Phase 1. Agreement was **Pearson
r = 0.9980**, mean signed difference **+0.00073**, median absolute difference 0.0084, worst case
0.0389 — unbiased, straddling zero in all three backbones, and three orders of magnitude below the
smallest effect either phase was powered to detect. **Not one of the 63 cells agreed to 1e-6.**

## What this changes, and what it does not

- **Distributional results are sound.** Every seed is an honest independent draw. The
  hierarchical bootstrap over seeds, the criteria built on it, and the `NULL` verdict are all
  unaffected — none of them depends on any single run's value.
- **Run-level provenance is weaker than the pre-registration implies.** A reader re-running one
  cell from `phase1_run_manifest.csv` will not recover its recorded value unless they run
  single-threaded.
- **Reproducing the sweep means reproducing the distribution, not the manifest.** Re-running with
  12 workers yields different cell values and the same conclusions.
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
