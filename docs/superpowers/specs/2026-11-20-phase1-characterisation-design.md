# Phase 1 — Characterisation: design

**Date:** 2026-11-20
**Status:** awaiting approval
**Gated by:** Phase 0 verdict `PARTIAL` (prereg `fca49820…`, decided 2026-11-20 03:18)

---

## 1. Intent and why

Phase 0 returned `PARTIAL`: Criterion A passed (the winning exploration setting flips across
landscape clusters), Criterion B failed (no frozen diagnostic tracked the winner). The frozen
pre-registration's mandated consequence is explicit and binding:

> The landscape dependence is real but our diagnostics do not capture it. Proceed to Phase 1 for
> **C1 (characterisation) only**; **redesign the diagnostic feature set before attempting the
> controller**. **Downgrade venue expectation** to a benchmark/characterisation contribution.

So Phase 1 answers three questions and builds nothing that depends on their answers:

1. **Does the flip survive at scale?** Phase 0 confirmed it on 2 cluster pairs, in 1 of 3
   backbones, with both pairs sharing the `isomer` cluster. That is thin. 23 oracles says whether
   it generalises.
2. **Can a redesigned feature set predict the winner?** Phase 0's screen failed as pre-registered,
   but its best candidate (`gini`, ρ = −0.655, p = 0.0893, mechanistically correct sign) was
   underpowered at n = 8 rather than absent.
3. **Is the flip worth more than the information needed to exploit it?** Not asked in Phase 0, and
   on Phase 0's own numbers the answer looks like *no*: a perfect cluster-conditional chooser buys
   +0.0074 AUC (reinvent), +0.0050 (augmented_memory), +0.0000 (graph_ga), while the diagnostic
   probe costs 300 of 1000 oracle calls. **If that holds at 23 oracles, the project pivots on
   economics regardless of questions 1 and 2.** This is the most decision-relevant quantity in the
   phase and it must be pre-registered, not discovered.

## 2. Goals

- **C1** Characterise exploration-setting dependence across 23 PMO oracles × 3 backbones.
- **C2** Confirmatory test of a redesigned diagnostic feature set, α = 0.05.
- **C3** Quantify **cross-validated headroom** (best-fixed vs landscape-conditional) against the
  probe's cost in the same units.
- **C4** Cross-**method** selection: does the best `(backbone, setting)` pair flip across
  landscapes? Phase 0's method-level margins are an order of magnitude larger than its
  within-method knob effects, so this is where the characterisation contribution actually lives.

C1–C4 run on **one** set of runs. No analysis requires its own sweep.

## 3. Non-goals and anti-goals

**Non-goals** (deferred, not forgotten): the controller (forbidden by the verdict), Genetic-GFN and
Mol GA (new backbone code; the proposal schedules them, this phase defers them to keep the build
small), the docking oracle, backbone transfer, the 10,000-call PMO budget.

**Anti-goals** — Phase 1 fails, even if it runs clean, if any of these happen:

- Criterion B is re-tested on a backbone chosen *after* seeing which backbone flipped. This is the
  exact defect Phase 0 disclosed, and repeating it knowingly would be worse than the original.
- The 15 new oracles' cluster tags are assigned by hand after Phase 0's results are known.
- The gini hypothesis is "confirmed" using data that includes the 8 oracles that generated it.
- A `GO`-shaped conclusion is reached without C3 clearing the probe cost.

## 4. Constraints

| | |
|---|---|
| Hardware | one RTX 4070 Super, 12 GB; 16 cores; Windows 11 |
| Measured Phase 0 throughput | 286.5 s/run mean, 12 workers, ~4.9 GB VRAM at 12 concurrent |
| Budget | 1000 true oracle calls per run (same as Phase 0, for comparability) |
| Code | reuse `phase0_lib`; the notebook stays small and readable |
| Environment | the `lce` env exactly as Phase 0 left it — **including `scikit-learn==1.2.2`** |

> The sklearn pin is load-bearing. On 1.9.1, `drd2` loads without error and returns `0.0` for every
> molecule. Re-check the four probe values in `src/phase0_source.py` before any Phase 1 run.

## 5. Approaches considered

**(a) Extend Phase 0's architecture — chosen.** A new jupytext source of truth emits
`Phase 1/landscape_phase1.ipynb` and `src/phase1_lib.py`; both import `phase0_lib` for the
backbones, the oracles, the budget semantics and the metric. New code is only: 23-oracle cluster
tags, the redesigned feature set, the four analyses, and a v2 pre-registration. ~35 cells.

**(b) Self-contained Phase 1 notebook.** Rejected: duplicates ~2,000 lines of backbone code and
forks `top_auc`, the one function verified to 1e-12 against the published formula. Two copies of a
metric is how published numbers stop being comparable.

**(c) Scripts only, no notebook.** Rejected: the `.ipynb` was requested, and Phase 0's notebook is
how the narrative gets read.

## 6. The design

### 6.1 Discovery / confirmation split

Phase 0's 8 oracles generated the gini hypothesis, so they cannot confirm it.

| Set | Oracles | Role |
|---|---|---|
| **Discovery** | the 8 from Phase 0 | hypothesis source; runs **reused as-is**, not re-run |
| **Confirmation** | the 15 remaining PMO oracles | carries every confirmatory test in C2 |

Primary analyses use the 15 held-out oracles. All 23 are reported as a secondary, clearly labelled
descriptive result. This cuts new compute to **405 runs ≈ 2.7 h** and makes "confirmatory" true
rather than decorative.

### 6.2 Grid

15 oracles × 3 backbones (`reinvent`, `augmented_memory`, `graph_ga`) × 3 settings × 3 seeds =
**405 new runs** at 1000 calls. Plus Phase 0's 216 = 621 total for the secondary report.
12 workers, single process, `ThreadPoolExecutor` — one shared CUDA context, as Phase 0 established
(a process pool would mean 12 contexts and OOM on 12 GB).

### 6.3 Cluster tags — frozen from an external source

The 15 new oracles are tagged **before any run**, from PMO's own published oracle categories, and
the source is recorded in the pre-registration alongside each assignment. Where PMO's category is
ambiguous the oracle is tagged `unassigned` and excluded from cluster-level tests rather than
placed by judgement. Hand-assignment after seeing Phase 0's results is an anti-goal (§3).

### 6.4 Redesigned diagnostic features

Phase 0's failures to fix:

- **Drop the degenerate absolute thresholds.** `frac_above_0.7` and `frac_above_0.9` were constant
  (no oracle exceeded 0.5) and consumed multiplicity budget while being mathematically incapable
  of correlating with anything.
- **Thresholds become quantiles** of the probe's own observed reward distribution, so they cannot
  go degenerate on a harder oracle suite.
- **One pre-specified primary hypothesis.** `gini`, directional (negative), tested at α = 0.05
  **without** multiplicity penalty — legitimate because Phase 0 is prior evidence, not this phase's
  data. Every other feature is secondary under Benjamini–Hochberg FDR, not Holm: Phase 0's 13
  features are heavily correlated measures of reward concentration, and Holm over correlated tests
  is needlessly conservative.
- **The backbone is named explicitly:** `augmented_memory` is primary, justified a priori by Phase
  0 having confirmed its flips; `reinvent` secondary; `graph_ga` descriptive only. Phase 0's prereg
  said "the LM backbone", which is ambiguous between two LM backbones, and the implementation's
  choice of `reinvent` landed on the arm with zero confirmed flips.

### 6.5 C3, the economics test

Runs persist `(score, call_index)` pairs so AUC is exactly computable at any truncation. Then:

- **headroom** = AUC@1000(landscape-conditional choice) − AUC@1000(best single fixed setting),
  with the conditional choice selected on **held-out seeds** (fit on 2, evaluate on the 3rd,
  rotated) to remove the winner's curse that inflates Phase 0's naive +4.7%.
- **probe cost** = AUC@1000(best fixed) − AUC@700(best fixed), i.e. what 300 calls is worth in the
  metric's own units.
- **Pre-registered rule:** if the bootstrap CI for (headroom − probe cost) does not exclude 0, the
  landscape-conditional controller is not viable at this budget, and that is reported as the
  phase's headline result regardless of C1, C2 and C4.

### 6.6 Integrity fixes carried over

1. Hash the **computed** feature constant, not a parallel declaration in the prereg dict.
2. Pin `mol_opt` to SHA `2da631b` instead of tracking upstream HEAD.
3. Keep the `drd2`/`jnk3` probe-value tripwire; assert it before the sweep, not after.

### 6.7 Deliverables

| Path | Role |
|---|---|
| `src/phase1_source.py` | source of truth, jupytext percent |
| `src/build_phase1.py` | build step → notebook + library + scripts |
| `Phase 1/landscape_phase1.ipynb` | the deliverable, ~35 cells |
| `src/phase1_lib.py` | *generated*, importable, no side effects |
| `Phase 1/preregistration_phase1.json` | frozen v2 rule; **committed before any run** |
| `scripts/run_sweep_phase1.py` | *generated*, resumable |
| `tests/test_phase1.py` | assertions on the new features, CV headroom, FDR, tags |

### 6.8 Known touch points in `phase0_lib` (my calls, flagged for checking)

Found while builder-checking this document, both load-bearing:

1. **`execute_run` will `KeyError` on every new oracle.** `src/phase0_lib.py:1513` does
   `ORACLE_TO_CLUSTER[oracle_name]`, and that dict is derived from Phase 0's pre-registration, which
   lists only 8 oracles. Phase 1 therefore needs its own thin `execute_run` wrapper taking the
   Phase 1 tag map. `phase0_lib` is **not** edited — it is the artifact a frozen verdict was
   produced under, and editing it retroactively would invalidate Phase 0's provenance.
2. **Run records must not share a directory.** `RUNS_DIR = results/runs` and `load_results()` globs
   all of it, so 405 Phase 1 records would silently mix with Phase 0's 216 under a different prereg
   hash. Phase 1 writes to `results/runs_phase1/` and its loader reads only that, with the
   secondary all-23 report explicitly unioning both and filtering on `prereg_sha256`.

### 6.9 Left to the builder

Cell-level narrative structure; figure styling (follow Phase 0's); the exact quantile breakpoints
for the redesigned features; `tests/test_phase1.py`'s internal organisation; whether
`build_phase1.py` is a separate file or a flag on the existing builder.

## 7. Runtime

| | |
|---|---|
| New runs | 405 |
| Estimated wall clock | ~2.7 h at 12 workers (from Phase 0's measured 286.5 s/run) |
| VRAM | ~4.9 GB of 12 GB at 12 concurrent — unchanged from Phase 0 |
| Reused | Phase 0's 216 runs, untouched |
