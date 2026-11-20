# Phase 1 Characterisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 1 — a notebook that characterises exploration-setting dependence across 23 PMO oracles, confirms a redesigned diagnostic on 15 held-out oracles, and tests whether the flip is worth more than the probe it costs.

**Architecture:** Three layers. `src/phase1_stats.py` is pure numpy/scipy/pandas with **no** torch, rdkit, TDC or `phase0_lib` import, so every statistical claim is testable on any machine. `src/phase1_source.py` is the jupytext source of truth that imports `phase0_lib` for oracles/backbones/metric and `phase1_stats` for analysis. `src/build_phase1.py` emits the notebook, the library and the sweep script.

**Tech Stack:** python 3.11, numpy 1.26.4, scipy, pandas, matplotlib; torch 2.14.1+cu126 and rdkit only at sweep time; conda env `lce` with **scikit-learn pinned to 1.2.2**.

**Spec:** [docs/superpowers/specs/2026-11-20-phase1-characterisation-design.md](../specs/2026-11-20-phase1-characterisation-design.md)

## Global Constraints

- Budget: **1000** true oracle calls per run. Seeds: **0, 1, 2**. Settings: `explore_low`, `explore_mid`, `explore_high`.
- Backbones: `reinvent`, `augmented_memory`, `graph_ga`. No new backbones.
- `src/phase0_lib.py` and `src/phase0_source.py` are **never modified** — Phase 0's verdict was produced under them.
- Phase 1 runs write to `results/runs_phase1/`, never `results/runs/`.
- C2 primary: feature `gini`, directional (negative), α = 0.05, backbone `augmented_memory`. Secondary features: Benjamini–Hochberg FDR at q = 0.05.
- C2 confirmatory tests use **only the 15 confirmation oracles**. All-23 results are secondary and labelled.
- The pre-registration hash must cover the **computed** feature tuple, not a copy of it.
- `mol_opt` pinned to SHA `2da631b`.
- `scikit-learn==1.2.2` tripwire: `drd2` must score the four probe molecules non-zero before any sweep.

## Review Focus

Input classes the spec implies but no happy-path test exercises:

1. **A cluster with fewer than 2 oracles** — the hierarchical bootstrap resamples oracles within cluster; with n=1 it degenerates silently to seed-only. Must raise or report `insufficient`, not emit a confident CI. → Task 2.
2. **All three settings tie for a winner** — `argmax` picks the first silently, manufacturing a flip from noise. → Task 2.
3. **A degenerate (constant) feature** — Phase 0 shipped two; ρ is NaN and must be excluded from FDR *before* the correction, not counted as a non-significant test. → Task 3.
4. **Fewer than 3 seeds present for an oracle** — held-out-seed CV needs ≥2 fit + 1 eval; with 2 seeds it must report `insufficient` rather than fit and evaluate on the same seed. → Task 4.
5. **A run record missing `score_trace`** (every Phase 0 record) — AUC@700 is then unavailable and the probe cost must fall back to the coarse `history` grid, flagged as approximate, never silently treated as exact. → Task 4.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/phase1_stats.py` | pure statistics: flip test, FDR, CV headroom, probe cost, verdict. No heavy imports. |
| `src/phase1_source.py` | jupytext source of truth for the notebook. |
| `src/build_phase1.py` | build step → notebook + `src/phase1_lib.py` + `scripts/run_sweep_phase1.py`. |
| `tests/test_phase1.py` | assertions on `phase1_stats` only, so they run without torch/rdkit/TDC. |
| `Phase 1/landscape_phase1.ipynb` | *generated* deliverable. |
| `Phase 1/preregistration_phase1.json` | *written on first run*, committed before any sweep. |

---

### Task 1: `phase1_stats` scaffold and feature redesign

**Files:**
- Create: `src/phase1_stats.py`
- Create: `tests/test_phase1.py`

**Interfaces:**
- Produces: `FEATURES: tuple[str, ...]`; `quantile_features(rewards: np.ndarray) -> dict[str, float]`; `features_hash(features: tuple) -> str` (sha256 of the canonical JSON of the tuple).

- [ ] **Step 1:** Write failing tests: `test_no_degenerate_features` (a constant reward array yields no NaN for any quantile feature), `test_features_hash_is_order_sensitive`, `test_gini_of_uniform_is_near_zero`, `test_gini_of_single_spike_is_near_one`.
- [ ] **Step 2:** Run `python tests/test_phase1.py` — expect ImportError.
- [ ] **Step 3:** Implement. Thresholds are quantiles of the observed reward vector (q = 0.5, 0.75, 0.9, 0.99), never absolute, so no feature can go constant on a harder suite. Keep Phase 0's `gini`, `reward_mean`, `reward_std`, `reward_skew`, `reward_kurtosis`, `dynamic_range`; drop `frac_above_0.7` and `frac_above_0.9`.
- [ ] **Step 4:** Run tests — expect PASS.
- [ ] **Step 5:** Commit.

### Task 2: Criterion A at scale — the flip test

**Files:**
- Modify: `src/phase1_stats.py`
- Modify: `tests/test_phase1.py`

**Interfaces:**
- Consumes: Task 1.
- Produces: `winning_setting(df, backbone, by="cluster") -> dict[str, str]`; `flip_pairs(df, backbone, n_boot=10000, rng=None) -> list[dict]` with keys `c1,c2,e1,e2,D,lo,hi,confirmed,status`.

- [ ] **Step 1:** Write failing tests: `test_pure_noise_confirms_no_flip` (random data, 40 trials, zero confirmed), `test_planted_flip_is_confirmed`, `test_single_oracle_cluster_reports_insufficient` (status `insufficient`, `confirmed` False), `test_exact_tie_is_not_a_flip`.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement. Same interaction contrast and hierarchical bootstrap as Phase 0 (`D = [AUC(c1,e1)−AUC(c1,e2)] + [AUC(c2,e2)−AUC(c2,e1)]`, resample oracles within cluster then seeds within oracle, 95% percentile CI excludes 0). A cluster with <2 oracles yields status `insufficient`; an exact argmax tie yields no pair.
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Commit.

### Task 3: Criterion B — one primary hypothesis plus BH-FDR

**Files:**
- Modify: `src/phase1_stats.py`
- Modify: `tests/test_phase1.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `exact_spearman(x, y, alternative="two-sided") -> (rho, p)`; `bh_fdr(pvals: dict[str, float], q=0.05) -> dict[str, dict]`; `criterion_B(df, diagnostics, primary="gini", backbone="augmented_memory", alpha=0.05, q=0.05) -> dict`.

- [ ] **Step 1:** Write failing tests: `test_bh_rejects_more_than_holm_on_same_pvals`, `test_constant_feature_excluded_before_correction` (a NaN-ρ feature must not appear in the FDR denominator), `test_primary_is_directional` (positive ρ on a negative-direction primary ⇒ not significant however small p is), `test_exact_spearman_matches_scipy_on_n8`.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement. `gini` is tested one-sided negative at α = 0.05 with no multiplicity penalty; all remaining non-degenerate features go through BH at q = 0.05. Exact permutation p-values by full enumeration while n ≤ 8, Monte Carlo (100,000 draws, fixed seed) above that — 15! is not enumerable.
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Commit.

### Task 4: Criterion C — cross-validated headroom vs probe cost

**Files:**
- Modify: `src/phase1_stats.py`
- Modify: `tests/test_phase1.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `auc_at(score_trace, n_calls, freq_log=10, top_n=10) -> float`; `cv_headroom(df, backbone, by="cluster") -> dict` with `headroom, lo, hi, folds, status`; `probe_cost(df, backbone, probe_calls=300) -> dict`; `criterion_C(df, backbone, probe_calls=300) -> dict`.

- [ ] **Step 1:** Write failing tests: `test_auc_at_full_budget_matches_recorded_auc` (within 1e-9 on a synthetic trace), `test_auc_at_is_monotone_in_budget`, `test_cv_headroom_of_setting_independent_data_is_near_zero`, `test_cv_headroom_below_naive_headroom_on_noisy_data` (the winner's-curse check), `test_two_seeds_reports_insufficient`, `test_missing_score_trace_marks_approximate`.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement. Conditional setting chosen on 2 seeds, evaluated on the held-out 3rd, rotated over all 3 folds; headroom = mean(conditional) − mean(best single fixed setting), bootstrap CI over oracles. `probe_cost` = AUC@1000 − AUC@700 for the best fixed setting. `criterion_C` returns the CI of (headroom − probe_cost) and `viable` iff that CI excludes 0 and is positive.
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Commit.

### Task 5: Criterion D and the Phase 1 verdict

**Files:**
- Modify: `src/phase1_stats.py`
- Modify: `tests/test_phase1.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `criterion_D(df) -> dict` (best `(backbone, setting)` per cluster, flip-tested across methods); `phase1_verdict(A, B, C, D) -> dict` returning one of `CONTROLLER-VIABLE`, `CHARACTERISATION-ONLY`, `NULL`.

- [ ] **Step 1:** Write failing tests: `test_verdict_null_when_A_fails`, `test_verdict_characterisation_when_C_not_viable_despite_A_and_B` (the economics veto), `test_verdict_controller_viable_requires_A_B_and_C`, `test_criterion_D_action_space_is_backbone_setting_pairs`.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement. Verdict: `NULL` if A fails; else `CONTROLLER-VIABLE` iff B primary passes **and** C is viable; else `CHARACTERISATION-ONLY`.
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Commit.

### Task 6: Notebook source, builder, and the v2 pre-registration

**Files:**
- Create: `src/phase1_source.py`, `src/build_phase1.py`
- Generate: `Phase 1/landscape_phase1.ipynb`, `src/phase1_lib.py`, `scripts/run_sweep_phase1.py`

**Interfaces:**
- Consumes: Tasks 1–5 via `from phase1_stats import *`.
- Produces: `PREREG1` dict + `Phase 1/preregistration_phase1.json`; `ORACLE_TAGS`, `DISCOVERY`, `CONFIRMATION`; `execute_run1(...)`, `run_sweep1(...)`, `load_results1()`.

- [ ] **Step 1:** Write failing tests in `tests/test_phase1.py`: `test_prereg_hash_covers_computed_features` (mutating `FEATURES` changes the hash), `test_discovery_and_confirmation_are_disjoint`, `test_confirmation_set_has_15_oracles`, `test_notebook_builds_and_every_cell_parses`.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement the source in 10 parts: 0 prereg v2, 1 environment + sklearn tripwire + mol_opt SHA assert, 2 oracle inventory and cluster tags derived from TDC/`mol_opt` metadata, 3 redesigned diagnostics, 4 the runner (own `RUNS1_DIR`, persists `score_trace`), 5–8 criteria A–D, 9 verdict and figures. Builder mirrors `src/build_notebook.py`: parse percent cells, exclude `@notebook-only` from the library, `ast.parse` every cell before writing.
- [ ] **Step 4:** Run `python src/build_phase1.py && python tests/test_phase1.py` — expect notebook written and all assertions pass.
- [ ] **Step 5:** Commit.
