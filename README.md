# Landscape-Conditional Exploration for Sample-Efficient Molecular Optimization

Phase 0 — the pre-registered falsification.

Target venue: **ICML 2027** (main track), per
`Landscape-Conditional-Exploration_Research-Proposal.pdf`. Part 12 of the notebook sets out the
positioning argument, the objection most likely to sink the paper, and the timeline problem.

**[GUIDE.md](GUIDE.md) explains what the notebook does and which proposal section each part
satisfies.** Start there.

## What is here

| Path | What it is |
|---|---|
| `Phase 0/landscape_phase0.ipynb` | **The deliverable.** 72 cells, end to end. |
| `GUIDE.md` | What it does, proposal §→part mapping, deviations, how to run. |
| `src/phase0_source.py` | **Source of truth**, jupytext percent format. Edit this, never the `.ipynb`. |
| `src/phase0_lib.py` | *Generated.* Importable library (notebook minus demo cells). |
| `src/build_notebook.py` | **The build step.** Emits the notebook, the library and the assets below. |
| `tests/test_core.py` | 56 assertions on the metric, statistics and build artifacts. No torch/rdkit needed. |
| `environment.yml`, `scripts/run_sweep.py`, `scripts/slurm_phase0.sh`, `.gitignore` | *Generated.* |

Everything marked *generated* comes from one build step, so nothing is kept in sync by hand:

```bash
python src/build_notebook.py
```

The `.ipynb` is a build product — edits made inside Jupyter are overwritten on the next build.

Written on first run: `preregistration.json` (**commit it**), plus `results/`, `figures/`,
`vendor/`, `data/`.

## Quickstart

```bash
conda env create -f environment.yml && conda activate lce   # after the first notebook run
python tests/test_core.py                                   # ~90s, needs only numpy/scipy/pandas
jupyter lab "Phase 0/landscape_phase0.ipynb"
```

Run the notebook top to bottom once. It vendors `mol_opt` (for the ChEMBL prior and the GB-GA
operators), prefetches and times the TDC oracles, runs smoke tests on both backbones, and
computes the landscape diagnostics. Then launch the sweep on the cluster:

```bash
python scripts/run_sweep.py --workers 8          # resumable; safe to interrupt
sbatch scripts/slurm_phase0.sh                   # or under Slurm
```

Come back to Part 10 for the verdict.

## Hardware

**VRAM is not the constraint.** The only thing on the GPU is a 4.18 M-parameter SMILES RNN:

| | |
|---|---|
| Peak VRAM, one training run | **~1.0–1.5 GB** |
| RTX 4070 Super (12 GB) | ~6–8 concurrent runs |
| 2080 Ti (11 GB) | ~5–6 concurrent runs |

Every PMO oracle and all of Graph GA are CPU-only, so throughput is set by **core count**.
Set `--workers` from cores, not VRAM. Part 1 of the notebook measures this on your actual
hardware rather than trusting the table.

Full Phase-0 sweep: 8 oracles × 3 backbones × 3 settings × 3 seeds = **216 runs** of 1000
oracle calls. Roughly 1 hour at 8 workers, 4–8 hours serial.

## The rule that governs this phase

Part 0 freezes a decision rule and writes `preregistration.json` with a SHA-256. Part 10
**refuses to produce a verdict** if that hash no longer matches. H0 survives only if both hold:

- **Criterion A** — the winning exploration setting flips across ≥2 landscape cluster pairs
  within one backbone, confirmed by a *hierarchical* bootstrap (oracles resampled within
  cluster, then seeds within oracle).
- **Criterion B** — at least one cheap diagnostic tracks the winner across oracles (exact
  permutation Spearman, Holm-corrected, α=0.10 as a declared **screen**).

Outcomes: **GO** → Phase 1. **PARTIAL** (A only) → characterisation only; redesign the
diagnostic. **PIVOT** (A fails) → the premise is dead; stop and write up the null.

**Commit `preregistration.json` before running anything.** Its entire value is being
timestamped earlier than the results.

## Before ICML 2027: two decisions that cannot wait

1. **Add an online bandit baseline to Phase 2** (EXP3 or successive halving over the settings
   as arms). The proposal's baselines (a)–(d) do not answer *"why predict the landscape when
   you could just adapt online?"*, which is the first thing an ICML reviewer will ask.
2. **Settle the timeline before launching the sweep.** The proposal's dates are abstract
   16 Jan / paper 22 Jan 2027 (verify against the official CFP). From 19 Nov 2026 that is ~58
   days against an 8-week (56-day) research plan — zero slack. Either compress (run Phases 0
   and 1 in parallel, scope Phase 3 down to backbone transfer, draft related work during the
   sweep) or re-target NeurIPS 2027, which the proposal already names as the fallback.

## Two things worth knowing before you read any number

1. **AUC-top-10 has a ramp-in artifact.** mol_opt's integral starts from `prev=0`, so a
   constant-0.42 run scores 0.4179. We reproduce the published formula exactly (verified to
   1e-12 in `tests/test_core.py`), which means absolute AUC values are *not* mean rewards and
   must not be described as such.
2. **Criterion A's bootstrap is hierarchical by design, and that costs power.** With 2 oracles
   per cluster, resampling oracles is weak — deliberately. Seed-only resampling confirmed 5/40
   pure-noise trials; hierarchical confirmed 0/40. The test is biased toward PIVOT rather than
   toward a false GO.

## Deviations from published work, stated up front

- **Selective Memory Purge** is a scaffold-cap adaptation, not a line-for-line port of Guo &
  Schwaller. Held fixed within each setting, so it does not affect the rank-flip test.
- **Budget check** uses `>=` where mol_opt uses `>`, capping the buffer at exactly
  `max_oracle_calls`. Effect on AUC ~1e-3.
- **Phase 0 varies one knob per backbone**, not the full grid in the proposal — a multi-knob
  ladder makes a flip uninterpretable. The grid belongs in Phase 1.
- **REINVENT's character-level tokenizer** is inherited unchanged to stay comparable with the
  pretrained prior. Held fixed across all settings, so it cannot confound the rank-flip test.
