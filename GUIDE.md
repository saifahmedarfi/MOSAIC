# Phase 0 — What this code is, and what it satisfies

A guide to `landscape_phase0.ipynb` and its supporting files.

Companion to `Landscape-Conditional-Exploration_Research-Proposal.pdf` (ICML 2027, main track).

---

## 1. In one paragraph

The proposal's central claim (H0) is that **no single exploration setting is best across oracle
landscapes** — the winner *flips* depending on the shape of the reward, a cheap probe can predict
the flip, and a controller exploiting that prediction beats any fixed choice. The proposal also
says, correctly, that this premise should be **cheaply killed in week 1** if it is false.

This notebook is that week-1 experiment. It runs 8 PMO oracles spanning 4 landscape types
against 3 backbones at 3 exploration settings and 3 seeds — **216 runs** of 1000 oracle calls —
then applies a decision rule that was **frozen and cryptographically hashed before any result
existed**, and prints one of three verdicts: **GO**, **PARTIAL**, or **PIVOT**.

It does not build the controller. Building the controller before the gate returns would defeat
the purpose of having a gate.

---

## 2. Which part of the proposal each section satisfies

| Proposal § | Requirement | Where it lives | Status |
|---|---|---|---|
| §7.1 | Pre-registered decision rule, "fixed before any result is inspected" | **Part 0** | Done — and *enforced*, not merely promised (see §4 below) |
| §7.1 | 6–8 oracles spanning PMO clusters | **Part 0** `PREREGISTRATION["oracles"]` | Done — 8 oracles, 4 clusters, 2 each |
| §11 | Compute feasibility on consumer GPUs | **Part 1** | Done — analytic *and* measured VRAM |
| §8 | Reproducibility: public `mol_opt` code, released configs and seeds | **Part 2** | Done — shallow clone, SHA-256 checksummed |
| §8 | Primary metric: AUC of top-10 vs. true oracle calls (PMO convention) | **Part 3** | Done — reimplemented, verified to 1e-12 against the published function |
| §6.1 | Exploration knobs for REINVENT / Augmented Memory | **Parts 5, 7** | Done — σ primary; replay, augmentation, SMP implemented |
| §6.1 | Exploration knobs for Graph GA | **Parts 6, 7** | Done — mutation rate primary |
| §6.1 | Genetic-GFN inverse temperature β | — | **Deferred to Phase 1**, as the proposal schedules it |
| §6.2 | Reward-distribution shape: skew, kurtosis, fraction above threshold | **Part 8** | Done |
| §6.2 | Local ruggedness via small-edit autocorrelation | **Part 8** | Done |
| §6.2 | Basin width around early hits | **Part 8** | Done |
| §6.2 | Weak prior from PMO oracle-type tags, *validated against* computed diagnostics | **Part 8** | Done — cluster tags kept strictly separate from computed features |
| §7.1 | Methods × settings × 3 seeds at ≈1,000 calls | **Part 9** | Done — resumable runner |
| §8 | ≥3 seeds with variance; statistical tests | **Part 10** | Done — hierarchical bootstrap + exact permutation tests |
| §7.1 | The decision: H0 survives iff rank order flips **and** a diagnostic correlates | **Part 10** | Done — executed mechanically, not by eye |
| §10 | Step-0 novelty check | — | **Not done.** Literature search; not a coding task |
| §7.2 | Phase 1 — all 23 oracles, add Genetic-GFN and Mol GA | — | Deferred, gated on a GO verdict |
| §7.3 | Phase 2 — the controller and its ablations | — | Deferred, gated on a GO verdict |
| §7.4 | Phase 3 — transfer, docking oracle, sensitivity | — | Deferred |

**Scope summary:** Phase 0 is complete. Phases 1–3 are deliberately not built.

---

## 3. What each part does

**Part 0 — Pre-registration.** Writes `preregistration.json`: oracles, backbones, settings,
seeds, budget, both criteria with their statistical tests, the three verdicts and their
consequences, a list of prohibited post-hoc moves, and the frozen diagnostic feature list.
Hashes the whole thing with SHA-256.

**Part 1 — Environment.** Dependency check, device report, determinism. Estimates VRAM
analytically, then Part 5 measures it. Answers "will my GPU do this" with a number rather than
a shrug.

**Part 2 — Vendoring.** Shallow-clones `wenhao-gao/mol_opt` for four artifacts: `Prior.ckpt`
and `Voc` (the ChEMBL-pretrained prior), and `crossover.py` / `mutate.py` (the GB-GA graph
operators). Checksums them to `vendor/checksums.json`. We reimplement the *training loop*
because it needs instrumenting; we do not reimplement hundreds of lines of reaction SMARTS from
memory, which is how you get a silent bug that invalidates a backbone.

**Part 3 — Oracles and metric.** `BudgetedOracle` enforces PMO budget semantics exactly:
canonicalise then dedupe (a repeat costs nothing), invalid SMILES are free, `finish` fires on
unique valid molecules. This matters more than it sounds — a mode-collapsed generator emitting
one molecule 500 times burns almost no budget, so any metric computed over *proposals* rather
than *buffer growth* systematically flatters collapsed policies, which are exactly the policies
this project studies. Also: diversity, scaffold counts, #Circles, and oracle latency timing.

**Part 4 — Chemistry utilities.** Vocabulary and tokenizer (REINVENT's, reproduced so the prior
loads), SMILES randomization, Murcko scaffolds.

**Part 5 — Backbone A: the SMILES language model.** `MultiGRU` with `state_dict` keys matching
`mol_opt` so `Prior.ckpt` loads unmodified. The DAP objective, experience replay, Augmented
Memory's augmentation rounds, Selective Memory Purge. One function covers REINVENT and
Augmented Memory; `augmentation_rounds=0` recovers plain REINVENT.

**Part 6 — Backbone B: Graph GA.** Fitness-proportionate mating, vendored GB-GA operators,
truncation selection. Start population fixed per seed so the ladder is not confounded by where
it began.

**Part 7 — The exploration ladder.** Three settings per backbone. **One primary knob each**,
not a grid — see §5.

**Part 8 — Landscape diagnostics.** 13 frozen features from 300 budgeted calls: 200 prior
samples (reward-distribution shape) plus 100 local edits around the top 5 hits (ruggedness,
basin width, neighbour gain).

**Part 9 — Runner.** 216 runs, one JSON each, resumable. A sweep this size on a shared cluster
*will* be interrupted.

**Part 10 — Analysis and verdict.** Applies the frozen rule and prints GO / PARTIAL / PIVOT.
Two figures: the headline non-parallel-lines plot, and a per-oracle heatmap z-scored within
oracle so that oracle difficulty does not drown the ordering.

**Part 11 — Exports.** Writes `environment.yml` and the cluster scripts.

**Part 12 — ICML positioning.** The objection most likely to sink the paper, what to add to
Phase 2 because of it, and the timeline problem.

---

## 4. The two design decisions that matter most

### Pre-registration is enforced, not promised

The proposal says the decision rule is "fixed before any result is inspected." A sentence in a
PDF is not a mechanism. Here the rule is executable code, serialized to `preregistration.json`
with a SHA-256, and **Part 10 refuses to produce a verdict if that hash no longer matches**.
Every run record also carries the hash it was produced under, so results generated under a
different rule are flagged automatically.

This is cheap to build and disproportionately valuable: reviewers increasingly ask, and "the
rule is in git, timestamped before the results" is a complete answer.

> **Commit `preregistration.json` before running anything.** Its entire value is being
> timestamped earlier than the results.

### The diagnostic probe is method-agnostic by construction

The obvious way to compute a diagnostic is from an early REINVENT run. That would be circular:
"the diagnostic predicts which setting wins" could reduce to "REINVENT's early behaviour
predicts REINVENT's later behaviour", which is useless for choosing a configuration at
deployment.

So the probe uses a fixed policy that is **none of the competitors** — samples from the
untrained ChEMBL prior, plus local edits around whatever it finds. It would be identical if the
candidates were entirely different optimizers. All 300 probe calls are charged against the same
budget, so in Phase 2 the controller gets 300 diagnostic + 700 optimization against baselines
that get all 1000.

---

## 5. Where this knowingly departs from the proposal

Each of these is a judgement call, stated here so it can be overruled.

**Phase 0 varies one knob per backbone, not the full grid.** §6.1 lists several knobs each. The
Phase-0 question is narrow — *does the rank order flip?* — and a multi-knob ladder makes a flip
uninterpretable: you cannot tell whether σ flipped, the replay count flipped, or they
interacted. The grid belongs in Phase 1, where the goal is characterisation rather than
falsification.

**Criterion A uses a hierarchical bootstrap, which the proposal does not specify.** §7.1 says
"a margin exceeding seed variance." Resampling seeds alone treats the 2 oracles per cluster as
the entire population, but the claim is about landscape *clusters* and therefore must generalize
to other oracles of that type. On pure-noise synthetic data, seed-only resampling confirmed
**5/40** false flips; the hierarchical version confirmed **0/40**. The hierarchical test has low
power with 2 oracles per cluster — deliberately. It biases toward PIVOT rather than toward a
false GO.

**Criterion B is declared a screen at α=0.10, with Phase 1 confirmatory at α=0.05.** With n=8
oracles the test is underpowered; saying so in advance is what keeps Phase 1 confirmatory
instead of a second bite at the same apple.

**Selective Memory Purge is a scaffold-cap adaptation,** not a line-for-line port of Guo &
Schwaller's criterion. Held fixed within each setting, so it cannot affect the rank-flip test.
**Label it as an adaptation in the paper.**

**Budget check uses `>=` where mol_opt uses `>`,** capping the buffer at exactly
`max_oracle_calls`. Effect on AUC ≈ 1e-3.

---

## 6. Two things to know before reading any number

**AUC-top-10 has a ramp-in artifact.** mol_opt's trapezoidal integral starts from `prev = 0`,
asserting that mean top-10 reward is zero at zero oracle calls. A run with a *constant* reward
of 0.42 therefore scores **0.4179**. We reproduce the published formula exactly (verified to
1e-12 across 15 configurations in `tests/test_core.py`) because changing it would break
comparability with every published PMO number. Consequence: **absolute AUC values are not mean
rewards and must not be described as such in the paper.**

**A Phase-0 pass on Criterion B is weak evidence.** n=8. It is a screen. Phase 1, with 23
oracles, is the test to be nervous about.

---

## 7. Files

| File | Role |
|---|---|
| `landscape_phase0.ipynb` | **The deliverable.** 72 cells, run top to bottom. |
| `src/phase0_source.py` | **Source of truth**, jupytext percent format. Edit this, never the `.ipynb`. |
| `src/phase0_lib.py` | *Generated.* Importable library — the notebook minus demo cells, so importing it has no side effects. |
| `src/build_notebook.py` | **The build step.** Emits the notebook, the library, and every file below. Syntax-checks all cells first. |
| `tests/test_core.py` | 56 assertions. Runs on numpy/scipy/pandas alone — no torch, rdkit or GPU. |
| `environment.yml` | *Generated.* Conda environment. |
| `scripts/run_sweep.py` | *Generated.* CLI sweep runner, resumable. |
| `scripts/slurm_phase0.sh` | *Generated.* Slurm batch script. |
| `preregistration.json` | *Written on first notebook run.* **Commit this.** |
| `results/phase0_verdict.json` | *Written by Part 10.* **Commit this.** |

Everything marked *generated* comes from `src/phase0_source.py` via one build step, so there is
nothing to keep in sync by hand:

```bash
python src/build_notebook.py
```

The `.ipynb` is a build product. Edits made inside Jupyter will be overwritten on the next
build — change `src/phase0_source.py` instead.

---

## 8. Running it

```bash
# 1. environment (the notebook also has an install cell for a first run)
conda env create -f environment.yml && conda activate lce

# 2. verify the statistics before trusting any result (~90s, no GPU needed)
python tests/test_core.py

# 3. run the notebook top to bottom once: vendors mol_opt, prefetches and times
#    the oracles, smoke-tests both backbones, computes the diagnostics
jupyter lab landscape_phase0.ipynb

# 4. COMMIT preregistration.json NOW, before the sweep

# 5. launch the sweep on the cluster, not in a notebook kernel
python scripts/run_sweep.py --workers 8      # resumable; safe to interrupt
sbatch scripts/slurm_phase0.sh               # or under Slurm

# 6. return to Part 10 for the verdict
```

**Hardware.** Peak VRAM for one run is **~1.0–1.5 GB** — the only GPU resident is a 4.18 M-param
SMILES RNN. Every PMO oracle and all of Graph GA are CPU-only. A 12 GB card holds 6–8 concurrent
runs; an 11 GB 2080 Ti holds 5–6. **Set `--workers` from core count, not VRAM.** Full sweep:
~1 hour at 8 workers, 4–8 hours serial.

---

## 9. What to do with the verdict

| Verdict | Meaning | Action |
|---|---|---|
| **GO** | A and B both pass | Proceed to Phase 1 as written. |
| **PARTIAL** | Flip is real, diagnostics miss it | Phase 1 for characterisation (C1) only. Redesign the feature set before attempting the controller. Downgrade venue expectation. |
| **PIVOT** | No flip | **Stop.** The premise is dead. Switch to the proxy-gap direction. Write up the null — it is informative, because the field has assumed this dependence since PMO without testing it directly. |

Follow it, including when it is unwelcome. That is what the pre-registration is for.

---

## 10. Open items

1. **Step-0 novelty check (§10).** Not done — a literature search, not a coding task.
   "Landscape-aware / instance-adaptive method selection for molecular optimization" is a
   crowded enough phrase that it is worth knowing *now* whether it was published recently.
2. **Add an online bandit baseline to Phase 2.** The proposal's baselines (a)–(d) do not answer
   *"why predict the landscape when you could adapt online?"* — the first thing an ICML reviewer
   will ask. See Part 12.
3. **Settle the ICML-vs-NeurIPS timeline before launching the sweep.** The proposal's dates
   (abstract 16 Jan, paper 22 Jan 2027 — verify against the official CFP) leave ~58 days against
   an 8-week plan. Zero slack. See Part 12.
