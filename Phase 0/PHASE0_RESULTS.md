# Phase 0 — Pre-Registered Falsification: Results

**Landscape-Conditional Exploration** · Phase 0 — the pre-registered falsification

| | |
|---|---|
| **Verdict** | **`PARTIAL`** |
| Decided at | 2026-11-20 03:18:07 |
| Pre-registration | `sha256 fca4982097668bd980197cf9f067c474d500434357331975b0d93c82a0ee329c` (v1.1.0, frozen 2026-11-19) |
| Prereg status at decision time | `verified` — in-code rule matched the frozen file (no `REFUSED`) |
| Sweep completion | **216 / 216 runs ok (100.0%)**, 0 failed |
| Verdict is provisional? | **No** — the sweep is complete |
| Machine | NVIDIA GeForce RTX 4070 SUPER (Ada sm_89, 11.99 GB), 16 cores, Windows 11 |
| Provenance | prereg commit `0a28b5a` → verdict commit `47dba26` |

The pre-registration was committed **before any run existed** (`results/runs/` empty at `0a28b5a`),
so the ordering claim is verifiable from git history rather than asserted.

---

## Two caveats that must accompany any number in this document

**1. AUC-top-10 is not a mean reward.** The PMO integral starts from `prev = 0`, so it ramps in
from zero and a *constant-0.42* run scores **0.4179**. The formula is reproduced to 1e-12 against
`mol_opt/main/optimizer.py::top_auc` on purpose. Absolute AUC values in this document must never be
described as mean rewards, and a mean AUC near 0.41 does **not** mean "average reward 0.41".

**2. Criterion B at n = 8 oracles is a screen, not a result.** It is explicitly underpowered, which
is why the pre-registration set α = 0.10 here and reserved α = 0.05 for the confirmatory Phase 1
test over 23 oracles. Criterion B failing is "no signal worth chasing *yet, with these features*",
not "no signal exists". Declaring this split in advance is what keeps Phase 1 confirmatory instead
of a second bite at the same apple.

---

## Mandated action for `PARTIAL`

Taken verbatim from the frozen pre-registration (`verdict_consequences.PARTIAL`) and GUIDE.md §9:

> The landscape dependence is real but our diagnostics do not capture it. Proceed to Phase 1 for
> **C1 (characterisation) only**; **redesign the diagnostic feature set before attempting the
> controller**. **Reframe the contribution** as benchmark/characterisation rather than method.

Concretely: the controller is **not** unblocked. Phase 1 proceeds as characterisation; the feature
set must be redesigned before any controller work; the contribution drops from a method
contribution to a benchmark/characterisation contribution.

---

## Criterion A — does the winning exploration setting flip across landscape clusters?

**Result: PASS.** Threshold was ≥ 2 confirmed flipped cluster pairs within a single backbone;
the best backbone has exactly 2.

Test: for each ordered cluster pair with differing winners, the interaction contrast
`D = [AUC(c1,e1) − AUC(c1,e2)] + [AUC(c2,e2) − AUC(c2,e1)]`, with a **hierarchical** bootstrap
(10,000 resamples: oracles within cluster, then seeds within oracle). The hierarchical 95% CI
decides; the seed-only CI is reported as an optimistic bound only.

### augmented_memory — 2 confirmed pairs (this is what carries the verdict)

Winners: bioactivity `low` · isomer `low` · mpo `mid` · similarity `mid`

| Pair | D | hierarchical 95% CI (decides) | seed-only 95% CI | Confirmed |
|---|---|---|---|---|
| isomer(low) vs mpo(mid) | +0.0871 | **[+0.0211, +0.1732]** | [+0.0266, +0.1373] | **YES** |
| isomer(low) vs similarity(mid) | +0.0915 | **[+0.0300, +0.1764]** | [+0.0334, +0.1395] | **YES** |
| bioactivity(low) vs mpo(mid) | +0.0287 | [−0.0494, +0.1266] | [−0.0404, +0.1027] | no |
| bioactivity(low) vs similarity(mid) | +0.0331 | [−0.0412, +0.1320] | [−0.0344, +0.1066] | no |

### reinvent — 0 confirmed pairs

Winners: bioactivity `high` · isomer `low` · mpo `high` · similarity `mid`

| Pair | D | hierarchical 95% CI (decides) | seed-only 95% CI | Confirmed |
|---|---|---|---|---|
| bioactivity(high) vs isomer(low) | +0.0926 | [−0.0502, +0.3017] | [−0.0535, +0.2247] | no |
| isomer(low) vs mpo(high) | +0.0936 | [−0.0409, +0.2869] | [−0.0413, +0.2159] | no |
| isomer(low) vs similarity(mid) | +0.0180 | [−0.0560, +0.1008] | [−0.0612, +0.0789] | no |
| bioactivity(high) vs similarity(mid) | +0.0163 | [−0.1027, +0.1699] | [−0.0702, +0.1084] | no |
| mpo(high) vs similarity(mid) | +0.0143 | [−0.0267, +0.0601] | [−0.0124, +0.0425] | no |

### graph_ga — 0 confirmed pairs

Winners: **`low` in all four clusters** — no cluster pair has differing winners, so there are no
flip candidates at all. The non-neural control shows no landscape-dependent ranking whatsoever.

### Honest reading of Criterion A

The criterion passes exactly as pre-registered, but three things should be stated plainly rather
than left for a reviewer to notice:

- **The flip is backbone-specific.** It appears in `augmented_memory` only. `reinvent` confirms
  nothing, and `graph_ga` has no differing winners to test. A claim of "landscape-conditional
  exploration" that holds for one of three backbones is weaker than the proposal's framing.
- **Both confirmed pairs involve the same cluster.** `isomer` is one half of both confirmed pairs,
  and its winner (`low`) is also where its margin is largest (0.6185 vs 0.5392/0.5293). The
  evidence is closer to "the isomer cluster behaves differently" than to "each landscape type has
  its own optimum".
- **Power is low by construction.** With 2 oracles per cluster, the hierarchical bootstrap is
  deliberately conservative, which biases toward `PIVOT` rather than toward a false `GO`. The two
  confirmed CIs clear zero but not by much (lower bounds +0.0211 and +0.0300).

---

## Criterion B — does any cheap diagnostic track the winning setting?

**Result: FAIL.** α = 0.10, Holm–Bonferroni across 13 features, exact permutation p-values
(all 8! = 40,320 permutations enumerated), backbone = `reinvent`, n = 8 oracles.

Winner per oracle (reinvent): albuterol `high` · celecoxib `mid` · drd2 `high` · isomers_c7h8n2o2
`low` · isomers_c9h10n2o2pf2cl `mid` · jnk3 `mid` · ranolazine `high` · sitagliptin `low`

| Feature | Spearman ρ | exact p | Holm threshold | Significant |
|---|---|---|---|---|
| gini | −0.655 | 0.0893 | 0.0077 | no |
| basin_width | +0.580 | 0.1429 | 0.0083 | no |
| reward_mean | +0.504 | 0.2071 | 0.0091 | no |
| frac_above_0.5 | +0.480 | 0.3214 | 0.0125 | no |
| reward_skew | −0.441 | 0.2786 | 0.0100 | no |
| ruggedness_absdelta | −0.428 | 0.2929 | 0.0111 | no |
| reward_kurtosis | −0.365 | 0.3714 | 0.0143 | no |
| reward_std | +0.164 | 0.7071 | 0.0167 | no |
| ruggedness_corr | +0.151 | 0.7286 | 0.0200 | no |
| neighbor_gain | −0.076 | 0.8643 | 0.0250 | no |
| dynamic_range | +0.038 | 0.9429 | 0.0333 | no |
| frac_above_0.7 | NaN | 1.0000 | 0.0500 | no |
| frac_above_0.9 | NaN | 1.0000 | 0.1000 | no |

The strongest candidate is **gini (ρ = −0.655, p = 0.0893)**: higher reward-concentration
("needle-like") landscapes associate with *lower* exploration winning. The sign is mechanistically
sensible, and p = 0.0893 would clear an uncorrected α = 0.10 — but it is an order of magnitude above
its Holm threshold of 0.0077, so it fails as pre-registered.

### Two observations about Criterion B, neither of which rescues the verdict

- **Two features are degenerate.** `frac_above_0.7` and `frac_above_0.9` returned ρ = NaN: no
  oracle in the suite ever exceeded those reward thresholds, so the features are constant and
  mathematically incapable of correlating with anything. They still consume Holm budget. **This
  does not change the outcome**: dropping them leaves 11 features and lifts gini's threshold only
  from 0.0077 to 0.0091, still far below p = 0.0893. Worth fixing in Phase 1 for cleanliness, not
  as a route to a different verdict.
- **Criterion B was screened on a backbone that confirmed no flip.** The pre-registration fixes
  Criterion B to "the LM backbone", and the implementation uses `reinvent` — yet `reinvent` produced
  **0** confirmed flipped pairs, while the 2 confirmed pairs came from `augmented_memory`. So the
  diagnostic screen was run against the backbone whose rankings were *not* shown to flip. This is a
  genuine design mismatch and is the single most useful thing to fix in Phase 1's pre-registration.
  It is recorded here as an observation **only**. Re-running Criterion B on `augmented_memory` to
  see whether it passes would be exactly the post-hoc, outcome-dependent analysis the
  pre-registration prohibits, and it has **not** been done.

---

## Supporting results

### Mean AUC-top-10 by cluster × setting × backbone

Each cell averages 6 runs (2 oracles × 3 seeds). Bold marks the row winner.

**reinvent** (overall mean 0.3901)

| Cluster | low | mid | high | Winner |
|---|---|---|---|---|
| similarity | 0.3792 | **0.3865** | 0.3865 | mid |
| isomer | **0.4997** | 0.4890 | 0.4177 | low |
| mpo | 0.3449 | 0.3422 | **0.3565** | high |
| bioactivity | 0.3578 | 0.3522 | **0.3684** | high |

**augmented_memory** (overall mean 0.4131)

| Cluster | low | mid | high | Winner |
|---|---|---|---|---|
| similarity | 0.4306 | **0.4429** | 0.4053 | mid |
| isomer | **0.6185** | 0.5392 | 0.5293 | low |
| mpo | 0.3406 | **0.3485** | 0.3334 | mid |
| bioactivity | **0.3616** | 0.3408 | 0.2667 | low |

**graph_ga** (overall mean 0.4378)

| Cluster | low | mid | high | Winner |
|---|---|---|---|---|
| similarity | **0.4002** | 0.3852 | 0.3850 | low |
| isomer | **0.6102** | 0.6038 | 0.6011 | low |
| mpo | **0.3700** | 0.3368 | 0.3512 | low |
| bioactivity | **0.4112** | 0.4111 | 0.3879 | low |

Note the non-neural control (`graph_ga`) has the **highest** overall mean AUC of the three
backbones (0.4378), and prefers `low` exploration uniformly.

### Per-oracle mean AUC, reinvent (3 seeds per cell)

| Oracle | low | mid | high | Winner |
|---|---|---|---|---|
| albuterol_similarity | 0.4315 | 0.4347 | **0.4350** | high |
| celecoxib_rediscovery | 0.3270 | **0.3384** | 0.3380 | mid |
| drd2 | 0.5344 | 0.4912 | **0.5845** | high |
| isomers_c7h8n2o2 | **0.4738** | 0.4427 | 0.3158 | low |
| isomers_c9h10n2o2pf2cl | 0.5257 | **0.5353** | 0.5197 | mid |
| jnk3 | 0.1812 | **0.2132** | 0.1524 | mid |
| ranolazine_mpo | 0.4788 | 0.4826 | **0.5213** | high |
| sitagliptin_mpo | **0.2109** | 0.2018 | 0.1917 | low |

### Per-oracle AUC spread across all 216 runs

| Oracle | Cluster | n | mean | min | max |
|---|---|---|---|---|---|
| albuterol_similarity | similarity | 27 | 0.4622 | 0.4123 | 0.5266 |
| celecoxib_rediscovery | similarity | 27 | 0.3381 | 0.2888 | 0.3686 |
| isomers_c7h8n2o2 | isomer | 27 | 0.5594 | 0.1212 | 0.7785 |
| isomers_c9h10n2o2pf2cl | isomer | 27 | 0.5314 | 0.4657 | 0.5989 |
| sitagliptin_mpo | mpo | 27 | 0.1964 | 0.1313 | 0.2556 |
| ranolazine_mpo | mpo | 27 | 0.4979 | 0.4438 | 0.5365 |
| drd2 | bioactivity | 27 | 0.5323 | 0.3185 | 0.6953 |
| jnk3 | bioactivity | 27 | 0.1917 | 0.1281 | 0.3186 |

Overall: min 0.1212, max 0.7785, mean 0.4137, **zero exactly-zero AUCs**.

### Landscape diagnostics (300 budgeted calls × 3 seeds × 8 oracles)

| Oracle | Cluster | reward_mean | frac>0.5 | gini | ruggedness_corr | basin_width | neighbor_gain |
|---|---|---|---|---|---|---|---|
| albuterol_similarity | similarity | 0.251 | 0.002 | 0.134 | 0.316 | 0.690 | 0.060 |
| celecoxib_rediscovery | similarity | 0.144 | 0.000 | 0.244 | 0.390 | 0.597 | 0.017 |
| isomers_c7h8n2o2 | isomer | 0.005 | 0.000 | 0.980 | 0.384 | 0.447 | 0.244 |
| isomers_c9h10n2o2pf2cl | isomer | 0.024 | 0.003 | 0.916 | 0.424 | 0.487 | 0.176 |
| sitagliptin_mpo | mpo | 0.010 | 0.000 | 0.909 | 0.205 | 0.287 | 0.077 |
| ranolazine_mpo | mpo | 0.068 | 0.003 | 0.597 | 0.578 | 0.679 | 0.095 |
| drd2 | bioactivity | 0.007 | 0.000 | 0.769 | 0.252 | 0.363 | 0.191 |
| jnk3 | bioactivity | 0.017 | 0.000 | 0.628 | 0.345 | 0.473 | 0.034 |

The clusters do separate on the frozen features (gini: isomer 0.980/0.916 vs similarity
0.134/0.244), which is why Criterion B was worth testing — but separation of clusters is not the
same as correlation with the *winning setting*, which is what Criterion B measures and what failed.

---

## Sweep execution

| | |
|---|---|
| Grid | 8 oracles × 3 backbones × 3 settings × 3 seeds = **216** |
| Budget | 1000 true oracle calls per run (all 216 runs confirmed at 1000) |
| Completed | 216 / 216 (**100.0%**), status `ok` for every run |
| Missing / duplicate grid cells | 0 / 0 |
| Wall clock | **87.9 min** (launched 01:47:44, finished 03:15:48, 2026-11-20) |
| Total run-seconds | 61,877 (mean 286.5 s per run) — ~11.7× speedup from 12 workers |
| Workers | **12**, single process |
| Launch | `nohup`, detached, `LCE_ROOT` set, local (not sbatch, not a notebook kernel) |
| Exit | 0 |

### VRAM: estimated vs measured vs observed

| Source | Figure |
|---|---|
| **Part 1 analytic estimate** | **894.8 MB = 0.87 GB** per LM run (params 16.0 + prior 16.0 + grads 16.0 + Adam 31.9 + activations 315.0 + CUDA context 500.0) → "~11 concurrent" |
| **Part 5 measured** | **peak allocated 354.9 MB**, peak reserved **420.0 MB** (batch = 64) |
| **Observed during the sweep** | **~4.9 GB of 11.99 GB** steady at 12 concurrent workers (≈41%), stable across all 216 runs with no drift |

The analytic estimate was conservative by roughly 2.5× on the model itself.

### Why no backbone split was needed

The sweep parallelises with `ThreadPoolExecutor`, not a process pool (`src/phase0_lib.py:1572`), so
all workers share **one** CUDA context and one copy of the loaded oracle models. The ~500 MB context
is therefore paid once, not per worker:

```
total VRAM ~= 0.5 GB (shared context) + N x 0.42 GB (per-run reserved)
N = 12  ->  ~5.5 GB projected   vs   ~4.9 GB actually observed
```

Against the ≈9 GB ceiling set for this run, `measured peak × workers = 0.42 × 12 = 5.0 GB` stays
well under, so the sweep ran as a single process over all three backbones. Had it used a process
pool, 12 workers would have meant 12 CUDA contexts (~12 GB) and OOM on a 12 GB card — the split by
backbone would then have been mandatory.

Worker count was set from core count (16 cores → 12), **not** from VRAM. The library's own heuristic
(`src/phase0_source.py:2089`, `min(cores−2, 0.85·VRAM/1.3)`) printed `CPU allows 14, VRAM (12 GB)
allows 7 -> VRAM-bound at 7` and was overridden by the `--workers 12` CLI flag.

---

## Environment

Built as conda env `lce` (python 3.11.18); Jupyter kernel registered as **`mosaic`**.

| Package | Version | Note |
|---|---|---|
| torch | 2.14.1+cu126 | `cuda.is_available() True`, sm_89 native, real kernel verified on device |
| scikit-learn | **1.2.2 (<1.3)** | **pinned — see "silent corruption" below** |
| numpy | 1.26.4 | forced by the sklearn pin |
| pandas | 3.0.6 | forced by the sklearn pin |
| scipy | 1.17.1 | |
| rdkit | 2026.09.2 | survived the numpy downgrade |
| PyTDC | 1.1.15 | installed `--no-deps`, with a patched `rdkit.six` import |
| setuptools | 80.9.0 | `<81`, because `tdc.metadata` imports `pkg_resources` |

GPU driver 610.88 / CUDA UMD 13.3. Note `pytorch-cuda` has no 13.x conda package and the conda
`pytorch` channel is frozen near torch 2.5, so "match the driver" was resolved as "newest build the
driver supports" (CUDA 13.x drivers are backward compatible with older runtimes) — hence pip cu126
wheels. `environment.yml`'s pin was updated 11.8 → 12.4 and the working pip recipe documented in
`src/phase0_source.py`.

### The one environment fact that could have invalidated the science

TDC ships the bioactivity oracles as scikit-learn models pickled before sklearn 1.3, which added
`missing_go_to_left` to the decision-tree node dtype. On sklearn 1.9.1:

- **`jnk3`** raised `ValueError: node array from the pickle has an incompatible dtype` — loud, obvious.
- **`drd2`** loaded with **no error at all** and then returned **0.0 for every molecule**.

`drd2` is the dangerous one. The bioactivity cluster would have scored all-zero reward, Criterion A's
cluster comparison would have been computed over garbage, and nothing anywhere would have raised.
Same four probe molecules, before and after the pin:

| | sklearn 1.9.1 | sklearn 1.2.2 |
|---|---|---|
| `drd2` | `[0.0, 0.0, 0.0, 0.0]` | `[0.0052, 0.0057, 0.0003, 0.0891]` |
| `jnk3` | *crash on load* | `[0.0, 0.0, 0.0, 0.01]` |

These probe values are recorded in `src/phase0_source.py` as a regression tripwire. **After any
environment change, re-check them: an all-zero `drd2` means the sklearn pin has regressed and every
bioactivity number is void.** In this run, `drd2` finished with mean AUC 0.5323 over 27 runs
(range 0.3185–0.6953), confirming the oracle was live throughout.

---

## Defects found and fixed during execution

Six distinct defects had to be fixed before Phase 0 could run correctly on this machine. Each would
have either blocked the run or silently corrupted it.

| # | Defect | Consequence if unfixed | Fix |
|---|---|---|---|
| 1 | `build_notebook.py` uses `read_text()`/`write_text()` with no `encoding`; on Windows this round-trips the UTF-8 source through cp1252 | every em-dash corrupted (`—` → `â€”`) in generated artifacts | run everything under `PYTHONUTF8=1` |
| 2 | PyTDC ≥ 0.4 hard-depends on `tiledbsoma` / `cellxgene-census` / `gget` (single-cell tooling the project never imports); `tiledbsoma` has no Windows wheel and cannot build from source | environment impossible to create | `pip install --no-deps` + only the 10 light deps the oracle path actually imports |
| 3 | setuptools ≥ 81 removed `pkg_resources`, which `tdc.metadata` imports | TDC import failure | pin `setuptools==80.9.0` |
| 4 | numpy's BLAS fails to delay-load (`0xc06d007f`) when the env's `python.exe` is invoked directly without conda activation — imports succeed, but any `np.cov`/`corrcoef` call hard-crashes the process | **the statistics gate crashed inside `scipy.stats.spearmanr` with no output at all** (stdout was block-buffered and lost), and the sweep would have died the same way | invoke everything via `conda run -n lce`; add `Library\bin` to the `mosaic` kernel's PATH |
| 5 | mol_opt is vendored with `git clone --depth 1`, tracking upstream HEAD; current upstream `mutate.py` uses an absolute import (`import main.graph_ga.crossover`) but the project only put `main/graph_ga/` on `sys.path` | **`ModuleNotFoundError: No module named 'main'` — Graph GA, 72 of 216 runs, could not import at all** | also put the mol_opt root on `sys.path` (`src/phase0_source.py:549`) |
| 6 | TDC (through 1.1.15, the latest) does `from rdkit.six import iteritems`; `rdkit.six` was removed in rdkit 2020.03, and the bare `except:` around it re-raises as a misleading "Please install rdkit" | oracle loading failure, misdiagnosed as a missing rdkit | patch the import to the py3 equivalent `def iteritems(d): return iter(d.items())` (one site, `oracle.py:384`; `.orig` backup kept) |

Two further notes on the runbook itself, for whoever runs this next:

- **`RUN_PHASE0.md`'s `PY=<env>/bin/python` idiom is Linux-only** and actively dangerous here — see
  defect 4. Use `conda run`.
- **`RUN_PHASE0.md` Stage 1 expects `git status` to come back empty after a rebuild. It cannot.**
  The committed notebook was normalized by Jupyter after building (literal UTF-8, cell `id`s,
  different key order), while the builder emits `json.dumps` defaults (`\u2014`, no ids). No builder
  run on any OS reproduces it byte-for-byte. Integrity was instead verified semantically:
  **72/72 cells and all metadata identical** between the committed and rebuilt notebook.
- The test suite contains **64** assertions, not the 56 the runbook states. All 64 passed, and were
  re-run and re-passed after the sklearn/numpy downgrade.

---

## Open items — flagged, deliberately not acted on

1. **mol_opt is not pinned to a commit.** It is cloned `--depth 1` from upstream HEAD (currently
   `2da631b`). `vendor/checksums.json` records hashes only *after* cloning, so it detects drift
   retroactively rather than preventing it. For a pre-registered study whose primary metric is
   specified as matching `mol_opt/main/optimizer.py::top_auc` *exactly*, that dependency is a moving
   target — defect 5 above is precisely this risk materialising. **Recommend pinning the SHA.**
2. **The pre-registration hash does not cover what it claims.** `PREREGISTRATION.prohibited` states
   the diagnostic feature list is "frozen in Part 8 and hashed with this document", but
   `prereg_hash()` hashes only the `PREREGISTRATION` dict. The Part 8 feature list is **not** in the
   hash, so adding or removing a feature would not trip the mismatch guard. Not changed here
   (invariant: never alter a frozen criterion or feature list); **worth closing for Phase 1.**
3. **Criterion B's backbone choice** — see "Two observations" above. It screened `reinvent`, which
   confirmed no flip, while the flips came from `augmented_memory`. Fix in Phase 1's
   pre-registration, not retroactively here.
4. **Two degenerate diagnostic features** (`frac_above_0.7`, `frac_above_0.9`) are constant and
   consume Holm budget. Removing them would not change this verdict.
5. **Not coding tasks, per the runbook's §10** — still open and not decided here: the step-0 novelty
   check (literature search), adding the online bandit baseline to Phase 2, and settling the
   the schedule question (~58 days against a 56-day plan, zero slack).

---

## Artifacts

| Path | Contents |
|---|---|
| `results/phase0_verdict.json` | the verdict, both criteria's full statistics, prereg hash, run counts (committed, `47dba26`) |
| `preregistration.json` | the frozen decision rule (committed, `0a28b5a`, before any run existed) |
| `results/runs/*.json` | 216 per-run records (gitignored) |
| `results/diagnostics.json` | landscape diagnostics, 8 oracles (gitignored) |
| `figures/h1_setting_by_cluster.png` | setting × cluster AUC, the Criterion A picture |
| `figures/per_oracle_reinvent.png` | per-oracle heatmap, reinvent |
| `figures/diagnostics_by_cluster.png` | diagnostic features by cluster |
| `logs/sweep.log` | full sweep log, 216 runs |
| `logs/verdict.log` | full verdict computation output |
| `logs/test_core2.log` | 64/64 statistics assertions on the final stack |
| `vendor/checksums.json` | sha256 of the 4 vendored mol_opt files (`Prior.ckpt`, `Voc`, `crossover.py`, `mutate.py`) |

## Reproducing this

```bash
cd <repo-root>
CONDA=<conda-prefix>/Scripts/conda.exe

# 1. rebuild generated artifacts from the single editable source
PYTHONUTF8=1 python src/build_notebook.py

# 2. statistics gate (64 assertions) -- MUST pass before any sweep
LCE_ROOT="$PWD" PYTHONUTF8=1 "$CONDA" run --no-capture-output -n lce \
    python -u tests/test_core.py

# 3. notebook pass 1: vendoring, oracles, smoke tests, diagnostics, prereg
LCE_ROOT="$PWD" PYTHONUTF8=1 "$CONDA" run --no-capture-output -n lce \
    python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 "Phase 0/landscape_phase0.ipynb"

# 4. the sweep (resumable; completed runs are skipped; never pass --force)
LCE_ROOT="$PWD" PYTHONUTF8=1 nohup "$CONDA" run --no-capture-output -n lce \
    python -u scripts/run_sweep.py --workers 12 > logs/sweep.log 2>&1

# 5. the verdict
cat > /tmp/verdict.py <<'PY'
import sys; sys.path.insert(0, r"<repo-root>\src")
from phase0_lib import (load_results, compute_all_diagnostics, phase0_verdict,
                        plot_setting_by_cluster, plot_per_oracle_heatmap)
df   = load_results()                 # results/runs/*.json
diag = compute_all_diagnostics()      # cached in results/diagnostics.json
out  = phase0_verdict(df, diag)       # writes results/phase0_verdict.json
plot_setting_by_cluster(df); plot_per_oracle_heatmap(df, backbone="reinvent")
print(out["verdict"])
PY
LCE_ROOT="$PWD" PYTHONUTF8=1 "$CONDA" run --no-capture-output -n lce \
    python -u /tmp/verdict.py
```

Every command must go through `conda run` (defect 4). `PYTHONUTF8=1` everywhere (defect 1).

---

*Generated 2026-11-20 from the artifacts listed above. Every number in this document comes from a
command whose output is in `logs/`; no value is estimated or carried over from the runbook.*
