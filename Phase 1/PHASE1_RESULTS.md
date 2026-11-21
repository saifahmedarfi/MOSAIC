# Phase 1 — Pre-Registered Confirmation: Results

**Landscape-Conditional Exploration** · Phase 1 (characterisation + economics), gated on Phase 0's `PARTIAL`

| | |
|---|---|
| **Verdict** | **`NULL`** |
| Decided at | 2026-11-20 18:22:51 |
| Pre-registration | `sha256 bad00ca8186a58d74c1bcaa5615f648d4fd4c0b2aac78f44d52386c50a9d2094` (v2.0.0, frozen 2026-11-20) |
| Prereg status at decision time | `verified` — in-code rule matched the frozen file (no `REFUSED`) |
| Diagnostic feature set | `sha256 ddf2761f218a77c2849c64627168a6b26d21b338df5918f2af02203c3cd1263b` (14 features), matched |
| Gated by | Phase 0 `sha256 fca4982…`, verdict `PARTIAL` — recorded inside the Phase 1 prereg |
| Sweep completion | **405 / 405 runs ok (100.0%)**, 0 failed, 0 missing cells, 0 duplicates |
| Verdict is provisional? | **No** — the sweep is complete |
| Machine | NVIDIA GeForce RTX 4070 SUPER (Ada sm_89, 11.99 GB), 16 cores, Windows 11 |
| Provenance | prereg commit `858229b` at **14:24:28**, first sweep write at **14:24:37** (9 s later) |

The pre-registration was committed **before any Phase 1 run existed** — `results/runs_phase1/` did
not exist at `858229b`, and the sweep log was created nine seconds after that commit. As in Phase 0,
the ordering claim is verifiable from timestamps rather than asserted.

---

## Three caveats that must accompany any number in this document

**1. AUC-top-10 is not a mean reward.** The PMO integral starts from `prev = 0`, so it ramps in from
zero and a *constant-0.42* run scores **0.4179**. Absolute AUC values here must never be described as
mean rewards.

**2. `valsartan_smarts` is a dead oracle.** All 27 of its runs scored exactly **0.0000**, and its
diagnostic probe returned all-zero rewards (`gini = 0`, `reward_mean = 0`, `dynamic_range = 0`). This
is a known property of the PMO oracle, not a bug in this stack. It is tagged `unassigned` and is
therefore excluded from every cluster-level test; it survives only in the whole-set descriptive
tables, where it drags the per-oracle means down. Flagged, not removed — removing it after seeing the
data would be exactly the kind of post-hoc filtering the pre-registration forbids.

**3. Criterion B ran on 9 oracles, not the 15 its prose names.** The frozen prereg text says
`oracle_set: "confirmation (15 held-out oracles) only"`, but the frozen *code path* passes
`analysis_frame(...)`, which drops `unassigned` oracles before any test — leaving 9. The code is what
executed, so n = 9 is what is reported. **This is a documentation defect in the pre-registration, not
a deviation from it**: the prose and the code disagreed before any Phase 1 result existed, and the
code was not touched to produce this number. It is recorded here rather than silently reconciled.
The effect is to make Criterion B *less* powered than advertised, which cuts against the hypothesis,
not for it.

---

## Mandated action for `NULL`

Taken verbatim from the frozen pre-registration (`verdict_consequences.NULL`):

> The flip does not generalise beyond Phase 0's isomer-driven pairs. The premise is dead at this
> budget; write up the null.

Concretely: **Phase 2 (the controller) is not unblocked, and under this pre-registration it never
will be at this budget.** The deliverable is the null result plus the characterisation. The verdict
binds; it was not adjusted, and no criterion, α, feature list, oracle set, seed set or budget was
changed to reach it.

---

## Criterion A — does the flip survive on held-out oracles?

**Result: FAIL.** Threshold was ≥ 2 confirmed flipped cluster pairs within a single backbone; the
best backbone has **0**.

Test: the interaction contrast `D = [AUC(c1,e1) − AUC(c1,e2)] + [AUC(c2,e2) − AUC(c2,e1)]` with a
hierarchical bootstrap (10,000 draws: oracles within cluster, then seeds within oracle). Phase 1
additionally requires **(ii) the flip pattern re-selects in ≥ 95% of draws** — the requirement added
because Phase 0's procedure confirms ~4 flips per 40 trials on pure noise and adding (ii) returns 0
per 40.

This criterion fails for **two independent reasons, and both must be stated.**

### Reason 1 — structural: the held-out set cannot supply two testable pairs

The confirmation set is the 15 PMO oracles that Phase 0 did not use. Its cluster composition is an
accident of what PMO contains, not a design choice:

| Cluster | Oracles in the confirmation set | Testable? |
|---|---|---|
| `mpo` | 5 — amlodipine, fexofenadine, osimertinib, perindopril, zaleplon | yes |
| `similarity` | 3 — mestranol, thiothixene, troglitazone | yes |
| `bioactivity` | **1** — gsk3b | **no** — a cluster with < 2 oracles is `insufficient` by construction |
| `isomer` | **0** | **no** — PMO holds exactly 2 isomer oracles and Phase 0 used both |
| `unassigned` | 6 — deco_hop, median1, median2, qed, scaffold_hop, valsartan_smarts | excluded by rule |

Two testable clusters admit **at most one** cluster pair. The frozen threshold asks for two. So the
threshold could not have been met by any data whatsoever. This was established **before the sweep
finished**, by running `criterion_A` on a synthetic, noiseless, maximally-strong planted flip
(`D = +1.2000`): it returned `n_confirmed = {reinvent: 1, augmented_memory: 1, graph_ga: 1}` against
`min_pairs = 2`, i.e. `passes = False` on data engineered to pass.

### Reason 2 — empirical: the one pair that *was* testable does not replicate

This is the part that matters, and it is why no amendment could have rescued the criterion.

| Backbone | Testable pairs | Pair | D | 95% CI (hierarchical) | Replication | Status |
|---|---|---|---|---|---|---|
| `reinvent` | 0 | bioactivity vs mpo | +0.1178 | — | — | insufficient |
| | | bioactivity vs similarity | +0.1156 | — | — | insufficient |
| `augmented_memory` | 1 | bioactivity vs mpo | +0.0413 | — | — | insufficient |
| | | **mpo(low) vs similarity(mid)** | **+0.0030** | **[−0.0102, +0.0170]** | **0.28** | **not confirmed** |
| `graph_ga` | 1 | bioactivity vs mpo | +0.0240 | — | — | insufficient |
| | | bioactivity vs similarity | +0.0126 | — | — | insufficient |
| | | **mpo(low) vs similarity(high)** | **+0.0014** | **[−0.0110, +0.0127]** | **0.21** | **not confirmed** |

The one pair that could be tested was tested, on both backbones where it existed. Its contrast is
**+0.003 and +0.001 AUC** — within the seed noise, which has a median within-cell standard deviation
of **0.0063**. Its bootstrap CI spans zero. Its replication rate is **0.21–0.28** against a
pre-registered requirement of **0.95**.

**Therefore: even with `min_pairs` lowered to 1 — the amendment that the structural problem would
have invited — Criterion A still fails.** The verdict is `NULL` on the evidence, not merely on a
threshold technicality. No pre-registration amendment was made, and none was needed.

For comparison, the winning settings by cluster (the quantity whose disagreement defines a flip):

| Backbone | bioactivity | mpo | similarity |
|---|---|---|---|
| `reinvent` | explore_high | explore_low | explore_low |
| `augmented_memory` | explore_mid | explore_low | explore_mid |
| `graph_ga` | explore_mid | explore_low | explore_high |

The winners *do* differ by cluster in every backbone — which is precisely why Phase 0's weaker test
passed and Phase 1's replication requirement does not. The differences are not stable across
resamples of the oracles within a cluster.

![Criterion A](figures/phase1_setting_by_cluster.png)

The lines are very nearly parallel for `mpo` and `similarity` (the two populated clusters); the
visible kink is in `bioactivity`, which is a single oracle and therefore has no within-cluster
resampling at all.

---

## Criterion B — does a cheap diagnostic track the winning setting?

**Result: FAIL**, on the primary and on all 13 secondaries.

The primary was a single directional hypothesis carried over from Phase 0's screen and frozen in
advance: **`gini`, negative direction, α = 0.05, backbone `augmented_memory`, no multiplicity
penalty** — legitimate without a penalty precisely because Phase 0 is prior evidence rather than this
phase's data, which is also why it had to be confirmed on oracles that did not generate it.

| | Phase 0 (discovery, n = 8) | Phase 1 (confirmation, n = 9) |
|---|---|---|
| `gini` vs winning-setting index | ρ = **−0.655**, p = 0.0893 | ρ = **+0.410**, p = 0.8679 |
| Decision | screen passed at α = 0.10? no (Holm) | **FAIL** at α = 0.05, one-sided negative |

**The sign reversed.** The one-sided p of 0.8679 is what a correlation pointing the wrong way looks
like under a directional test. This is a clean non-replication, not a near miss.

### Secondaries — Benjamini–Hochberg, q = 0.05

| Feature | ρ | p | Significant |
|---|---|---|---|
| dynamic_range | +0.392 | 0.2993 | no |
| reward_skew | +0.356 | 0.3415 | no |
| reward_std | +0.321 | 0.3912 | no |
| reward_kurtosis | +0.321 | 0.3930 | no |
| ruggedness_absdelta | +0.294 | 0.4355 | no |
| ruggedness_corr | +0.187 | 0.6297 | no |
| neighbor_gain | +0.080 | 0.8389 | no |
| basin_width | +0.080 | 0.8435 | no |
| reward_mean | −0.036 | 0.9348 | no |
| q50_gap, q75_gap, q90_gap, q99_gap | — | — | **degenerate, excluded before correction** |

**Nothing survives, and nothing comes close** — the smallest uncorrected p is 0.30, before any
correction is applied.

The four quantile-gap features — the headline redesign mandated by Phase 0's `PARTIAL` — are
**degenerate**: they returned no finite variation and were excluded before the correction. The reason
is in the diagnostic code path: `phase0_lib.compute_diagnostics` does not return the raw reward
vector on every path, and when it does not, `compute_diagnostics1` records `NaN` for the quantile
gaps rather than inventing them. That fallback fired for every oracle. **So the redesigned feature
set was not actually tested in this phase.** That is a defect in the Phase 1 build, found after the
fact, and it is reported rather than patched-and-rerun: rerunning a diagnostic after seeing a failed
result is p-hacking regardless of how good the engineering reason is. It also does not change the
verdict, which Criterion A had already decided.

![Criterion B primary](figures/phase1_criterionB_primary.png)
![Diagnostics by cluster](figures/phase1_diagnostics_by_cluster.png)

---

## Criterion C — is the flip worth the probe? *(the economics veto)*

**Result: FAIL on all three backbones**, and this is the most decisive number in the study.

This is the question Phase 0 never asked. Headroom is measured with **held-out-seed selection** (the
conditional setting is chosen on 2 seeds and scored on the rotated-out 3rd), which removes the
winner's curse that inflated Phase 0's estimate. The probe cost is measured in the metric's own units
from the persisted traces: `AUC@1000 − AUC@700`, since a 300-call probe is 30% of the budget.

| Backbone | Best fixed setting | Conditional (CV) | **Headroom** | 95% CI | **Probe cost** | **Net** | 95% CI | Viable |
|---|---|---|---|---|---|---|---|---|
| `reinvent` | explore_high = 0.4359 | 0.4367 | **+0.0008** | [−0.0029, +0.0048] | +0.0237 | **−0.0230** | [−0.0266, −0.0190] | **no** |
| `augmented_memory` | explore_mid = 0.4393 | 0.4352 | **−0.0041** | [−0.0146, +0.0027] | +0.0270 | **−0.0311** | [−0.0416, −0.0243] | **no** |
| `graph_ga` | explore_high = 0.4297 | 0.4273 | **−0.0025** | [−0.0061, +0.0014] | +0.0224 | **−0.0249** | [−0.0285, −0.0210] | **no** |

Read the first two numeric columns against each other. **A perfect landscape-conditional chooser,
given free information, buys between −0.004 and +0.001 AUC.** On two of three backbones the
conditional policy is *worse* than simply fixing one setting — that is what a selection rule looks
like when it is fitting noise. The probe that would supply the information costs **0.022–0.027 AUC**,
roughly **6–30× the entire gain it is meant to enable**, and every net CI excludes zero on the
negative side.

![Criterion C](figures/phase1_economics.png)

This result is robust to Criterion A. Even if the flip were real and perfectly detectable, paying 300
of 1000 calls to detect it loses AUC at this budget. **The controller is not viable at a 1000-call
budget** — and that conclusion does not depend on a single thing that Criterion A did or did not
show.

---

## Criterion D — selection across methods *(reported, not part of the verdict rule)*

**Result: flips — 3 distinct winning arms across 3 clusters.** Criterion D does not enter the verdict
rule; it is the pre-registered check on whether the *method*, rather than the knob, is the thing
worth selecting per landscape.

| Cluster | Winning arm | Top 3 arms by mean AUC |
|---|---|---|
| bioactivity | `reinvent:explore_high` | reinvent:high = 0.455, augmented_memory:mid = 0.454, augmented_memory:low = 0.414 |
| mpo | `reinvent:explore_low` | reinvent:low = 0.529, graph_ga:low = 0.528, graph_ga:high = 0.528 |
| similarity | `augmented_memory:explore_mid` | augmented_memory:mid = 0.291, augmented_memory:low = 0.289, reinvent:low = 0.287 |

The margins deserve the same scepticism Criterion A got: first-to-second gaps are **0.001–0.002 AUC**
in `mpo` and `similarity`, against a median within-cell seed SD of 0.0063. Criterion D's bare count
of distinct winners is satisfied; whether those winners would survive the replication requirement
that Criterion A imposes was not tested, because the rule does not ask it to be.

---

## Supporting results

### Mean AUC-top-10 by backbone × cluster × setting (analysis frame, 243 runs)

| Backbone | Cluster | explore_low | explore_mid | explore_high | n per cell |
|---|---|---|---|---|---|
| augmented_memory | bioactivity | 0.4138 | **0.4543** | 0.3445 | 3 |
| augmented_memory | mpo | **0.5261** | 0.5252 | 0.5251 | 15 |
| augmented_memory | similarity | 0.2892 | **0.2912** | 0.2821 | 9 |
| graph_ga | bioactivity | 0.3728 | **0.3960** | 0.3864 | 3 |
| graph_ga | mpo | **0.5283** | 0.5275 | 0.5276 | 15 |
| graph_ga | similarity | 0.2802 | 0.2780 | **0.2810** | 9 |
| reinvent | bioactivity | 0.3424 | 0.3560 | **0.4548** | 3 |
| reinvent | mpo | **0.5286** | 0.5236 | 0.5232 | 15 |
| reinvent | similarity | 0.2874 | 0.2845 | **0.2842** | 9 |

In the two clusters with more than one oracle, the three settings differ by **0.001–0.009 AUC**. The
only large spreads (0.11 for reinvent bioactivity) are in the single-oracle cluster.

### Overall — the knob does almost nothing, and neither does the method

| Backbone | mean | std | min | max | n |
|---|---|---|---|---|---|
| augmented_memory | 0.4137 | 0.2292 | 0.0 | 0.9253 | 135 |
| graph_ga | 0.4103 | 0.2310 | 0.0 | 0.9232 | 135 |
| reinvent | 0.4113 | 0.2283 | 0.0 | 0.9275 | 135 |

| Setting | mean | std | n |
|---|---|---|---|
| explore_low | 0.4120 | 0.2294 | 135 |
| explore_mid | 0.4125 | 0.2290 | 135 |
| explore_high | 0.4109 | 0.2301 | 135 |

| Backbone × setting | explore_low | explore_mid | explore_high |
|---|---|---|---|
| augmented_memory | 0.4153 | **0.4189** | 0.4070 |
| graph_ga | 0.4100 | 0.4105 | **0.4106** |
| reinvent | 0.4106 | 0.4082 | **0.4150** |

The total spread across all nine arms is **0.0119 AUC**. The std column (0.229) is dominated by
*which oracle* a run was on, not by any choice this study manipulates. Note also the contrast with
Phase 0, where `graph_ga` had the highest overall mean (0.4378 vs 0.4131 vs 0.3901) — on the 15
held-out oracles the three backbones are within **0.0034** of each other.

### Per-oracle mean AUC-top-10 (all 15 confirmation oracles)

| Oracle | Cluster | augmented_memory | graph_ga | reinvent |
|---|---|---|---|---|
| amlodipine_mpo | mpo | 0.4671 | 0.4846 | 0.4740 |
| deco_hop | unassigned | 0.5747 | 0.5760 | 0.5693 |
| fexofenadine_mpo | mpo | 0.6419 | 0.6497 | 0.6308 |
| gsk3b | bioactivity | 0.4042 | 0.3851 | 0.3844 |
| median1 | unassigned | 0.1935 | 0.1815 | 0.1940 |
| median2 | unassigned | 0.1716 | 0.1724 | 0.1726 |
| mestranol_similarity | similarity | 0.3387 | 0.3391 | 0.3390 |
| osimertinib_mpo | mpo | 0.7316 | 0.7330 | 0.7306 |
| perindopril_mpo | mpo | 0.3995 | 0.4068 | 0.3986 |
| qed | unassigned | 0.9226 | 0.9159 | 0.9214 |
| scaffold_hop | unassigned | 0.4494 | 0.4461 | 0.4457 |
| thiothixene_rediscovery | similarity | 0.2940 | 0.2754 | 0.2904 |
| troglitazone_rediscovery | similarity | 0.2298 | 0.2248 | 0.2267 |
| **valsartan_smarts** | unassigned | **0.0000** | **0.0000** | **0.0000** |
| zaleplon_mpo | mpo | 0.3874 | 0.3649 | 0.3917 |

![Per-oracle AUC](figures/phase1_per_oracle_augmented_memory.png)

### Noise floor — why the effect sizes above are not interpretable as effects

| Quantity | Value |
|---|---|
| Within-cell seed SD (median over 135 cells) | **0.0063** |
| Within-cell seed SD (mean) | 0.0116 |
| Within-cell seed SD (max) | 0.1037 |
| Median AUC spread across the 3 settings, per (oracle, backbone) | **0.0234** |
| Winning setting counts across all 45 (oracle, backbone) combinations | low 19, mid 15, high 10, tie 1 |

The winning-setting counts are close to the 15/15/15 a fair coin would produce. A χ²-style read of
19/15/10 against uniform is unremarkable at n = 45.

![AUC spread](figures/phase1_auc_spread.png)

### Landscape diagnostics (300 budgeted calls × 3 probe seeds × 15 oracles)

| Oracle | Cluster | gini | reward_mean | reward_std | dynamic_range | ruggedness_corr | basin_width | neighbor_gain |
|---|---|---|---|---|---|---|---|---|
| amlodipine_mpo | mpo | 0.357 | 0.2208 | 0.1430 | 0.3733 | 0.4261 | 0.7533 | 0.0144 |
| deco_hop | unassigned | 0.034 | 0.5167 | 0.0502 | 0.0558 | 0.2771 | 1.0000 | 0.0100 |
| fexofenadine_mpo | mpo | 0.509 | 0.2301 | 0.2083 | 0.5003 | 0.5622 | 0.9494 | 0.0213 |
| gsk3b | bioactivity | 0.604 | 0.0276 | 0.0386 | 0.2867 | 0.4523 | 0.5067 | 0.0647 |
| median1 | unassigned | 0.321 | 0.0651 | 0.0370 | 0.1238 | 0.4622 | 0.7233 | 0.0203 |
| median2 | unassigned | 0.140 | 0.1067 | 0.0270 | 0.0932 | 0.4912 | 0.6900 | 0.0076 |
| mestranol_similarity | similarity | 0.204 | 0.1660 | 0.0597 | 0.1822 | 0.4050 | 0.7174 | 0.0332 |
| osimertinib_mpo | mpo | 0.625 | 0.1819 | 0.2208 | 0.7022 | 0.3699 | 0.8400 | 0.0225 |
| perindopril_mpo | mpo | 0.352 | 0.1834 | 0.1148 | 0.2844 | 0.3138 | 0.6345 | 0.0142 |
| qed | unassigned | 0.100 | 0.7329 | 0.1328 | 0.1688 | 0.1183 | 0.6512 | 0.0079 |
| scaffold_hop | unassigned | 0.037 | 0.3731 | 0.0252 | 0.0830 | 0.2938 | 0.8767 | 0.0145 |
| thiothixene_rediscovery | similarity | 0.160 | 0.1607 | 0.0460 | 0.1343 | 0.4471 | 0.6167 | 0.0110 |
| troglitazone_rediscovery | similarity | 0.140 | 0.1355 | 0.0337 | 0.0971 | 0.3186 | 0.6767 | 0.0229 |
| **valsartan_smarts** | unassigned | **0.000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| zaleplon_mpo | mpo | 0.689 | 0.0631 | 0.0909 | 0.4329 | 0.3232 | 0.3433 | 0.0360 |

The four `q*_gap` columns are omitted because every value is `NaN` — see Criterion B.

Note that `gini` does separate the clusters reasonably well (mpo 0.35–0.69 vs similarity 0.14–0.20).
It simply does not predict which *setting* wins, which is the thing Criterion B asked.

---

## The training record — 405 runs, 7,240 logged update steps

Every run persists its full per-step history and its complete score trace. These are flattened into
two committed tables rather than left inside 405 JSON files.

| Backbone | min steps | median | max | mean | median wall (s) |
|---|---|---|---|---|---|
| augmented_memory | 18 | **30** | 53 | 30.8 | 132.8 |
| reinvent | 17 | **18** | 20 | 17.7 | 50.6 |
| graph_ga | 5 | **5** | 6 | 5.1 | 52.1 |

"Steps" are optimiser update steps, not epochs over a fixed dataset — the budget is 1000 oracle
calls, and each backbone spends them at its own batch granularity. All 405 runs spent exactly
**1000** calls (`n_calls` min = max = 1000), so the metric is comparable across every cell.

### Proposal economy (mean per run)

| Backbone | proposals | invalid | duplicate | purged | yield rate |
|---|---|---|---|---|---|
| augmented_memory | 1973.5 | 940.8 | 1.6 | 24.8 | **0.565** |
| graph_ga | 1624.0 | 0.0 | 581.5 | 0.0 | **0.617** |
| reinvent | 1133.4 | 99.6 | 1.3 | 0.0 | **0.884** |

The two LM backbones share a prior but not a yield: `augmented_memory` discards ~48% of its proposals
as invalid SMILES, `reinvent` ~9%. `graph_ga` never emits an invalid molecule (its operators are
graph-level) but re-proposes ~36% duplicates.

![Training curves](figures/phase1_training_curves.png)

The trajectory plot is the null result drawn as a picture: across 15 oracles and 3 seeds, the three
exploration settings produce curves that are indistinguishable until ~500 calls and separated by less
than the shaded ±1 SD band thereafter. The dashed line at 700 calls is the probe-cost boundary from
Criterion C — the vertical gap between the curve at 700 and at 1000 is the AUC a probe would forfeit.

![Training dynamics](figures/phase1_training_dynamics.png)

---

## Sweep execution

| | |
|---|---|
| Grid | 15 confirmation oracles × 3 backbones × 3 settings × 3 seeds = **405** |
| Budget | 1000 true oracle calls per run (all 405 confirmed at exactly 1000) |
| Completed | 405 / 405 (**100.0%**), status `ok` for every run |
| Missing / duplicate grid cells | **0 / 0** (verified by set difference against the full product) |
| Wall clock | **172.5 min** (launched 14:24:37, finished 17:17:08, 2026-11-20) |
| Total run-seconds | **122,285 s = 34.0 GPU-hours**, mean 301.9 s per run — **11.8× speedup** from 12 workers |
| Workers | **12**, single process, `ThreadPoolExecutor` (one shared CUDA context) |
| VRAM observed | ~3.3–5.7 GB of 11.99 GB at 12 concurrent workers |
| Launch | `nohup`, detached, `LCE_ROOT` set, local (not sbatch, not a notebook kernel) |
| Exit | **`SWEEP1_EXIT=0`**, read from the log, not from the wrapper's status |
| Diagnostics | 15 oracles × 3 probe seeds, 4.4–24.8 s per oracle, written to `results/diagnostics_phase1.json` |

The worker count was set from cores (16 → 12), overriding the library's VRAM heuristic, which printed
`CPU allows 14, VRAM (12 GB) allows 7 -> VRAM-bound at 7`. The observed ceiling of 5.7 GB confirms
the heuristic is conservative by roughly 2× for this workload, as it was in Phase 0.

One run took **10,334 s** (2.9 h) against a median of 132.8 s. That is worker contention on a shared
GPU, not a stuck job — it completed with status `ok` and a normal AUC.

---

## Environment

Unchanged from Phase 0 — same conda env `lce` (python 3.11.18), same Jupyter kernel `mosaic`, same
vendored `mol_opt` clone with the four checksummed files.

| Package | Version | Note |
|---|---|---|
| torch | 2.14.1+cu126 | `cuda.is_available() True`, sm_89 |
| scikit-learn | **1.2.2 (<1.3)** | **pinned — the tripwire below depends on it** |
| numpy | 1.26.4 | forced by the sklearn pin |
| rdkit | 2026.09.2 | |
| PyTDC | 1.1.15 | installed `--no-deps` |

**The sklearn tripwire was asserted before the sweep, and passed.** On sklearn ≥ 1.3, TDC's `drd2`
oracle loads with no error and returns `0.0` for every molecule, because the tree node dtype gained a
`missing_go_to_left` field that silently breaks the pre-1.3 pickles. The Phase 1 gate scored four
probe SMILES through `drd2` and asserted at least one non-zero **before** any run was launched. Had
it regressed, the entire bioactivity cluster would have scored zero and nothing would have raised —
exactly the failure mode that `valsartan_smarts` demonstrates is otherwise invisible.

---

## Defects found and fixed during execution

All four were found and fixed **before** the sweep produced any record, so none of them could have
been influenced by a result.

1. **`pmo_oracle_names()` returned `['hparams_default', 'hparams_tune', 'theme']`.** It called
   `retrieve_all_benchmarks("PMO")`, which raises `TypeError` (TDC has no PMO benchmark group), caught
   the exception, and fell through to a glob over YAML stems. The notebook died with
   `ValueError: ('hparams_default', 'does not match to available values.')`. Fixed by parsing the
   `oracle_array` out of the vendored `mol_opt/make_run_prod.sh` — the suite's own production sweep
   script — which yields exactly 23 names and contains all 8 Phase 0 oracles. `dst/evaluate.py`'s list
   was rejected because it has only 21 and omits `sitagliptin_mpo`. **The pre-registration contains no
   oracle list, so this fixed a build defect without touching a pre-registered quantity.**
2. **`ImportError: cannot import name 'CONFIRMATION' from 'phase1_lib'`.** The oracle-set constants
   sat in a `# %% @notebook-only` cell, so the generated library never defined them and the sweep CLI
   could not import them. Moved into the library.
3. **`NameError: name 'P0' is not defined`.** `import phase0_lib as P0` was itself marked
   `@notebook-only`, so every runner in the generated library referenced an undefined name. Moved into
   the library after verifying the test suite only `read_text()`s the library and never imports it.
4. **`RUNS1_DIR = None` in the generated library.** `run_path1()` and `load_results1()` would raise
   `TypeError` outside a notebook kernel — which is exactly where the sweep runs. Replaced with the
   real paths.

Two further defects are reported above rather than fixed, because fixing them after seeing a result
would invalidate the result: the degenerate `q*_gap` features (Criterion B) and the prereg prose/code
disagreement over Criterion B's oracle count (caveat 3).

The kernel also had to be named explicitly: the notebook declares kernelspec `lce`, which does not
exist on this machine (`No such kernel named lce`); passing
`--ExecutePreprocessor.kernel_name=mosaic` resolves it, and `mosaic` is additionally the hardened
kernel whose PATH includes `Library\bin`, which prevents the MSVC delay-load crash (`0xC06D007F`) in
numpy's BLAS.

---

## What this phase decided, and what it did not

**Decided.** At a 1000-call budget, on 15 held-out PMO oracles, with 3 backbones × 3 exploration
settings × 3 seeds: the landscape-conditional exploration premise does not hold. The flip does not
replicate (A), no pre-specified diagnostic predicts the winner (B), and — independently of both —
conditional selection cannot repay the probe that would inform it (C).

**Not decided, and deliberately not acted on here.**

1. **The step-0 novelty check.** Per-instance algorithm selection has a long history in optimisation
   (portfolio solvers for SAT and TSP). Whether the molecular-optimisation version is already
   published is still unverified. It is a literature search, not a coding task.
2. **The online-bandit baseline.** EXP3 or successive halving over the settings as arms. Criterion C
   makes this question *sharper*, not softer: a bandit pays no upfront probe cost, so the 0.022–0.027
   AUC that sank the probe-based controller is not a cost it would incur.
3. **Whether a larger budget changes Criterion C.** The probe cost is a fixed 300 calls; at 10,000
   calls it would be 3% of the budget rather than 30%. The headroom, however, measured −0.004 to
   +0.001 — so the probe is only half the problem, and the smaller half.
4. **Whether a redesigned diagnostic could pass Criterion B.** Untested here, because the quantile-gap
   features were degenerate. Any retest needs a fresh pre-registration and oracles not used in Phase 0
   or Phase 1 — of which PMO has none left.
5. **mol_opt SHA pinning** for the vendor directory, and the Phase 0 `prereg_hash()` scope gap. Both
   carried over from Phase 0, both still open.

---

## Artifacts

| Path | Contents |
|---|---|
| `results/phase1_verdict.json` | the verdict, all four criteria's full statistics, the structural note, prereg hash, run counts |
| `Phase 1/preregistration_phase1.json` | the frozen decision rule (committed `858229b`, 9 s before the first run) |
| `results/runs_phase1/*.json` | **405** per-run records, each with full history and 1000-point score trace (14 MB, gitignored) |
| `results/phase1_run_manifest.csv` | 405 rows — one per run, 22 columns of outcome and cost |
| `results/phase1_training_history.csv` | **7,240** rows — every logged update step of every run |
| `results/diagnostics_phase1.json` | landscape diagnostics, 15 oracles × 3 probe seeds |
| `results/phase1_tables.json` | every supporting table in this document, as data |
| `scripts/phase1_verdict.py` | the exact analysis driver — **hand-written, not a product of `build_phase1.py`** |
| `figures/phase1_setting_by_cluster.png` | Criterion A — setting × cluster, per backbone |
| `figures/phase1_economics.png` | Criterion C — headroom vs probe cost |
| `figures/phase1_criterionB_primary.png` | Criterion B — gini vs winning setting |
| `figures/phase1_diagnostics_by_cluster.png` | 8 diagnostic features by cluster |
| `figures/phase1_per_oracle_augmented_memory.png` | per-oracle AUC × setting, primary backbone |
| `figures/phase1_training_curves.png` | optimisation trajectories, mean ± 1 SD |
| `figures/phase1_training_dynamics.png` | policy and scaffold entropy vs budget |
| `figures/phase1_auc_spread.png` | AUC distribution by backbone, and cost vs outcome |
| `logs/sweep_phase1.log` | full sweep log, 405 runs, ends `SWEEP1_EXIT=0` |
| `logs/phase1_diag.log` | diagnostics computation |
| `logs/phase1_verdict.log` | full criteria + verdict output — every number in this document |
| `PHASE0_RESULTS.md` | the Phase 0 deliverable this phase was gated on |

## Reproducing this

```bash
cd <repo-root>
CONDA=<conda-prefix>/Scripts/conda.exe

# 1. rebuild generated artifacts from the single editable source
PYTHONUTF8=1 python src/build_phase1.py

# 2. notebook pass 1: vendoring, oracles, the sklearn tripwire, the frozen prereg
LCE_ROOT="$PWD" PYTHONUTF8=1 "$CONDA" run --no-capture-output -n lce \
    python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=mosaic "Phase 1/landscape_phase1.ipynb"

# 3. the sweep (resumable; completed runs are skipped; never pass --force)
LCE_ROOT="$PWD" PYTHONUTF8=1 nohup "$CONDA" run --no-capture-output -n lce \
    python -u scripts/run_sweep_phase1.py --workers 12 > logs/sweep_phase1.log 2>&1
echo "SWEEP1_EXIT=$?" >> logs/sweep_phase1.log      # check THIS, not the wrapper's status

# 4. diagnostics, criteria A-D, the verdict and all eight figures.
#    Diagnostics are cached -- delete results/diagnostics_phase1.json to recompute them
#    (~2.5 min); everything else runs in about 90 s.
LCE_ROOT="$PWD" PYTHONUTF8=1 "$CONDA" run --no-capture-output -n lce \
    python -u scripts/phase1_verdict.py > logs/phase1_verdict.log 2>&1
echo "VERDICT_EXIT=$?" >> logs/phase1_verdict.log
```

Step 4 re-runs end to end from a clean checkout and reproduces `NULL` with every figure; that was
verified, not assumed. It refuses to produce a verdict at all if either the pre-registration hash or
the diagnostic-feature hash fails to match the frozen file.

Every command goes through `conda run` (the MSVC delay-load defect), with `PYTHONUTF8=1` everywhere.

---

*Generated 2026-11-20 from the artifacts listed above. Every number in this document comes from a
command whose output is in `logs/`; no value is estimated, and none is carried over from the runbook
or from Phase 0 except where explicitly labelled as a Phase 0 comparison.*
