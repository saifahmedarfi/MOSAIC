# Phase 3 — Pareto-Front Selection vs Scalarized Selection: Design

**Status:** design, awaiting approval. Nothing is built.
**Date:** 2026-11-22 (revised the same day after a builder check; see §10)
**Target:** one RTX 4070 Super (12 GB), 16 cores, Windows 11, 12-worker thread pool.
**Cost:** ~5 min (Stage A) + ~1 h (Stage B).

---

## 1. Intent

### 1.1 The premise

**Hypervolume is where this project's variance actually is.** Phase 2 scored three backbones on
seven metrics in a common unit (per-oracle standardised z, §6.0). The two members of its primary
pair came out like this:

| backbone | `auc_top10` | `hypervolume` |
|---|---:|---:|
| Augmented Memory | +0.0646 | +0.4682 |
| Graph GA | +0.0043 | −0.3744 |
| REINVENT | −0.0688 | −0.0938 |
| **spread** | **0.133** | **0.843** |

Hypervolume separates the field **6.3× more widely than AUC-top-10 does, and it reorders it** —
`graph_ga` moves from the middle to last. Phase 1 had already established that AUC-top-10 at 1000
calls compresses everything this project manipulates into seed noise: total spread across nine arms
was 0.0119 AUC. Phase 2's contribution, even without a verdict, is the observation that a
front-based measure does not.

Phase 3 asks the question that follows: **if the front is where the differences are visible, does
selecting on the front rather than on the scalar produce a better one — and what does that cost on
the scalarized metric the field reports?**

This also tests the parent proposal's own stated criticism of the field directly rather than by
proxy. `Q UNIMOL OPT_proposal.md` §3: *"Multi-objective optimization is frequently oversimplified
through weighted scoring functions that obscure real trade-offs between competing goals."* Nothing
in Phases 0–2 tested that. Phase 3 does.

### 1.2 What this phase does NOT rest on — a correction

An earlier draft of this document claimed Phase 2 measured that "roughly half the achievable Pareto
quality is discarded by scalarizing," from `scalarization_loss` = 0.513, CI [0.367, 0.668]. **That
was a misreading and the premise has been replaced.** As implemented
(`phase2_source.population_metrics`):

```python
thr = scores[order[min(10, len(order)) - 1]]   # the 10th-best scalar score in the buffer
sc_loss = np.mean(scores[mask] < thr)          # fraction of FRONT molecules below it
```

It measures *what fraction of Pareto-optimal molecules the scalar top-10 cut excludes*, and it is
substantially a function of front cardinality. Measured over Phase 2's 405 runs:
`corr(front_size, scalarization_loss) = 0.577`, and per oracle it tracks front size monotonically:

| oracle | mean front size | mean `scalarization_loss` |
|---|---:|---:|
| `amlodipine_mpo` | 1.8 | 0.381 |
| `median1` | 6.7 | 0.149 |
| `ranolazine_mpo` | 30.2 | 0.805 |
| `sitagliptin_mpo` | 106.0 | **0.916** |

20% of runs have a front of ≤2 molecules, where the statistic can only take the values 0, 0.5 or 1,
and the 0.513 headline averages a 0.149–0.916 spread. Worse, an optimizer that successfully
*enlarges* the front pushes `scalarization_loss` **up**, so a goal phrased as "how much of the
0.513 is left" would read success as failure.

`scalarization_loss` therefore appears **nowhere** in Phase 3 — not as a premise, not as a goal,
not as a reported metric. Phase 2's own write-up should be read with §1.2 in hand.

### 1.3 What it satisfies in the parent proposal

§4's Optimization Layer lists "explicit Pareto-front optimization" — unbuilt. §7 promises
"standardized Pareto-based evaluation protocols"; Phase 2 built the measurement half and no
optimizer. §3's criticism of weighted scoring is tested directly. It does not touch §5 (quantum),
the Representation Layer or the Memory Layer.

---

## 2. Goals, non-goals, anti-goals

### Goals

- **G1.** Test whether NSGA-II selection over the verified component decomposition beats scalarized
  selection on **hypervolume**, at equal oracle budget.
- **G2.** Measure what that costs on **`auc_top10`** — the metric Phases 0–2 reported.
- **G3.** Prove the reimplemented GA is faithful to vendored `graph_ga` before any arm contrast is
  interpreted.

The earlier draft's fourth goal (residual scalarization loss) is **removed**, per §1.2.

### Non-goals

- Re-testing landscape-conditional exploration. Phase 1's `NULL` binds and is not reopened.
- Any controller or method-selection claim. Phase 1 foreclosed it.
- Generalising to the RL/LM backbones. Pareto-ifying REINVENT or `augmented_memory` needs an
  invented reward scheme; declined (§5, approach 2).
- Rescuing Phase 2's overlap gate or producing a Phase 2 verdict.
- Budget scaling. The post-Phase-1 memo says do not.
- Any claim about black-box multi-objective optimization (§3.5).

### Anti-goals — failure even if the code works

- **A1. A circular result.** "Optimize hypervolume, win on hypervolume" is not a finding. Defeated
  by Criterion Q being **co-primary** (§6.2).
- **A2. A result caused by the reimplementation.** Defeated by the anchor arm and a halting gate
  (§6.3).
- **A3. A degenerate secondary reported as a finding.** This is Phase 2's defect 5 — a `novelty`
  result that rested on one oracle read as nine. Phase 3's answer is to carry **no** optional
  secondary at all (§6.4).
- **A4. Any claim of seed-level reproducibility.** Phase 2 established runs are not individually
  reproducible. No bit-exact gate may be written (§3.3).
- **A5. Any threshold moved after seeing the data it judges.**

---

## 3. Constraints

### 3.1 Frozen ground

`src/phase0_lib.py`, `src/phase0_source.py`, `src/phase1_source.py`, `src/phase1_stats.py`,
`src/phase1_lib.py` and everything under `Phase 1/` must not be modified or deleted — the guard in
`tests/test_phase2.py` fails on any `M` or `D`; additions are permitted. `src/phase2_*.py` are not
guarded but are **read-only here**: Phase 3 imports them and never edits them.

**Consequence driving the whole design:** the Pareto operator cannot be patched into
`phase0_lib.run_ga_optimizer`. It must be a reimplemented loop, which creates a
reimplementation confound, which is why there are three arms (§4.2).

### 3.2 Environment

`scikit-learn == 1.2.2` exactly (on ≥1.3 TDC's `drd2`/`jnk3` return 0.0 for everything, silently;
the tripwire carries forward), `numpy == 1.26.4`, Python 3.11, torch cu126, rdkit ≥2023.09. 12
workers, one process, `ThreadPoolExecutor`, one shared CUDA context.

### 3.3 No bit-exact gate, by construction

`set_all_seeds` seeds process-global RNGs and the sweep runs 12-wide, so concurrent runs reseed each
other; a 1e-6 tolerance is unsatisfiable in principle. Phase 3's fidelity gate is **distributional
and frozen before any Phase 3 run exists** (§6.3). That is not Phase 2 §9 option 3, which would
have been a gate rewritten after seeing the data it judges.

### 3.4 Oracle reuse — L1 carries forward

All 23 PMO oracles are spent (8 in Phase 0, 15 in Phase 1, 9 reused in Phase 2). Phase 3 uses the
same 9, because they are the only ones with a verified component decomposition. There is no
held-out set inside PMO and Phase 3 does not pretend otherwise. The mitigation is that Phase 3
tests an optimizer's selection rule, not a landscape predictor, so reuse biases nothing it
estimates — but the write-up concedes it in its own voice.

### 3.5 Equal budget, privileged information, and the operational rule

`phase2_components.component_vector` reproduces each oracle to ≤1.1e-16, so components are
available at **zero oracle cost**. Both arms therefore consume identical budget.

**The operational rule that keeps that true, and which goes in the pre-registration:** the treatment
arm may compute component vectors **only for molecules already present in `oracle.mol_buffer`** —
molecules the run has already paid for. No component-based pre-screening of unscored candidates.
Without this rule the treatment arm could component-score arbitrarily many offspring and the budget
comparison would be void. *(My call — the builder check raised it and the design had no rule.)*

The remaining asymmetry is stated as a **scope condition**, not hidden: the treatment arm sees more
information per call. That is defensible — GuacaMol publishes these decompositions and a
practitioner optimizing their own MPO objective knows its components — but it means Phase 3 tests
*"does decomposing a **known** MPO objective and searching its front beat collapsing it"*, and the
result is titled that way.

### 3.6 Five selection events

`graph_ga` at budget 1000 with population 100 / offspring 200 spends its budget in **5–6
generations** (verified: Phase 2's 135 `graph_ga` runs have `steps` min 5, median 5, max 6). The
selection operator therefore fires about five times.

**This is accepted rather than engineered around** (PI decision, §9). The configuration stays
identical to Phases 0–2 so results remain commensurable, and the consequence is pre-registered: a
null may mean "Pareto selection does not help" *or* "five selection events are too few to tell",
and Phase 3 cannot distinguish them. See L3.

---

## 4. Design

### 4.1 Where the scalar enters `graph_ga`

| Line | Decision | Current rule | Pareto replacement |
|---|---|---|---|
| `phase0_lib.py:1194` | mating pool | fitness-proportionate on scalar (`probs = scores / total`) | NSGA-II binary tournament on (rank, crowding) |
| `phase0_lib.py:1214` | survival | `argsort(combined_scores)[::-1][:population_size]` | NSGA-II environmental selection: fill by front, break the last front by crowding distance |

Nothing else in the loop reads the scalar. Start population from `ZINC_POOL` at the seed,
`_reproduce` via vendored `crossover.py`/`mutate.py`, and the `oracle.finish` budget check are
reproduced unchanged.

### 4.1.1 Zero-coordinate molecules — a pre-registered modelling choice

`pareto_front` keeps non-dominated points including those with a zero component;
`hypervolume` then drops every point with a zero coordinate
(`phase2_stats.py:158`, "a zero coordinate encloses no volume"). Because every one of these nine
oracles aggregates its components by **geometric mean**, a zero component means an oracle score of
exactly **0.0**. Such molecules are not an exploration frontier; they are known-worthless under the
project's own objective, and the control arm already excludes them automatically — fitness-
proportionate selection gives a 0.0-scoring molecule probability zero.

Plain NSGA-II on raw vectors would promote them to rank 1. These are not rare: Phase 2's
reproduction gate records `frac nonzero` of 0.594 for `sitagliptin_mpo` and 0.747 for
`zaleplon_mpo`.

**Frozen rule:** both the mating pool and the survival step **restrict to molecules with all
components > 0** before non-dominated sorting. This is the apples-to-apples choice against the
control arm, it is fixed *a priori* on the argument above rather than tuned, and it is in the
pre-registration. The test suite must show the raw-vector variant is implementable and that the
frozen rule is the one used; it must **not** compare their outcomes, which would be tuning.
*(My call — the builder check found the misalignment and §7 did not delegate it.)*

### 4.2 Three arms

| Arm | What it is | Why |
|---|---|---|
| **anchor** | `phase0_lib.run_ga_optimizer`, untouched | fidelity reference |
| **control** | Phase 3's reimplemented loop, **scalar** selection | the real baseline |
| **treatment** | Phase 3's reimplemented loop, **NSGA-II** selection | the hypothesis |

Treatment vs control carries the verdict and isolates the selection operator. Anchor vs control is a
**halting gate** (§6.3): a reimplementation that does not reproduce vendored `graph_ga`
distributionally makes the contrast meaningless.

### 4.3 Metric list

Enumerated here because its sha256 goes in the pre-registration, and because Phase 2's defect 6 was
a metric that was a deterministic rescaling of another consuming its own FDR budget.

```
METRICS = ("auc_top10", "hypervolume", "front_size")
```

`auc_top10` carries Criterion Q, `hypervolume` carries Criterion P, `front_size` is descriptive and
enters no rule. **`front_coverage` is deliberately excluded** — it is `front_size` / n and would be
perfectly collinear. A pairwise-collinearity check over the list is a test, run before the hash.

### 4.4 Two stages

**Stage A — the pilot sweep.** A real three-arm sweep on **3 of the 9 oracles at 5 seeds: 45 runs,
≈5 minutes** at 12 workers (graph_ga averaged 54.5 s per run threaded in Phase 2).

The earlier draft proposed a free post-hoc analysis of Phase 2's persisted buffers instead. **That
was unexecutable** and is withdrawn: `top_auc` (`phase0_lib.py:476`) sorts the buffer by call index
and is a pure function of the `(score, call_index)` pairs, and `population_metrics` computes
`hypervolume` over the **whole buffer**. A post-hoc selector changes neither, so the pilot could
have produced neither threshold. A real sweep exercises the operator and costs five minutes. *(PI
decision, §9.)* A welcome side effect: Phase 3 no longer depends on `results/runs_phase2/`, which is
not in the repository.

Pilot oracles, chosen before it runs to span the dimensionality range, since hypervolume behaviour
depends on dimension: **`median1`** (2 components), **`fexofenadine_mpo`** (3), **`osimertinib_mpo`**
(4).

Stage A produces `δ_P` and the Stage B seed count, and nothing else — **not `ε_Q`**, which §6.2
fixes at 0.14. It is exploratory by
construction and reaches no verdict.

**The ordering contract**, which is the integrity-critical part:

```
1. Stage A pilot sweep runs                -> Phase 3/phase3_pilot.json
2. phase3_pilot.json committed
3. pre-registration written USING those numbers
4. pre-registration frozen, hashed, COMMITTED   <-- before any Stage B record exists
5. Stage B sweep runs
6. analysis; verdict or REFUSED
```

**Stage B — the sweep.** 9 oracles × 3 arms × seeds, `explore_mid`, budget 1000. Seeds set by
Stage A's power analysis, bounded to **[20, 30]**, recorded with its ruling as Phase 2 did
(`seed_ruling`). 540 runs at 20 seeds ≈1.3 h; 810 at 30 ≈2 h.

Stage A's power analysis must satisfy **two** requirements, not one: enough power on P and Q at
the observed effect, **and** a Criterion F interval that fits inside ±0.25 at the chosen seed
count. The second is what sets the floor at 20 — see §6.3.

**Circularity control.** Stage A sets thresholds on 3 oracles that Stage B then tests. Both
criteria are therefore pre-registered **evaluated two ways**: over all 9 oracles, and over the 6
Stage A did not touch (`amlodipine_mpo`, `median2`, `perindopril_mpo`, `ranolazine_mpo`,
`sitagliptin_mpo`, `zaleplon_mpo`). The all-9 result is primary; the held-out-6 result is reported
beside it and any disagreement between them is reported as a finding, not reconciled. Note the
held-out 6 contain no 3-component oracle, since `fexofenadine_mpo` is the only one.

### 4.5 Files

| File | Role |
|---|---|
| `src/phase3_source.py` | **the only editable source.** Percent-format cells; `@notebook-only` cells excluded from the library |
| `src/build_phase3.py` | generates the notebook, `src/phase3_lib.py`, `scripts/run_sweep_phase3.py` |
| `src/phase3_pareto.py` | non-dominated sort, crowding distance, NSGA-II selection, the reimplemented GA loop |
| `src/phase3_stats.py` | the criteria. Pure numpy. **Imports `phase2_stats`** for `canonical_json`, `pareto_front`, `hypervolume`, the z/score core and the bootstrap cell machinery — import, never copy, never edit |
| `scripts/phase3_pilot.py` | Stage A |
| `tests/test_phase3.py` | plain script harness, `check(cond, label)`, as `tests/test_phase1.py` and `test_phase2.py` |
| `Phase 3/landscape_phase3.ipynb` | generated. Named for series consistency with Phases 0–2, by the principal investigator's choice (2026-11-23), overriding my `pareto_phase3.ipynb` |
| `Phase 3/preregistration_phase3.json` | frozen, hashed |
| `Phase 3/phase3_pilot.json` | Stage A output, committed before the prereg |

Everything in `Phase 3/` is new, so nothing touches guarded ground.

---

## 5. Approaches considered

**1 — Narrow: `graph_ga` only, three arms.** Chosen. One swap point, everything else fixed, cheap
enough to afford generous seeds. Trades away generality.

**2 — Broad: Pareto variants of all three backbones.** Rejected. REINVENT and `augmented_memory`
consume a scalar reward, so this means inventing a reward scheme (hypervolume-improvement, or
Pareto-rank shaping) — a research contribution of its own, with no precedent here, multiplying the
ways the phase fails for reasons unrelated to the hypothesis, at 15–30 h instead of 1–2.

**3 — Free post-hoc pilot on Phase 2's buffers.** Chosen first, then **withdrawn as unexecutable**
(§4.4). Kept in the record because the reason matters: both scored metrics are buffer-level and a
selector cannot move them.

**4 — Pilot only, no optimizer.** Rejected. Cannot support the Optimization Layer claim, which is
the point.

**5 — Shrink population/offspring to buy more generations.** Considered after the builder check
(§3.6) and rejected by the PI: it would break commensurability with Phases 0–2, since the anchor arm
would then compare against a configuration the project has never measured.

---

## 6. The frozen rule

### 6.0 The common unit

Phase 2's §5.2.1 unit and its **shared-oracle hierarchical bootstrap**:
`z(m,o,a) = [X(m,o,a) − mean_a X(m,o,·)] / SD_pool(m,o)`, `score(m,a) = mean_o z`. The oracle list
is resampled **once per draw and shared across all three arms**; seeds are resampled independently
within each cell. `N_BOOT = 10_000`. Resampling oracles independently per arm would make a null
true by construction — Phase 2 §5.2.1 explains why, and `phase1_stats.flip_pairs` is the wrong
precedent.

Because `score` is already `mean_o z` in pooled-seed-SD units, every threshold below is **in those
units and is not multiplied by an SD again**. *(My call, confirming an ambiguity the builder check
flagged.)*

Including the anchor arm in the centring is harmless: the per-oracle mean cancels in any arm-vs-arm
difference, and the pooled SD is a within-cell seed SD that only gains degrees of freedom.

### 6.1 Criterion P — primary: does Pareto selection buy hypervolume?

**≥95% of bootstrap draws satisfy `score_hv(treatment) − score_hv(control) ≥ δ_P`.**

`δ_P` is set by Stage A and frozen. The margin sits **inside** the quantile — there is no separate
condition on the point estimate. *(My call: the earlier draft stated two conditions, one on the
point estimate and one on sign, which is a materially different and weaker test. P and Q now use
identical semantics.)*

### 6.2 Criterion Q — co-primary: what does it cost on the scalarized metric?

**≥95% of bootstrap draws satisfy `score_auc(control) − score_auc(treatment) ≤ ε_Q`.**

**`ε_Q` is a fixed, pre-registered `0.14`, not derived from the pilot** (decided 2026-11-22,
§10). It is the design's 95% resolution limit: the bootstrap SE of an arm difference is
≈`sqrt(2 / (n_seeds · n_oracles))` = 0.086 at 30 seeds with 9 oracles, so ≈0.14 z is the
smallest loss this design can distinguish from zero. Criterion Q therefore asserts: **the AUC
loss is indistinguishable from zero at the finest resolution available.** Margin inside the
quantile, as P.

A non-inferiority tolerance must not be derived from the loss it judges. An earlier
implementation set `ε_Q = 1.5 ×` the pilot's observed loss, which would have meant a
worse-performing Pareto arm earning a more forgiving bar — and, with only 3 pilot oracles, from
a very noisy estimate. Since §6.5 declares `PARETO-TRADES` the modal outcome *because* Q decides
the phase, a tolerance that grows with the observed loss tips the phase toward its most
favourable verdict by construction. Rejected for that reason.

**What Stage A still does for Q:** it does not set `ε_Q`. It establishes that Stage B has the
power to test it — planting a true loss of **zero** and requiring Q to pass at ≥ the target
power at the chosen seed count, which is what a non-inferiority power calculation means. The
pilot's observed point loss is reported beside that as the verdict it anticipates, which is
descriptive and gates nothing: an observed loss above 0.14 predicts `PARETO-TRADES` and is a
finding in advance, not an unresolvable design.

**This is the anti-circularity guard and it is co-primary, not secondary.** An optimizer that wins
on the metric it optimizes has shown nothing. The defensible claim is that Pareto search buys front
quality *without* surrendering the scalarized performance the field reports.

### 6.3 Criterion F — fidelity gate, halting

**The 95% bootstrap CI of `score_metric(anchor) − score_metric(control)` lies entirely within
[−0.25, +0.25]**, for both `auc_top10` and `hypervolume`.

This is an equivalence test in the TOST sense, and more seeds make it **easier** to pass — which is
the right direction. *(My call. The earlier draft also required the CI to contain 0, which is
accepting a point null and gets **harder** with more seeds, so the gate would have become more
likely to halt the phase exactly as the pilot bought margin. The builder check caught it. Clause
removed.)*

A failure halts the phase and yields `REFUSED`. The 0.25 figure is checked against Stage A's
measured seed SD before freezing, and may be **tightened, never loosened**.

**The bound implies a seed floor, measured from the implemented criterion** (added 2026-11-22,
§10). With `anchor == control` exactly, 9 oracles and no arm×oracle interaction — the most
favourable case available — the F interval's half-width is:

| seeds | 10 | 15 | 20 | 25 | 30 |
|---|---:|---:|---:|---:|---:|
| CI half-width | 0.307 | 0.249 | 0.208 | 0.189 | 0.170 |
| independently reproduced | 0.310 | 0.248 | 0.212 | 0.192 | 0.172 |
| passes at 0.25 | no | **marginal** | yes | yes | yes |

Swept over 6 fixture RNGs × 3 bootstrap seeds at `n_boot = 10000`, **15 seeds passes 7 of 18
times** (`lo` ∈ [−0.253, −0.236], `hi` ∈ [0.240, 0.257]) while **20 seeds passes 18 of 18**
(`lo` ∈ [−0.219, −0.205], `hi` ∈ [0.202, 0.215]).

So at 15 seeds a **perfectly faithful reimplementation is a coin flip against its own fidelity
gate** — decided by arithmetic rather than by infidelity, which is the dead-gate failure that cost
Phase 2 its verdict. Because the bound may only be tightened, the remedy is seeds, and the Stage B
floor is therefore **20**.

An earlier wording of this section said flatly that a faithful reimplementation "fails" at 15.
That overstated a marginal measurement taken at one fixture configuration, and is corrected here:
7-of-18 is the measured fact, and it is a stronger argument for the floor than a single
half-width was.

Relatedly, the bootstrap SE of an arm difference is ≈`sqrt(2 / (n_seeds · n_oracles))` = 0.086 at
30 seeds, so **no margin below ≈0.14 z is testable at 95%**. That is the resolution Stage A must
set `δ_P` against, and a `δ_P` below it means `resolvable = False`. `ε_Q` is fixed AT this floor
(§6.2).

### 6.4 No optional secondaries

The earlier draft carried a "front extent" criterion R as a practitioner-preference proxy. **It is
removed.** Three reasons, all from the builder check and all verified: front sizes average 1.8–2.8
on three of the nine oracles, so it is unmeasurable on much of the grid; there is no honest way to
aggregate a per-dimension quantity across oracles with 2, 3 and 4 components of different physical
meaning; and no "surviving oracle" definition exists for it, so the ≥3-survivor guard could not be
implemented. Phase 2's `novelty` is the precedent for what shipping such a metric costs.

`front_size` is reported descriptively, with no threshold and no rule.

Any fraction-over-oracles statistic that Phase 3 does report requires **≥3 surviving oracles** and
prints the survivor count beside it — Phase 2's defect 5 corrected at the root, where the guard
fired only at *zero* survivors and a single survivor produced a p = 0.0003 that read as
nine-of-nine.

### 6.5 Verdicts

| Verdict | Condition | Consequence |
|---|---|---|
| `PARETO-RECOVERS` | P and Q both pass | Front quality is buyable at no scalarized cost. The parent proposal's Optimization Layer is justified empirically. |
| `PARETO-TRADES` | P passes, Q fails | Front quality is purchasable only against scalarized performance. Report the exchange rate — a useful result, not a win. |
| `PARETO-NULL` | P fails | No recoverable headroom at 1000 calls **within five selection events** (§3.6). Phase 2's hypervolume spread is then a property of the backbones, not of what selection can reach. |
| `REFUSED` | prereg or metric-list hash mismatch, Criterion F fails, or <5 oracles usable | The mechanism working. Not a bug to fix. |

**Declared in advance: `PARETO-TRADES` is the modal outcome I expect**, with `PARETO-NULL` a close
second for the mechanical reason in §3.6. P is likely to pass — the treatment arm optimizes what P
measures — so Q is where the phase is decided. Writing this down now is what stops it being spun as
a win later.

### 6.6 Stop conditions

Before Stage B: Stage A shows the available effect unresolvable at 30 seeds (then Phase 3 reports
the pilot as a measurement and does not spend the sweep); the `drd2` tripwire fails; fewer than 5 of
9 oracles pass a re-run of Phase 2's reproduction gate; the pre-registration hash does not verify.
After Stage B: Criterion F fails.

---

## 7. What is left to the builder

- NSGA-II tie-breaking, and the crowding-distance convention at front boundaries (infinite distance
  for extremes is standard; state the choice in code). **Not** the zero-coordinate rule — §4.1.1
  freezes that.
- Whether `phase3_stats` re-exports `phase2_stats`'s helpers through a thin wrapper or imports them
  directly. Import; never copy; never edit `phase2_stats`.
- Figure selection and layout.
- The pilot's power-analysis implementation, following `scripts/phase2_power.py`.
- `δ_P` and the seed count are **not** the builder's: Stage A produces them. `ε_Q` is neither's
  — §6.2 fixes it at 0.14.

## 8. Limitations to carry into the write-up

1. **L1 — oracle reuse.** All 9 oracles appeared in Phases 0–2. No held-out set exists in PMO.
2. **L2 — privileged information.** The treatment arm sees the component decomposition (§3.5). The
   result is about *known* MPO objectives and must be titled that way.
3. **L3 — five selection events.** The operator fires ~5 times (§3.6). A `PARETO-NULL` cannot be
   distinguished from "too few generations to tell." Accepted deliberately to keep the configuration
   commensurable with Phases 0–2.
4. **L4 — one optimizer family.** `graph_ga` only. Nothing generalises to the RL/LM backbones.
5. **L5 — partial scalarization on 2 of 9 oracles.** For `sitagliptin_mpo` and `zaleplon_mpo` one
   "component" is `_isomer(...)`, itself a geometric mean of per-element Gaussians plus an
   atom-count Gaussian. It is treated as **one** Pareto dimension, because decomposing it would
   break the reproduction gate that makes the components trustworthy. *(My call.)*
6. **L6 — runs are not individually reproducible** (Phase 2's finding). Distributional results are
   sound; cell-level provenance is not.
7. **L7 — the premise is weaker than the earlier draft claimed.** §1.1 rests on a Phase 2
   observation from a phase that returned `REFUSED`; §1.2 records what was withdrawn and why.
8. **L8 — the novelty check is still not done.** It has gated the write-up since Phase 0 and now
   has three claims to check. A literature search, not compute; it gates framing, not execution.
9. **L9 — `environment.yml` is generated and its generator is frozen.** Pins corrected by hand
   2026-11-22; re-running `src/build_notebook.py` reverts them and reintroduces the silent-`drd2`
   hazard.
10. **L10 — what the prereg hash does not cover.** As Phase 2 L10: the hash covers the decision
    rule, not the code computing its inputs. `phase3_pareto.py` and `phase2_components.py` are
    outside it; the reproduction and fidelity gates stand in their place.

---

## 9. Decision record

| Decision | By whom | When |
|---|---|---|
| Hypothesis: Pareto-front selection vs scalarized selection | principal investigator | 2026-11-22 |
| Premise reframed to Phase 2's hypervolume spread; `scalarization_loss` removed entirely | principal investigator, after I reported the misreading | 2026-11-22 |
| Stage A = real 45-run pilot sweep, replacing the unexecutable post-hoc analysis | principal investigator | 2026-11-22 |
| Five selection events accepted as a limitation rather than engineered around | principal investigator | 2026-11-22 |
| Three arms rather than two, absorbing the frozen-`phase0_lib` reimplementation confound | my call | 2026-11-22 |
| Criterion Q co-primary, as the anti-circularity guard | my call | 2026-11-22 |
| Fidelity gate distributional, TOST form, "CI contains 0" clause removed | my call | 2026-11-22 |
| P and Q both put the margin inside the 95% quantile | my call | 2026-11-22 |
| Zero-coordinate molecules excluded before non-dominated sorting | my call | 2026-11-22 |
| Criterion R dropped; no optional secondaries at all | my call | 2026-11-22 |
| Components computable only for molecules already in `mol_buffer` | my call | 2026-11-22 |
| `isomer` kept as one Pareto dimension | my call | 2026-11-22 |
| Metric list fixed at 3, `front_coverage` excluded as collinear | my call | 2026-11-22 |
| Notebook named `pareto_phase3.ipynb` | my call, correctable | 2026-11-22 |
| Renamed `landscape_phase3.ipynb` for series consistency | principal investigator, overriding the above | 2026-11-23 |

## 10. Revision note

**Second amendment, 2026-11-22, during implementation.** Task 5 derived `ε_Q` as 1.5 × the
pilot's observed AUC loss. That makes a non-inferiority tolerance a function of the quantity it
judges, and tips the phase toward `PARETO-RECOVERS` by construction. On the principal
investigator's decision, `ε_Q` is now a **fixed pre-registered 0.14** — the design's resolution
limit — and Stage A's role for Q is reduced to establishing the power to test it (§6.2, §7).
`δ_P` still comes from the pilot, CI-shrunk and halved, floored at 0.14.

**Amendment, 2026-11-22, during implementation.** Task 4 implemented Criterion F and measured
it: at 15 seeds a faithful reimplementation passes the gate only 7 times in 18 across fixture and
bootstrap seeds, against 18 of 18 at 20 seeds (§6.3's table). The approved seed
bound of [15, 30] therefore permitted a seed count at which the fidelity gate could not pass,
which would have halted Phase 3 on arithmetic. The floor is raised to **20**, and Stage A must
now check F passability as well as P/Q power. No threshold moved: F's bound is still 0.25, and
§6.3 permits only tightening. Two further defects Task 4 found are recorded in §10 of the plan's
ledger rather than here, being plan rather than spec defects: a plan test assertion that could
never match, and `synth3`'s unit convention.

A builder check on the first draft (2026-11-22) returned ten questions. Two were fatal as written:
Stage A could produce neither threshold (§4.4), and the premise misread `scalarization_loss`
(§1.2). Both were verified directly against the source and the Phase 2 manifest before being
accepted. Three went to the principal investigator (§9); the remaining seven are settled above and
marked *(my call)*. Two further items the check raised and resolved itself need no decision: the
hypervolume reference point is already fixed (exact HV against the origin, components in [0,1],
reused from `phase2_stats`), and including the anchor arm in the z-centring is harmless (§6.0).
