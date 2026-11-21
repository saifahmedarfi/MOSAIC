# Post-Phase-1 Decision Memo

> **This is a decision memo, not an implementation plan.** No code is specified, because the
> mandated action is a write-up and because the choice of what Phase 2 becomes is the
> principal investigator's, not the implementer's. When that choice is made, it needs its own
> spec and its own plan through the normal route.

**Status:** Phase 1 verdict is `NULL`. Phase 2-as-the-controller is foreclosed by the frozen
pre-registration. This memo records what was decided, what is now unavailable, and the four
directions that remain, with costs.

**Inputs:** `Phase 1/PHASE1_RESULTS.md`, `Phase 1/phase1_verdict.json`,
`Phase 1/phase1_run_manifest.csv` (405 rows, independently recomputed 2026-11-20),
`GUIDE.md` §2 and §9, `Q UNIMOL OPT_proposal.md`.

---

## 1. What is settled

At a 1000-call budget, on 15 held-out PMO oracles, 3 backbones × 3 exploration settings × 3
seeds:

| Criterion | Result | What it establishes |
|---|---|---|
| A — flip replicates | FAIL (0 confirmed on every backbone) | The Phase 0 flip does not survive a replication requirement. The one testable pair gave D = +0.003/+0.001, replication 0.21–0.28 vs required 0.95. |
| B — diagnostic predicts | FAIL (primary and all 13 secondaries) | `gini` reversed sign: ρ = −0.655 (Phase 0) → +0.410 (Phase 1). Smallest uncorrected secondary p = 0.30. |
| C — economics | FAIL on all 3 backbones | Held-out-seed headroom −0.004 to +0.001 AUC, CIs spanning zero. Probe cost 0.022–0.027. |
| D — method-level selection | "flips" by argmax count only | Margins 0.0003–0.0020 vs seed SD 0.0063. Not a rescue. |

**The load-bearing finding is C, and it does not depend on A or B.** A free, perfect
landscape oracle buys nothing measurable at this budget. The probe cost is the second-order
term. Any write-up must lead with headroom, not with probe cost — the probe-cost proxy
(`AUC@1000 − AUC@700`) ignores that probe molecules enter the top-10 buffer and is therefore
somewhat overstated, while the headroom result is robust.

**The method-level reframe is also dead.** Backbone spread fell from 0.048 (Phase 0, 8
oracles) to 0.0034 (Phase 1, 15 held-out oracles). `graph_ga`'s Phase 0 lead was an 8-oracle
artifact.

## 2. Three defects that must travel with the result

Stating these first is cheaper than having them found.

1. **Criterion A's threshold was unsatisfiable.** The confirmation set admits at most one
   testable cluster pair (`bioactivity` holds 1 oracle; PMO's 2 `isomer` oracles were both
   spent in Phase 0); the frozen rule required two. This was proved before the sweep finished
   by running `criterion_A` on a planted noiseless D = +1.2, which returned `passes = False`.
   It is salvaged only by the empirical failure of the one pair that *was* testable — so
   lowering `min_pairs` to 1 changes nothing. No amendment was made.
2. **The redesigned features were never tested.** All four `q*_gap` features returned NaN,
   because `phase0_lib.compute_diagnostics` does not return the raw reward vector on every
   path and `compute_diagnostics1` correctly refuses to invent one. These four were the
   headline response to Phase 0's `PARTIAL`. Criterion B is therefore part non-replication
   (real, clean, and sufficient on its own via the `gini` sign reversal) and part non-test.
3. **One asymmetry favours the null.** `cv_headroom` selects `best_fixed_setting` in-sample
   while scoring the conditional policy out-of-sample. True headroom may be ~0.001–0.002
   higher than reported — still an order of magnitude below the probe cost.

Not fixing 2 after seeing the result is deliberate. Re-running a diagnostic after a failed
test is p-hacking regardless of the engineering justification.

## 3. What is now unavailable

- **The controller.** `GUIDE.md` §7.3 gates Phase 2 on `GO`. The frozen consequence clause
  reads: *"The premise is dead at this budget; write up the null."* Building it would void
  every hash in both results documents.
- **PMO as a confirmation set.** 8 oracles in Phase 0, 15 in Phase 1 — all 23 are spent. Any
  further confirmatory test on this hypothesis family needs oracles from outside PMO.
- **Re-testing Criterion B on a fixed feature set.** Needs a fresh pre-registration *and*
  oracles untouched by Phases 0–1. There are none in PMO.

## 4. What remains, ranked

### (i) Write up the null and the economics framework — **do this now**

The mandated action, and the cheapest real contribution on the table. Two claims:

- The field has assumed landscape-dependent exploration since PMO without testing it
  directly. A pre-registered test that reverses its own discovery-phase screen is a result.
- *Selection headroom must be measured against the cost of the information that enables
  selection.* This is reusable beyond molecular optimization and no part of it depends on the
  hypothesis having failed. It is arguably the more durable contribution.

Cost: writing only. No new compute. Everything needed is in `Phase 1/`.

**Blocking prerequisite:** the step-0 novelty check, still not done (`GUIDE.md` §10.1). Both
claims above are novelty claims. Per-instance algorithm selection has a long history in
optimization (portfolio solvers for SAT/TSP); whether the molecular version is already
published is unverified. This is a literature search, not a coding task, and it gates the
framing of the write-up — not whether the write-up happens.

### (ii) Pivot to the parent proposal's unbuilt layers — **the real Phase 2**

`Q UNIMOL OPT_proposal.md` is a benchmarking-framework proposal; landscape-conditional
exploration was one slice of its Optimization layer. Unbuilt and specified: the
Representation layer (§4), Pareto-based multi-objective evaluation rather than scalarized MPO
(§3, §7), the Memory layer reuse ablation (§4), quantum components as testable hypotheses
(§5), GuacaMol/MOSES proper (§7).

Why this is the right direction rather than a retreat: Phase 1's null is partly a
*ceiling* result. The total spread across all nine arms was 0.0119 AUC — AUC-top-10 at 1000
calls on PMO compresses everything the study manipulated into the seed noise. A wider action
space and a metric that exposes trade-offs (Pareto coverage) is where differences can exist
at all.

The apparatus transfers: the resumable runner, persisted score traces, the hash-enforced
prereg pattern, the hierarchical bootstrap, `cv_headroom`, and 59 passing assertions.

Cost: a new spec and plan. New oracles required.

### (iii) The online bandit baseline — **narrow, honest, cheap**

EXP3 or successive halving over the settings as arms. Criterion C sharpens rather than
softens this: a bandit pays no upfront probe, so the 0.022–0.027 that sank the probe-based
controller is not a cost it incurs. Still listed third because headroom ≈ 0 means a bandit
has almost nothing to win either — it can at best recover a gain that measured between
−0.004 and +0.001.

Worth doing only as a one-paragraph addition to (i): it closes the obvious reviewer question
*"why predict the landscape when you could adapt online?"* (`GUIDE.md` §10.2) without
claiming a win.

### (iv) Budget scaling — **do not do this**

The probe is a fixed 300 calls, so at 10,000 calls it costs 3% rather than 30%. But headroom
measured −0.004 to +0.001, so the probe is the smaller half of the problem. Scaling 10× to
test whether a quantity that measured zero becomes non-zero is a ~340 GPU-hour bet against a
pre-registered null, and the burden of proof is now high. If it is ever run, it needs a fresh
pre-registration, new oracles, and an explicit statement that it is testing a budget-scaling
hypothesis and not re-litigating Phase 1.

## 5. Recommendation

Do (i) now, with (iii) folded in as a paragraph. Then treat (ii) as Phase 2 under the parent
proposal, with its own spec, its own frozen pre-registration, and oracles from outside PMO.

Run the novelty check before writing, because it determines the framing of (i) and may
reposition (ii).

## 6. Housekeeping, independent of the above

- **`GUIDE.md` still names venues and submission dates** (§10.2, §10.3) and is public. Phase 0
  and Phase 1 results documents were stripped; this one was not.
- Six Phase 0 and fifteen Phase 1 result artifacts are untracked locally. The Phase 0 six must
  reach GitHub from the Windows clone via `git pull --no-rebase && git push`; a rebase would
  rewrite `0a28b5a`/`47dba26`, the hashes `PHASE0_RESULTS.md` cites as provenance.
- `mol_opt` is still unpinned (SHA `2da631b`), and the Phase 0 `prereg_hash()` scope gap is
  still open. Both carried over from Phase 0.
