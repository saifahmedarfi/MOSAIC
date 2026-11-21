# SDD ledger — plan: docs/superpowers/plans/2026-11-20-phase2-metric-sensitivity.md

Spec: docs/superpowers/specs/2026-11-20-phase2-metric-sensitivity-design.md (read, authoritative)
Branch: phase2-metric-sensitivity, forked from main at 5700883
Test runner: ~/miniconda3/envs/research-base/bin/python tests/test_phase2.py (plain script, NOT pytest)

Ruling: work on branch `phase2-metric-sensitivity` in the existing working tree rather than a
separate git worktree — the deliverable is `Phase 2/landscape_phase2.ipynb` at the user's own
path, and relocating the tree would move it. Cost if wrong: a merge at the end instead of a
fast-forward.

## Tasks

## Pre-flight scan

Pairs sharing a file or interface:

| Pair | Produces → consumes | Finding |
|---|---|---|
| 1→2 | `metric_score`, `z_scores`, `METRICS`, constants → bootstrap | agrees |
| 2→3 | `criterion_0`, `winner_replication` → criteria A/B/C | agrees |
| 1→3 | `z_scores`, `metric_score` → criterion_B/C | agrees |
| 1→4 | `GATE_TOL`, `MIN_ORACLES` → gate logic | agrees |
| 3→5 | `criterion_0/A/C`, `phase2_verdict` → simulate | agrees |
| 4→6 | `phase2_components.run_gate` → source Part 3 | agrees |
| 5→6 | chosen seed count → `PREREG2["seeds"]` | **ordering constraint**: Task 6 cannot write PREREG2 until Task 5 has been run for real, not merely tested. Plan's Task 5 Step 4 does run it. |
| 1..5→6 | all stats/components → notebook | agrees |
| all | `tests/test_phase2.py` appended by every task | agrees; each task adds its own `test_*` function to the same `main()` tuple |

Task self-consistency:

| Task | Finding |
|---|---|
| 1 | Creates `tests/test_phase2.py` but plan does not state the harness scaffolding. **Ruling:** Task 1 also creates the `check()`/`main()` harness in the shape of `tests/test_phase1.py`. Cost if wrong: later tasks each re-invent it. |
| 1 | Test fixtures label backbones `A`/`B`/`C` while Global Constraints name `reinvent`/`augmented_memory`/`graph_ga`. **Ruling:** the constraint binds the sweep grid, not synthetic test fixtures; labels in tests are free. Cost if wrong: a reviewer flags cosmetic inconsistency. |
| 1 | `pooled_seed_sd` has no direct test, only indirect via the zero-variance `z_scores` case. **Ruling:** acceptable — the degenerate path is what matters and it is covered. |
| 2 | Plan's test asserts the shared resample *works* but never that an independent resample *fails*, which spec §9 explicitly mandates. **Ruling:** gap in the plan against the spec. Added to Task 2's dispatch as a required test; spec wins. |
| 3 | agrees — `criterion_C(df, c0, cA)` matches its test's call shape; BH smallest threshold `q/m` is correct |
| 4 | agrees — `gate_decision` arity matches; reproduction gate correctly untested locally (no TDC) |
| 5 | agrees |
| 6 | Tests reference `L.PREREG2_SHA256`, absent from the task's Interfaces block. **Ruling:** add it — the frozen constant read from the committed JSON, against which `prereg2_hash()` is compared. |
| 6 | `test_build_artifacts`'s persistence assertion (`"smiles" in src.split("score_trace")[0][-400:]`) is brittle. **Ruling:** implementer asserts the persisted record shape directly instead. Cost if wrong: a weak test passes on the wrong code. |

Nothing in the plan mandates anything the review rubric treats as a defect.

Task 1: dispatched (sonnet), BASE 5700883. Briefs 1-6 all extracted.

### Task 1 rulings (from implementer concerns)

Ruling: there are **four** secondary metrics, not five — `front_size`, `front_coverage`,
`novelty`, `diversity`. The spec's §5.2 table has seven rows; two are the primary pair and
`scalarization_loss` is Criterion G2's descriptive measurement. The implementer caught a real
defect in the spec and the plan, both now corrected (spec §6 twice, plan Task 1 interfaces).
The smallest BH threshold is 0.05/4 = 0.0125, not 0.01; the Kendall-τ impossibility argument is
unaffected (min attainable p = 0.167 either way). Cost if wrong: the BH denominator in Task 3.

Ruling: `z_scores`/`pooled_seed_sd` defining a cell as (oracle, backbone) and ignoring `setting`
is correct for Phase 2, whose grid is `explore_mid` only. Task 6's dispatch must state that any
frame handed to the stats layer is filtered to a single setting — this matters for the Phase 1
overlap gate, whose source data carries three settings. Cost if wrong: seed SD inflated by
setting variance, widening Criterion C's threshold.

Ruling: the implementer wrote tests and implementation together and established RED by moving
the module aside, proving only `ModuleNotFoundError` rather than per-assertion failure. Noted as
a process deviation, not sent to a fix round — the tests exist, pass, and are independently
re-runnable. Cost if wrong: an assertion that never actually discriminated.

### Task 1 review (b8ecf7e) — Needs fixes

Spec compliant. Verified by the reviewer, no defect: the relative zero-variance guard
(`sd > 1e-12 * max(1.0, abs(mean))`, returns exactly 0.0), all constant values, numpy-and-stdlib
imports only, no protected file touched, harness shape matches test_phase1.py. The tests include
a brute-force coordinate-grid cross-check in 3-D and 4-D that would catch a 2-D-only algorithm.

Two Important findings -> fix round 1 (resumed original implementer):
1. `pareto_front` is a Python row loop called at every recursion level, so hypervolume is ~n^2
   in 3-D and ~n^3 in 4-D on large fronts. Reviewer measured 3-D n=1000 = 12.2 s, 4-D n=400 =
   12.2 s on an anti-chain front; 4-D n=1000 extrapolates to minutes. The brief requires fronts
   up to 1000 points, and no test exercised n >= 100.
2. The non-ok-row test assertion is vacuous: injecting 1e6 inflates the pooled SD too, so the
   leaked z stays ~0.4 and passes `< 100` either way.

Ruling: upgraded one Minor into the round — `hypervolume` must raise on NaN rather than drop the
row via `(pts > 0).all(axis=1)`. A NaN component means a failed RDKit call, and silently dropping
the molecule yields a wrong hypervolume with no diagnostic; the project's standing rule is that a
metric records NaN rather than inventing a value, and a silently smaller front is an invented
value. Cost if wrong: one extra guard on a path Task 4's tested [0,1] contract already protects.

Task 1: minor (deferred): no test distinguishes relative from absolute tolerance at large magnitude
Task 1: minor (deferred): `canonical_json` duplicated from the phase1 pattern
Task 1: minor (deferred): `_ok` ignores `setting`; docstring warning suggested

Task 1: fix round 1/5 dispatched -> implementer returned 06d1a2a (3 findings claimed fixed;
38 assertions, -W error clean; 4-D n=1000 anti-chain 0.42 s, 3-D 0.00 s). Scoped re-review
dispatched against b8ecf7e..06d1a2a. Note: that range also contains 48dbf16, the docs-only
four-secondaries correction, flagged to the re-reviewer so it does not read as task scope.
Risk flagged to re-reviewer: the fix split one slice recursion into dedicated 2-D/3-D sweeps
plus a 4-D recursion, so there are now three code paths where there was one.

Task 1: fix round 1/5 (3 addressed, 0 open; commits b8ecf7e..06d1a2a)
  Re-reviewer independently ran 3000 randomised cases against a brute-force coordinate-grid
  union (d=2,3,4; n=1..11; a half with 0.25-grid coordinates to force ties; a fifth with a
  duplicated point): 0 mismatches at 1e-10. It also verified the 2-D closed form (n+1)/(2n) is a
  real derivation rather than a restatement of the implementation.
Task 1: minor (deferred): the committed 3-D 1000-point test asserts bounds and a time limit, not
  exact values, so a large-n regression in `_hv3` that stayed inside (0,1) would not be caught.
  A small randomised brute-force 3-D test with ties would close it. FLAG FOR FINAL REVIEW as the
  one deferred minor worth fixing before merge -- `hypervolume` is what Criterion A rests on.
Task 1: complete (commits 5700883..06d1a2a, review clean)

Task 2: dispatched (sonnet), BASE 06d1a2a. Carried into the dispatch: the spec-mandated
sharing test the plan omitted (an independent-per-backbone resample must FAIL it), and a
performance note that a pandas-per-draw bootstrap at N_BOOT=10000 is unusably slow.

### Task 2 implemented (a6ec31d), review dispatched

Reported: 55 assertions, 2.4 s, -W error clean. The mandated sharing test discriminates —
shared resample replication 1.000 vs 0.487 for the independent-per-backbone variant; pure noise
0.467. Full N_BOOT criterion_0 in ~0.35 s, so the vectorisation note landed.

Implementer concerns, and how each was handled:
1. The local "independent" variant cannot centre per oracle (each backbone draws different
   oracles), so it scores mean(cell mean / pooled SD). Accepted as a faithful interpretation —
   the inability to centre IS the defect that unshared resampling causes.
2. The vectorised per-draw z was NOT cross-checked against a pandas reference on a resampled
   frame. Named as THE risk for the reviewer: if the fast bootstrap path and Task 1's tested
   `z_scores`/`metric_score` are two independent implementations of the same arithmetic, they can
   drift, and every replication figure would be wrong while all behavioural tests still pass.
3. A draw whose own top score is tied counts for no backbone — implementer's choice, unspecified
   by the brief. Consistent with the frozen "exact tie -> winner None" rule. Ruling: stands;
   flagged to the reviewer for documentation rather than behaviour.
4. Sandbox had to be disabled for python (seccomp). Environmental, not a code issue.

### Task 2 review (a6ec31d) — Needs fixes

Spec compliant. Verified by the reviewer at file:line: the oracle index is drawn once per draw
with shape (n_boot, n_o) and shared across backbones; seeds resampled independently per cell;
argmax re-selected per draw; exact tie -> winner None, replication 0.0, stable False;
`criterion_0` iterates PRIMARY_PAIR only; `stable` uses >= REPLICATION_RATE.

Three "cannot verify from diff" items, all resolved by the controller, all clean:
- imports are numpy + stdlib only (pandas is duck-typed on the frame; constraint is "only
  these", not "all of these")
- REPLICATION_RATE=0.95 and N_BOOT=10_000 confirmed defined in Task 1 at src/phase2_stats.py:36-37
- `git diff --name-only 5700883 HEAD` over the five protected paths: empty. No venue references.

One Important -> fix round 1 (resumed original implementer):
- `winner_replication` hand-rolls a second copy of the z arithmetic (lines 77-85) alongside
  `z_scores`/`metric_score` (189-209). Reviewer read both and found no divergence today, but
  nothing pins them, and the behavioural tests would still pass if the fast path drifted.
  Fix directed at removing the duplication (shared array core, point estimate calls it on the
  identity resample) rather than testing around it.

Ruling: upgraded two Minors into the round.
- The exact `v == top` tie check does not enforce the frozen rule "never resolve a tie by array
  order" — float roundoff from the centring subtraction defeats it, and the only tie test
  exercises the all-zero-z path. A frozen rule that the code does not actually enforce is a spec
  gap, not a nicety. Cost if wrong: one np.isclose and one test.
- `np.nanmean` on an all-NaN slice emits a RuntimeWarning that breaks `-W error`. Sparsity is
  expected, not hypothetical: the reproduction gate can exclude oracles before the sweep.

Ruling: the implementer's choice that a draw with a tied top score counts for no backbone
stands — consistent with the frozen tie rule and the conservative reading.

Task 2: minor (deferred): noise threshold asserts only < 0.95; < 0.7 would guard regressions better
Task 2: minor (deferred): the `one`-state test relies on its fixed seed
Task 2: minor (deferred): independent-variant helper needs a comment that losing centring is a
  consequence of losing the shared list, not a second deliberate difference

Task 2: fix round 1/5 dispatched -> implementer returned f3820a4 (3 findings claimed fixed;
64 assertions, -W error clean). Deduplication reported successful: z_scores, metric_score,
pooled_seed_sd and winner_replication all route through one `_z_and_score`, point estimate as
the identity resample. Replication figures unchanged (shared 1.000, independent 0.487). Tie
tolerance np.isclose(rtol=atol=1e-9); the tie test's A-B gap is -3.55e-15, which the old `==`
would have missed -- that upgraded Minor was real and is now demonstrably caught.

Scoped re-review dispatched against a6ec31d..f3820a4 with three named risks, since this fix
rewrote the module's numerical core rather than patching around it:
  (a) is the deduplication real, or does a second path remain -- and would the "spy" test fail
      if one path stopped calling the core, or does it merely assert the core is callable?
  (b) is the equivalence test's reference genuinely independent arithmetic, or derived from the
      implementation (in which case it proves nothing)?
  (c) the warning suppression now sits inside the shared core, so it affects z_scores,
      metric_score and pooled_seed_sd too -- is it narrowly scoped to the all-NaN slice, or a
      blanket catch_warnings that would mask real numerical problems in every caller?
      Implementer disclosed it did NOT verify the sparse test fails without the guard.
Unchanged outputs after a core rewrite treated as consistent with a correct refactor, not proof.

Task 2: fix round 1/5 (3 addressed, 0 open; commits a6ec31d..f3820a4)
  Re-reviewer confirmed the deduplication is real at file:line — z_scores, metric_score and
  pooled_seed_sd retain no arithmetic of their own; winner_replication calls the core twice
  (point estimate via _point_arrays, bootstrap via the draw arrays), both built by _cell_stats.
  The spy test patches the module global, which winner_replication resolves at call time, so it
  genuinely intercepts; boot_calls would drop below 2 if a path stopped using the core. The
  equivalence reference `_naive_scores` is independent arithmetic (plain loops, own SD, own
  tolerance, own centring), and the ragged frame carries a missing cell, unequal seed counts, a
  1-seed cell and a zero-SD oracle. Dropping centring on partial-coverage oracles would break it.
Task 2: minor (deferred): the warning suppression in `_z_and_score` is broader than the all-NaN
  case — np.errstate(all="ignore") plus a blanket RuntimeWarning ignore now also covers overflow,
  underflow and invalid for z_scores, metric_score and pooled_seed_sd. Practical risk is low
  (inputs filtered finite by _ok, division guarded by np.where(sd==0,1,sd)), but narrowing to
  message="Mean of empty slice" would scope it to the intended case. FLAG FOR FINAL REVIEW.
Task 2: minor (deferred): the per-draw tie tolerance is not pinned directly by any test
Task 2: complete (commits 06d1a2a..f3820a4, review clean, 4 minors deferred)

Task 3: dispatched (sonnet), BASE f3820a4. Carried into the dispatch: the four-not-five
secondaries correction (BH denominator must derive from len(per_metric), not a literal), the
instruction to route bootstraps through the existing _z_and_score core rather than reintroduce
the duplicate arithmetic Task 2's review removed, and the frozen verdict branch order
(state -> C -> A).

### Task 3 implemented (c75bbbc) — rulings on four concerns

Ruling (concern 1, real contradiction in my own documents): the spec's §6 **table governs**. The
plan's prose said "state first, then C, then A", which would have routed both-stable + A-fails +
C-fails to COSMETIC. The table says both-stable + A-fails is METRIC-ROBUST whatever C says, and
the table is right: when the two metrics agree on the winner there is no ranking change whose
magnitude C could measure. The implementer followed the table and was correct. Plan prose
corrected. Cost if wrong: one verdict branch mislabelled in the frozen rule.

Ruling (concern 2): Criterion B's test is **two-sided**, and the spec now says so explicitly. A
metric agreeing with auc_top10 more often than chance is as interpretable as one disagreeing more
often, and B is reported rather than verdict-bearing, so there is no directional hypothesis to
protect. Recorded in the spec so the pre-registration freezes it rather than leaving it implicit.

Ruling (concern 3): the implementer refactored `winner_replication`'s resampling loop into a
shared `_boot_cells` used by A, B and G2. Direction is right — it continues Task 2's
deduplication rather than reversing it — but it modifies code Task 2's review already passed.
Named as THE risk for this review: is the refactor genuinely behaviour-preserving? "The original
64 assertions stayed green" is necessary, not sufficient.

Ruling (concern 4): loose tolerances on the Criterion C tests are accepted. Measured gaps were
2.23 and 0.12 against nominals of 2.5 and 0.4 because pooled SD is estimated from 5 seeds; both
sit far from the 1.0 threshold, so the pass/fail assertions are not near-misses.

### Task 3 review (c75bbbc) — Approved, no Critical/Important

Named risk closed properly: `_boot_cells` never draws oracles — `idx` is drawn once per draw by
the caller and passed in, every backbone's cell filled via argwhere over that same `idx`. The
reviewer checked RNG call order and shapes and found the extraction draw-for-draw bit-identical
to the pre-refactor loop, so it is behaviour-preserving rather than merely test-green.
Also verified: Criterion A re-selects both argmaxes per draw; the weak-effect A test is a real
discriminator (point winners differ but replication stays under 0.95, where a frozen-winners
implementation would report 1.0); degenerate metrics removed before bh_fdr with three kinds
covered; the bootstrap seed is indexed over SECONDARY_METRICS so a skipped degenerate metric does
not shift other metrics' RNG streams.

Controller resolved the one "cannot verify" item: Criterion C's strict `gap > C_THRESHOLD` is
correct. The spec's §6 formula is `> 1.0` and its prose is "by more than one pooled seed SD".
No change needed; the tests never probe exactly 1.0, which is acceptable for a strict reading.

Ruling: upgrading three Minors into a fix round despite the Approved verdict, because this is a
decision rule that will be frozen and published and the cost is one cheap round.
  (a) No data-driven end-to-end test composes criterion_0 + criterion_A into METRIC-ROBUST,
      the branch that fires whenever the methods are distinguishable but metric-invariant.
      Dispatch proved only by hand-constructed dicts; criteria proved only separately.
  (b) `tests/test_phase2.py` asserts a literal `BH_Q / 2` denominator — the exact pattern the
      review brief was told to flag, and the sibling test already derives it.
  (c) `criterion_B` calls np.percentile on a possibly-empty draws array when every bootstrap
      draw loses all surviving oracles. Only reachable on a tiny frame, but it is a crash path.
Cost if wrong: one round's dispatch on a task already approved.

Task 3: minor (deferred): C's 0.4-gap case does not assert winner_hv == "B"
Task 3: minor (deferred): G2's CI test asserts only lo < mean < hi and nonzero width; asserting
  the width is near the oracle-level SD would discriminate an unshared/non-hierarchical resample
Task 3: minor (deferred): criteria A/B point winners use jointly-finite runs (_joint_cells) while
  criterion_0 and criterion_C use each metric's own finite runs, so cA["winner_hv"] can disagree
  with c0's hypervolume winner when NaN patterns differ. Unlikely on the real frame; document in
  Task 4. FLAG FOR FINAL REVIEW.

Task 3: fix round 1/5 dispatched -> implementer returned c757ae3 (3 addressed; 123 assertions,
up from 119). Scoped re-review dispatched against c75bbbc..c757ae3 with two named risks:
  (a) Minor 3's test monkeypatches _differing_fraction to return all-NaN draws -- does that
      prove the guard is reachable on a real data path, or only that it fires when forced?
      A real tiny frame would be the stronger test if one can reach the branch.
  (b) Minor 1's new test must call the REAL criterion_0 and criterion_A on a frame, not assemble
      dicts -- the finding existed precisely because dispatch-only tests do not pin composition.

Task 3: fix round 1/5 (3 addressed, 0 open; commits c75bbbc..c757ae3)
  METRIC-ROBUST test confirmed genuinely data-driven: it calls the real criterion_0 and
  criterion_A on a planted frame and asserts state=="both" and passes is False from those
  outputs. BH denominator now derived both ways the finding allowed. criterion_B appends to
  `degenerate` and continues before np.percentile, so a dropped metric never reaches bh_fdr.
  Re-reviewer judged the monkeypatch in the empty-bootstrap test probably the only practical
  route (the point estimate must survive while all n_boot draws lose every oracle) and the mock
  faithful (it NaNs only array results, leaving the scalar point estimate intact).
  NOTE: the re-reviewer's Bash was unavailable, so it reviewed from the diff only and ran nothing.
  Controller verified directly instead: 123 assertions pass under -W error, test_phase1 still
  passes 59, and `git diff --name-only 5700883 HEAD` over the five protected paths is empty.
Task 3: minor (deferred): long `for t in (...)` line in main() -- cosmetic
Task 3: minor (deferred): a real tiny frame would beat the monkeypatch for the empty-bootstrap
  case if one can be found that reaches the branch
Task 3: complete (commits f3820a4..c757ae3, review clean, 5 minors deferred)

Task 4: dispatched (sonnet), BASE c757ae3. The dispatch carries the GuacaMol component table
extracted from BenevolentAI/guacamol standard_benchmarks.py + common_scoring_functions.py during
design recon, so the implementer does not have to rediscover modifier parameters.

### Task 4 implemented (1d8b656) — rulings on four concerns

Reported: 236 assertions (was 123), -W error clean. Implementer curl'd GuacaMol's raw master
sources (score_modifier, fingerprints, descriptors, chemistry, common_scoring_functions,
standard_benchmarks) because the package was not installed, and confirmed all nine component
definitions matched the table supplied in the dispatch. An arithmetic-mean impostor oracle is
caught on all nine, so the gate demonstrably discriminates.

Ruling (concern 2): the probe set's low-end pool is a frozen committed literal, NOT a ChEMBL
draw. Accepted, and the spec is amended to say so. `phase2_components.py` must import without
TDC or torch to keep the gate logic testable off the execution machine, and the prior sits
behind both; a committed literal is also strictly more reproducible than a seeded draw from a
downloaded corpus. The implementer added highly lipophilic molecules because without them
ranolazine's logP MaxGaussian(7,1) never leaves the bottom of its range and the probe fails the
gate-on-the-gate. That is the intended design, not gaming it: the gate compares our gmean against
TDC's scalar at 1e-6, so a probe set that spans more of each component's range can only make a
wrong decomposition EASIER to catch. Measured per-oracle min_range 0.675-1.0, frac_nonzero
0.59-0.99, comfortably above the 0.5 and 0.20 floors.

Ruling (concern 3): "non-zero" read as score > GATE_TOL, and a deviation outranking
`probe_too_weak`, both accepted. A deviation found on a weak probe is still a real deviation, so
reporting "deviates" is the correct precedence.

Ruling (concern 1): the real TDC gate cannot run here and will execute on the Windows machine as
the notebook's pre-sweep step. Not a defect; it is why the gate takes the oracle as a parameter.

Concern 4 (legacy count-based Morgan API with rdkit's deprecation log blocked) sent to the
reviewer as a named risk rather than ruled: matching GuacaMol's fingerprint API is right for
reproduction fidelity, but the log suppression needs its scope checked -- Task 2 already produced
one over-broad warning suppression.

### Task 4 review (1d8b656) — Approved with one Important

Reviewer checked all nine oracles parameter-by-parameter against the dispatch table and found no
divergence: modifier forms (Min is 1 for x<=mu, Max is 1 for x>=mu, Clipped a linear ramp from
lower_x=0 clipped to [0,1]), all eleven reference SMILES, the isomer sub-score as ONE component
(geometric mean of per-element Gaussian sigma 1 plus total atoms including hydrogens at sigma 2),
component counts 2/3/4/2/4/4/2/2/2, AGGREGATION a nine-entry dict all "geometric", sitagliptin's
logP/TPSA targets computed at runtime (a second-molecule test would fail a hard-coded constant),
no TDC at module level (asserted via sys.modules), no rdkit in phase2_stats.py.
The BlockLogs scope came back TIGHT, contra my suspicion: it is a scoped context manager wrapping
only the fingerprint call and the sanitisation of generated probe edits. `_parse_mol` and
`component_vector` sit outside it, so malformed-SMILES errors still log and invalid SMILES raise.

Controller resolved the load-bearing "cannot verify" item: fetched GuacaMol's raw
utils/fingerprints.py and confirmed `get_AP` is `GetAtomPairFingerprint(mol, maxLength=10)` —
the non-default value the implementer used — with ECFP4/ECFP6 as `GetMorganFingerprint(mol, 2)`
and `(mol, 3)` and FCFP4 as radius 2 `useFeatures=True`. `src/phase2_components.py:_fp` matches
line for line. CLEAN. The second item (whether TDC's bound oracle uses count-based fingerprints)
is only settleable by the TDC-side gate and is the already-ruled limitation.

One Important -> fix round 1 (resumed original implementer): no test would catch a SUBTLY wrong
decomposition. The only impostor is the arithmetic mean, which tests aggregation alone; swapping
ECFP4/ECFP6, FCFP4 for ECFP4, AP maxLength 10 for 30, _clipped(0.7) for (0.8), or any TPSA/logP
mu/sigma would all pass the committed suite. Root cause: the reference-self-similarity assertions
are insensitive to upper_x AND to fingerprint kind, because self-similarity is 1 either way, and
the probe/component tests compare ours to ours. This is spec anti-goal A2 with no test behind it.
Fix directed at hand-computed values at NON-TRIVIAL similarity plus a parametrised mutation test
that perturbs one parameter and asserts the arithmetic assertions then fail.

Ruling: upgraded one Minor into the round — `except Exception: pass` at phase2_components.py:308
can hide an AttributeError by quietly shrinking the probe set. Swallowed errors are on this
project's blocking list, not its polish list. Narrow to (ValueError, RuntimeError).

Task 4: minor (deferred): _PRIOR_SEED and the permutation only reorder a de-duplicated list
Task 4: minor (deferred): redundant SanitizeMol after MolFromSmiles (mirrors GuacaMol; stays)
Task 4: minor (deferred): empty string "" parses to an empty Mol and scores rather than raising
Task 4: minor (deferred): the tolerance-boundary test restates the tolerance

Task 4: fix round 1/5 dispatched -> implementer returned 7a0745f (2 addressed; 262 assertions,
up from 236; 21 mutations tested, all detected, none survived -- covering every mutation named
plus AP maxLength 9, median1 ECFP4->ECFP6, osimertinib FCFP4->ECFP6, MinGaussian mu and sigma
changes, a Max/Min swap and a fluorine mu change). Component table now built by
_build_components() so the mutation test can rebuild it; `emit` narrowed to (ValueError,
RuntimeError).

Implementer's residual concern, accepted as an accurate statement of scope rather than a finding:
the independent expectations restate the same GuacaMol parameters, so they guard against
transcription errors and regressions but not against having misread GuacaMol in the first place.
That risk sits with the unrun TDC gate -- and is lower than it sounds, since the controller
independently verified the fingerprint parameters against GuacaMol's raw source.

Scoped re-review dispatched against 1d8b656..7a0745f with four named risks, because a mutation
test is easy to write in a way that proves nothing:
  (a) does each mutation actually REACH the arithmetic, or is a fingerprint / reference
      fingerprint / modifier closure cached at import time so the mutation silently misses?
  (b) is the "independent" expectation genuinely independent (RDKit's newer generator API), or
      recomputed by calling the module's own component_vector/_fp?
  (c) are the test molecules really at non-trivial similarity and OFF the modifier plateaus --
      an assertion at Tanimoto 1.0 or where Min/MaxGaussian returns 1.0 is insensitive to the
      very parameters the fix exists to pin, which was the original defect?
  (d) does the mutation test assert the committed assertions FAIL, or merely that a value differs?

Task 4: fix round 1/5 (2 addressed, 0 open; commits 1d8b656..7a0745f)
  Re-reviewer verified all four named risks rather than the happy path:
  (a) mutations DO reach the arithmetic -- `_build_components()` resolves `_tanimoto`, `_clipped`,
      `_max_gaussian` etc. through module globals on every call; `_tanimoto` computes ref_fp at
      build time via global `_fp` and its closure calls global `_fp` at score time, so an AP or
      kind patch reaches both sides; `component_vector` reads the module-global COMPONENTS.
      Restoration in `finally` plus a post-restore baseline check.
  (b) expectations ARE independent: `_independent_expected` builds them from
      `rdFingerprintGenerator` (newer API) with GetMorganFeatureAtomInvGen for FCFP4 and
      maxDistance=10 for AP, using its own Gaussian/Min/Max/clip formulas, never calling the
      module's component_vector or _fp.
  (c) molecules ARE off-plateau: mid-similarity neighbours and analogues, not the references;
      >=80% of values asserted inside (1e-9, 0.999) and the osimertinib ECFP6 MinGaussian
      asserted to actually bite.
  (d) the mutation test asserts FAILURE (`len(failed) > 0`) via `_decomposition_failures`, not
      mere difference.
  Re-reviewer also ran the suite itself: 262 assertions, 21 "mutation detected" lines.
Task 4: minor (deferred): amlodipine_mpo, perindopril_mpo, sitagliptin_mpo and zaleplon_mpo have
  no non-trivial hand-computed point (only the other five do)
Task 4: minor (deferred): ranolazine's TPSA mu (95) is not mutated, only its sigma
Task 4: minor (deferred): the off-plateau check is an aggregate 80% threshold, not per case
Task 4: complete (commits c757ae3..7a0745f, review clean, 7 minors deferred)

Task 5: dispatched (sonnet), BASE 7a0745f.

### Task 5 implemented (2a8f78a) — gate PASSED but its specification was too weak

Reported: 269 assertions, -W error clean. Gate exited 0, chose n=5, no halt. Ran from the Phase 1
manifest (explore_mid only, 14 usable oracles after excluding valsartan_smarts for ~0 SD), median
pooled seed SD 0.0087. Power (n_rep=200, 9 oracles, n_boot=2000): 0 SD -> 0.010/0.005/0.005;
1 SD -> 1.000/1.000/1.000; 2 SD -> 1.000/1.000/1.000 at n=5/10/15. Full gate 588 s; suite now 83 s.

Ruling: the implementer's concern 1 identifies a REAL DEFECT IN MY SPEC §5.3.1, not in its code.
A planted 1 SD is ~0.0087 AUC, but Phase 1's observed method gaps were ~0.0034 AUC, about 0.4 SD.
So the gate as specified demonstrates the pipeline is not structurally insensitive and says
nothing about whether a realistically-sized disagreement is detectable. Defining power as "any
verdict other than BENCHMARK-INSENSITIVE" is a bar n=5 clears trivially. This is the weak-bar
form of exactly the failure the gate exists to prevent. Correcting it before the freeze is
legitimate; correcting it after the sweep would not be. Cost if wrong: one more gate run.

Fix round 1 dispatched, all four items REPORTING ONLY -- the frozen halt condition, the criteria,
REPLICATION_RATE and C_THRESHOLD are untouched:
  1. add a 0.4-SD (Phase-1-sized) row, the ratio derived from data where possible
  2. report the per-cell verdict DISTRIBUTION, since concern 2 found METRIC-DEPENDENT appears in
     only ~55% of replicates at 1 SD with the rest COSMETIC, and that this does not improve with
     n because Criterion C's 1-z-unit bar is effect-size-limited rather than seed-limited
  3. report the per-n detectable-effect FLOOR (smallest planted effect reaching 80% power) --
     this is the number that licenses the study's central claim, because every verdict branch is
     a bounding statement and BENCHMARK-INSENSITIVE only means something with an X attached
  4. keep choose_seeds' frozen rule intact; report what n would be chosen at 0.4 SD and at the
     floor, and let the controller rule on the final seed count

Ruling: excluding valsartan_smarts is correct -- Phase 1 recorded all 27 of its runs scoring
exactly 0.0, so it carries no usable noise scale.
Ruling: the added exit code 2 ("2 SD reachable but nothing reaches target at 1 SD") is a sensible
distinction and stays.
Task 5: minor (deferred): the test suite is now 83 s, trending slow

### Task 5 fix round 1 (ab51f96) — the gate's real answer, and two rulings

273 assertions, -W error clean. Gate run 149 s (down from 588 s: B and G2 skipped in the
simulation, correctly, since phase2_verdict's branch logic reads only c0/cA/cC, plus a process
pool). No halt. The frozen 1-SD rule still chose n=5. No rule, criterion or constant changed.

Phase-1-sized effect derived from the manifest: spread 0.00339 / median pooled SD 0.00873 =
**0.388 SD**. (Phase 1's *recorded* 0.0063 SD would have given 0.54 SD; the manifest-derived
0.00873 is the right denominator because it is explore_mid-only and pooled the same way the
analysis pools.)

Power:            0 SD 0.010/0.005/0.005 | 0.39 SD 0.33/0.67/0.90 | 1 SD 1.0 | 2 SD 1.0
Detectable floor: n=5 -> 0.6 SD, n=10 -> 0.6 SD, n=15 -> 0.4 SD
At 0.39 SD: COSMETIC in 0.32/0.66/0.90 of replicates; METRIC-DEPENDENT never exceeds 0.01.
METRIC-DEPENDENT's own floor is ~1.2 SD at EVERY n and does not move with seeds.

RULING 1 — seed count is **15**, not the 5 the frozen rule selected. At the effect size Phase 1
actually observed, n=5 has power 0.33 and n=15 has 0.90. Grid becomes 9 x 3 x 15 = 405 runs,
~3 h, a cost §5.3 already priced. This is the legitimate use of the gate: the choice is driven
entirely by Phase 0/1 data before any Phase 2 run exists. The RULE was not amended to reach a
verdict, and the halt condition, criteria, REPLICATION_RATE and C_THRESHOLD are untouched.
Cost if wrong: 2 extra GPU hours.

RULING 2 — C_THRESHOLD stays at 1.0. It is NOT lowered to make METRIC-DEPENDENT reachable at a
Phase-1-sized effect. Criterion C asks whether a ranking change exceeds one pooled seed SD; an
effect of 0.39 SD is by definition smaller than the noise between replicate runs of the same
configuration, and COSMETIC ("the winner moves, by less than seed noise") is the CORRECT verdict
for it. Lowering the threshold would be adjusting a criterion to change a verdict, which this
project's standing rules forbid outright. The design is working, not failing.
Consequence, now declared in spec §5.3.3 and L6 rather than left to be discovered in the results:
the realistic outcome space is BENCHMARK-INSENSITIVE or COSMETIC, and a reader must be told that
COSMETIC was the most likely non-null outcome BEFORE seeing it.

Spec amended: §5.3.2 (gate result + seed ruling + power table), §5.3.3 (what each verdict needs
in effect-size terms), L6 (rewritten with the measured numbers), §5.3 grid table (15 seeds,
405 runs, ~3 h).

Process note: Task 5 went implementation -> fix round without an intervening task review, because
the implementer's own disclosed concern WAS the finding. Correcting by dispatching the task review
now over the full range 7a0745f..ab51f96 (both commits) rather than a scoped re-review of a round
that never had a review.

### Task 5 review (7a0745f..ab51f96) — Needs fixes, 4 Important

Verified sound by the reviewer: the B/G2 skip is correct (phase2_stats.py:504-516 reads only
c0["state"], cA["passes"], cC["passes"] for the verdict NAME; cB/cG2 feed only descriptive fields
at :520-521, so the stubs cannot change a name). Nothing hard-coded — manifest read at runtime,
explore_mid + status=="ok" filter applied, valsartan excluded via sd >= MIN_SD. No seeding hazard:
each simulate() builds its own default_rng(seed), all cells share one seed deliberately as common
random numbers, and a test asserts the parallel grid equals the serial grid. Oracle means averaged
over backbones so the planted null is a true null. Only scripts/phase2_power.py and
tests/test_phase2.py touched, so the criteria, phase2_verdict, REPLICATION_RATE and C_THRESHOLD
are untouched.

Four Important -> fix round 2 (resumed original implementer):
1. The headline 0.388 SD is a ratio of MISMATCHED BASES (phase2_power.py:188-197): the spread uses
   all settings and all oracles INCLUDING valsartan's all-zero runs, while the SD is explore_mid
   without valsartan. This is the number my 15-seed ruling was published on. Directed to recompute
   the spread on the same explore_mid, valsartan-excluded rows and report old vs corrected.
   Direction unpredictable: a zero-difference oracle shrinks the spread, three settings inflate it.
   The 15-seed ruling is robust either way since power is monotone in n, but the PUBLISHED figure
   must be right, so spec §5.3.2/§5.3.3 get updated from the corrected run.
2. main()'s exit codes 1 and 2 are untested. choose_seeds reaches halt=True in a test, but nothing
   asserts main() returns non-zero — and a halt that silently exits 0 is the one failure this gate
   exists to prevent. A regression to `return 0` would pass the whole suite today.
3. The 2-SD end is unpinned: the test asserts only big["power"] > null["power"], which a feeble
   simulation would also pass. Directed to assert power >= 0.9 at 2 SD, n=15 (reported cell 1.000).
4. Nothing is persisted. main() only prints, so there is no artifact to commit beside the
   pre-registration and the "reproduces within noise" claim is uncheckable. Directed to write a
   JSON into Phase 2/ with the table, verdict distributions, floors, data source and effective
   parameters (seed, n_rep, floor_rep, n_boot, candidates, derived effect + both bases).
Also folded in: the docstring's "0.0063" and FALLBACK_SD=0.0063 vs the runtime 0.00873 must each
be labelled for what they are (Phase 1's recorded within-cell median vs the manifest-derived
pooled SD over usable oracles).

Task 5: minor (deferred): `0.2 < effect_sd < 0.6` magic band tied to today's manifest
Task 5: minor (deferred): p1e rounded to 2dp used as a dict key beside 0.0/1.0/2.0 — collision risk
Task 5: minor (deferred): the fallback path can let main() exit 0 on synthetic data
Task 5: minor (deferred): floor_rep=100 gives ~0.04 MC SE on an 0.80 threshold, 0.2-SD steps
  coarser still; state it beside the floor output

### Task 5 fix round 2 (b4bcd20) — 3 of 4 accepted, effect size still wrong

285 assertions. main()'s exit codes 1/2/0 now tested (exit-2 case built explicitly), power >= 0.9
at 2 SD n=15 pinned, and the gate persists Phase 2/phase2_power_gate.json with table, verdict
distributions, floors, data source and effective parameters. Floor scan reproduced identically on
re-run. Those three Important findings are closed.

Effect size corrected to 1.310 SD (spread 0.01143 / median pooled SD 0.00873, both on explore_mid
ok rows over 14 usable oracles) with 0.388 retained as superseded. At 1.31 SD: power 1.000 at
every n, METRIC-DEPENDENT 0.94/0.98/0.99, and the smallest sufficient n becomes 5, not 15.

Ruling: 1.310 SD is STILL not the right quantity -> fix round 3. The implementer's own concern
shows why: the figure is dominated by gsk3b, whose pooled seed SD is ~0.1 and whose method lead
is ~0.1, both roughly 10x every other oracle. Excluding it by hand gives 0.59 SD. The underlying
defect is that numerator and denominator remain on different footings -- a raw-AUC spread
aggregated across oracles divided by a MEDIAN of per-oracle pooled SDs, with a 10x outlier in
both. Not a meaningful ratio.
The principled unit is the one §5.2.1 already uses: standardise WITHIN each oracle
(spread_o / sd_o), then aggregate. gsk3b then contributes ~1.0 like any other oracle instead of
swamping the numerator. Directed to report the median AND the full per-oracle distribution so
outlier dominance is visible, plus power at the 25th percentile as the conservative realistic
effect -- a study that only works on its most favourable oracles is not a study. Explicitly told
not to exclude gsk3b by hand and not to aim for a value.
Cost if wrong: one more gate run (~150 s) and a re-ruling on seeds.

The 15-seed ruling is now OPEN pending that table. At ~0.4 SD n=5 has power 0.33 vs n=15 0.90 and
the two extra GPU hours are decisive; at ~1.3 SD n=5 already reaches 1.000 with METRIC-DEPENDENT
0.94 and n=15 buys nothing. Implementer's concern 2 is correct that the superseded 0.388 no
longer supports the ruling.

### Task 5 fix round 3 (e6663cc) — effect size settled, seed ruling FINAL

286 assertions, gate 148 s, exit 0, JSON updated. Per-oracle standardised spread over the 14
usable oracles: min 0.708, p25 0.965, median 1.476, p75 2.136, max 3.874. gsk3b is now 1.06, an
ordinary oracle. Both earlier figures kept in the JSON as superseded. Gate does not halt; the
frozen rule still selects 5.

The implementer then found the deepest issue of the four rounds, and it is a real statistical
point: THE STATISTIC IS NOISE-INFLATED. The range of 3 backbone means each from 3 seeds has a null
median of 0.975 by Monte Carlo (expected range of 3 draws ~1.69 SD, divided by sqrt(3) ~ 0.98).
So p25 = 0.965 is INDISTINGUISHABLE FROM PURE NOISE and the median of 1.476 exceeds the null by
only ~0.5. Honest reading of Phase 1's between-method differences: ~0.5 SD at the median, near
zero at the lower quartile. The uncorrected 0.97 is generous, not conservative.

RULING (FINAL) — seed count is 15, decided on DOMINANCE rather than on the point estimate, which
is why refining the effect size stops here. n=15 is never worse than n=5 at any effect size
tested and is decisively better across exactly the range the noise correction points to:
at 0.4-0.6 SD, n=5 reaches 0.37-0.80 while n=15 reaches 0.92-1.00, and the detectable floor drops
from 0.6 to 0.4 SD. Above ~1 SD both saturate, so the extra seeds cost nothing in validity and
buy insurance against the effect being at the low end, which is where it most likely is.
405 runs, ~3 h, a cost §5.3 already priced. choose_seeds was NOT amended -- it still returns 5 --
and the ruling overrides its output in the pre-registration, in the open, with the table as
justification. Cost if wrong: 2 GPU hours.

Ruling: no round 4 for a noise-corrected point estimate. The decision is settled by dominance,
the correction factor (null median 0.975) is already measured and now recorded in the spec, and
the code's actual job -- choose a seed count, halt or not, persist the evidence -- is done.

Spec amended: §5.3.2 now carries the four-attempt progression table, the per-oracle distribution,
the noise-inflation disclosure and the dominance ruling; §5.3.3 and L6 restated against the
corrected numbers.

Task 5: fix rounds 2-3 (4 addressed, 0 open; commits ab51f96..e6663cc)
  Re-reviewer verified per-oracle standardisation is genuinely per-oracle: phase1_sized_effect
  computes (bm.max()-bm.min())/sds[names.index(o)] per oracle, dividing by that oracle's own SD
  BEFORE aggregation, both sides from explore_mid ok rows of the usable oracles. gsk3b = 1.0601 in
  the artifact, following from the code rather than only the report. Exit-code tests call the REAL
  main() with only run_grid/choose_seeds stubbed, so the `code = 1 if halt else (2 if chosen is
  None else 0)` logic is real code and a regression to `return 0` would fail. JSON carries table,
  verdict counts, floor_scan, floor, floor_metric_dependent, data_source, full params (incl. seed,
  n_rep, floor_rep, n_boot, real_n_boot, candidates, target_power, scan_effects) and
  phase1_sized_effect with its bases and per_oracle dict.
  INDEPENDENT VERIFICATION of the number the spec now publishes: re-reviewer reran the null-median
  statistic itself -- 3 backbones, 3 seeds, range of means over pooled SD, 400k draws, different
  seed -- and got 0.972 against the persisted 0.9749. It also confirmed the manifest has exactly
  3 ok runs per oracle/backbone on explore_mid, so the 3-seed assumption matches the data.
Task 5: minor (deferred): on the fallback path p1 is None, so p1e/p25e both fall back to 0.4 and
  phase1_sized_effect is stored as null; the fallback path has no test
Task 5: minor (deferred): the headline-effect range check is weak (effect_sd > 0 and
  0 < null_median < 2); the median-of-per-oracle check pins the real relationship
Task 5: minor (deferred): phase1_sized_effect hardcodes default_rng(0)/200k draws for null_median
  and that seed is not recorded in params (deterministic, but unrecorded)
Task 5: complete (commits 7a0745f..e6663cc, review clean, 4 rounds, 7 minors deferred)

Task 6: dispatched (sonnet), BASE e6663cc. FINAL TASK -- the notebook.

### Task 6 implemented (29439e8) — the notebook exists

405 assertions (up from 286). Notebook 36 cells, library 7 code cells, plus
scripts/run_sweep_phase2.py and Phase 2/preregistration_phase2.json (v3.0.0, seeds=15, 405 runs).
Notebook NOT executed — TDC, results/ and the GPU are on the other machine. Reported as such.

Ruling (concern 1): the implementer is RIGHT and my spec L8 was WRONG. Overlap is 63 cells, not
81. Verified independently by the controller: of the 9 candidates, `ranolazine_mpo` and
`sitagliptin_mpo` have no rows in Phase 1's manifest because they were PHASE 0 oracles; the other
seven give 7 x 3 x 3 = 63 explore_mid ok cells, which the manifest confirms exactly. Correcting
the prereg text before freezing was the right call. Spec L8 amended.

Ruling (concern 6, GPU nondeterminism vs the 1e-9 tolerance): tolerance raised to **1e-6
absolute per cell**, with the mean signed difference and max deviation across the 63 cells
reported and persisted. Bit-exactness is unavailable — GPU reduction order is nondeterministic
unless forced globally, which Phase 1 did not do — so 1e-9 would halt on floating-point dust.
What would actually invalidate the cross-phase argument is a SYSTEMATIC shift, which the signed
mean catches. Setting this before any Phase 2 run exists is legitimate; loosening it after a halt
would not be, and the spec now says so explicitly. Cost if wrong: a real small discrepancy slips
through, which the reported max deviation would still surface.

Finding (concern 4), recorded as spec L9: `src/phase1_lib.py` AS COMMITTED IS STALE relative to
`src/phase1_source.py`. The source has `import phase0_lib as P0` at line 228 in a library cell —
added when Phase 1's execution fixed this very defect — but the committed generated library has
zero occurrences while referencing P0 at six call sites. Importing it and calling
pmo_oracle_names / execute_run1 / load_results1 raises NameError. This is a defect in ALREADY
PUSHED Phase 1 code. Ruling: do NOT regenerate it inside Phase 2 — it is outside this plan and
touching a prior phase's committed artifact risks muddying Phase 1's provenance. Fix is
`python src/build_phase1.py` + commit, as Phase 1 housekeeping. Phase 2's workaround (setting
L1.P0 itself) sent to the reviewer to judge.

Ruling (concern 5): the extra scope is not extra — a persisted reproduction-gate JSON and a sweep
that refuses below MIN_ORACLES or on a prereg mismatch are what design §5's four-gates section
requires. Accepted.
Concerns 2 (lazy phase0_lib via p0()) and 3 (code before tests, RED by removing artifacts) sent
to the reviewer as named risks.

Interruption: the Task 6 reviewer (a56423b) died mid-run on a 5-hour rate limit. Resumed after
reset with the same range and priorities. Branch verified intact first: 14 Phase 2 commits
b8ecf7e..29439e8 on phase2-metric-sensitivity, notebook present at Phase 2/landscape_phase2.ipynb.
Controller's spec/plan corrections from the rulings committed as c62c4a9 (docs only, outside the
review range; the reviewer was told to ignore it).

### Task 6 review (e6663cc..29439e8) — Needs fixes, 4 Important, 0 Critical

Verified correct by the reviewer at file:line: PREREG2["seeds"]==15 with seed_rule_output 5
recorded openly, version 3.0.0, budget 1000, explore_mid, HV reference "origin"; the 63-cell
overlap with corrected prose and a test asserting it against the real manifest;
CANDIDATE_ORACLES = sorted(PC.COMPONENTS) cross-checked by a function that raises GateHalt (and a
test confirming YAML stems fail the check); the full buffer persisted as [smiles, score,
call_index] triples with a test on record SHAPE and call_index ORDERING rather than source text,
and no component vectors stored; every constant, import and runner in library cells with
RUNS2_DIR a real path and a regex test on the cell tagging itself; the Phase 1 frame filtered to
explore_mid+ok with a test perturbing other settings; the reproduction gate halting below
MIN_ORACLES; the git-diff guard comparing against 5700883. phase2_stats.py,
phase2_components.py, phase2_power.py and phase2_power_gate.json absent from the diff, so
choose_seeds was not "fixed". Praised: the population-metric test computes hypervolume, the
brute-force front, novelty and diversity independently of the implementation.

Controller resolved all the "cannot verify" items:
- P0.make_oracle(name, budget=, freq_log=) and P0.run_backbone(backbone, orc, setting, seed,
  verbose=) match Phase 1's calls EXACTLY (phase0_lib.py:584, :1300 vs phase1_source.py:412,415).
  CLEAN.
- Phase 1's manifest WAS untracked -- only Phase 1/landscape_phase1.ipynb was tracked -- so the
  test would fail on a clean checkout. Ruling: committed as 8a9d67b. This does not touch the
  provenance hold, which concerns commit hashes being rewritten by a rebase on the machine that
  produced the verdicts, not files added here. Other Phase 1 artifacts stay untracked.
- Reviewer's Bash failed (seccomp) so it ran nothing. Controller ran the suite: 405 assertions
  under -W error, Phase 1's 59 still green.

Four Important -> fix round 1 (resumed original implementer):
1. The overlap ruling never reached the implementer: OVERLAP_TOL is still 1e-9, frozen into the
   prereg JSON and the prose, with no signed mean and nothing persisted. CONTROLLER EXPLICITLY
   AUTHORISED THE RE-FREEZE (new PREREG2_SHA256) because no Phase 2 run exists --
   results/runs_phase2/ is absent and LAUNCH_SWEEP2 is False. Legitimate now, forbidden after a
   run. Tests must straddle 1e-6 with clear margins (5e-7 / 5e-6), not add exactly 1e-6 to a float.
2. THE SWEEP CAN FAIL EXACTLY AS PHASE 1's RUNNER DID. p0() is first called inside worker threads,
   so a missing torch/TDC or wrong LCE_ROOT makes every job print CRASHED while the CLI reports
   "sweep finished" and exits 0 having written nothing. The accessor is sound; nothing triggers it
   in the main thread. Fix: p0() + tripwire before the pool, non-zero exit on any crash or
   non-ok record.
3. The sklearn tripwire guards only the notebook, not the sweep -- a script outside the kernel is
   exactly where valsartan_smarts produced 27 silent 0.0 runs in Phase 1. Folded into 2.
4. The overlap gate does not refuse the verdict: a could_not_run only prints and still writes
   phase2_verdict.json. :688 also lets V2 proceed with OVERLAP possibly None, raising at :704.
5. Resume trusts any file: status "error" records are skipped as done, so a transient OOM becomes
   permanent, and an interrupted budget-200 dry-run record would be accepted as a real run.
   Fix: require status=="ok" AND budget==PREREG2["budget"]; dry-run against a temp RUNS2_DIR.

Task 6: minor (deferred): replace the L1.P0 monkeypatch with direct TDC enumeration (judged an
  acceptable contained stopgap -- it halts rather than corrupts -- but it mutates another module's
  global, is untested, and enables the very fallback that produced YAML stems in Phase 1 defect 4)
Task 6: minor (deferred): no test pins that importing phase2_lib needs neither torch nor TDC
Task 6: minor (deferred): no assertion that persisted records lack component keys
Task 6: minor (deferred): surviving_oracles not bound to the prereg hash or sklearn version
Task 6: minor (deferred): analysis_frame2 and plot_scores_by_metric are notebook-only, untestable
Task 6: minor (deferred): population_metrics reaches into the private PC._REFS

Task 6: fix round 1/5 (5 addressed, 0 open; commits 29439e8..d3a4657)
  430 assertions (was 405). Re-freeze applied with controller authorisation: overlap tolerance now
  1e-6 absolute per cell, PREREG2_SHA256 42243abd... (was e8bb792e...), nothing else in PREREG2
  changed but two descriptive fields. Overlap gate reports the mean signed difference and persists
  to Phase 2/phase2_overlap_gate.json including failures. run_sweep2 calls p0() and the drd2
  tripwire in the MAIN THREAD before the pool, and the CLI exits 1 if any job lacks a complete ok
  record. Part 10 returns REFUSED when the overlap gate is missing or did not pass. Resume accepts
  only status=="ok" at the full budget, so error and budget-200 records are retried. The dry run
  goes through dry_run_check against a temp directory, restored even on interrupt.
  Controller verified directly: 430 assertions under -W error, Phase 1's 59 green, Phase 0/1
  sources untouched vs 5700883, prereg2_hash() == PREREG2_SHA256 == 42243abd..., grid 9 x 3 x 15
  = 405, setting explore_mid, RUNS2_DIR a real absolute path, overlap tolerance 1e-06
  "absolute, per cell" with "mean signed difference (phase1 - phase2)" in also_reported.
Task 6: complete (commits e6663cc..d3a4657, review clean, 6 minors deferred)

ALL SIX TASKS COMPLETE. Final whole-branch review dispatched on opus over 5700883..d3a4657
(17 commits), pointed at the ~20 deferred-minor ledger lines for triage, with the verified facts
listed so it does not re-derive them.

### FINAL WHOLE-BRANCH REVIEW (5700883..d3a4657, opus, 17 commits) — fix before merge

No Critical. The reviewer read phase2_stats.py, phase2_components.py, phase2_power.py,
phase2_source.py, build_phase2.py, run_sweep_phase2.py, the spec and all 1701 lines of the test
suite in full, and ran two focused checks of its own.

Cross-cutting verdict: THE STATISTICAL LAYER COMPOSES. All five named properties verified at
file:line — one normalisation applied once; the shared oracle resample ENFORCED BY DATA LAYOUT
(argwhere(idx==oi) fills every backbone column for the same (draw,slot) pairs, so no per-backbone
loop exists to drift); both argmaxes re-selected per draw with a non-unique argmax counting as
failure to replicate (conservative direction); Criterion C's gap and threshold in the same z unit;
the five verdict branches matching §6's table exhaustively.

Two independent checks it ran:
- `_hv3` at large n: built an independent exact 3-D algorithm and compared at n=5..1000 on
  anti-chains, uniform clouds, triplicate duplicates, 200 points sharing z, 200 sharing x, a
  3x3x3 grid and 500 coarsely-rounded points. Max discrepancy 3.3e-16, plus a 2M-draw Monte Carlo
  agreeing. So the flagged hypervolume deferred minor names a real coverage gap but THE BUG IT
  FEARS DOES NOT EXIST — no fix needed.
- Generated-artifact staleness: re-ran parse_percent and reproduced write_library/write_notebook
  in memory. phase2_lib.py, landscape_phase2.ipynb and run_sweep_phase2.py are BYTE-IDENTICAL to
  a rebuild, prereg2_hash()==PREREG2_SHA256==committed _sha256, status "verified", and the code
  dict and committed file differ in no key. The committed power gate records
  data_source "manifest" — the real gate, not the synthetic fallback.
It also confirmed nothing moved under cover of the tolerance re-freeze: replication_rate 0.95,
c_threshold 1.0, bh_q 0.05, n_boot 10000, alpha 0.05, seeds 15, seed_rule_output 5, budget 1000,
min_surviving_oracles 5, gate_tolerance 1e-6, hv_reference_point "origin", metric list and hash
all match phase2_stats's constants and the spec.

Five Important -> ONE fix wave (no second wave per the process), resumed the Task 6 implementer:
I1. load_results2 admits any ok record whatever budget or prereg wrote it. record_complete has
    the right predicate but is consulted only on resume. `--budget 200` would leave 405 records
    that RESULTS2 feeds to Criteria 0/A/B/C as one z score, no halt. Filter on budget AND
    prereg_sha256; restrict analysis_frame2 to surviving_oracles().
I2. n_runs_expected is wrong on the designed exclusion path: a COMPLETE 7-oracle sweep would
    publish 315/405 and print "INCOMPLETE SWEEP". Derive SWEEP2 and _n_exp from gate survivors.
I3. The overlap gate returns "passed" for any non-empty merge, so drift leaving 3 of 63 cells
    reads as a pass on 3 cells. The 63-cell test runs HERE, not on the execution machine where
    the gate decides. Compute n_expected and refuse on a partial merge.
I4. build_phase2.main() ignores write_library's failure return, writes the sweep script anyway
    and exits 0, leaving a stale library — L9's defect exactly. Return 1, and add a test
    comparing a regeneration to the committed artifacts.
I5. Three things §5.1 calls frozen are not in the prereg: component lists/modifiers/reference
    SMILES, the probe set, and GATE_MIN_RANGE=0.5 / GATE_MIN_NONZERO=0.20. The suite only
    BRACKETS the thresholds (range in (0.2,0.8], so 0.3 passes everything), and those two
    numbers decide which oracles survive. NOT added to the frozen file; pinned by tests plus
    stamping the prereg and component-table hashes into the reproduction-gate JSON and checking
    them in surviving_oracles().
Plus four cheap triage picks: assert winner_hv=="B" in C's 0.4-gap case (the only guard that
COSMETIC, the likeliest non-null verdict, is exercised with a real non-zero gap); ranolazine TPSA
mu to the mutation table; a test that the committed power gate's data_source is "manifest"; and
the criterion_C docstring (state "one" is equally reached when hypervolume is the unstable one).

Ruling: §5.1's freezing claim corrected by a new spec L10 rather than by editing the frozen file.
The file is frozen; the correction belongs in the limitations where a reader will see it. Cost if
wrong: a reader must read L10 to know what the hash does not cover.
Ruling: the reviewer's triage of all ~30 deferred minors accepted as written. Two were promoted
into the wave (C's 0.4-gap winner assertion; surviving_oracles not bound to the prereg hash,
folded into I5); one was already closed; the rest stay deferred with its reasoning on record.

### Fix wave re-review (d3a4657..6350779) — All findings addressed, 0 open

All five Important and all four one-liners ADDRESSED at file:line. Verified by the re-reviewer:
load_results2 drops wrong-budget and foreign-prereg records and reports them in the metric_error
style, with no path found where a legitimate record is dropped; SWEEP2/_n_exp now follow
surviving_oracles() with a test asserting the 7-oracle path gives 315 and the normal path 405;
phase1_overlap_gate computes n_expected and persists "incomplete" + raises GateHalt on a partial
merge, tested at 62/63, 9/63, a 6-survivor 54-cell frame and a float seed column; build_phase2
main() returns 1 before any write and a test regenerates into a temp dir asserting byte equality
with all three committed artifacts; GATE_MIN_RANGE==0.5 and GATE_MIN_NONZERO==0.20 pinned exactly
along with nine per-oracle probe sha256 literals and COMPONENTS_SHA, with save_gate stamping
prereg_sha256 + components_sha256 and surviving_oracles halting on mismatch or a missing stamp.
components_hash hashes every component VALUE on the probe set, so changed modifier parameters or
reference SMILES are detected behaviourally rather than by name.

All four disclosed departures judged SOUND, including the one the controller flagged: moving
Criterion C's "small" case from seed 13 to seed 1. The fixture is an invented plant, not a result
under test; at seed 13 noise made C the HV winner so the case never exercised the planted
sub-threshold effect, which is the finding itself. Seed 1 gives B with gap 0.44, `passes` false,
the original assertion unchanged plus a real `gap > 0.1` bound. Nothing else in the diff was
re-seeded. Residual weakness: the test depends on one seed.

Three Low items, adjudicated by the controller (no second fix wave):
Ruling: PARK the `components_hash` cache keyed on `id(PC.COMPONENTS)`. A freed-and-reused id could
return a stale hash, but it only bites if surviving_oracles() is called inside a mutation loop,
which only the test suite does; production binds COMPONENTS once at import. Real, not reachable
on the production path. Cost if wrong: a stale gate file accepted during a future mutation-style
test, caught by that test's own baseline check.
Ruling: PARK the overlap gate now halting when a metric_error or failed Phase 2 run shrinks the
merge below n_expected. Strict, but it is exactly the "refuse rather than credit a partial merge"
behaviour I1/I3 asked for, and a failed run is something the operator needs to know about before
a verdict anyway. Cost if wrong: an operator must fix or force-rerun one bad run before the
overlap gate will pass.
Ruling: the "unstamped reproduction-gate JSON halts surviving_oracles()" item is MOOT. Verified
no such file is committed (`git ls-files "Phase 2/"` returns only the notebook, the power gate and
the pre-registration), and Part 3 generates it stamped on the execution machine. No action.

PHASE 2 BUILD COMPLETE. 19 commits, 5700883..6350779. 473 assertions.
