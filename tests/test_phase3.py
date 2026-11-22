#!/usr/bin/env python
"""Phase 3 tests: NSGA-II primitives and the criteria (numpy/pandas only).

    python -u tests/test_phase3.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import phase2_stats as PS  # noqa: E402

from phase3_pareto import (  # noqa: E402
    non_dominated_sort, crowding_distance, positive_mask,
    scalar_mating_pool, scalar_survival, nsga2_mating_pool, nsga2_survival,
)
from phase3_stats import (  # noqa: E402
    METRICS, ARMS, F_BOUND, REPLICATION_RATE, metrics_hash3, check_collinear,
    arm_difference, criterion_P, criterion_Q, criterion_F, phase3_verdict, VERDICTS3,
)

N = 0


def check(cond, label):
    global N
    N += 1
    if not cond:
        raise AssertionError(f"FAILED: {label}")
    print(f"  ok  {label}")


def test_non_dominated_sort_hand_computed():
    # (0.9,0.1) and (0.1,0.9) mutually non-dominated; (0.5,0.5) dominated by neither;
    # (0.05,0.05) dominated by all three.
    pts = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.05, 0.05]])
    fronts = non_dominated_sort(pts)
    check(sorted(fronts[0].tolist()) == [0, 1, 2], "front 0 is the three mutually non-dominated")
    check(fronts[1].tolist() == [3], "front 1 is the dominated point")
    check(sum(len(f) for f in fronts) == 4, "every index assigned exactly once")


def test_non_dominated_sort_exact_duplicates_share_a_front():
    pts = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    fronts = non_dominated_sort(pts)
    check(len(fronts) == 1 and len(fronts[0]) == 3, "exact ties all in front 0")


def test_crowding_distance_boundaries_infinite_no_nan():
    pts = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    d = crowding_distance(pts)
    check(np.isinf(d[0]) and np.isinf(d[2]), "extremes are infinite")
    check(np.isfinite(d[1]), "interior point is finite")
    check(not np.isnan(d).any(), "no NaN")


def test_crowding_distance_tiny_fronts():
    for n in (1, 2):
        d = crowding_distance(np.linspace(0.1, 0.9, n * 2).reshape(n, 2))
        check(np.isinf(d).all(), f"all infinite at n={n}")
        check(not np.isnan(d).any(), f"no NaN at n={n}")


def test_positive_mask_excludes_zero_components():
    comps = np.array([[1.0, 0.5], [1.0, 0.0], [0.0, 0.0], [1e-12, 0.5]])
    check(positive_mask(comps).tolist() == [True, False, False, True], "only all-positive kept")


def test_non_dominated_sort_three_fronts():
    # chain of strictly dominated points: 0 dominates 1 dominates 2
    pts = np.array([[0.2, 0.2], [0.9, 0.9], [0.5, 0.5], [0.95, 0.1]])
    fronts = non_dominated_sort(pts)
    check(len(fronts) == 3, "three fronts reached")
    check(sorted(fronts[0].tolist()) == [1, 3], "front 0 is {1, 3}")
    check(fronts[1].tolist() == [2], "front 1 is {2}")
    check(fronts[2].tolist() == [0], "front 2 is {0}")


def test_crowding_distance_normalises_by_range_unequal_spacing():
    pts = np.array([[0, 1], [.2, .8], [.5, .5], [1, 0]], dtype=float)
    d = crowding_distance(pts)
    check(np.isinf(d[[0, 3]]).all(), "extremes infinite")
    check(np.allclose(d[[1, 2]], [1.0, 1.6]), "interior distances 1.0 and 1.6")
    pts2 = pts * 0.5  # scaling every objective must not change distances
    check(np.allclose(crowding_distance(pts2)[[1, 2]], [1.0, 1.6]), "scale invariant")


def test_crowding_distance_constant_objective_no_nan():
    pts = np.array([[.1, .5], [.2, .5], [.4, .5], [.9, .5]])
    d = crowding_distance(pts)
    check(not np.isnan(d).any(), "no NaN with a constant objective")
    check(np.isinf(d[[0, 3]]).all(), "extremes infinite")
    check(np.allclose(d[[1, 2]], [.375, .875]), "interior .375 and .875")


def test_crowding_distance_all_identical_points():
    d = crowding_distance(np.full((4, 2), 0.5))
    check(not np.isnan(d).any(), "no NaN for identical points")
    check(np.isinf(d[0]) and np.isinf(d[3]) and d[1] == 0 and d[2] == 0, "[inf, 0, 0, inf]")


def test_scalar_rules_match_phase0_lib_exactly():
    # phase0_lib._make_mating_pool draws with np.random.choice on the global RNG;
    # compare distributions, not draws: a zero-score member must never be selected
    # once another member has positive score.
    scores = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(0)
    idx = scalar_mating_pool(3, scores, 2000, rng)
    check((idx == 2).mean() > 0.99, "fitness-proportionate concentrates on the positive score")
    check(scalar_survival(np.array([0.1, 0.9, 0.5]), 2).tolist() == [1, 2], "survival is top-k by score")


def test_scalar_mating_pool_all_zero_is_uniform():
    idx = scalar_mating_pool(3, np.zeros(3), 3000, np.random.default_rng(0))
    check(all(0.25 < (idx == i).mean() < 0.42 for i in range(3)), "uniform when every score is zero")


def test_nsga2_survival_prefers_front_0_then_crowding():
    comps = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.05, 0.05]])
    keep, fell_back = nsga2_survival(comps, np.array([0.3, 0.3, 0.5, 0.05]), 3)
    check(not fell_back, "no fallback when positives exist")
    check(sorted(keep.tolist()) == [0, 1, 2], "front 0 fills the population, dominated point dropped")


def test_nsga2_excludes_zero_component_molecules():
    # index 1 has a zero component: NSGA-II rank 1 on raw vectors, but oracle score 0.0.
    comps = np.array([[0.4, 0.4], [1.0, 0.0], [0.3, 0.3]])
    keep, fell_back = nsga2_survival(comps, np.array([0.4, 0.0, 0.3]), 2)
    check(1 not in keep.tolist(), "zero-component molecule never survives")
    check(not fell_back, "not a fallback, just a filter")


def test_nsga2_falls_back_to_scalar_when_no_molecule_is_all_positive():
    comps = np.array([[1.0, 0.0], [0.0, 0.5], [0.0, 0.0]])
    scores = np.array([0.0, 0.0, 0.0])
    keep, fell_back = nsga2_survival(comps, scores, 2)
    check(fell_back, "fallback reported")
    check(len(keep) == 2, "population still filled")
    pool, fb2 = nsga2_mating_pool(comps, scores, 5, np.random.default_rng(0))
    check(fb2 and len(pool) == 5, "mating pool also falls back and still returns offspring_size")


def test_scalar_mating_pool_proportional_and_with_replacement():
    scores = np.array([1.0, 3.0])
    idx = scalar_mating_pool(2, scores, 20000, np.random.default_rng(1))
    check(len(idx) == 20000, "more draws than members: sampled with replacement")
    check(abs((idx == 1).mean() - 0.75) < 0.02, "probability proportional to score")


def test_scalar_survival_returns_all_when_population_exceeds_n():
    check(sorted(scalar_survival(np.array([0.3, 0.1]), 5).tolist()) == [0, 1], "k > n returns everyone")


def test_nsga2_survival_breaks_last_front_by_descending_crowding():
    # one front of 4 on a line: extremes (0, 3) have inf crowding; interior 1 is crowded, 2 is not.
    comps = np.array([[0.1, 0.9], [0.5, 0.5], [0.55, 0.45], [0.9, 0.1]])
    comps[1] = [0.30, 0.70]
    comps[2] = [0.65, 0.35]
    keep, _ = nsga2_survival(comps, np.ones(4), 3)
    check(0 in keep.tolist() and 3 in keep.tolist(), "boundary points (inf crowding) kept")
    # point 1 gap (0.1->0.65)/0.8, point 2 gap (0.3->0.9)/0.8: point 2 has the larger distance
    check(sorted(keep.tolist()) == [0, 2, 3], "least-crowded interior point wins the last slot")


def test_nsga2_survival_fills_front_by_front_and_returns_original_indices():
    # index 0 has a zero component and is excluded, so mapping back must skip it.
    comps = np.array([[1.0, 0.0], [0.9, 0.9], [0.5, 0.5], [0.2, 0.2], [0.1, 0.1]])
    keep, fb = nsga2_survival(comps, np.array([0.0, .9, .5, .2, .1]), 3)
    check(not fb and keep.tolist() == [1, 2, 3], "fronts taken in rank order, indices into original population")


def test_nsga2_survival_shortfall_filled_from_excluded_by_score():
    comps = np.array([[0.5, 0.5], [0.0, 0.9], [0.9, 0.0]])
    keep, fb = nsga2_survival(comps, np.array([0.5, 0.1, 0.2]), 3)
    check(not fb and len(keep) == 3 and keep[0] == 0, "population filled when positives are fewer")
    check(keep.tolist() == [0, 2, 1], "shortfall filled by descending score")


def test_nsga2_mating_pool_excludes_zero_component_and_uses_original_indices():
    comps = np.array([[1.0, 0.0], [0.5, 0.5], [0.4, 0.4], [0.0, 1.0]])
    pool, fb = nsga2_mating_pool(comps, np.array([0, .5, .4, 0]), 4000, np.random.default_rng(0))
    check(not fb and len(pool) == 4000, "no fallback, offspring_size returned")
    check(set(pool.tolist()) <= {1, 2}, "only all-positive members, as original indices")


def test_nsga2_mating_pool_tournament_prefers_better_front():
    # index 0 is front 0; indices 1..3 are dominated. Uniform draw would pick 0 w.p. 0.25;
    # a binary tournament picks it whenever it is in the pair: 1 - 0.75^2 = 0.4375.
    comps = np.array([[0.9, 0.9], [0.5, 0.5], [0.4, 0.4], [0.3, 0.3]])
    pool, _ = nsga2_mating_pool(comps, np.ones(4), 20000, np.random.default_rng(0))
    check(abs((pool == 0).mean() - 0.4375) < 0.02, "tournament win rate of the front-0 member is 0.4375")
    check((pool == 3).mean() < 0.07, "worst member rarely wins (expected 1/16)")


def test_nsga2_mating_pool_tournament_tiebreaks_on_crowding():
    # one front of 4: points 0 and 3 have inf crowding, 1 and 2 finite.
    comps = np.array([[0.1, 0.9], [0.4, 0.6], [0.6, 0.4], [0.9, 0.1]])
    pool, _ = nsga2_mating_pool(comps, np.ones(4), 20000, np.random.default_rng(0))
    # P(interior member wins) = P(both draws equal that member) + ties between equals ~ small
    check(((pool == 0) | (pool == 3)).mean() > 0.6, "boundary (inf crowding) members win same-front tournaments")


def test_scalar_mating_pool_nonfinite_total_is_uniform():
    # the uniform branch is only reachable via a non-finite total (zeros stay positive via MINIMUM)
    idx = scalar_mating_pool(3, np.array([np.nan, 1.0, 2.0]), 3000, np.random.default_rng(0))
    check(all(0.25 < (idx == i).mean() < 0.42 for i in range(3)), "uniform when total is non-finite")


def test_nsga2_fallback_delegates_to_scalar_rules_with_nonzero_scores():
    comps = np.array([[1.0, 0.0], [0.0, 0.5], [0.0, 0.0]])  # no all-positive row
    scores = np.array([0.1, 0.9, 0.5])
    keep, fb = nsga2_survival(comps, scores, 2)
    check(fb and keep.tolist() == [1, 2], "fallback survival is scalar top-k by score")
    pool, fb2 = nsga2_mating_pool(comps, scores, 4000, np.random.default_rng(0))
    check(fb2 and len(pool) == 4000, "fallback pool reports and returns offspring_size")
    # fitness-proportionate: P = .1/1.5, .9/1.5, .5/1.5 = .067, .6, .333
    check(abs((pool == 1).mean() - 0.6) < 0.03 and abs((pool == 2).mean() - 1 / 3) < 0.03,
          "fallback pool is fitness-proportionate on scores")


# A zero-component row cannot dominate a positive row, so it never changes positive ranks;
# but it sits in front 0 and so changes crowding (range and boundary status) unless it is
# masked out BEFORE sorting. A,B,C positive; Z has a zero component.
_MASK_COMPS = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2], [1.0, 0.0]])


def test_nsga2_survival_masks_before_sorting():
    # masked first: A and C are the boundaries (inf), B is interior -> keep {A, C}.
    # rank-all-then-filter: Z is a boundary, C interior (1.25) < B (1.5) -> would keep {A, B}.
    keep, fb = nsga2_survival(_MASK_COMPS, np.array([.5, .5, .5, 0.0]), 2)
    check(not fb and sorted(keep.tolist()) == [0, 2], "crowding computed without the zero-component row")


def test_nsga2_mating_pool_masks_before_sorting():
    # masked first: A, C inf, B finite -> C wins 4/9; rank-all-then-slice -> C wins 1/9.
    pool, fb = nsga2_mating_pool(_MASK_COMPS, np.array([.5, .5, .5, 0.0]), 30000, np.random.default_rng(0))
    check(not fb and 3 not in pool.tolist(), "zero-component row never drawn")
    check(abs((pool == 2).mean() - 4 / 9) < 0.02, "C tournament win rate 4/9 (crowding without Z)")
    check(abs((pool == 1).mean() - 1 / 9) < 0.02, "B tournament win rate 1/9")


# ---------------------------------------------------------------- Task 4: the criteria
def _cell_values(mu, noise, n, rng):
    """`n` seed values whose cell mean is exactly `noise * mu` and whose sample SD (ddof=1)
    is exactly `noise`, so the spec 6.0 z unit divides the scale straight back out."""
    e = rng.standard_normal(n)
    e = e - e.mean()
    sd = float(e.std(ddof=1)) if n > 1 else 0.0
    if sd > 0.0:
        e = e / sd
    return noise * (mu + e)


def synth3(means, oracles=tuple(f"o{i}" for i in range(9)), seeds=tuple(range(30)),
           noise=1.0, metric="auc_top10", setting="explore_mid", rng=None):
    """Synthetic Phase 3 analysis frame, keyed on `arm` (never `backbone`).

    `means[arm]` is that arm's true cell mean IN THE SPEC 6.0 z UNIT. The per-seed value is
    `noise * (mu + e)` with `e` standardised to mean exactly 0 and sample SD exactly 1 inside
    the cell, so the within-cell pooled seed SD is exactly `noise` and
    `score(a) - score(b)` is exactly `means[a] - means[b]`. Every Phase 3 threshold is in that
    unit and is never multiplied by an SD again (spec 6.0), so the fixture and the criteria
    speak the same language and each test's numbers are the numbers its comment claims.
    `noise` is then a pure rescaling of the metric, which z-standardisation is invariant to by
    construction; `noise=0.0` collapses every cell to a constant and exercises the zero-SD
    path, exactly as `tests/test_phase2.py`'s `synth` does.

    Two deliberate departures from that `synth`: the axis column is `arm`, and the metrics
    OTHER than `metric` carry mean-zero noise rather than NaN -- Criterion F reads both
    members of the primary pair at once, and a collinearity check needs columns with variance.
    """
    rng = rng or np.random.default_rng(0)
    seeds = tuple(seeds)
    rows = []
    for o in oracles:
        for arm, mu in means.items():
            cell = {m: _cell_values(0.0, noise, len(seeds), rng) for m in METRICS}
            cell[metric] = _cell_values(mu, noise, len(seeds), rng)
            for k, s in enumerate(seeds):
                row = {"oracle": o, "arm": arm, "setting": setting, "seed": s, "status": "ok"}
                for m in METRICS:
                    row[m] = float(cell[m][k])
                rows.append(row)
    return pd.DataFrame(rows)


def _independent_oracle_draws(df, metric, arm_a, arm_b, n_boot=2000, seed=0):
    """The WRONG scheme (spec 6.0), kept only to show the sharing test bites: a SEPARATE
    oracle list per arm. The per-oracle centring then compares arms measured on different
    oracles, so the across-oracle main effect stays in z instead of cancelling.
    `phase1_stats.flip_pairs` is this precedent and it is the wrong one."""
    d = df.assign(backbone=df["arm"].to_numpy())
    oracles, arms, vals = PS._cells(d, metric)
    n_o, n_a = len(oracles), len(arms)
    rng = np.random.default_rng(seed)
    per_arm = []
    for bi in range(n_a):
        idx = rng.integers(0, n_o, size=(n_boot, n_o))      # one list PER ARM, not per draw
        per_arm.append([a[:, :, bi] for a in PS._boot_cells(n_o, n_a, [vals], idx, rng)[0]])
    stacked = [np.stack([c[k] for c in per_arm], axis=-1) for k in range(4)]
    _, score = PS._z_and_score(*stacked)
    return score[:, arms.index(arm_a)] - score[:, arms.index(arm_b)]


def test_criterion_P_passes_only_when_the_margin_is_inside_the_quantile():
    # treatment beats control by 1.0 z with negligible noise -> passes at delta_p = 0.5
    df = synth3({"anchor": 0.0, "control": 0.0, "treatment": 1.0}, noise=0.01, metric="hypervolume")
    check(criterion_P(df, 0.5, n_boot=2000)["passes"], "clear gain passes a 0.5 margin")
    check(not criterion_P(df, 1.5, n_boot=2000)["passes"], "same data fails a 1.5 margin")


def test_criterion_P_is_not_a_sign_test():
    # a gain that is positive in >95% of draws but smaller than delta_p must FAIL:
    # the margin sits inside the quantile, not beside it.
    df = synth3({"anchor": 0.0, "control": 0.0, "treatment": 0.2}, noise=0.01, metric="hypervolume")
    r = criterion_P(df, 0.5, n_boot=2000)
    check(r["diff"] > 0 and not r["passes"], "positive but below the margin fails")


def test_criterion_Q_is_non_inferiority_not_superiority():
    df = synth3({"anchor": 0.0, "control": 0.3, "treatment": 0.0}, noise=0.01, metric="auc_top10")
    check(criterion_Q(df, 0.5, n_boot=2000)["passes"], "a 0.3 loss passes a 0.5 tolerance")
    check(not criterion_Q(df, 0.1, n_boot=2000)["passes"], "a 0.3 loss fails a 0.1 tolerance")


def test_criterion_F_is_TOST_and_gets_easier_with_more_seeds():
    # anchor == control in truth; more seeds tightens the CI, so F must pass more readily.
    narrow = criterion_F(synth3({"anchor": 0.0, "control": 0.0, "treatment": 0.0},
                                seeds=tuple(range(30)), noise=0.3), n_boot=2000)
    wide = criterion_F(synth3({"anchor": 0.0, "control": 0.0, "treatment": 0.0},
                              seeds=(0, 1, 2), noise=0.3), n_boot=2000)
    check(narrow["passes"], "30 seeds passes")
    check(narrow["width"] < wide["width"], "more seeds gives a narrower CI")


def test_criterion_F_fails_a_genuinely_divergent_reimplementation():
    df = synth3({"anchor": 0.0, "control": 1.0, "treatment": 1.0}, noise=0.01)
    check(not criterion_F(df, n_boot=2000)["passes"], "a 1.0 z divergence halts the phase")


def test_verdict_table():
    mk = lambda p: {"passes": p}
    check(phase3_verdict(mk(True), mk(True), mk(True))["verdict"] == "PARETO-RECOVERS", "P and Q pass")
    check(phase3_verdict(mk(True), mk(False), mk(True))["verdict"] == "PARETO-TRADES", "Q fails")
    check(phase3_verdict(mk(False), mk(True), mk(True))["verdict"] == "PARETO-NULL", "P fails")
    check(phase3_verdict(mk(True), mk(True), mk(False))["verdict"] == "REFUSED", "F fails overrides all")


def test_check_collinear_catches_a_rescaled_metric():
    # Phase 2 defect 6: front_coverage was front_size/1000 and consumed its own FDR budget.
    df = synth3({"anchor": 0.0, "control": 0.0, "treatment": 0.0})
    df["front_coverage"] = df["front_size"] / 1000.0
    # The brief wrote the expected pair as ("front_size", "front_coverage"); sorted() puts
    # "front_coverage" first, so that literal could never match. Order-free, same intent.
    check(("front_coverage", "front_size") in
          [tuple(sorted(p)) for p in check_collinear(df, ("front_size", "front_coverage"))],
          "perfect rescaling detected")
    check(check_collinear(df, METRICS) == [], "the frozen metric list is collinearity-free")


def test_criteria_do_not_emit_nan_on_tiny_fronts():
    df = synth3({"anchor": 0.0, "control": 0.0, "treatment": 0.5}, noise=0.0, metric="hypervolume")
    df.loc[df["arm"] == "treatment", "front_size"] = 1.0
    for r in (criterion_P(df, 0.1, n_boot=500), criterion_Q(df, 0.5, n_boot=500)):
        check(not np.isnan(r["diff"]), "no NaN with front_size == 1")


# --- beyond the brief: the mutants the brief's nine do not reach -----------------------
def test_criterion_P_margin_is_inside_the_quantile_not_beside_the_point_estimate():
    # diff is exactly 1.0 and the 5th percentile of the draws is ~0.86. A margin between the
    # two must FAIL: "point estimate >= delta_p" is a different, weaker test.
    df = synth3({"anchor": 0.0, "control": 0.0, "treatment": 1.0}, metric="hypervolume")
    r = criterion_P(df, 0.93, n_boot=4000)
    check(r["diff"] >= 0.93, "the point estimate clears the margin")
    check(r["lo"] < 0.93 and not r["passes"], "but the quantile does not, so P fails")


def test_criterion_Q_margin_is_inside_the_quantile_not_beside_the_point_estimate():
    df = synth3({"anchor": 0.0, "control": 0.3, "treatment": 0.0})
    r = criterion_Q(df, 0.35, n_boot=4000)
    check(r["diff"] <= 0.35, "the observed loss is inside the tolerance")
    check(r["hi"] > 0.35 and not r["passes"], "but the quantile is not, so Q fails")


def test_criterion_F_needs_both_metrics_inside_the_bound():
    # hypervolume diverges, auc_top10 does not: `all` fails where `any` would pass.
    df = synth3({"anchor": 0.0, "control": 1.0, "treatment": 1.0}, noise=0.01, metric="hypervolume")
    r = criterion_F(df, n_boot=2000)
    check(not r["passes"], "a divergence on either metric halts the phase")
    check(not r["per_metric"]["hypervolume"]["inside"], "hypervolume is the one outside")
    check(r["per_metric"]["auc_top10"]["inside"], "auc_top10 is inside, and does not rescue it")


def test_criterion_F_has_no_contains_zero_clause():
    # anchor - control = 0.12 z with a CI of about [0.04, 0.20]: well inside +-0.25 and
    # entirely ABOVE zero. TOST must pass. A "CI contains 0" clause would fail it, and would
    # get harder with more seeds -- the direction the spec 6.3 revision removed.
    df = synth3({"anchor": 0.12, "control": 0.0, "treatment": 0.0}, seeds=tuple(range(120)))
    r = criterion_F(df, n_boot=1000)
    auc = r["per_metric"]["auc_top10"]
    check(auc["lo"] > 0.0, "the CI excludes zero")
    check(auc["lo"] >= -F_BOUND and auc["hi"] <= F_BOUND, "and lies entirely inside the bound")
    check(r["passes"], "equivalence passes without containing zero")


def test_criterion_F_bound_is_a_flat_0_25_in_z_units():
    # F_BOUND is already in the spec 6.0 pooled-seed-SD z unit and is never multiplied by an
    # SD again. `noise` rescales the raw metric but not z, so these two verdicts are fixed;
    # a bound scaled by any measured SD (either direction) flips one of them.
    inside = criterion_F(synth3({"anchor": 0.1, "control": 0.0, "treatment": 0.0},
                                seeds=tuple(range(60)), noise=0.3), n_boot=2000)
    outside = criterion_F(synth3({"anchor": 0.5, "control": 0.0, "treatment": 0.0},
                                 seeds=tuple(range(60)), noise=0.3), n_boot=2000)
    check(inside["passes"], "a 0.1 z deviation is equivalent at a flat 0.25 bound")
    check(0.0 < inside["per_metric"]["auc_top10"]["hi"] < F_BOUND, "and its CI stops short of 0.25")
    check(not outside["passes"], "a 0.5 z deviation is not, at that same flat bound")


def test_verdict_reads_F_before_P():
    # Spec 6.3: a failed fidelity gate halts the phase whatever P and Q say. With P also
    # failing, "F first" is REFUSED and "P first" would report a null it cannot support.
    mk = lambda p: {"passes": p}
    check(phase3_verdict(mk(False), mk(False), mk(False))["verdict"] == "REFUSED",
          "F fails and P fails: REFUSED, not PARETO-NULL")
    check(phase3_verdict(mk(False), mk(True), mk(False))["verdict"] == "REFUSED",
          "an unfaithful reimplementation cannot produce a null either")
    v = phase3_verdict(mk(True), mk(False), mk(True))
    check(v["consequence"] == VERDICTS3["PARETO-TRADES"] and v["metrics_sha256"] == metrics_hash3(),
          "the verdict carries its consequence and the metric-list hash")


def test_bootstrap_shares_one_oracle_list_across_arms():
    # A 24 z across-oracle main effect, far larger than the 1.0 z arm effect. Sharing the
    # oracle list centres it out exactly as the point estimate does; a per-arm list leaves it
    # in and the criterion collapses.
    plain = synth3({"anchor": 0.0, "control": 0.0, "treatment": 1.0}, metric="hypervolume")
    df = plain.copy()
    offs = {o: 3.0 * i for i, o in enumerate(sorted(df["oracle"].unique()))}
    df["hypervolume"] = df["hypervolume"].to_numpy() + df["oracle"].map(offs).to_numpy()
    shared, base = criterion_P(df, 0.5, n_boot=2000), criterion_P(plain, 0.5, n_boot=2000)
    check(shared["passes"], "a 1.0 z gain survives a 24 z oracle main effect")
    check(abs(shared["lo"] - base["lo"]) < 1e-9 and abs(shared["hi"] - base["hi"]) < 1e-9,
          "the oracle main effect cancels draw by draw: identical CI with and without it")
    indep = _independent_oracle_draws(df, "hypervolume", "treatment", "control", n_boot=2000)
    check(float(np.percentile(indep, 5)) < 0.5,
          "a per-arm oracle list leaves the main effect in and the 5th percentile collapses")
    check((shared["hi"] - shared["lo"]) * 10 < float(np.percentile(indep, 97.5) - np.percentile(indep, 2.5)),
          "sharing the oracle list is what makes the interval usable")


def test_thresholds_are_in_z_units_and_never_multiplied_by_an_sd_again():
    df = synth3({"anchor": 0.0, "control": 0.0, "treatment": 1.0}, noise=0.01, metric="hypervolume")
    big = df.copy()
    big["hypervolume"] = big["hypervolume"] * 100.0      # same z, 100x the raw SD
    a, b = criterion_P(df, 0.5, n_boot=1000), criterion_P(big, 0.5, n_boot=1000)
    check(abs(a["diff"] - b["diff"]) < 1e-9 and abs(a["diff"] - 1.0) < 1e-9,
          "score difference is 1.0 z at either raw scale")
    check(a["passes"] and b["passes"], "rescaling the metric cannot change the verdict")


def test_metrics_hash3_pins_the_frozen_three():
    check(METRICS == ("auc_top10", "hypervolume", "front_size"), "the frozen metric list")
    check(ARMS == ("anchor", "control", "treatment"), "the three arms")
    check(F_BOUND == 0.25 and REPLICATION_RATE == 0.95, "the frozen bound and rate")
    check(len(metrics_hash3()) == 64 and metrics_hash3() == metrics_hash3(METRICS), "sha256, stable")
    check(metrics_hash3(METRICS + ("front_coverage",)) != metrics_hash3(),
          "adding a metric changes the hash")
    check(set(VERDICTS3) == {"PARETO-RECOVERS", "PARETO-TRADES", "PARETO-NULL", "REFUSED"},
          "the verdict table is spec 6.5's four")


def test_the_public_frame_keeps_the_arm_column():
    df = synth3({"anchor": 0.0, "control": 0.0, "treatment": 1.0}, seeds=(0, 1, 2), metric="hypervolume")
    r = arm_difference(df, "hypervolume", "treatment", "control", n_boot=200)
    check(r["n_oracles"] == 9 and np.isfinite(r["diff"]), "arm_difference reads an `arm` frame")
    check("arm" in df.columns and "backbone" not in df.columns,
          "the frame is not renamed in place: the adapter is internal")
    raised = False
    try:
        arm_difference(df.rename(columns={"arm": "backbone"}), "hypervolume",
                       "treatment", "control", n_boot=10)
    except Exception:
        raised = True
    check(raised, "a frame keyed on `backbone` is refused, not silently mis-analysed")


def main():
    print(f"numpy {np.__version__}, pandas {pd.__version__}")
    test_non_dominated_sort_hand_computed()
    test_non_dominated_sort_exact_duplicates_share_a_front()
    test_crowding_distance_boundaries_infinite_no_nan()
    test_crowding_distance_tiny_fronts()
    test_positive_mask_excludes_zero_components()
    test_non_dominated_sort_three_fronts()
    test_crowding_distance_normalises_by_range_unequal_spacing()
    test_crowding_distance_constant_objective_no_nan()
    test_crowding_distance_all_identical_points()
    test_scalar_rules_match_phase0_lib_exactly()
    test_scalar_mating_pool_all_zero_is_uniform()
    test_nsga2_survival_prefers_front_0_then_crowding()
    test_nsga2_excludes_zero_component_molecules()
    test_nsga2_falls_back_to_scalar_when_no_molecule_is_all_positive()
    test_scalar_mating_pool_proportional_and_with_replacement()
    test_scalar_mating_pool_nonfinite_total_is_uniform()
    test_scalar_survival_returns_all_when_population_exceeds_n()
    test_nsga2_survival_breaks_last_front_by_descending_crowding()
    test_nsga2_survival_fills_front_by_front_and_returns_original_indices()
    test_nsga2_survival_shortfall_filled_from_excluded_by_score()
    test_nsga2_mating_pool_excludes_zero_component_and_uses_original_indices()
    test_nsga2_mating_pool_tournament_prefers_better_front()
    test_nsga2_mating_pool_tournament_tiebreaks_on_crowding()
    test_nsga2_fallback_delegates_to_scalar_rules_with_nonzero_scores()
    test_nsga2_survival_masks_before_sorting()
    test_nsga2_mating_pool_masks_before_sorting()
    test_criterion_P_passes_only_when_the_margin_is_inside_the_quantile()
    test_criterion_P_is_not_a_sign_test()
    test_criterion_Q_is_non_inferiority_not_superiority()
    test_criterion_F_is_TOST_and_gets_easier_with_more_seeds()
    test_criterion_F_fails_a_genuinely_divergent_reimplementation()
    test_verdict_table()
    test_check_collinear_catches_a_rescaled_metric()
    test_criteria_do_not_emit_nan_on_tiny_fronts()
    test_criterion_P_margin_is_inside_the_quantile_not_beside_the_point_estimate()
    test_criterion_Q_margin_is_inside_the_quantile_not_beside_the_point_estimate()
    test_criterion_F_needs_both_metrics_inside_the_bound()
    test_criterion_F_has_no_contains_zero_clause()
    test_criterion_F_bound_is_a_flat_0_25_in_z_units()
    test_verdict_reads_F_before_P()
    test_bootstrap_shares_one_oracle_list_across_arms()
    test_thresholds_are_in_z_units_and_never_multiplied_by_an_sd_again()
    test_metrics_hash3_pins_the_frozen_three()
    test_the_public_frame_keeps_the_arm_column()
    print(f"{N} assertions passed")


if __name__ == "__main__":
    main()
