#!/usr/bin/env python
"""Phase 3 tests: NSGA-II primitives (numpy only).

    python -u tests/test_phase3.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase3_pareto import (  # noqa: E402
    non_dominated_sort, crowding_distance, positive_mask,
    scalar_mating_pool, scalar_survival, nsga2_mating_pool, nsga2_survival,
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
    print(f"{N} assertions passed")


if __name__ == "__main__":
    main()
