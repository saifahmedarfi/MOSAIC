#!/usr/bin/env python
"""Phase 3 tests: NSGA-II primitives (numpy only).

    python -u tests/test_phase3.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase3_pareto import non_dominated_sort, crowding_distance, positive_mask  # noqa: E402

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


def main():
    print(f"numpy {np.__version__}, pandas {pd.__version__}")
    test_non_dominated_sort_hand_computed()
    test_non_dominated_sort_exact_duplicates_share_a_front()
    test_crowding_distance_boundaries_infinite_no_nan()
    test_crowding_distance_tiny_fronts()
    test_positive_mask_excludes_zero_components()
    print(f"{N} assertions passed")


if __name__ == "__main__":
    main()
