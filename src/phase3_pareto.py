"""NSGA-II primitives for Phase 3 Pareto-front selection.

Maximisation semantics throughout: every component is in [0,1], higher is better.
numpy only.
"""
from typing import List

import numpy as np


def non_dominated_sort(points: np.ndarray) -> List[np.ndarray]:
    """Fast non-dominated sort (O(n^2) dominance counts).

    Returns fronts as arrays of row indices, front 0 first; every row index appears in
    exactly one front. Exact duplicates do not dominate each other, so they share a front.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return []
    # dom[i, j] is True when i dominates j: >= everywhere and > somewhere.
    ge = (pts[:, None, :] >= pts[None, :, :]).all(axis=2)
    gt = (pts[:, None, :] > pts[None, :, :]).any(axis=2)
    dom = ge & gt
    dom_count = dom.sum(axis=0)  # number of points dominating j
    fronts: List[np.ndarray] = []
    current = np.flatnonzero(dom_count == 0)
    while len(current):
        fronts.append(current)
        dom_count = dom_count - dom[current].sum(axis=0)
        dom_count[current] = -1  # mark assigned so it is never picked again
        current = np.flatnonzero(dom_count == 0)
    return fronts


def crowding_distance(points: np.ndarray) -> np.ndarray:
    """Crowding distance for the points of one front; one value per row.

    Each objective is normalised by its range across the front. Boundary rows get np.inf.
    An objective with exactly zero range contributes 0 to every row.
    """
    pts = np.asarray(points, dtype=float)
    n, m = pts.shape
    d = np.zeros(n)
    for k in range(m):
        order = np.argsort(pts[:, k], kind="stable")
        col = pts[order, k]
        rng = col[-1] - col[0]
        d[order[0]] = np.inf
        d[order[-1]] = np.inf
        if rng == 0.0:
            continue
        d[order[1:-1]] += (col[2:] - col[:-2]) / rng
    return d


def positive_mask(comps: np.ndarray) -> np.ndarray:
    """True where every component is > 0.

    Frozen scientific rule, not an optimisation: every oracle aggregates its components by
    geometric mean, so a molecule with any zero component scores exactly 0.0 on the real
    objective. Kept literal on purpose.
    """
    return (np.asarray(comps) > 0).all(axis=1)
