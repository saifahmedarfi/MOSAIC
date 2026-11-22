"""NSGA-II primitives for Phase 3 Pareto-front selection.

Maximisation semantics throughout: every component is in [0,1], higher is better.
numpy only.
"""
from typing import List, Tuple

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


def scalar_mating_pool(n_pop: int, scores: np.ndarray, offspring_size: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Fitness-proportionate sampling of indices; transcribes phase0_lib._make_mating_pool.

    MINIMUM is added to every score; sampling is with replacement; uniform when the total
    is non-positive or non-finite.
    """
    MINIMUM = 1e-10
    sc = np.array(scores, dtype=np.float64) + MINIMUM
    total = sc.sum()
    if total <= 0 or not np.isfinite(total):
        probs = np.ones(len(sc)) / len(sc)
    else:
        probs = sc / total
    return rng.choice(n_pop, p=probs, size=offspring_size, replace=True)


def scalar_survival(scores: np.ndarray, population_size: int) -> np.ndarray:
    """Top-k indices by score; transcribes phase0_lib's np.argsort(...)[::-1][:population_size]."""
    return np.argsort(scores)[::-1][:population_size]


def _rank_and_crowding(comps: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    fronts = non_dominated_sort(comps)
    rank = np.zeros(len(comps), dtype=int)
    crowd = np.zeros(len(comps))
    for r, f in enumerate(fronts):
        rank[f] = r
        crowd[f] = crowding_distance(comps[f])
    return fronts, rank, crowd


def nsga2_mating_pool(comps: np.ndarray, scores: np.ndarray, offspring_size: int,
                      rng: np.random.Generator) -> Tuple[np.ndarray, bool]:
    """Binary tournament on (front_rank, -crowding), with replacement, over all-positive members.

    Returns (indices into the given population, fell_back). When no member has all components
    positive, delegates to scalar_mating_pool and reports fell_back=True.
    """
    comps = np.asarray(comps, dtype=float)
    pos = np.flatnonzero(positive_mask(comps))
    if len(pos) == 0:
        return scalar_mating_pool(len(comps), scores, offspring_size, rng), True
    _, rank, crowd = _rank_and_crowding(comps[pos])
    a = rng.integers(0, len(pos), size=offspring_size)
    b = rng.integers(0, len(pos), size=offspring_size)
    a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] >= crowd[b]))
    return pos[np.where(a_wins, a, b)], False


def nsga2_survival(comps: np.ndarray, scores: np.ndarray,
                   population_size: int) -> Tuple[np.ndarray, bool]:
    """Fill by front over all-positive members; last front broken by descending crowding.

    Returns (indices, fell_back). When no member has all components positive, delegates to
    scalar_survival and reports fell_back=True. If fewer all-positive members exist than
    population_size, the shortfall is filled from the excluded members by descending score.
    """
    comps = np.asarray(comps, dtype=float)
    pos = np.flatnonzero(positive_mask(comps))
    if len(pos) == 0:
        return scalar_survival(np.asarray(scores), population_size), True
    fronts, _, crowd = _rank_and_crowding(comps[pos])
    chosen: List[int] = []
    for f in fronts:
        room = population_size - len(chosen)
        if room <= 0:
            break
        if len(f) <= room:
            chosen.extend(f.tolist())
        else:
            order = np.argsort(-crowd[f], kind="stable")
            chosen.extend(f[order[:room]].tolist())
    keep = pos[np.array(chosen, dtype=int)]
    short = population_size - len(keep)
    if short > 0:
        rest = np.setdiff1d(np.arange(len(comps)), keep)
        sc = np.asarray(scores, dtype=float)[rest]
        keep = np.concatenate([keep, rest[np.argsort(sc)[::-1][:short]]])
    return keep, False
