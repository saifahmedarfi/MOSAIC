"""Phase 2 statistics -- pure numpy/scipy/pandas.

Deliberately imports NOTHING heavy: no torch, no rdkit, no TDC, no phase0_lib and no
phase1_lib. That is what lets the whole Phase 2 decision rule be tested on any machine,
with no GPU and no oracle downloads:

    python tests/test_phase2.py

This layer holds exact hypervolume and the noise-normalised common unit (spec 5.2.1) that
makes metrics of different scale comparable across oracles. Everything is maximisation,
every component lives in [0, 1], and the hypervolume reference point is the origin.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any, Dict, Sequence, Tuple

import numpy as np

# Spec 5.2, in order.
METRICS: Tuple[str, ...] = (
    "auc_top10",
    "hypervolume",
    "front_size",
    "front_coverage",
    "novelty",
    "diversity",
    "scalarization_loss",
)
PRIMARY_PAIR: Tuple[str, str] = ("auc_top10", "hypervolume")
SECONDARY_METRICS: Tuple[str, ...] = tuple(
    m for m in METRICS if m not in PRIMARY_PAIR and m != "scalarization_loss"
)

REPLICATION_RATE: float = 0.95
N_BOOT: int = 10_000
C_THRESHOLD: float = 1.0
BH_Q: float = 0.05
MIN_ORACLES: int = 5
GATE_TOL: float = 1e-6

# Relative zero-variance tolerance. np.std of 50 identical 0.3 values is ~1e-17, not 0, so
# an absolute `std > 0` passes and divides noise by noise. Phase 1 shipped that bug.
_REL_TOL = 1e-12


# --------------------------------------------------------------------------- hashing
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def metrics_hash(metrics: Sequence[str] = METRICS) -> str:
    """sha256 of the metric tuple the code actually uses (same pattern as features_hash)."""
    return hashlib.sha256(canonical_json(list(metrics)).encode()).hexdigest()


# --------------------------------------------------------------------------- geometry
def _check_no_nan(pts: np.ndarray) -> None:
    # A NaN component means a failed computation upstream. Dropping that molecule would
    # return a smaller front with no diagnostic -- an invented value -- so refuse instead.
    if np.isnan(pts).any():
        raise ValueError("NaN in points: a failed component must not be silently dropped")


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Non-dominated rows under maximisation, shape (k, d). Duplicates collapse to one."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be 2-D (n, d)")
    _check_no_nan(pts)
    if len(pts) == 0:
        return pts.copy()
    pts = np.unique(pts, axis=0)
    # A dominator has a strictly larger coordinate sum, so visiting rows by descending sum
    # means every dominator is already in the front when a dominated row is examined.
    pts = pts[np.argsort(-pts.sum(axis=1), kind="stable")]
    front = np.empty_like(pts)
    k = 0
    for p in pts:
        if k == 0 or not (front[:k] >= p).all(axis=1).any():
            front[k] = p
            k += 1
    return front[:k].copy()


def _hv2(pts: np.ndarray) -> float:
    """Exact 2-D HV, O(n log n): sweep x downwards carrying the running max of y."""
    order = np.argsort(-pts[:, 0], kind="stable")
    x, y = pts[order, 0], np.maximum.accumulate(pts[order, 1])
    return float(((x - np.append(x[1:], 0.0)) * y).sum())


def _hv3(pts: np.ndarray) -> float:
    """Exact 3-D HV: sweep z downwards, keeping the 2-D (x, y) staircase and its area."""
    from bisect import bisect_left
    pts = pts[np.argsort(-pts[:, 2], kind="stable")]
    zs = pts[:, 2].tolist()
    xs: list = []           # staircase, x ascending, y strictly descending
    ys: list = []
    area, total = 0.0, 0.0
    for i, (x, y, _) in enumerate(pts.tolist()):
        idx = bisect_left(xs, x)
        if idx < len(xs) and ys[idx] >= y:
            pass            # dominated in (x, y) by a point already at or above this z
        else:
            hi = idx + 1 if idx < len(xs) and xs[idx] == x else idx
            lo = idx
            while lo > 0 and ys[lo - 1] <= y:
                lo -= 1
            left = xs[lo - 1] if lo > 0 else 0.0
            right_y = ys[hi] if hi < len(xs) else 0.0
            old, prev = 0.0, left
            for j in range(lo, hi):
                old += ys[j] * (xs[j] - prev)
                prev = xs[j]
            old += right_y * (x - prev)
            area += y * (x - left) - old
            xs[lo:hi] = [x]
            ys[lo:hi] = [y]
        nxt = zs[i + 1] if i + 1 < len(zs) else 0.0
        total += area * (zs[i] - nxt)
    return total


def _hv(pts: np.ndarray) -> float:
    """Exact hypervolume against the origin by dimension sweep.

    d = 2 and 3 use dedicated sweeps; for d >= 4 slice along the last coordinate, where
    between consecutive levels the cross-section is the (d-1)-dimensional HV of every point
    at or above that level. Polynomial in the number of points, exponential only in d.
    """
    n, d = pts.shape
    if d == 1:
        return float(pts[:, 0].max())
    if d == 2:
        return _hv2(pts)
    if d == 3:
        return _hv3(pts)
    pts = pareto_front(pts)
    pts = pts[np.argsort(-pts[:, -1], kind="stable")]
    total = 0.0
    for i in range(len(pts)):
        lower = pts[i + 1, -1] if i + 1 < len(pts) else 0.0
        height = pts[i, -1] - lower
        if height > 0.0:
            total += _hv(pts[: i + 1, :-1]) * height
    return total


def hypervolume(points: np.ndarray) -> float:
    """HV of the front against the origin. Exactly 0.0 for an empty or all-origin set."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 1:
        raise ValueError("points must be 2-D (n, d) with d >= 1")
    _check_no_nan(pts)
    pts = pts[(pts > 0.0).all(axis=1)]      # a zero coordinate encloses no volume
    if len(pts) == 0:
        return 0.0
    return float(_hv(pts))


# ----------------------------------------------------------------------- common unit
# ONE implementation of the arithmetic. The point estimate (z_scores / metric_score) and
# every bootstrap draw (winner_replication) both reduce to `_z_and_score` over arrays of cell
# summaries shaped (..., oracle, backbone); the point estimate is the identity resample.
def _ok(df, metric: str, oracle: str):
    d = df[(df["oracle"] == oracle) & (df["status"] == "ok")]
    return d[np.isfinite(d[metric].to_numpy(dtype=float))]


def _cell_stats(v: np.ndarray):
    """(mean, sum of squares about the mean, dof, n) over the last axis of `v`."""
    n = v.shape[-1]
    m = v.mean(axis=-1)
    return m, ((v - m[..., None]) ** 2).sum(axis=-1), np.full(m.shape, n - 1.0), np.full(m.shape, float(n))


def _pooled_sd(mean, ss, dof, cnt):
    """Pooled seed SD per oracle (reduces the last, backbone, axis). Exactly 0.0 when below a
    RELATIVE tolerance of the metric's own magnitude. A cell with < 2 seeds adds 0 SS, 0 df."""
    tot = dof.sum(axis=-1)
    sd = np.sqrt(np.where(tot > 0, ss.sum(axis=-1) / np.maximum(tot, 1.0), 0.0))
    grand = np.nansum(mean * cnt, axis=-1) / np.maximum(cnt.sum(axis=-1), 1.0)
    return np.where(sd > _REL_TOL * np.maximum(1.0, np.abs(grand)), sd, 0.0)


def _z_and_score(mean, ss, dof, cnt):
    """z (..., oracle, backbone) and score (..., backbone). An absent cell has mean NaN,
    ss = dof = cnt = 0 and yields z NaN; a backbone with no cell scores NaN."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN slices are expected
        centred = mean - np.nanmean(mean, axis=-1, keepdims=True)
        sd = _pooled_sd(mean, ss, dof, cnt)[..., None]
        z = np.where(sd > 0.0, centred / np.where(sd == 0.0, 1.0, sd), 0.0)
        z = np.where(np.isnan(mean), np.nan, z)
        return z, np.nanmean(z, axis=-2)


def _cells(df, metric: str):
    """Oracles with data, backbones with data (both sorted) and {(oi, bi): values}."""
    ok = df[df["status"] == "ok"]
    oracles = [o for o in sorted(ok["oracle"].unique()) if not _ok(df, metric, o).empty]
    vals = {}
    for oi, o in enumerate(oracles):
        for b, g in _ok(df, metric, o).groupby("backbone"):
            vals[(oi, b)] = g[metric].to_numpy(dtype=float)
    bbs = sorted({b for _, b in vals})
    return oracles, bbs, {(oi, bbs.index(b)): v for (oi, b), v in vals.items()}


def _point_arrays(oracles, bbs, vals):
    shape = (len(oracles), len(bbs))
    mean, ss = np.full(shape, np.nan), np.zeros(shape)
    dof, cnt = np.zeros(shape), np.zeros(shape)
    for (oi, bi), v in vals.items():
        mean[oi, bi], ss[oi, bi], dof[oi, bi], cnt[oi, bi] = (x[()] for x in _cell_stats(v))
    return mean, ss, dof, cnt


def pooled_seed_sd(df, metric: str, oracle: str) -> float:
    """Within-cell seed SD of `metric` on `oracle`, pooled across backbones.

    A cell is (oracle, backbone). Pooling is sqrt of the df-weighted mean variance. Returns
    exactly 0.0 when the SD is below a RELATIVE tolerance of the metric's own magnitude.
    """
    d = df[df["oracle"] == oracle]
    oracles, bbs, vals = _cells(d, metric)
    if not oracles:
        return 0.0
    return float(_pooled_sd(*_point_arrays(oracles, bbs, vals))[0])


def z_scores(df, metric: str) -> Dict[str, Dict[str, float]]:
    """{oracle: {backbone: z}}, z = (cell mean - mean across backbones) / pooled seed SD."""
    oracles, bbs, vals = _cells(df, metric)
    if not oracles:
        return {}
    z, _ = _z_and_score(*_point_arrays(oracles, bbs, vals))
    return {o: {b: float(z[oi, bi]) for bi, b in enumerate(bbs) if not np.isnan(z[oi, bi])}
            for oi, o in enumerate(oracles)}


def metric_score(df, metric: str) -> Dict[str, float]:
    """{backbone: mean z over the oracles on which that backbone has data}."""
    oracles, bbs, vals = _cells(df, metric)
    if not oracles:
        return {}
    _, score = _z_and_score(*_point_arrays(oracles, bbs, vals))
    return {b: float(score[bi]) for bi, b in enumerate(bbs)}


# ------------------------------------------------------------- shared-oracle bootstrap
_TIE_TOL = 1e-9     # scores are in z units (O(1)); roundoff from centring is ~1e-16


def winner_replication(df, metric: str, n_boot: int = N_BOOT, seed: int = 0) -> Dict[str, Any]:
    """How often the point-estimate winner on `metric` is also the winner of a resample.

    Each draw resamples the oracle list ONCE with replacement and uses that same list for
    every backbone; within each selected (oracle, backbone) cell the seeds are then resampled
    with replacement independently. The across-oracle main effect (Phase 1: SD 0.229, far
    above any method difference) therefore cancels in the per-oracle centring exactly as it
    does in the point estimate. Resampling oracles independently per backbone would leave it
    in and push every draw's argmax toward chance, failing Criterion 0 mechanically.

    The winner is the argmax of `metric_score`, RE-SELECTED inside every draw (freezing the
    point-estimate winner and resampling around it is positive by construction). The draw uses
    the SAME `_z_and_score` core as the point estimate. A draw whose own top score is tied
    (within _TIE_TOL) counts for no backbone. A tie in the point estimate gives winner None,
    replication 0.0 and stable False; array order never breaks a tie.
    """
    oracles, bbs, vals = _cells(df, metric)
    if not oracles:
        return {"winner": None, "replication": 0.0, "per_backbone": {}, "stable": False}
    n_o, n_b = len(oracles), len(bbs)
    _, point = _z_and_score(*_point_arrays(oracles, bbs, vals))
    top = np.max(point)
    tied = np.isclose(point, top, rtol=_TIE_TOL, atol=_TIE_TOL)
    winner = None if tied.sum() > 1 else bbs[int(np.argmax(point))]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_o, size=(n_boot, n_o))      # ONE oracle list per draw
    shape = (n_boot, n_o, n_b)
    mean, ss = np.full(shape, np.nan), np.zeros(shape)
    dof, cnt = np.zeros(shape), np.zeros(shape)
    for oi in range(n_o):
        pos = np.argwhere(idx == oi)                    # (draw, slot) pairs that chose oracle
        if len(pos) == 0:
            continue
        for bi in range(n_b):
            v = vals.get((oi, bi))
            if v is None:
                continue
            samp = v[rng.integers(0, len(v), size=(len(pos), len(v)))]   # seeds, per cell
            for arr, x in zip((mean, ss, dof, cnt), _cell_stats(samp)):
                arr[pos[:, 0], pos[:, 1], bi] = x
    _, score = _z_and_score(mean, ss, dof, cnt)         # (n_boot, n_b)
    score = np.where(np.isnan(score), -np.inf, score)
    best = score.max(axis=1, keepdims=True)
    unique = np.isclose(score, best, rtol=_TIE_TOL, atol=_TIE_TOL).sum(axis=1) == 1
    arg = score.argmax(axis=1)
    per_bb = {b: float(((arg == bi) & unique).mean()) for bi, b in enumerate(bbs)}
    replication = 0.0 if winner is None else per_bb[winner]
    return {"winner": winner, "replication": replication, "per_backbone": per_bb,
            "stable": winner is not None and replication >= REPLICATION_RATE}


def criterion_0(df, n_boot: int = N_BOOT, seed: int = 0) -> Dict[str, Any]:
    """Is there anything to rank? Evaluated on the PRIMARY PAIR only, never the secondaries."""
    per = {m: winner_replication(df, m, n_boot=n_boot, seed=seed) for m in PRIMARY_PAIR}
    stable = [m for m in PRIMARY_PAIR if per[m]["stable"]]
    return {"per_metric": per, "n_stable": len(stable),
            "state": ("neither", "one", "both")[len(stable)], "stable_metrics": stable}
