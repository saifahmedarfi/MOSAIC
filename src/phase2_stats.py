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
def pareto_front(points: np.ndarray) -> np.ndarray:
    """Non-dominated rows under maximisation, shape (k, d). Duplicates collapse to one."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be 2-D (n, d)")
    if len(pts) == 0:
        return pts.copy()
    pts = np.unique(pts, axis=0)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        # j dominates i if j >= i everywhere and j > i somewhere (rows are unique)
        dom = (pts >= pts[i]).all(axis=1)
        dom[i] = False
        if dom.any():
            keep[i] = False
    return pts[keep]


def _hv(pts: np.ndarray) -> float:
    """Exact hypervolume against the origin by recursive dimension sweep.

    Slice along the last coordinate: between consecutive distinct levels the cross-section
    is the (d-1)-dimensional HV of every point at or above that level. Polynomial in the
    number of points, exponential only in d (2..4 here).
    """
    n, d = pts.shape
    if n == 0:
        return 0.0
    if d == 1:
        return float(pts[:, 0].max())
    pts = pareto_front(pts)
    order = np.argsort(-pts[:, -1], kind="stable")
    pts = pts[order]
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
    pts = pts[(pts > 0.0).all(axis=1)]      # a zero coordinate encloses no volume
    if len(pts) == 0:
        return 0.0
    return float(_hv(pts))


# ----------------------------------------------------------------------- common unit
def _ok(df, metric: str, oracle: str):
    d = df[(df["oracle"] == oracle) & (df["status"] == "ok")]
    return d[np.isfinite(d[metric].to_numpy(dtype=float))]


def pooled_seed_sd(df, metric: str, oracle: str) -> float:
    """Within-cell seed SD of `metric` on `oracle`, pooled across backbones.

    A cell is (oracle, backbone). Pooling is sqrt of the df-weighted mean variance. Returns
    exactly 0.0 when the SD is below a RELATIVE tolerance of the metric's own magnitude.
    """
    d = _ok(df, metric, oracle)
    ss, dof = 0.0, 0
    for _, g in d.groupby("backbone"):
        v = g[metric].to_numpy(dtype=float)
        if len(v) >= 2:
            ss += float(((v - v.mean()) ** 2).sum())
            dof += len(v) - 1
    if dof == 0:
        return 0.0
    sd = float(np.sqrt(ss / dof))
    mean = float(d[metric].mean())
    return sd if sd > _REL_TOL * max(1.0, abs(mean)) else 0.0


def z_scores(df, metric: str) -> Dict[str, Dict[str, float]]:
    """{oracle: {backbone: z}}, z = (cell mean - mean across backbones) / pooled seed SD."""
    out: Dict[str, Dict[str, float]] = {}
    for oracle in sorted(df.loc[df["status"] == "ok", "oracle"].unique()):
        d = _ok(df, metric, oracle)
        if d.empty:
            continue
        means = d.groupby("backbone")[metric].mean()
        sd = pooled_seed_sd(df, metric, oracle)
        centred = means - means.mean()
        out[oracle] = {b: (0.0 if sd == 0.0 else float(c / sd)) for b, c in centred.items()}
    return out


def metric_score(df, metric: str) -> Dict[str, float]:
    """{backbone: mean z over the oracles on which that backbone has data}."""
    acc: Dict[str, list] = {}
    for per_bb in z_scores(df, metric).values():
        for b, z in per_bb.items():
            acc.setdefault(b, []).append(z)
    return {b: float(np.mean(v)) for b, v in sorted(acc.items())}
