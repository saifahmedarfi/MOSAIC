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


def _boot_cells(n_o: int, n_b: int, cellvals, idx: np.ndarray, rng) -> list:
    """Hierarchical draws. `idx` (n_boot, n_slot) names the oracle chosen for each slot (ONE list
    per draw, shared by every backbone). Within each chosen (oracle, backbone) cell the seeds
    are resampled with replacement, with the SAME seed indices for every dict in `cellvals`
    (so two metrics measured on the same runs are resampled jointly). Returns one
    (mean, ss, dof, cnt) tuple per dict, arrays shaped (n_boot, n_slot, n_b)."""
    n_boot, n_slot = idx.shape
    shape = (n_boot, n_slot, n_b)
    out = [(np.full(shape, np.nan), np.zeros(shape), np.zeros(shape), np.zeros(shape))
           for _ in cellvals]
    for oi in range(n_o):
        pos = np.argwhere(idx == oi)                    # (draw, slot) pairs that chose oracle
        if len(pos) == 0:
            continue
        for bi in range(n_b):
            if (oi, bi) not in cellvals[0]:
                continue
            k = len(cellvals[0][(oi, bi)])
            pick = rng.integers(0, k, size=(len(pos), k))        # seeds, per cell
            for vals, arrs in zip(cellvals, out):
                for arr, x in zip(arrs, _cell_stats(vals[(oi, bi)][pick])):
                    arr[pos[:, 0], pos[:, 1], bi] = x
    return out


def _unique_argmax(score: np.ndarray):
    """(argmax, unique) over the last axis; NaN never wins and a top score tied within
    _TIE_TOL is not unique, so array order never breaks a tie."""
    score = np.where(np.isnan(score), -np.inf, score)
    best = score.max(axis=-1, keepdims=True)
    unique = np.isclose(score, best, rtol=_TIE_TOL, atol=_TIE_TOL).sum(axis=-1) == 1
    return score.argmax(axis=-1), unique


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
    _, score = _z_and_score(*_boot_cells(n_o, n_b, [vals], idx, rng)[0])    # (n_boot, n_b)
    arg, unique = _unique_argmax(score)
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


# ------------------------------------------------------------------- decision criteria
def _both_finite(df, metrics: Sequence[str]):
    """Ok runs on which every named metric is finite, so the metrics share one set of runs
    and can be resampled jointly."""
    d = df[df["status"] == "ok"]
    return d[np.isfinite(d[list(metrics)].to_numpy(dtype=float)).all(axis=1)]


def _joint_cells(df, metrics: Sequence[str]):
    """Oracles, backbones and one {(oi, bi): values} dict per metric over a SHARED index
    space, built from runs finite on every metric so that dict values align seed by seed."""
    d = _both_finite(df, metrics)
    oracles = sorted(d["oracle"].unique())
    bbs = sorted(d["backbone"].unique())
    cellvals = [{} for _ in metrics]
    for (o, b), g in d.groupby(["oracle", "backbone"]):
        for vals, m in zip(cellvals, metrics):
            vals[(oracles.index(o), bbs.index(b))] = g[m].to_numpy(dtype=float)
    return oracles, bbs, cellvals


def criterion_A(df, n_boot: int = N_BOOT, seed: int = 0) -> Dict[str, Any]:
    """Do the primary pair pick different winners, and does that survive resampling?

    Both argmaxes are RE-SELECTED inside every draw, on one shared oracle list and one shared
    seed resample (the two metrics are measured on the same runs). A draw counts only when
    both argmaxes are unique and differ; freezing the point-estimate winners instead would
    be positive by construction on a pure-noise frame.
    """
    auc_m, hv_m = PRIMARY_PAIR
    oracles, bbs, cellvals = _joint_cells(df, PRIMARY_PAIR)
    if not oracles:
        return {"winner_auc": None, "winner_hv": None, "differ": False,
                "replication": 0.0, "passes": False}
    n_o, n_b = len(oracles), len(bbs)
    winners = []
    for vals in cellvals:
        _, point = _z_and_score(*_point_arrays(oracles, bbs, vals))
        tied = np.isclose(point, np.max(point), rtol=_TIE_TOL, atol=_TIE_TOL)
        winners.append(None if tied.sum() > 1 else bbs[int(np.argmax(point))])
    w_auc, w_hv = winners
    differ = w_auc is not None and w_hv is not None and w_auc != w_hv

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_o, size=(n_boot, n_o))
    (arg_a, uni_a), (arg_h, uni_h) = (
        _unique_argmax(_z_and_score(*c)[1]) for c in _boot_cells(n_o, n_b, cellvals, idx, rng))
    replication = float((uni_a & uni_h & (arg_a != arg_h)).mean())
    return {"winner_auc": w_auc, "winner_hv": w_hv, "differ": bool(differ),
            "replication": replication,
            "passes": bool(differ and replication >= REPLICATION_RATE)}


def bh_fdr(pvals: Dict[str, float], q: float = BH_Q) -> Dict[str, Dict[str, Any]]:
    """Benjamini-Hochberg step-up. `threshold` is q * rank / m for that key's rank; every key
    up to the largest rank whose p is within its threshold is significant."""
    m = len(pvals)
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    thr = {k: q * (r + 1) / m for r, k in enumerate(order)}
    cut = max((r for r, k in enumerate(order) if pvals[k] <= thr[k]), default=-1)
    return {k: {"threshold": thr[k], "significant": r <= cut} for r, k in enumerate(order)}


_B_NULL = 2.0 / 3.0     # P(within-oracle winners differ) for unrelated winners, 3 backbones


def _differing_fraction(z_a: np.ndarray, z_m: np.ndarray) -> np.ndarray:
    """Fraction of oracles whose unique within-oracle winner differs between two z arrays
    (..., oracle, backbone); oracles with a tied winner on either side do not survive.
    NaN where nothing survives."""
    arg_a, uni_a = _unique_argmax(z_a)
    arg_m, uni_m = _unique_argmax(z_m)
    surv = uni_a & uni_m
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(surv.sum(axis=-1) > 0,
                        (surv & (arg_a != arg_m)).sum(axis=-1) / surv.sum(axis=-1), np.nan)


def criterion_B(df, n_boot: int = N_BOOT, seed: int = 0) -> Dict[str, Any]:
    """Secondary metrics: per metric, the fraction of surviving oracles whose within-oracle
    winner differs from auc_top10's, tested against 2/3 by hierarchical bootstrap and
    corrected by Benjamini-Hochberg. Reported, never verdict-bearing.

    A metric with no oracle in which both it and auc_top10 have a unique winner has no finite
    variation to test; it is named in `degenerate_metrics` and removed BEFORE bh_fdr so it
    cannot consume FDR budget. p is two-sided: twice the smaller bootstrap tail beyond 2/3.
    """
    base = PRIMARY_PAIR[0]
    per, pvals, degenerate = {}, {}, []
    for k, m in enumerate(SECONDARY_METRICS):
        oracles, bbs, (v_auc, v_m) = _joint_cells(df, (base, m))
        point = np.nan
        if oracles:
            z = [_z_and_score(*_point_arrays(oracles, bbs, v))[0] for v in (v_auc, v_m)]
            point = float(_differing_fraction(*z))
        if np.isnan(point):
            degenerate.append(m)
            continue
        rng = np.random.default_rng([seed, k])
        n_o = len(oracles)
        idx = rng.integers(0, n_o, size=(n_boot, n_o))
        zb = [_z_and_score(*c)[0] for c in _boot_cells(n_o, len(bbs), [v_auc, v_m], idx, rng)]
        draws = _differing_fraction(*zb)
        draws = draws[~np.isnan(draws)]
        if len(draws) == 0:         # no draw kept a surviving oracle: nothing to test
            degenerate.append(m)
            continue
        lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
        tail = min((draws <= _B_NULL + _TIE_TOL).sum(), (draws >= _B_NULL - _TIE_TOL).sum())
        p = float(min(1.0, 2.0 * (tail + 1) / (len(draws) + 1)))
        per[m] = {"fraction": point, "lo": lo, "hi": hi, "p": p}
        pvals[m] = p
    for m, r in bh_fdr(pvals).items():
        per[m].update(r)
    return {"per_metric": per, "degenerate_metrics": degenerate}


def criterion_C(df, c0: Dict, cA: Dict) -> Dict[str, Any]:
    """Magnitude veto. gap = |score_HV(b_HV) - score_HV(comparator)| in the 5.2.1 z unit, so
    the bar is C_THRESHOLD itself. The comparator is cA's AUC winner, except in Criterion 0
    state 'one', where auc_top10 has no stable global winner and it is the per-oracle AUC
    argmax: the mean over oracles (with a unique AUC winner) of the HV z of that winner."""
    hv_m, auc_m = PRIMARY_PAIR[1], PRIMARY_PAIR[0]
    b_hv = cA.get("winner_hv")
    gap = 0.0
    if b_hv is not None:
        hv_score = metric_score(df, hv_m)
        if c0["state"] == "one":
            z_hv, z_auc = z_scores(df, hv_m), z_scores(df, auc_m)
            picks = []
            for o, zs in z_auc.items():
                top = max(zs.values())
                best = [b for b, v in zs.items() if abs(v - top) <= _TIE_TOL]
                if len(best) == 1 and o in z_hv and best[0] in z_hv[o]:
                    picks.append(z_hv[o][best[0]])
            if picks:
                gap = abs(hv_score[b_hv] - float(np.mean(picks)))
        elif cA.get("winner_auc") is not None:
            gap = abs(hv_score[b_hv] - hv_score[cA["winner_auc"]])
    return {"gap": float(gap), "threshold": C_THRESHOLD, "passes": bool(gap > C_THRESHOLD)}


def criterion_G2(df, n_boot: int = N_BOOT, seed: int = 0) -> Dict[str, Any]:
    """Mean scalarization_loss across runs with a hierarchical bootstrap 95% CI. Descriptive:
    it enters no rule, and a measurement cannot come back empty, so no data raises."""
    oracles, bbs, vals = _cells(df, "scalarization_loss")
    if not oracles:
        raise ValueError("scalarization_loss has no finite value in any ok run")
    n_o = len(oracles)
    mean, _, _, cnt = _point_arrays(oracles, bbs, vals)
    point = float(np.nansum(mean * cnt) / cnt.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_o, size=(n_boot, n_o))
    bm, _, _, bc = _boot_cells(n_o, len(bbs), [vals], idx, rng)[0]
    draws = np.nansum(np.where(bc > 0, bm, 0.0) * bc, axis=(1, 2)) / bc.sum(axis=(1, 2))
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    return {"mean": point, "lo": lo, "hi": hi}


# ------------------------------------------------------------------------------ verdict
VERDICTS: Dict[str, str] = {
    "METRIC-DEPENDENT": "The conclusion depends on the metric. The benchmark, not the method, has been doing the deciding. Write the evaluation-framework contribution.",
    "METRIC-RESOLVING": "One metric resolves the methods and the other cannot. The benchmark's choice of metric decides whether a difference is visible at all.",
    "METRIC-ROBUST": "A winner exists and survives the change of metric. PMO's AUC-top-10 is adequate for these methods, and Phase 1's null stands as a clean refutation rather than a measurement artifact.",
    "COSMETIC": "The winner moves, by less than seed noise. Report as a caution about metric choice, not as a finding.",
    "BENCHMARK-INSENSITIVE": "No metric distinguishes these three backbones at 1000 calls. Phase 1's null is then partly a measurement-ceiling result, and the contribution is about the benchmark's resolution.",
}


def phase2_verdict(c0: Dict, cA: Dict, cB: Dict, cC: Dict, cG2: Dict) -> Dict[str, Any]:
    """Spec 6. Criterion 0 is read first: with neither metric stable A and C cannot pass, and
    with exactly one stable A has no pair of argmaxes to compare so C alone decides. Only when
    both are stable does A apply, and A failing is METRIC-ROBUST whatever C says."""
    state = c0["state"]
    if state == "neither":
        name = "BENCHMARK-INSENSITIVE"
    elif state == "one":
        name = "METRIC-RESOLVING" if cC["passes"] else "COSMETIC"
    elif not cA["passes"]:
        name = "METRIC-ROBUST"
    else:
        name = "METRIC-DEPENDENT" if cC["passes"] else "COSMETIC"
    return {"verdict": name, "consequence": VERDICTS[name], "metrics_sha256": metrics_hash(),
            "c0_state": state, "c0_n_stable": c0.get("n_stable"),
            "a_passes": bool(cA["passes"]), "c_passes": bool(cC["passes"]),
            "b_n_significant": sum(bool(d.get("significant")) for d in cB["per_metric"].values()),
            "b_degenerate": list(cB["degenerate_metrics"]), "g2_mean": cG2["mean"]}


# ---------------------------------------------------------------------------------------------
# Reproduction gate (spec 5.1). Pure logic: the rdkit/TDC work lives in phase2_components.py.
# ---------------------------------------------------------------------------------------------
GATE_MIN_RANGE: float = 0.5      # every component must span at least this much across the probe
GATE_MIN_NONZERO: float = 0.20   # at least this fraction of probe molecules must score non-zero


def gate_decision(ours: np.ndarray, theirs: np.ndarray, component_values: np.ndarray,
                  tol: float = GATE_TOL) -> Dict[str, Any]:
    """Decide whether our scalarisation reproduces the benchmark oracle AND whether the probe
    set was strong enough for that to mean anything.

    ours, theirs: (n,) scalar scores from our components and from the TDC oracle.
    component_values: (n, k) our component vectors on the same molecules.

    A geometric mean is zero whenever any component is zero, so a probe set of poor molecules
    agrees 0-vs-0 and proves nothing. "Non-zero" therefore means above `tol`: a score below the
    comparison tolerance is indistinguishable from zero by this very gate.

    Status precedence: a deviation is reported as "deviates" even on a weak probe (the
    decomposition is wrong regardless); only an agreeing-but-unexercised probe is "probe_too_weak".
    """
    ours = np.asarray(ours, dtype=float)
    theirs = np.asarray(theirs, dtype=float)
    comp = np.asarray(component_values, dtype=float)
    if ours.ndim != 1 or ours.shape != theirs.shape or comp.ndim != 2 or comp.shape[0] != ours.shape[0]:
        raise ValueError(f"shape mismatch: ours {ours.shape}, theirs {theirs.shape}, components {comp.shape}")
    if ours.size == 0:
        raise ValueError("empty probe set")
    _check_no_nan(comp)

    diff = np.abs(ours - theirs)
    # NaN on either side is a failure to reproduce, never an agreement.
    max_dev = float("inf") if np.isnan(diff).any() else float(diff.max())
    reproduces = bool(max_dev <= tol)

    ranges = comp.max(axis=0) - comp.min(axis=0)
    min_range = float(ranges.min())
    frac_nonzero = float(np.mean(np.nan_to_num(ours, nan=0.0) > tol))
    exercises = bool(min_range >= GATE_MIN_RANGE and frac_nonzero >= GATE_MIN_NONZERO)

    status = "deviates" if not reproduces else ("ok" if exercises else "probe_too_weak")
    return {"max_deviation": max_dev, "reproduces": reproduces,
            "component_ranges": [float(r) for r in ranges], "min_range": min_range,
            "frac_nonzero": frac_nonzero, "probe_exercises": exercises, "status": status}


def gate_summary(per_oracle: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-oracle gate results. Only status "ok" survives; every other oracle is
    excluded by name with its reason recorded. `enough` is the launch condition."""
    surviving, excluded = [], {}
    for name in sorted(per_oracle):
        g = per_oracle[name]
        st = g.get("status")
        if st == "ok":
            surviving.append(name)
        elif st == "deviates":
            excluded[name] = f"deviates from the benchmark oracle: max |diff| = {g.get('max_deviation', float('nan')):.6g}"
        elif st == "probe_too_weak":
            excluded[name] = (f"probe set too weak to verify (min component range {g.get('min_range', float('nan')):.3g}, "
                              f"non-zero fraction {g.get('frac_nonzero', float('nan')):.3g})")
        else:
            excluded[name] = f"unrecognised gate status {st!r}"
    return {"surviving": surviving, "excluded": excluded, "enough": len(surviving) >= MIN_ORACLES}
