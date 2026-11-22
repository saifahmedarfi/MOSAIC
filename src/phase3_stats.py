"""Phase 3 statistics -- the three criteria and the verdict table (spec 6.0-6.5).

Pure numpy/pandas, and it imports `phase2_stats` and nothing heavy: no torch, no rdkit, no
TDC, no phase0_lib, no phase2_source. That is what lets the whole Phase 3 decision rule be
tested on any machine:

    python -u tests/test_phase3.py

Everything arithmetical is Phase 2's. The common unit (spec 6.0) is
`z(m,o,a) = [X(m,o,a) - mean_a X(m,o,.)] / SD_pool(m,o)` and `score(m,a) = mean_o z`, computed
by `phase2_stats._z_and_score`; the shared-oracle hierarchical bootstrap is
`phase2_stats._boot_cells`. Neither is re-derived here. Phase 2 killed its own duplicate-z bug
by collapsing every path onto a single core, and a second copy of that arithmetic in Phase 3
would be the same defect reintroduced. `phase2_stats` is read-only from here: never edited.

Three things this module exists to get right, each of which has a test that fails when it
breaks:

1. **Criteria P and Q put the margin INSIDE the quantile.** >=95% of bootstrap draws must
   satisfy `diff >= delta_P` (P) and `diff <= eps_Q` (Q). There is no separate condition on
   the point estimate and there is no sign test: a gain that is positive in 99% of draws but
   smaller than `delta_P` fails.
2. **Thresholds are already in pooled-seed-SD z units** and are never multiplied by an SD
   again (spec 6.0). Criterion F's bound is therefore a flat 0.25.
3. **Criterion F is an equivalence test in the TOST sense**: the 95% CI of
   `score(anchor) - score(control)` must lie entirely inside `[-bound, +bound]`, for BOTH
   members of the primary pair. More seeds make it EASIER to pass, which is the right
   direction; the earlier draft's "and the CI contains 0" clause was removed by spec 6.3
   because it accepts a point null and gets harder with more seeds.

The axis column is `arm` ("anchor" / "control" / "treatment"), never `backbone`: all three
arms ARE graph_ga, so calling that axis `backbone` would be false in the manifest and in the
analysis frame. `phase2_stats._cells` hardcodes `groupby("backbone")` (phase2_stats.py:207),
so `_by_arm` below renames for delegation only and the rename never leaves this module.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import phase2_stats as PS

# Spec 4.3, enumerated because its sha256 goes in the pre-registration. `front_coverage` is
# deliberately ABSENT: it is front_size / n, a deterministic rescaling, and in Phase 2
# (defect 6) it consumed its own FDR budget while carrying no information of its own.
# `auc_top10` carries Criterion Q, `hypervolume` carries Criterion P, `front_size` is
# descriptive and enters no rule.
METRICS: Tuple[str, ...] = ("auc_top10", "hypervolume", "front_size")

# Spec 4.2. anchor = untouched phase0_lib.run_ga_optimizer, control = the reimplemented loop
# with scalar selection, treatment = the reimplemented loop with NSGA-II selection.
ARMS: Tuple[str, str, str] = ("anchor", "control", "treatment")

P_METRIC: str = "hypervolume"                           # spec 6.1
Q_METRIC: str = "auc_top10"                             # spec 6.2
F_METRICS: Tuple[str, str] = ("auc_top10", "hypervolume")   # spec 6.3, both of them

F_BOUND: float = 0.25           # spec 6.3; may be tightened before freezing, never loosened
REPLICATION_RATE: float = PS.REPLICATION_RATE           # 0.95, spec 6.1 / 6.2
N_BOOT: int = PS.N_BOOT                                 # 10_000, spec 6.0
MIN_ORACLES: int = PS.MIN_ORACLES                       # 5; <5 usable oracles is REFUSED

# |Pearson r| at or above this is a deterministic relation for the purpose of spec 4.3.
COLLINEAR_R: float = 0.999


def metrics_hash3(metrics: Sequence[str] = METRICS) -> str:
    """sha256 of Phase 3's metric tuple, through `phase2_stats`' own canonical JSON hashing."""
    return PS.metrics_hash(metrics)


# ------------------------------------------------------------------- the arm/backbone adapter
def _by_arm(df):
    """Rename `arm` -> `backbone` for delegation to `phase2_stats`, and nowhere else.

    `phase2_stats._cells` hardcodes `groupby("backbone")` (phase2_stats.py:207). Phase 3's
    public frame, its manifest and its figures name that axis `arm`, because all three arms
    are the same backbone and calling them backbones would be a false label. This adapter is
    the whole of the reconciliation: it returns a renamed COPY, so the caller's frame is
    untouched, and a frame that does not carry `arm` is refused rather than silently
    mis-analysed (a frame keyed on `backbone` is a Phase 2 frame).
    """
    if "arm" not in getattr(df, "columns", ()):
        raise ValueError("a Phase 3 frame must carry an `arm` column, not `backbone`")
    return df.assign(backbone=df["arm"].to_numpy()).drop(columns=["arm"])


# ----------------------------------------------------------------------- collinearity (4.3)
def check_collinear(df, metrics: Sequence[str] = METRICS,
                    r: float = COLLINEAR_R) -> List[Tuple[str, str]]:
    """Pairs among `metrics` whose |Pearson r| over the ok runs reaches `r`.

    Spec 4.3: run this BEFORE the metric list is hashed, so what gets frozen is known to be
    collinearity-free. Phase 2's defect 6 is the precedent -- `front_coverage` was
    `front_size` / n, so it could not disagree with it, and it still spent FDR budget.

    A metric with no variance (below `phase2_stats._REL_TOL` of its own magnitude, the
    relative tolerance that killed Phase 1's divide-noise-by-noise bug) has no correlation to
    measure: 0/0. It is not flagged, because a constant column is a degenerate measurement,
    not evidence of a deterministic relation to some other column.
    """
    d = df[df["status"] == "ok"] if "status" in getattr(df, "columns", ()) else df
    names = list(metrics)
    out: List[Tuple[str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = d[names[i]].to_numpy(dtype=float)
            b = d[names[j]].to_numpy(dtype=float)
            keep = np.isfinite(a) & np.isfinite(b)
            a, b = a[keep], b[keep]
            if len(a) < 3:
                continue
            if any(x.std() <= PS._REL_TOL * max(1.0, abs(float(x.mean()))) for x in (a, b)):
                continue
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                cc = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(cc) and abs(cc) >= r:
                out.append((names[i], names[j]))
    return out


# --------------------------------------------------------- the shared-oracle arm contrast
def arm_difference(df, metric: str, arm_a: str, arm_b: str, n_boot: int = N_BOOT,
                   seed: Any = 0, margin: float = 0.0) -> Dict[str, Any]:
    """`score(metric, arm_a) - score(metric, arm_b)` with a shared-oracle hierarchical
    bootstrap (spec 6.0). The arm axis replaces Phase 2's backbone axis; nothing else differs.

    Each draw resamples the oracle list ONCE with replacement and uses that same list for
    every arm; within each selected (oracle, arm) cell the seeds are then resampled with
    replacement, independently per cell. The per-oracle centring in `_z_and_score` therefore
    compares the three arms on the SAME oracle in every draw, so the across-oracle main effect
    cancels exactly as it does in the point estimate. Resampling oracles independently per arm
    would leave that main effect inside z and make a null true by construction;
    `phase1_stats.flip_pairs` is the wrong precedent and `tests/test_phase3.py`'s
    `_independent_oracle_draws` shows what it costs.

    `margin` is the ONLY hook the criteria need: it is the value the draws are compared
    against, so the margin sits inside the quantile rather than beside it. It defaults to 0.0,
    where `draws_ge` / `draws_le` are plain sign fractions and descriptive only -- Criterion P
    is not a sign test (spec 6.1).

    Returns `diff`, `lo`/`hi` (the 2.5 / 97.5 percentiles of the bootstrap difference),
    `draws_ge` = fraction of draws with `diff >= margin`, `draws_le` = fraction with
    `diff <= margin`, and `n_oracles` = oracles with usable data. Everything is NaN when an
    arm has no usable cell: an absent measurement is never reported as agreement.
    """
    oracles, arms, vals = PS._cells(_by_arm(df), metric)
    out: Dict[str, Any] = {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                           "draws_ge": float("nan"), "draws_le": float("nan"),
                           "n_oracles": len(oracles), "n_draws": 0,
                           "metric": metric, "arms": (arm_a, arm_b), "margin": float(margin)}
    if not oracles or arm_a not in arms or arm_b not in arms:
        return out

    ia, ib = arms.index(arm_a), arms.index(arm_b)
    n_o, n_a = len(oracles), len(arms)
    _, point = PS._z_and_score(*PS._point_arrays(oracles, arms, vals))
    out["diff"] = float(point[ia] - point[ib])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_o, size=(n_boot, n_o))      # ONE oracle list per draw, all arms
    _, score = PS._z_and_score(*PS._boot_cells(n_o, n_a, [vals], idx, rng)[0])   # (n_boot, n_a)
    draws = score[..., ia] - score[..., ib]
    draws = draws[np.isfinite(draws)]       # a draw that lost an arm entirely tests nothing
    if len(draws) == 0:
        return out
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    out.update({"lo": lo, "hi": hi, "n_draws": int(len(draws)),
                "draws_ge": float((draws >= margin).mean()),
                "draws_le": float((draws <= margin).mean())})
    return out


# ------------------------------------------------------------------- the criteria (6.1-6.3)
def criterion_P(df, delta_p: float, n_boot: int = N_BOOT, seed: Any = 0) -> Dict[str, Any]:
    """Spec 6.1, primary: >= REPLICATION_RATE of draws satisfy
    `score_hv(treatment) - score_hv(control) >= delta_p`.

    The margin is INSIDE the quantile. There is no separate condition on the point estimate
    and no sign test: a gain positive in 99% of draws but smaller than `delta_p` fails, and
    `delta_p` is in the spec 6.0 z unit, so it is used flat and never scaled by an SD.
    `delta_p` is Stage A's, frozen in the pre-registration, and is never adjusted here.
    """
    r = arm_difference(df, P_METRIC, "treatment", "control", n_boot=n_boot, seed=seed,
                       margin=delta_p)
    rate = r["draws_ge"]
    return {**r, "rate": rate, "threshold": REPLICATION_RATE,
            "passes": bool(np.isfinite(rate) and rate >= REPLICATION_RATE)}


def criterion_Q(df, eps_q: float, n_boot: int = N_BOOT, seed: Any = 0) -> Dict[str, Any]:
    """Spec 6.2, co-primary: >= REPLICATION_RATE of draws satisfy
    `score_auc(control) - score_auc(treatment) <= eps_q`.

    Non-inferiority, not superiority, and co-primary rather than secondary: it is the
    anti-circularity guard (anti-goal A1). An optimizer that wins on the metric it optimizes
    has shown nothing; the defensible claim is front quality bought without surrendering the
    scalarized performance the field reports. Same semantics as P, margin inside the quantile.
    """
    r = arm_difference(df, Q_METRIC, "control", "treatment", n_boot=n_boot, seed=seed,
                       margin=eps_q)
    rate = r["draws_le"]
    return {**r, "rate": rate, "threshold": REPLICATION_RATE,
            "passes": bool(np.isfinite(rate) and rate >= REPLICATION_RATE)}


def criterion_F(df, bound: float = F_BOUND, n_boot: int = N_BOOT,
                seed: Any = 0) -> Dict[str, Any]:
    """Spec 6.3, the halting fidelity gate: the 95% bootstrap CI of
    `score_metric(anchor) - score_metric(control)` lies entirely within `[-bound, +bound]`,
    for BOTH members of the primary pair.

    This is an equivalence test in the TOST sense. More seeds narrow the CI and so make it
    EASIER to pass, which is the right direction for a gate that must not become more likely
    to halt the phase exactly as the pilot buys margin. There is deliberately NO clause
    requiring the CI to contain 0: that accepts a point null and gets HARDER with more seeds
    (spec 6.3, clause removed after the builder check).

    Both metrics must be inside -- `all`, not `any`. A metric with no usable cell cannot be
    shown equivalent to anything, so it is `inside: False` and the gate fails closed; a
    fidelity gate that passed for want of data would be the opposite of a gate.

    `width` is the widest CI across the metrics, i.e. the binding one. A failure yields
    REFUSED whatever P and Q say, and the 0.25 bound is checked against Stage A's measured
    seed SD before freezing: it may be tightened, never loosened.
    """
    per: Dict[str, Dict[str, Any]] = {}
    for k, m in enumerate(F_METRICS):
        r = arm_difference(df, m, "anchor", "control", n_boot=n_boot, seed=[seed, k])
        lo, hi = r["lo"], r["hi"]
        inside = bool(np.isfinite(lo) and np.isfinite(hi) and lo >= -bound and hi <= bound)
        per[m] = {**r, "width": float(hi - lo), "inside": inside}
    widths = [per[m]["width"] for m in F_METRICS if np.isfinite(per[m]["width"])]
    return {"per_metric": per, "bound": float(bound),
            "width": max(widths) if widths else float("nan"),
            "n_oracles": max((per[m]["n_oracles"] for m in F_METRICS), default=0),
            "passes": bool(all(per[m]["inside"] for m in F_METRICS))}


# ------------------------------------------------------------------------- verdict (6.5)
VERDICTS3: Dict[str, str] = {
    "PARETO-RECOVERS": "Front quality is buyable at no scalarized cost. The parent proposal's Optimization Layer is justified empirically.",
    "PARETO-TRADES": "Front quality is purchasable only against scalarized performance. Report the exchange rate -- a useful result, not a win.",
    "PARETO-NULL": "No recoverable headroom at 1000 calls within five selection events (spec 3.6). Phase 2's hypervolume spread is then a property of the backbones, not of what selection can reach.",
    "REFUSED": "The mechanism working, not a bug to fix: a prereg or metric-list hash mismatch, a failed Criterion F, or fewer than five usable oracles.",
}


def phase3_verdict(cP: Dict, cQ: Dict, cF: Dict) -> Dict[str, Any]:
    """Spec 6.5. Criterion F is read FIRST: a reimplementation that does not reproduce
    vendored graph_ga distributionally makes every arm contrast meaningless, so a failed
    fidelity gate is REFUSED whatever P and Q say. P failing is PARETO-NULL whatever Q says;
    only with P passing does Q choose between RECOVERS and TRADES.
    """
    if not cF["passes"]:
        name = "REFUSED"
    elif not cP["passes"]:
        name = "PARETO-NULL"
    else:
        name = "PARETO-RECOVERS" if cQ["passes"] else "PARETO-TRADES"
    return {"verdict": name, "consequence": VERDICTS3[name],
            "metrics_sha256": metrics_hash3(),
            "f_passes": bool(cF["passes"]), "p_passes": bool(cP["passes"]),
            "q_passes": bool(cQ["passes"])}
