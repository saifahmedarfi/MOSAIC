"""Phase 1 statistics -- pure numpy/scipy/pandas.

Deliberately imports NOTHING heavy: no torch, no rdkit, no TDC, and no phase0_lib.
That is what lets every statistical claim in Phase 1 be tested on any machine, with no
GPU and no oracle downloads:

    python tests/test_phase1.py

Phase 0's verdict was produced under src/phase0_lib.py, so that file is never modified.
Phase 1 reuses it at sweep time and does its own analysis here.
"""
from __future__ import annotations

import hashlib
import json
from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SETTINGS = ("explore_low", "explore_mid", "explore_high")
SETTING_INDEX = {s: i for i, s in enumerate(SETTINGS)}

# Quantile breakpoints. Phase 0 used ABSOLUTE thresholds (0.5/0.7/0.9); no oracle in the
# suite ever exceeded 0.5, so frac_above_0.7 and frac_above_0.9 were constant, returned
# rho = NaN, and still consumed Holm budget. Quantiles cannot go degenerate that way.
QUANTILES = (0.50, 0.75, 0.90, 0.99)

FEATURES: Tuple[str, ...] = (
    "reward_mean",
    "reward_std",
    "reward_skew",
    "reward_kurtosis",
    "gini",
    "dynamic_range",
    "q50_gap",
    "q75_gap",
    "q90_gap",
    "q99_gap",
    "ruggedness_corr",
    "ruggedness_absdelta",
    "basin_width",
    "neighbor_gain",
)

PRIMARY_FEATURE = "gini"
PRIMARY_DIRECTION = "negative"      # Phase 0: rho = -0.655, mechanistically sensible
PRIMARY_ALPHA = 0.05
SECONDARY_Q = 0.05


# --------------------------------------------------------------------------- hashing
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def features_hash(features: Sequence[str] = FEATURES) -> str:
    """sha256 of the COMPUTED feature tuple.

    Phase 0's prereg claimed the feature list was hashed, but hashed a separate copy of it
    inside the prereg dict, so the two could diverge without tripping the guard. Here the
    hash is taken from the tuple the code actually uses.
    """
    return hashlib.sha256(canonical_json(list(features)).encode()).hexdigest()


# --------------------------------------------------------------- distribution features
def gini(x: np.ndarray) -> float:
    """Reward concentration. 0 = flat, ->1 = one spike carries everything."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = np.sort(np.clip(x, 0.0, None))
    total = x.sum()
    if total <= 0:
        return 1.0                       # all-zero reward IS maximally concentrated
    n = x.size
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


def quantile_features(rewards: Sequence[float]) -> Dict[str, float]:
    """Reward-distribution shape. Every value finite for any non-empty input."""
    r = np.asarray(list(rewards), dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {k: float("nan") for k in
                ("reward_mean", "reward_std", "reward_skew", "reward_kurtosis",
                 "gini", "dynamic_range") + tuple(f"q{int(q*100)}_gap" for q in QUANTILES)}

    mean, std = float(r.mean()), float(r.std())
    out = {
        "reward_mean": mean,
        "reward_std": std,
        "gini": gini(r),
        "dynamic_range": float(r.max() - r.min()),
    }
    # Central moments, defined as 0 for a constant vector rather than NaN. The tolerance
    # is relative: np.std of 50 identical 0.3s is ~1e-17, not 0, and dividing a tiny
    # deviation by a tiny std yields garbage moments instead of the correct zero.
    if std > 1e-12 * max(1.0, abs(mean)):
        z = (r - mean) / std
        out["reward_skew"] = float((z ** 3).mean())
        out["reward_kurtosis"] = float((z ** 4).mean() - 3.0)
    else:
        out["reward_skew"] = 0.0
        out["reward_kurtosis"] = 0.0

    # Quantile GAPS, not fractions above an absolute cut: the distance from the quantile
    # to the max, normalised by the range. Scale-free and never constant unless the whole
    # reward vector is.
    rng = out["dynamic_range"]
    for q in QUANTILES:
        qv = float(np.quantile(r, q))
        out[f"q{int(q*100)}_gap"] = float((r.max() - qv) / rng) if rng > 0 else 0.0
    return out


# ------------------------------------------------------------------- aggregation helpers
def _cell(df, backbone: str, group: str, setting: str, by: str = "cluster") -> float:
    sel = df[(df["backbone"] == backbone) & (df[by] == group) & (df["setting"] == setting)]
    return float(sel["auc_top10"].mean()) if len(sel) else float("nan")


def winning_setting(df, backbone: str, by: str = "cluster") -> Dict[str, str]:
    """argmax setting per group. An EXACT tie returns None for that group.

    Phase 0 took argmax unconditionally, which silently picks the first of a tie and can
    manufacture a flip out of noise.
    """
    out: Dict[str, Optional[str]] = {}
    for g in sorted(df[by].dropna().unique()):
        means = [(_cell(df, backbone, g, s, by), s) for s in SETTINGS]
        means = [(m, s) for m, s in means if np.isfinite(m)]
        if not means:
            out[g] = None
            continue
        best = max(m for m, _ in means)
        tied = [s for m, s in means if m == best]
        out[g] = tied[0] if len(tied) == 1 else None
    return out


# ----------------------------------------------------------- Criterion A: the flip test
REPLICATION_RATE = 0.95


def flip_pairs(df, backbone: str, n_boot: int = 10000, by: str = "cluster",
               seed: int = 0, replication_rate: float = REPLICATION_RATE
               ) -> List[Dict[str, Any]]:
    """Hierarchical bootstrap of the interaction contrast, plus a replication requirement.

        D = [AUC(c1,e1) - AUC(c1,e2)] + [AUC(c2,e2) - AUC(c2,e1)]

    Oracles are resampled within each cluster with replacement, then seeds within each
    resampled oracle -- the hierarchical scheme Phase 0 froze, because the claim is about
    landscape CLUSTERS and must generalise to other oracles of the same type.

    Phase 1 adds a second requirement, and the reason matters. Phase 0 chose each cluster's
    winning setting by argmax on the full data and then bootstrapped D with that choice
    held fixed. D is then positive almost by construction, because e1 is *defined* as the
    setting that happened to win in c1. Measured here on pure noise, that procedure
    confirms roughly 4 flips in 40 trials. So each bootstrap draw re-selects the winners
    from the resampled data, and a pair is confirmed only if:

      1. the 95% percentile CI of D excludes 0  (Phase 0's test, reported for comparability)
      2. the flip PATTERN -- e1 best in c1 and e2 best in c2 -- replicates in at least
         `replication_rate` of draws

    A cluster holding fewer than 2 oracles gets status "insufficient" and is never
    confirmed: with one oracle the oracle-level resample is a no-op and the test silently
    degrades to the seed-only bootstrap, which Phase 0 measured at 5/40 false positives.
    """
    rng = np.random.default_rng(seed)
    win = winning_setting(df, backbone, by)
    groups = [g for g, e in win.items() if e is not None]
    members = {g: sorted(df[df[by] == g]["oracle"].unique()) for g in win}

    # Precompute per-(oracle, setting) seed vectors once; the bootstrap is pure numpy.
    sel = df[df["backbone"] == backbone]
    vals: Dict[Tuple[str, str], np.ndarray] = {}
    for (o, st), grp in sel.groupby(["oracle", "setting"]):
        vals[(o, st)] = grp["auc_top10"].to_numpy(dtype=float)

    def draw_cluster_means(cluster: str) -> Optional[np.ndarray]:
        """One hierarchical resample -> mean AUC per setting for this cluster."""
        picks = rng.choice(members[cluster], size=len(members[cluster]), replace=True)
        out = np.empty(len(SETTINGS))
        for j, st in enumerate(SETTINGS):
            per = [v[rng.integers(0, v.size, v.size)].mean()
                   for o in picks if (v := vals.get((o, st))) is not None and v.size]
            if not per:
                return None
            out[j] = np.mean(per)
        return out

    res: List[Dict[str, Any]] = []
    for i, c1 in enumerate(groups):
        for c2 in groups[i + 1:]:
            e1, e2 = win[c1], win[c2]
            if e1 == e2:
                continue
            rec = {"c1": c1, "c2": c2, "e1": e1, "e2": e2,
                   "n_oracles_c1": len(members[c1]), "n_oracles_c2": len(members[c2])}
            D = ((_cell(df, backbone, c1, e1, by) - _cell(df, backbone, c1, e2, by)) +
                 (_cell(df, backbone, c2, e2, by) - _cell(df, backbone, c2, e1, by)))
            rec["D"] = float(D)

            if min(len(members[c1]), len(members[c2])) < 2:
                rec.update(lo=float("nan"), hi=float("nan"), replication=float("nan"),
                           ci_excludes_zero=False, confirmed=False, status="insufficient")
                res.append(rec)
                continue

            a, b = SETTING_INDEX[e1], SETTING_INDEX[e2]
            draws = np.full(n_boot, np.nan)
            replicated = 0
            for k in range(n_boot):
                m1, m2 = draw_cluster_means(c1), draw_cluster_means(c2)
                if m1 is None or m2 is None:
                    continue
                draws[k] = (m1[a] - m1[b]) + (m2[b] - m2[a])
                if int(np.argmax(m1)) == a and int(np.argmax(m2)) == b:
                    replicated += 1
            good = draws[np.isfinite(draws)]
            if good.size == 0:
                rec.update(lo=float("nan"), hi=float("nan"), replication=float("nan"),
                           ci_excludes_zero=False, confirmed=False, status="insufficient")
                res.append(rec)
                continue
            lo, hi = np.percentile(good, [2.5, 97.5])
            rep = replicated / good.size
            ci_ok = bool(lo > 0 or hi < 0)
            rec.update(lo=float(lo), hi=float(hi), replication=float(rep),
                       ci_excludes_zero=ci_ok,
                       confirmed=bool(ci_ok and rep >= replication_rate),
                       status="ok")
            res.append(rec)
    return res


def criterion_A(df, backbones: Sequence[str], min_pairs: int = 2, **kw) -> Dict[str, Any]:
    per = {bb: flip_pairs(df, bb, **kw) for bb in backbones}
    conf = {bb: sum(1 for p in ps if p["confirmed"]) for bb, ps in per.items()}
    best = max(conf, key=conf.get) if conf else None
    return {"per_backbone": per, "n_confirmed": conf, "best_backbone": best,
            "passes": bool(best is not None and conf[best] >= min_pairs),
            "min_pairs": min_pairs}


# ------------------------------------------------- Criterion B: diagnostics vs winner
def exact_spearman(x: Sequence[float], y: Sequence[float],
                   alternative: str = "two-sided",
                   max_exact_n: int = 8, n_mc: int = 100_000,
                   seed: int = 0) -> Tuple[float, float]:
    """Spearman rho with a permutation p-value.

    Full enumeration while n <= max_exact_n (8! = 40,320); Monte Carlo above that, because
    Phase 1's confirmation set has 15 oracles and 15! is not enumerable.
    """
    from scipy.stats import rankdata

    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    n = xa.size
    if n < 3 or np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return float("nan"), 1.0

    rx, ry = rankdata(xa), rankdata(ya)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    rho = float(rx.dot(ry) / denom)

    if n <= max_exact_n:
        stats = np.array([rx.dot(ry[list(p)]) for p in permutations(range(n))]) / denom
    else:
        rng = np.random.default_rng(seed)
        stats = np.array([rx.dot(rng.permutation(ry)) for _ in range(n_mc)]) / denom

    if alternative == "greater":
        p = float((stats >= rho - 1e-12).mean())
    elif alternative == "less":
        p = float((stats <= rho + 1e-12).mean())
    else:
        p = float((np.abs(stats) >= abs(rho) - 1e-12).mean())
    return rho, p


def bh_fdr(pvals: Dict[str, float], q: float = SECONDARY_Q) -> Dict[str, Dict[str, Any]]:
    """Benjamini-Hochberg. Degenerate (NaN) features must be dropped BEFORE calling.

    Phase 0 used Holm over 13 heavily-correlated concentration measures, which is
    needlessly conservative; BH controls the false-discovery rate instead.
    """
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict[str, Any]] = {}
    crit = 0
    for i, (k, p) in enumerate(items, start=1):
        if p <= i / m * q:
            crit = i
    for i, (k, p) in enumerate(items, start=1):
        out[k] = {"p": p, "threshold": i / m * q, "significant": i <= crit, "rank": i}
    for k, v in pvals.items():
        if k not in out:
            out[k] = {"p": v, "threshold": float("nan"),
                      "significant": False, "rank": None, "degenerate": True}
    return out


def criterion_B(df, diagnostics, backbone: str = "augmented_memory",
                primary: str = PRIMARY_FEATURE, direction: str = PRIMARY_DIRECTION,
                alpha: float = PRIMARY_ALPHA, q: float = SECONDARY_Q,
                features: Sequence[str] = FEATURES, seed: int = 0) -> Dict[str, Any]:
    """One pre-specified primary hypothesis, then BH-FDR over the rest.

    `backbone` is named explicitly. Phase 0's prereg said "the LM backbone", which is
    ambiguous between reinvent and augmented_memory; the implementation chose reinvent,
    which had zero confirmed flips, so the diagnostic was screened against the arm whose
    rankings were never shown to flip.
    """
    win = winning_setting(df, backbone, by="oracle")
    oracles = [o for o in diagnostics["oracle"] if win.get(o) is not None]
    y = np.array([SETTING_INDEX[win[o]] for o in oracles], dtype=float)
    idx = diagnostics.set_index("oracle")

    alt = {"negative": "less", "positive": "greater"}.get(direction, "two-sided")
    rows, pvals, degenerate = {}, {}, []
    for f in features:
        if f not in idx.columns:
            continue
        xv = idx.loc[oracles, f].to_numpy(dtype=float)
        if not np.isfinite(xv).any() or np.ptp(xv[np.isfinite(xv)]) == 0:
            degenerate.append(f)
            rows[f] = {"rho": float("nan"), "p": 1.0, "degenerate": True}
            continue
        a = alt if f == primary else "two-sided"
        rho, p = exact_spearman(xv, y, alternative=a, seed=seed)
        rows[f] = {"rho": rho, "p": p, "degenerate": False}
        if f != primary:
            pvals[f] = p

    pr = rows.get(primary, {"rho": float("nan"), "p": 1.0})
    signed_ok = (np.isfinite(pr["rho"]) and
                 (pr["rho"] < 0 if direction == "negative" else pr["rho"] > 0))
    primary_pass = bool(signed_ok and pr["p"] <= alpha)

    fdr = bh_fdr(pvals, q=q)
    for f, r in fdr.items():
        rows[f].update(fdr_threshold=r["threshold"], significant=r["significant"])
    rows[primary].update(significant=primary_pass, alpha=alpha, directional=direction)

    return {"n_oracles": len(oracles), "backbone": backbone, "primary": primary,
            "primary_rho": pr["rho"], "primary_p": pr["p"], "primary_pass": primary_pass,
            "secondary_significant": sorted(f for f, r in fdr.items() if r["significant"]),
            "degenerate_features": degenerate, "per_feature": rows,
            "passes": primary_pass}


# ------------------------------- Criterion C: is the flip worth the probe it costs?
def auc_at(score_trace: Sequence[Sequence[float]], n_calls: int,
           freq_log: int = 10, top_n: int = 10) -> float:
    """AUC-top-k truncated at n_calls, reproducing mol_opt's integral exactly.

    `score_trace` is a sequence of (score, call_index) pairs in call order -- what Phase 1
    runs persist. Phase 0's records do not contain it (only top10_smiles and a coarse
    history grid), so for those runs the probe cost falls back to the history grid and is
    flagged approximate.

    Note the ramp-in: the integral starts from prev = 0, so a constant-0.42 run scores
    0.4179. Absolute values are NOT mean rewards.
    """
    pairs = [(float(s), int(c)) for s, c in score_trace if int(c) <= n_calls]
    if not pairs:
        return 0.0
    scores = [s for s, _ in pairs]
    total, prev, called = 0.0, 0.0, 0
    for idx in range(freq_log, min(len(scores), n_calls), freq_log):
        now = float(np.mean(sorted(scores[:idx], reverse=True)[:top_n]))
        total += freq_log * (now + prev) / 2
        prev = now
        called = idx
    now = float(np.mean(sorted(scores, reverse=True)[:top_n]))
    total += (len(scores) - called) * (now + prev) / 2
    return total / n_calls


def _oracle_setting_seed(df, backbone: str):
    """{oracle: {setting: {seed: auc}}} for one backbone."""
    out: Dict[str, Dict[str, Dict[int, float]]] = {}
    sel = df[df["backbone"] == backbone]
    for _, r in sel.iterrows():
        out.setdefault(r["oracle"], {}).setdefault(r["setting"], {})[int(r["seed"])] = \
            float(r["auc_top10"])
    return out


def cv_headroom(df, backbone: str, n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """Held-out-seed headroom: what a PERFECT per-oracle chooser would actually gain.

    Phase 0's naive headroom takes the max of three noisy 3-seed means, which is a
    winner's curse: the selected maximum is biased upward. Here the setting is chosen on
    two seeds and scored on the third, rotated over all three folds, so the estimate is
    unbiased.

    An oracle with fewer than 3 seeds is skipped and reported, never fitted and evaluated
    on the same seed.
    """
    data = _oracle_setting_seed(df, backbone)
    per_oracle, skipped = {}, []
    for o, by_setting in data.items():
        seeds = sorted(set().union(*(set(v) for v in by_setting.values())))
        if len(seeds) < 3 or not all(len(by_setting.get(s, {})) for s in SETTINGS):
            skipped.append(o)
            continue
        gains = []
        for held in seeds:
            fit = [s for s in seeds if s != held]
            means = {st: np.mean([by_setting[st][s] for s in fit if s in by_setting[st]])
                     for st in SETTINGS if st in by_setting}
            pick = max(means, key=means.get)
            gains.append(by_setting[pick].get(held, np.nan))
        per_oracle[o] = float(np.nanmean(gains))

    if not per_oracle:
        return {"status": "insufficient", "skipped": skipped,
                "headroom": float("nan"), "lo": float("nan"), "hi": float("nan")}

    fixed = {st: np.mean([np.mean(list(data[o][st].values()))
                          for o in per_oracle if st in data[o]]) for st in SETTINGS}
    best_fixed_setting = max(fixed, key=fixed.get)
    oracles = sorted(per_oracle)
    cond = np.array([per_oracle[o] for o in oracles])
    base = np.array([np.mean(list(data[o][best_fixed_setting].values()))
                     for o in oracles])
    diff = cond - base

    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(diff, size=diff.size, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"status": "ok", "headroom": float(diff.mean()),
            "lo": float(lo), "hi": float(hi),
            "best_fixed_setting": best_fixed_setting,
            "best_fixed_auc": float(base.mean()),
            "conditional_auc": float(cond.mean()),
            "n_oracles": len(oracles), "skipped": skipped,
            "per_oracle_gain": {o: float(g) for o, g in zip(oracles, diff)}}


def probe_cost(df, backbone: str, probe_calls: int = 300,
               budget: int = 1000) -> Dict[str, Any]:
    """What the diagnostic probe costs, in the metric's own units.

    The controller must spend `probe_calls` of its budget measuring the landscape, so its
    optimisation run is (budget - probe_calls) long. Cost = AUC@budget - AUC@(budget-probe).
    """
    sel = df[df["backbone"] == backbone]
    traces = [r for r in sel.to_dict("records") if r.get("score_trace")]
    if not traces:
        return {"status": "unavailable", "approximate": True, "cost": float("nan"),
                "reason": "no run record carries score_trace (Phase 0 records never do)"}
    full = np.array([auc_at(r["score_trace"], budget) for r in traces])
    short = np.array([auc_at(r["score_trace"], budget - probe_calls) for r in traces])
    return {"status": "ok", "approximate": False,
            "cost": float((full - short).mean()),
            "auc_full": float(full.mean()), "auc_short": float(short.mean()),
            "probe_calls": probe_calls, "n_runs": len(traces)}


def criterion_C(df, backbone: str, probe_calls: int = 300, budget: int = 1000,
                n_boot: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """Does landscape-conditional choice gain more than the probe costs?

    This is the question Phase 0 never asked. Criterion A tests whether a flip EXISTS;
    this tests whether exploiting it pays for the information it needs. If the CI of
    (headroom - probe_cost) does not exclude 0, the controller is not viable at this
    budget and that is the phase's headline result whatever A, B and D say.
    """
    H = cv_headroom(df, backbone, n_boot=n_boot, seed=seed)
    P = probe_cost(df, backbone, probe_calls=probe_calls, budget=budget)
    if H["status"] != "ok" or P["status"] != "ok":
        return {"status": "insufficient", "headroom": H, "probe": P,
                "viable": False, "net": float("nan")}
    net = H["headroom"] - P["cost"]
    return {"status": "ok", "headroom": H, "probe": P, "net": float(net),
            "net_lo": float(H["lo"] - P["cost"]), "net_hi": float(H["hi"] - P["cost"]),
            "viable": bool(H["lo"] - P["cost"] > 0)}


# ------------------------------------- Criterion D: selection across METHODS, not knobs
def criterion_D(df, by: str = "cluster", n_boot: int = 10000,
                seed: int = 0) -> Dict[str, Any]:
    """Best (backbone, setting) pair per landscape group.

    Phase 0's method-level margins were an order of magnitude larger than its within-method
    knob effects (graph_ga had the highest overall mean AUC and no landscape dependence at
    all), so the action space that matters is the pair, not the knob.
    """
    d = df.copy()
    d["arm"] = d["backbone"] + ":" + d["setting"]
    winners, per_group = {}, {}
    for g in sorted(d[by].dropna().unique()):
        sub = d[d[by] == g]
        means = sub.groupby("arm")["auc_top10"].mean().sort_values(ascending=False)
        per_group[g] = {k: float(v) for k, v in means.items()}
        if len(means) and (len(means) == 1 or means.iloc[0] != means.iloc[1]):
            winners[g] = means.index[0]
        else:
            winners[g] = None
    distinct = sorted({w for w in winners.values() if w})
    return {"winners": winners, "per_group": per_group,
            "n_distinct_winning_arms": len(distinct), "distinct_arms": distinct,
            "flips": bool(len(distinct) >= 2)}


# --------------------------------------------------------------------- the verdict
VERDICTS = {
    "CONTROLLER-VIABLE":
        "The flip generalises, a pre-specified diagnostic tracks it, and the gain exceeds "
        "the probe's cost. Phase 2 (the controller) is unblocked.",
    "CHARACTERISATION-ONLY":
        "The landscape dependence is real but either no diagnostic predicts it or the gain "
        "does not pay for the probe. Write the benchmark/characterisation contribution; do "
        "NOT build the controller.",
    "NULL":
        "The flip does not generalise beyond Phase 0's isomer-driven pairs. The premise is "
        "dead at this budget; write up the null.",
}


def phase1_verdict(A: Dict[str, Any], B: Dict[str, Any], C: Dict[str, Any],
                   D: Dict[str, Any]) -> Dict[str, Any]:
    if not A.get("passes"):
        v = "NULL"
    elif B.get("passes") and C.get("viable"):
        v = "CONTROLLER-VIABLE"
    else:
        v = "CHARACTERISATION-ONLY"
    return {"verdict": v, "consequence": VERDICTS[v],
            "criterion_A_passes": bool(A.get("passes")),
            "criterion_B_passes": bool(B.get("passes")),
            "criterion_C_viable": bool(C.get("viable")),
            "criterion_D_flips": bool(D.get("flips")),
            "features_sha256": features_hash()}
