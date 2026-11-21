#!/usr/bin/env python
"""Phase 2 statistics gate.

    python tests/test_phase2.py

The statistics tests need numpy/scipy/pandas alone -- no torch, no TDC, no GPU, no oracle
downloads. The component tests additionally need rdkit (available on research-base); the
reproduction gate against the real TDC oracle can only run where TDC is installed.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import phase2_stats as P  # noqa: E402

N = 0


def check(cond, label):
    global N
    N += 1
    if not cond:
        raise AssertionError(f"FAILED: {label}")
    print(f"  ok  {label}")


def synth(means, oracles=("o1", "o2"), seeds=(0, 1, 2), noise=0.0, metric="auc_top10",
          setting="explore_low", rng=None):
    """Synthetic run frame; `means[backbone]` is that backbone's true cell mean."""
    rng = rng or np.random.default_rng(0)
    rows = []
    for o in oracles:
        for bb, mu in means.items():
            for s in seeds:
                row = {"oracle": o, "backbone": bb, "setting": setting, "seed": s,
                       "status": "ok"}
                for m in P.METRICS:
                    row[m] = np.nan
                row[metric] = mu + noise * rng.standard_normal()
                rows.append(row)
    return pd.DataFrame(rows)


def test_hypervolume():
    print("\n[1] hypervolume")
    check(abs(P.hypervolume(np.array([[1.0, 1.0]])) - 1.0) < 1e-12, "HV of (1,1) is 1.0")
    check(abs(P.hypervolume(np.array([[0.5, 0.5]])) - 0.25) < 1e-12, "HV of (0.5,0.5) is 0.25")
    hv = P.hypervolume(np.array([[0.8, 0.2], [0.2, 0.8]]))
    check(abs(hv - (0.16 + 0.16 - 0.04)) < 1e-12, "HV of two trade-off points is the union")
    check(abs(P.hypervolume(np.array([[1.0, 1.0], [0.3, 0.3]])) - 1.0) < 1e-12,
          "a dominated point adds no hypervolume")
    check(P.hypervolume(np.zeros((5, 3))) == 0.0, "all-zero population has HV exactly 0.0")
    check(P.hypervolume(np.zeros((0, 3))) == 0.0, "empty front has HV exactly 0.0")
    check(abs(P.hypervolume(np.array([[1.0, 1.0, 1.0, 1.0]])) - 1.0) < 1e-12, "HV works in 4-D")

    # 3-D union by inclusion-exclusion by hand: boxes 0.8*0.5*0.2, 0.2*0.5*0.8
    a, b = (0.8, 0.5, 0.2), (0.2, 0.5, 0.8)
    want = np.prod(a) + np.prod(b) - np.prod(np.minimum(a, b))
    check(abs(P.hypervolume(np.array([a, b])) - want) < 1e-12, "3-D union matches by-hand value")

    # Exactness against a brute-force grid on random 3-D and 4-D fronts: HV is a union of
    # boxes at the point coordinates, so the coordinate-grid cell count is exact.
    rng = np.random.default_rng(1)
    for d in (3, 4):
        pts = rng.random((12, d))
        axes = [np.unique(np.concatenate([[0.0], pts[:, k]])) for k in range(d)]
        grids = np.meshgrid(*[ax[1:] for ax in axes], indexing="ij")
        widths = np.meshgrid(*[np.diff(ax) for ax in axes], indexing="ij")
        covered = np.zeros(grids[0].shape, dtype=bool)
        for p in pts:
            covered |= np.logical_and.reduce([grids[k] <= p[k] + 1e-15 for k in range(d)])
        brute = float((np.prod(widths, axis=0) * covered).sum())
        check(abs(P.hypervolume(pts) - brute) < 1e-10, f"{d}-D HV matches brute-force grid")

    # a zero coordinate encloses no volume, even next to a real point
    check(abs(P.hypervolume(np.array([[0.0, 0.9], [0.5, 0.5]])) - 0.25) < 1e-12,
          "a point with a zero coordinate contributes nothing")
    # order of input must not matter
    pts = rng.random((30, 3))
    check(abs(P.hypervolume(pts) - P.hypervolume(pts[::-1])) < 1e-12,
          "HV is invariant to row order")


def test_scale_and_nan():
    import time
    print("\n[2b] large fronts and NaN")
    n = 1000
    t = (np.arange(n) + 0.5) / n
    # 2-D anti-chain has a closed form: sum_i (1/n) * (1 - i/n)
    anti2 = np.column_stack([(np.arange(n) + 1) / n, 1 - np.arange(n) / n])
    check(abs(P.hypervolume(anti2) - (n + 1) / (2 * n)) < 1e-12, "2-D anti-chain of 1000 matches closed form")
    # mutually non-dominated fronts on a sphere surface, the worst case for front size
    rng = np.random.default_rng(5)
    for d, limit in ((3, 5.0), (4, 30.0)):
        g = np.abs(rng.standard_normal((n, d)))
        pts = g / np.linalg.norm(g, axis=1, keepdims=True)
        check(len(P.pareto_front(pts)) == n, f"{d}-D sphere points are a {n}-point anti-chain")
        t0 = time.perf_counter()
        hv = P.hypervolume(pts)
        dt = time.perf_counter() - t0
        check(0.0 < hv < 1.0 and dt < limit, f"{d}-D HV of a 1000-point front in {dt:.2f}s (< {limit}s)")
    # the fast sweeps must agree with brute force on a mid-sized anti-chain
    g = np.abs(rng.standard_normal((25, 4)))
    pts = g / np.linalg.norm(g, axis=1, keepdims=True)
    axes = [np.unique(np.concatenate([[0.0], pts[:, k]])) for k in range(4)]
    grids = np.meshgrid(*[ax[1:] for ax in axes], indexing="ij")
    widths = np.meshgrid(*[np.diff(ax) for ax in axes], indexing="ij")
    covered = np.zeros(grids[0].shape, dtype=bool)
    for p in pts:
        covered |= np.logical_and.reduce([grids[k] <= p[k] + 1e-15 for k in range(4)])
    check(abs(P.hypervolume(pts) - float((np.prod(widths, axis=0) * covered).sum())) < 1e-10,
          "4-D anti-chain HV matches brute-force grid")
    nanpts = np.array([[0.5, 0.5], [np.nan, 0.9]])
    for fn in (P.hypervolume, P.pareto_front):
        try:
            fn(nanpts)
            raised = False
        except ValueError:
            raised = True
        check(raised, f"{fn.__name__} raises ValueError on NaN instead of dropping the row")


def test_pareto_front():
    print("\n[2] pareto_front")
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.1, 0.1]])
    front = P.pareto_front(pts)
    check(len(front) == 3, "three of four points are non-dominated")
    check(not any((row == [0.1, 0.1]).all() for row in front), "the dominated point is excluded")
    check(front.shape == (3, 2), "front keeps shape (k, d)")


def test_common_unit():
    print("\n[3] the common unit")
    df = synth({"A": 0.6, "B": 0.5, "C": 0.4}, noise=0.02, rng=np.random.default_rng(2))
    z = P.z_scores(df, "auc_top10")
    for oracle, per_bb in z.items():
        check(abs(sum(per_bb.values())) < 1e-9, f"z sums to zero within oracle {oracle}")
    score = P.metric_score(df, "auc_top10")
    check(max(score, key=score.get) == "A", "the better backbone has the higher score")

    flat = synth({"A": 0.3, "B": 0.3, "C": 0.3})
    zf = P.z_scores(flat, "auc_top10")
    check(all(np.isfinite(v) and v == 0.0 for per in zf.values() for v in per.values()),
          "identical runs give z exactly 0.0, not inf or nan")
    # Float dust: values that differ at 1e-17 must still be treated as zero variance.
    dust = synth({"A": 0.3, "B": 0.3, "C": 0.3})
    dust["auc_top10"] = dust["auc_top10"] + np.tile([0.0, 1e-16, -1e-16], len(dust) // 3)
    check(P.pooled_seed_sd(dust, "auc_top10", "o1") == 0.0,
          "float-dust SD is exactly 0.0 under the relative tolerance")
    zd = P.z_scores(dust, "auc_top10")
    check(all(v == 0.0 for per in zd.values() for v in per.values()),
          "float-dust variance gives z exactly 0.0")

    # Pooled SD against a hand value: per-backbone seed SDs 1, 2, 2 (ddof=1) over 3 seeds
    hand = synth({"A": 0.0, "B": 0.0, "C": 0.0}, oracles=("o1",))
    for bb, sd in (("A", 1.0), ("B", 2.0), ("C", 2.0)):
        hand.loc[hand["backbone"] == bb, "auc_top10"] = [-sd, 0.0, sd]
    check(abs(P.pooled_seed_sd(hand, "auc_top10", "o1") - np.sqrt((1 + 4 + 4) / 3)) < 1e-12,
          "pooled SD equals sqrt of mean within-cell variance")

    # only status == "ok" rows enter: a poisoned failed row must change nothing at all
    bad = df.copy()
    bad.loc[bad.index[0], "status"] = "failed"
    bad.loc[bad.index[0], "auc_top10"] = 1e6
    clean = df.drop(index=df.index[0])
    check(P.z_scores(bad, "auc_top10") == P.z_scores(clean, "auc_top10"),
          "a non-ok row leaves z identical to the frame with that row dropped")
    check(P.pooled_seed_sd(bad, "auc_top10", "o1") == P.pooled_seed_sd(clean, "auc_top10", "o1")
          and P.pooled_seed_sd(bad, "auc_top10", "o1") < 1.0,
          "a non-ok row leaves the pooled SD at its clean value")
    # NaN metric values are skipped
    nan = df.copy()
    nan.loc[nan.index[1], "auc_top10"] = np.nan
    check(all(np.isfinite(v) for per in P.z_scores(nan, "auc_top10").values()
              for v in per.values()), "NaN metric values do not poison z")

    check(len(P.metrics_hash()) == 64, "metrics_hash is a sha256 hex digest")
    check(P.metrics_hash(("a", "b")) != P.metrics_hash(("b", "a")), "metrics_hash is order-sensitive")
    check(len(P.METRICS) == 7 and P.PRIMARY_PAIR == ("auc_top10", "hypervolume"),
          "seven metrics, with the primary pair as specified")
    check(not set(P.SECONDARY_METRICS) & (set(P.PRIMARY_PAIR) | {"scalarization_loss"}),
          "secondaries exclude the pair and scalarization_loss")
    check(set(P.SECONDARY_METRICS) | set(P.PRIMARY_PAIR) | {"scalarization_loss"}
          == set(P.METRICS), "pair + secondaries + scalarization_loss cover every metric")


def offset_frame(n_oracles=6, seeds=5, sd=0.02, edge=3.0, seed=7, offsets=None):
    """Per-oracle offsets spanning 0.1..0.9 shared by all backbones; A leads by edge*sd."""
    rng = np.random.default_rng(seed)
    offs = np.linspace(0.1, 0.9, n_oracles) if offsets is None else offsets
    rows = []
    for i, off in enumerate(offs):
        for bb, bump in (("A", edge * sd), ("B", 0.0), ("C", 0.0)):
            for s in range(seeds):
                row = {"oracle": f"o{i}", "backbone": bb, "setting": "x", "seed": s, "status": "ok"}
                for m in P.METRICS:
                    row[m] = np.nan
                row["auc_top10"] = off + bump + sd * rng.standard_normal()
                rows.append(row)
    return pd.DataFrame(rows)


def independent_replication(df, metric, n_boot, seed=0):
    """The WRONG scheme, kept only to show the sharing test bites: every backbone draws its
    own oracle list. With no shared list there is no common oracle to centre against, so a
    backbone's score is its mean of cell-mean / pooled-SD over its own oracles."""
    rng = np.random.default_rng(seed)
    oracles = sorted(df["oracle"].unique())
    bbs = sorted(df["backbone"].unique())
    vals = {(o, b): df[(df.oracle == o) & (df.backbone == b)][metric].to_numpy()
            for o in oracles for b in bbs}
    sds = {o: P.pooled_seed_sd(df, metric, o) for o in oracles}
    point = P.metric_score(df, metric)
    win = max(point, key=point.get)
    hits = 0
    for _ in range(n_boot):
        sc = []
        for b in bbs:
            zs = []
            for o in rng.choice(len(oracles), len(oracles)):
                v = vals[(oracles[o], b)]
                zs.append(rng.choice(v, len(v)).mean() / sds[oracles[o]])
            sc.append(np.mean(zs))
        hits += bbs[int(np.argmax(sc))] == win
    return hits / n_boot


def test_bootstrap_is_shared_across_backbones():
    print("\n[4] shared-oracle bootstrap")
    df = offset_frame()
    rep = P.winner_replication(df, "auc_top10", n_boot=2000)
    check(rep["winner"] == "A", "the within-oracle winner is found")
    check(rep["replication"] > 0.95, f"a consistent within-oracle winner replicates ({rep['replication']})")
    check(set(rep["per_backbone"]) == {"A", "B", "C"}, "per_backbone covers every backbone")
    check(rep["stable"] == (rep["replication"] >= P.REPLICATION_RATE),
          "stable mirrors the replication bar")
    ind = independent_replication(df, "auc_top10", 1000)
    print(f"      shared={rep['replication']:.3f}  independent={ind:.3f}")
    check(ind < 0.7, f"independent per-backbone resampling is near chance on the same frame ({ind})")
    check(rep["replication"] - ind > 0.3, "sharing the oracle list is what separates the two")


def _noise_frame(rng_seed, ors, noise=0.05):
    df = synth({"A": 0.5, "B": 0.5, "C": 0.5}, oracles=ors, seeds=range(5), noise=noise,
               rng=np.random.default_rng(rng_seed))
    df["hypervolume"] = synth({"A": 0.5, "B": 0.5, "C": 0.5}, oracles=ors, seeds=range(5),
                              noise=noise, metric="hypervolume",
                              rng=np.random.default_rng(rng_seed + 1))["hypervolume"]
    return df


def test_pure_noise_has_no_stable_winner():
    print("\n[5] pure noise")
    df = _noise_frame(11, tuple(f"o{i}" for i in range(9)))
    rep = P.winner_replication(df, "auc_top10", n_boot=2000)
    check(rep["replication"] < 0.95, f"pure noise has no stable winner ({rep['replication']})")
    check(not rep["stable"], "pure noise is not flagged stable")
    c0 = P.criterion_0(df, n_boot=2000)
    check(c0["state"] == "neither", "pure noise gives Criterion 0 state 'neither'")
    check(c0["n_stable"] == 0 and c0["stable_metrics"] == [], "n_stable is 0 and the list is empty")


def test_exact_tie():
    print("\n[6] exact tie")
    df = synth({"A": 0.3, "B": 0.3, "C": 0.3}, oracles=("o1", "o2", "o3"), seeds=(0, 1, 2))
    rep = P.winner_replication(df, "auc_top10", n_boot=200)
    check(rep["winner"] is None, "an exact three-way tie reports no winner")
    check(not rep["stable"], "an exact tie is not a stable winner")


def test_criterion_0_covers_only_the_primary_pair():
    print("\n[7] Criterion 0 scope")
    df = _noise_frame(21, tuple(f"o{i}" for i in range(9)))
    df["front_size"] = synth({"A": 0.9, "B": 0.5, "C": 0.5}, oracles=tuple(f"o{i}" for i in range(9)),
                             seeds=range(5), noise=0.02, metric="front_size",
                             rng=np.random.default_rng(23))["front_size"]
    check(P.winner_replication(df, "front_size", n_boot=500)["stable"],
          "front_size really has a stable winner in this frame")
    c0 = P.criterion_0(df, n_boot=2000)
    check(set(c0["per_metric"]) == set(P.PRIMARY_PAIR),
          "Criterion 0 is evaluated on the primary pair only")
    check(c0["state"] == "neither", "a stable secondary metric does not make Criterion 0 pass")

    both = offset_frame()
    both["hypervolume"] = both["auc_top10"]
    c1 = P.criterion_0(both, n_boot=1000)
    check(c1["state"] == "both" and c1["n_stable"] == 2
          and set(c1["stable_metrics"]) == set(P.PRIMARY_PAIR),
          "both metrics stable gives state 'both'")
    one = both.copy()
    one["hypervolume"] = np.random.default_rng(3).standard_normal(len(one))
    c2 = P.criterion_0(one, n_boot=1000)
    check(c2["state"] == "one" and c2["stable_metrics"] == ["auc_top10"],
          "exactly one stable metric gives state 'one'")


def _ragged_frame():
    """Missing cell (o1,C), unequal seed counts, a 1-seed cell, and a zero-SD oracle (o3)."""
    rng = np.random.default_rng(5)
    counts = {("o0", "A"): 5, ("o0", "B"): 4, ("o0", "C"): 5,
              ("o1", "A"): 3, ("o1", "B"): 5,
              ("o2", "A"): 5, ("o2", "B"): 2, ("o2", "C"): 1,
              ("o3", "A"): 3, ("o3", "B"): 3, ("o3", "C"): 3,
              ("o4", "A"): 4, ("o4", "B"): 4, ("o4", "C"): 2}
    rows = []
    for (o, b), n in counts.items():
        for s in range(n):
            row = {"oracle": o, "backbone": b, "setting": "x", "seed": s, "status": "ok"}
            for m in P.METRICS:
                row[m] = np.nan
            base = {"A": 0.6, "B": 0.5, "C": 0.45}[b] + 0.1 * int(o[1])
            row["auc_top10"] = 0.7 if o == "o3" else base + 0.05 * rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows)


def _naive_scores(df, metric):
    """From-scratch reference: plain loops over the frame, no shared helpers."""
    acc = {}
    for o in sorted(df.oracle.unique()):
        d = df[df.oracle == o]
        cells = {b: g[metric].to_numpy() for b, g in d.groupby("backbone")}
        ss = sum(((v - v.mean()) ** 2).sum() for v in cells.values() if len(v) >= 2)
        dof = sum(len(v) - 1 for v in cells.values() if len(v) >= 2)
        sd = np.sqrt(ss / dof) if dof else 0.0
        if sd <= 1e-12 * max(1.0, d[metric].mean()):
            sd = 0.0
        mu = np.mean([v.mean() for v in cells.values()])
        for b, v in cells.items():
            acc.setdefault(b, []).append(0.0 if sd == 0.0 else (v.mean() - mu) / sd)
    return {b: float(np.mean(z)) for b, z in sorted(acc.items())}


def test_point_estimate_and_bootstrap_share_one_core():
    print("\n[8] one core for point estimate and bootstrap")
    df = _ragged_frame()
    ref = _naive_scores(df, "auc_top10")
    got = P.metric_score(df, "auc_top10")
    check(set(got) == set(ref) and all(abs(got[b] - ref[b]) < 1e-12 for b in ref),
          "metric_score on a ragged frame equals an independent from-scratch computation")
    check(P.pooled_seed_sd(df, "auc_top10", "o3") == 0.0
          and all(v == 0.0 for v in P.z_scores(df, "auc_top10")["o3"].values()),
          "the zero-SD oracle gives exactly 0.0 z")
    check("C" not in P.z_scores(df, "auc_top10")["o1"], "the missing cell is absent, not zero")

    # the bootstrap must route through the same core as the point estimate
    calls = {"n": 0}
    real = P._z_and_score

    def spy(*a):
        calls["n"] += 1
        return real(*a)
    P._z_and_score = spy
    try:
        P.metric_score(df, "auc_top10")
        point_calls = calls["n"]
        P.winner_replication(df, "auc_top10", n_boot=50)
        boot_calls = calls["n"] - point_calls
    finally:
        P._z_and_score = real
    check(point_calls == 1 and boot_calls == 2,
          "winner_replication calls the same _z_and_score as metric_score (once for the point, once for the draws)")

    # a batch of identity resamples through the core reproduces the point estimate
    oracles, bbs, vals = P._cells(df, "auc_top10")
    arrs = P._point_arrays(oracles, bbs, vals)
    _, one = P._z_and_score(*arrs)
    _, many = P._z_and_score(*[np.broadcast_to(a, (4,) + a.shape).copy() for a in arrs])
    check(all(np.allclose(many[i], one, atol=1e-15) for i in range(4)),
          "the batched (bootstrap-shaped) core equals the unbatched point-estimate core")


def test_near_tie_with_nonzero_sd_is_a_tie():
    print("\n[9] tie from non-zero-SD data")
    rng = np.random.default_rng(9)
    base = 0.1 * rng.random(5)
    rows = []
    # A leads on o1 and trails on o2 by exactly the amount B does the reverse: scores tie
    # in theory, and may differ by roundoff in practice. C sits in the middle.
    spec = {"o1": {"A": 0.7, "B": 0.5, "C": 0.6}, "o2": {"A": 0.35, "B": 0.55, "C": 0.45}}
    for o, mus in spec.items():
        for b, mu in mus.items():
            for s in range(5):
                row = {"oracle": o, "backbone": b, "setting": "x", "seed": s, "status": "ok"}
                for m in P.METRICS:
                    row[m] = np.nan
                row["auc_top10"] = mu + base[s]
                rows.append(row)
    df = pd.DataFrame(rows)
    sc = P.metric_score(df, "auc_top10")
    check(P.pooled_seed_sd(df, "auc_top10", "o1") > 0.0, "this frame has non-zero seed SD")
    check(abs(sc["A"] - sc["B"]) < 1e-9, f"A and B tie to roundoff ({sc['A'] - sc['B']:.2e})")
    rep = P.winner_replication(df, "auc_top10", n_boot=200)
    check(rep["winner"] is None and not rep["stable"],
          "a roundoff-level tie with non-zero SD reports no winner")


def test_sparse_backbone_raises_no_warning():
    import warnings
    print("\n[10] sparse frames")
    df = offset_frame(n_oracles=4)
    df = df[~((df.backbone == "C") & (df.oracle != "o0"))]      # C has data in one oracle only
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rep = P.winner_replication(df, "auc_top10", n_boot=200)
    check(set(rep["per_backbone"]) == {"A", "B", "C"} and 0.0 <= rep["replication"] <= 1.0,
          "a backbone absent from most draws runs warning-free and returns a valid result")


# ------------------------------------------------------------------ Task 3: criteria
BBS = ("A", "B", "C")


def planted_frame(bumps, n_oracles=9, seeds=5, sd=0.02, seed=5):
    """`bumps[metric][backbone]` is that backbone's lead in units of `sd`, added to a
    per-oracle offset shared by every backbone. A callable bump takes the oracle index, which
    lets the within-oracle winner rotate. Metrics not named stay NaN."""
    rng = np.random.default_rng(seed)
    offs = {m: np.linspace(0.1, 0.9, n_oracles) for m in bumps}
    rows = []
    for i in range(n_oracles):
        for bb in BBS:
            for s in range(seeds):
                row = {"oracle": f"o{i}", "backbone": bb, "setting": "x", "seed": s, "status": "ok"}
                for m in P.METRICS:
                    row[m] = np.nan
                for m, b in bumps.items():
                    bump = b(i, bb) if callable(b) else b.get(bb, 0.0)
                    row[m] = offs[m][i] + bump * sd + sd * rng.standard_normal()
                rows.append(row)
    return pd.DataFrame(rows)


def _rot(shift):
    """Within-oracle winner is BBS[(i + shift) % 3], by 3 SD."""
    return lambda i, bb: 3.0 if BBS[(i + shift) % 3] == bb else 0.0


def test_criterion_A():
    print("\n[12] Criterion A")
    df = planted_frame({"auc_top10": {"A": 3.0}, "hypervolume": {"B": 3.0}})
    cA = P.criterion_A(df, n_boot=2000)
    check(cA["winner_auc"] == "A" and cA["winner_hv"] == "B", "the two metrics pick differently")
    check(cA["differ"] and cA["replication"] > 0.95, f"a planted disagreement replicates ({cA['replication']})")
    check(cA["passes"], "Criterion A passes on a planted disagreement")

    # same argmax pair, but the effect is far below the noise: the point winners differ yet the
    # disagreement does not replicate. A frozen-winners implementation would pass this.
    weak = planted_frame({"auc_top10": {"A": 0.15}, "hypervolume": {"B": 0.15}}, seed=9)
    cw = P.criterion_A(weak, n_boot=2000)
    check(cw["differ"], f"the weak frame's point winners differ ({cw['winner_auc']}, {cw['winner_hv']})")
    check(cw["replication"] < 0.95 and not cw["passes"],
          f"a disagreement that does not replicate fails ({cw['replication']})")

    # on pure noise two independent argmaxes differ about 2/3 of the time: far below the bar
    noise = _noise_frame(31, tuple(f"o{i}" for i in range(9)))
    cn = P.criterion_A(noise, n_boot=2000)
    check(cn["replication"] < 0.95 and not cn["passes"], f"pure noise fails Criterion A ({cn['replication']})")


def test_criterion_A_agreement_fails():
    print("\n[13] Criterion A, agreement")
    df = planted_frame({"auc_top10": {"A": 3.0}, "hypervolume": {"A": 3.0}})
    cA = P.criterion_A(df, n_boot=1000)
    check(cA["winner_auc"] == "A" and cA["winner_hv"] == "A", "both metrics pick A")
    check(not cA["differ"] and not cA["passes"], "agreement between metrics fails Criterion A")
    check(cA["replication"] < 0.05, f"re-selected argmaxes almost never differ ({cA['replication']})")


def test_metric_robust_end_to_end():
    print("\n[13b] METRIC-ROBUST from data")
    df = planted_frame({"auc_top10": {"A": 3.0}, "hypervolume": {"A": 3.0}})
    c0 = P.criterion_0(df, n_boot=2000)
    cA = P.criterion_A(df, n_boot=2000)
    check(c0["state"] == "both", "both primary metrics have a stable winner")
    check(cA["passes"] is False, "Criterion A fails on agreement")
    v = P.phase2_verdict(c0, cA, P.criterion_B(df, n_boot=200), P.criterion_C(df, c0, cA),
                         {"mean": 0.0, "lo": 0.0, "hi": 0.0})
    check(v["verdict"] == "METRIC-ROBUST", f"agreement end-to-end is METRIC-ROBUST ({v['verdict']})")


def test_bh_fdr():
    print("\n[14] Benjamini-Hochberg")
    r = P.bh_fdr({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5})
    check([r[k]["significant"] for k in "abcd"] == [True, True, True, False],
          "BH is not Bonferroni: p = .02 and .03 are significant at m = 4")
    check(abs(r["a"]["threshold"] - 0.05 / 4) < 1e-12 and abs(r["d"]["threshold"] - 0.05) < 1e-12,
          "thresholds are q * rank / m")
    # step-up: x's p exceeds its own rank-1 threshold but is rescued by y at rank 2
    r2 = P.bh_fdr({"x": 0.03, "y": 0.03, "z": 0.9})
    check(r2["x"]["significant"] and r2["y"]["significant"] and not r2["z"]["significant"],
          "BH is step-up: everything up to the largest passing rank is significant")
    check(P.bh_fdr({}) == {}, "an empty family is an empty result")


def _b_frame():
    """front_size constant; coverage shares auc's within-oracle winner, novelty is shifted by
    one backbone (never the same winner), diversity is pure noise."""
    df = planted_frame({"auc_top10": _rot(0), "front_coverage": _rot(0), "novelty": _rot(1),
                        "diversity": {}})
    df["front_size"] = 0.3
    return df


def test_criterion_B_excludes_degenerate_before_correction():
    print("\n[15] Criterion B")
    df = _b_frame()
    cB = P.criterion_B(df, n_boot=2000)
    check(cB["degenerate_metrics"] == ["front_size"], "a constant metric is named degenerate")
    check("front_size" not in cB["per_metric"], "a degenerate metric gets no test")
    check(set(cB["per_metric"]) == {"front_coverage", "novelty", "diversity"},
          "the other three secondaries are tested")
    thresholds = {m: d["threshold"] for m, d in cB["per_metric"].items()}
    check(abs(min(thresholds.values()) - P.BH_Q / len(cB["per_metric"])) < 1e-12,
          "BH thresholds are computed over the non-degenerate metrics only")
    pm = cB["per_metric"]
    check(pm["front_coverage"]["fraction"] == 0.0, "a metric with the same winners disagrees on 0 oracles")
    check(pm["novelty"]["fraction"] == 1.0, "a metric with shifted winners disagrees on every oracle")
    check(pm["front_coverage"]["significant"] and pm["novelty"]["significant"],
          "both extremes are significant against the 2/3 null")
    check(pm["front_coverage"]["hi"] < 2 / 3 < pm["novelty"]["lo"], "the CIs sit on either side of 2/3")
    check(not pm["diversity"]["significant"] and pm["diversity"]["lo"] < 2 / 3 < pm["diversity"]["hi"],
          f"noise is not significant and its CI straddles 2/3 ({pm['diversity']})")
    for d in pm.values():
        check(d["lo"] <= d["fraction"] <= d["hi"] and 0.0 < d["p"] <= 1.0, "fraction in CI, p in (0, 1]")

    # a metric that varies across oracles but never within one is also degenerate, as is an
    # all-NaN one; neither may take a BH slot
    df2 = _b_frame()
    df2["front_size"] = df2["oracle"].map(lambda o: 0.1 + 0.05 * int(o[1:]))
    df2["novelty"] = np.nan
    cB2 = P.criterion_B(df2, n_boot=500)
    check(cB2["degenerate_metrics"] == ["front_size", "novelty"], "within-oracle constant and all-NaN are degenerate")
    check(len(cB2["per_metric"]) == len(P.SECONDARY_METRICS) - len(cB2["degenerate_metrics"])
          and abs(min(d["threshold"] for d in cB2["per_metric"].values())
                  - P.BH_Q / (len(P.SECONDARY_METRICS) - len(cB2["degenerate_metrics"]))) < 1e-12,
          "BH denominator is the non-degenerate count")


def test_criterion_B_empty_bootstrap_is_degenerate():
    print("\n[15b] empty bootstrap")
    orig = P._differing_fraction
    P._differing_fraction = lambda a, m: (orig(a, m) if np.ndim(orig(a, m)) == 0
                                          else np.full(orig(a, m).shape, np.nan))
    try:
        cB = P.criterion_B(_b_frame(), n_boot=50)
    finally:
        P._differing_fraction = orig
    check(cB["per_metric"] == {} and "novelty" in cB["degenerate_metrics"],
          "no surviving oracle in any draw marks the metric degenerate instead of raising")


def test_criterion_C_threshold():
    print("\n[16] Criterion C")
    out = {}
    # seed 13 puts a noise-driven winner C ahead of B at the small edge, which would test a
    # different thing; seed 1 keeps B the HV winner so the gap is the planted one
    for name, edge, sd_ in (("big", 2.5, 13), ("small", 0.4, 1)):
        df = planted_frame({"auc_top10": {"A": 3.0}, "hypervolume": {"B": edge}}, seed=sd_)
        c0 = {"state": "both"}
        cA = P.criterion_A(df, n_boot=200)
        out[name] = P.criterion_C(df, c0, cA)
        out[name + "_hv_winner"] = cA["winner_hv"]
    check(out["big"]["threshold"] == P.C_THRESHOLD == 1.0, "threshold is C_THRESHOLD, no unit conversion")
    check(abs(out["big"]["gap"] - 2.5) < 0.6 and out["big"]["passes"], f"a 2.5-unit gap passes ({out['big']['gap']})")
    check(out["small_hv_winner"] == "B" and out["big_hv_winner"] == "B",
          "the HV winner is B (not A): the COSMETIC branch is exercised with a real non-zero gap")
    check(out["small"]["gap"] > 0.1 and abs(out["small"]["gap"] - 0.4) < 0.6 and not out["small"]["passes"],
          f"a 0.4-unit gap fails ({out['small']['gap']})")

    # one-stable state: the per-oracle AUC argmax rotates A, B, C, so cA's global winner is
    # meaningless. Here cA names B (the HV winner) as winner_auc; using it would give gap 0.
    df = planted_frame({"auc_top10": _rot(0), "hypervolume": {"B": 3.0}}, seed=17)
    cA = {"winner_auc": "B", "winner_hv": "B"}
    one = P.criterion_C(df, {"state": "one"}, cA)
    both = P.criterion_C(df, {"state": "both"}, cA)
    check(one["passes"] and one["gap"] > 1.0, f"state 'one' compares against the per-oracle AUC argmax ({one['gap']})")
    check(both["gap"] == 0.0 and not both["passes"], "state 'both' compares against cA['winner_auc']")
    none = P.criterion_C(df, {"state": "both"}, {"winner_auc": None, "winner_hv": None})
    check(none["gap"] == 0.0 and not none["passes"], "no winner on either side cannot pass")


def test_criterion_G2():
    print("\n[17] Criterion G2")
    df = planted_frame({"auc_top10": {}, "scalarization_loss": {}}, seed=19)
    df["scalarization_loss"] = 0.4 + 0.1 * (df["scalarization_loss"] - df["scalarization_loss"].mean())
    g = P.criterion_G2(df, n_boot=2000)
    check(abs(g["mean"] - df["scalarization_loss"].mean()) < 1e-12, "mean is the plain mean over runs")
    check(g["lo"] < g["mean"] < g["hi"], f"CI brackets the mean ({g['lo']:.4f}, {g['hi']:.4f})")
    check(g["hi"] - g["lo"] > 0.0, "the interval has width")
    empty = df.copy()
    empty["scalarization_loss"] = np.nan
    try:
        P.criterion_G2(empty, n_boot=10)
        raised = False
    except ValueError:
        raised = True
    check(raised, "a measurement cannot come back empty: no data raises")


C0 = {"neither": {"state": "neither", "n_stable": 0}, "one": {"state": "one", "n_stable": 1},
      "both": {"state": "both", "n_stable": 2}}


def _verdict(state, a, c):
    cB = {"per_metric": {}, "degenerate_metrics": []}
    return P.phase2_verdict(C0[state], {"passes": a}, cB, {"passes": c},
                            {"mean": 0.3, "lo": 0.2, "hi": 0.4})


def test_verdict_branches():
    print("\n[18] verdict branches")
    for a in (True, False):
        for c in (True, False):
            check(_verdict("neither", a, c)["verdict"] == "BENCHMARK-INSENSITIVE",
                  f"neither stable -> BENCHMARK-INSENSITIVE whatever A={a}, C={c}")
    check(_verdict("both", True, True)["verdict"] == "METRIC-DEPENDENT", "both, A, C -> METRIC-DEPENDENT")
    check(_verdict("both", False, True)["verdict"] == "METRIC-ROBUST", "both, A fails, C passes -> METRIC-ROBUST")
    check(_verdict("both", False, False)["verdict"] == "METRIC-ROBUST", "both, A fails, C fails -> METRIC-ROBUST")
    check(_verdict("both", True, False)["verdict"] == "COSMETIC", "both, A passes, C fails -> COSMETIC")
    for a in (True, False):
        check(_verdict("one", a, True)["verdict"] == "METRIC-RESOLVING",
              f"one stable, C passes -> METRIC-RESOLVING whatever A={a}")
        check(_verdict("one", a, False)["verdict"] == "COSMETIC",
              f"one stable, C fails -> COSMETIC whatever A={a}")
    check(set(P.VERDICTS) == {"METRIC-DEPENDENT", "METRIC-RESOLVING", "METRIC-ROBUST", "COSMETIC",
                              "BENCHMARK-INSENSITIVE"}, "exactly five verdicts")
    v = _verdict("both", True, True)
    check(v["consequence"] == P.VERDICTS["METRIC-DEPENDENT"] and v["metrics_sha256"] == P.metrics_hash(),
          "the verdict carries its consequence text and the metric-list hash")
    check(v["c0_state"] == "both" and v["a_passes"] is True and v["c_passes"] is True,
          "the verdict carries each criterion's headline")


def test_planted_disagreement_end_to_end():
    print("\n[19] planted disagreement end-to-end")
    df = planted_frame({"auc_top10": {"A": 3.0}, "hypervolume": {"B": 3.0}})
    c0 = P.criterion_0(df, n_boot=2000)
    cA = P.criterion_A(df, n_boot=2000)
    v = P.phase2_verdict(c0, cA, P.criterion_B(df, n_boot=200), P.criterion_C(df, c0, cA),
                         {"mean": 0.0, "lo": 0.0, "hi": 0.0})
    check(c0["state"] == "both" and v["verdict"] == "METRIC-DEPENDENT",
          f"a planted disagreement is METRIC-DEPENDENT end-to-end ({v['verdict']})")


def test_noise_gives_benchmark_insensitive_end_to_end():
    print("\n[20] pure noise end-to-end")
    ors = tuple(f"o{i}" for i in range(9))
    df = _noise_frame(41, ors)
    rng = np.random.default_rng(43)
    for m in P.SECONDARY_METRICS + ("scalarization_loss",):
        df[m] = 0.5 + 0.05 * rng.standard_normal(len(df))
    c0 = P.criterion_0(df, n_boot=2000)
    cA = P.criterion_A(df, n_boot=2000)
    cB = P.criterion_B(df, n_boot=1000)
    v = P.phase2_verdict(c0, cA, cB, P.criterion_C(df, c0, cA), P.criterion_G2(df, n_boot=1000))
    print(f"      noise verdict = {v['verdict']}, c0 = {c0['state']}")
    check(v["verdict"] == "BENCHMARK-INSENSITIVE", "pure noise end-to-end is BENCHMARK-INSENSITIVE")
    check(cB["degenerate_metrics"] == [] and not any(d["significant"] for d in cB["per_metric"].values()),
          "no secondary metric is significant on noise either")


def test_gate_logic():
    print("\n[21] reproduction gate logic")
    # exact agreement on a probe set that genuinely exercises the components
    comp = np.array([[0.0, 0.1], [0.5, 0.6], [1.0, 0.9]])   # ranges 1.0 and 0.8
    ours = np.array([0.0, 0.547, 0.949]); theirs = ours.copy()
    g = P.gate_decision(ours, theirs, comp)
    check(g["reproduces"] and g["probe_exercises"] and g["status"] == "ok", "agreement passes")
    check(np.allclose(g["component_ranges"], [1.0, 0.8]) and abs(g["min_range"] - 0.8) < 1e-12
          and abs(g["frac_nonzero"] - 2 / 3) < 1e-12, "ranges, min_range and frac_nonzero are reported")
    # a real deviation is caught
    bad = ours.copy(); bad[1] += 1e-5
    gb = P.gate_decision(bad, theirs, comp)
    check(gb["status"] == "deviates" and abs(gb["max_deviation"] - 1e-5) < 1e-12, "a 1e-5 deviation fails")
    # a deviation below tolerance is not a deviation
    sm = ours.copy(); sm[1] += 1e-8
    check(P.gate_decision(sm, theirs, comp)["status"] == "ok", "a 1e-8 deviation is within tolerance")
    # a deviating oracle on a weak probe is reported as deviating, not as a weak probe
    zc = np.zeros((20, 2)); z = np.zeros(20)
    check(P.gate_decision(z + 0.3, z, zc)["status"] == "deviates", "deviation outranks a weak probe")
    # a probe set of all-zero scores agrees trivially and must NOT credit
    g0 = P.gate_decision(z, z, zc)
    check(g0["reproduces"], "all-zero scores do agree")
    check(not g0["probe_exercises"] and g0["status"] == "probe_too_weak",
          "a probe set that exercises nothing is not credited as verification")
    # each half of the probe requirement fails independently
    narrow = np.array([[0.4, 0.1], [0.6, 0.9], [0.5, 0.5]])           # first range 0.2
    gn = P.gate_decision(np.array([.3, .5, .4]), np.array([.3, .5, .4]), narrow)
    check(gn["status"] == "probe_too_weak" and gn["frac_nonzero"] == 1.0, "one narrow component fails the range test")
    wide = np.array([[0.0, 0.0]] * 9 + [[1.0, 1.0]])                    # ranges 1, but 10% non-zero
    sc = np.array([0.0] * 9 + [1.0])
    gw = P.gate_decision(sc, sc, wide)
    check(gw["min_range"] == 1.0 and gw["status"] == "probe_too_weak", "full range but 10% non-zero fails")
    # NaN from our side must not be waved through as agreement
    nanv = ours.copy(); nanv[0] = np.nan
    check(P.gate_decision(nanv, theirs, comp)["status"] == "deviates", "a NaN score is a deviation")
    # shape mismatch is a hard error
    try:
        P.gate_decision(ours[:2], theirs, comp); ok = False
    except ValueError:
        ok = True
    check(ok, "mismatched lengths raise")


def test_gate_summary_refuses_below_minimum():
    print("\n[22] gate summary")
    four = {f"o{i}": {"status": "ok"} for i in range(4)}
    check(not P.gate_summary(four)["enough"], "four surviving oracles is not enough to launch")
    five = {f"o{i}": {"status": "ok"} for i in range(5)}
    check(P.gate_summary(five)["enough"], "five surviving oracles is enough")
    mixed = {**five, "bad": {"status": "deviates", "max_deviation": 0.3}}
    s = P.gate_summary(mixed)
    check("bad" in s["excluded"] and s["surviving"] == sorted(five) and s["enough"],
          "a deviating oracle is excluded by name")
    weak = {**four, "w": {"status": "probe_too_weak"}}
    sw = P.gate_summary(weak)
    check("w" in sw["excluded"] and not sw["enough"] and "w" not in sw["surviving"],
          "a probe_too_weak oracle is not credited as surviving")
    check("0.3" in P.gate_summary(mixed)["excluded"]["bad"], "the exclusion reason records the deviation")


def test_components_are_in_unit_range():
    print("\n[23] components")
    import phase2_components as C
    check(len(C.COMPONENTS) == 9, "nine candidate oracles are defined")
    check(set(C.AGGREGATION) == set(C.COMPONENTS) and set(C.AGGREGATION.values()) == {"geometric"},
          "every oracle has a geometric aggregation entry")
    check(all(2 <= len(v) <= 4 for v in C.COMPONENTS.values()),
          "component counts are 2..4, so the isomer sub-score is ONE component")
    check({o: len(v) for o, v in C.COMPONENTS.items()} == {
        "amlodipine_mpo": 2, "fexofenadine_mpo": 3, "osimertinib_mpo": 4, "perindopril_mpo": 2,
        "ranolazine_mpo": 4, "sitagliptin_mpo": 4, "zaleplon_mpo": 2, "median1": 2, "median2": 2},
        "component counts match the GuacaMol definitions per oracle")
    for oracle in C.COMPONENTS:
        v = C.component_vector(oracle, "CC(=O)Oc1ccccc1C(=O)O")   # aspirin
        check(v.shape == (len(C.COMPONENTS[oracle]),), f"{oracle} vector has one entry per component")
        check(((v >= 0.0) & (v <= 1.0)).all(), f"{oracle} components lie in [0,1]")
    check(C.scalar("median1", "CC(C)C1CCC(C)CC1O") > 0.3, "menthol scores on median1")


def test_component_arithmetic():
    print("\n[24] component arithmetic against hand-derived values")
    import phase2_components as C
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    ref = {"zaleplon_mpo": "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1",
           "amlodipine_mpo": r"Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC",
           "sitagliptin_mpo": "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F",
           "perindopril_mpo": "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"}
    # similarity is 1 and the formula/ring targets are met exactly -> every component is 1
    # The reference zaleplon is C17H15N5O, NOT the target formula C19H17N3O2, so its isomer
    # component is below 1. Hand-derived: element diffs C2,H2,N2,O1 at sigma 1 and total-atom
    # diff 3 at sigma 2 -> the geometric mean of five Gaussians:
    # (exp(-0.5*(4+4+4+1)) * exp(-0.5*(3/2)**2)) ** (1/5).
    v = C.component_vector("zaleplon_mpo", ref["zaleplon_mpo"])
    expect = (np.exp(-0.5 * 13) * np.exp(-0.5 * 1.5 ** 2)) ** (1 / 5)
    check(abs(v[0] - 1) < 1e-12 and abs(v[1] - expect) < 1e-12,
          f"zaleplon reference: sim 1 and isomer component {v[1]:.6f} equals the hand-derived value")
    iso = "O=C(c1ccccc1)Nc1ccc(cc1)C(=O)NC1=NC=CCC1"        # formula C19H17N3O2
    check(abs(C.component_vector("zaleplon_mpo", iso)[1] - 1) < 1e-12,
          "a C19H17N3O2 molecule scores 1 on the isomer component (hydrogens are counted)")
    # the amlodipine reference has 2 rings, the target is 3 -> Gaussian(3, .5) at 2 is exp(-2)
    check(np.allclose(C.component_vector("amlodipine_mpo", ref["amlodipine_mpo"]), [1, np.exp(-2)]),
          "amlodipine reference: sim 1 and ring-count Gaussian exp(-2)")
    check(abs(C.component_vector("amlodipine_mpo", "C1CC1C1CC1C1CC1")[1] - 1) < 1e-12, "three rings score 1")
    # perindopril reference has 0 aromatic rings: Gaussian(mu=2, sigma=.5) at 0 is exp(-8)
    v = C.component_vector("perindopril_mpo", ref["perindopril_mpo"])
    check(abs(v[0] - 1) < 1e-12 and abs(v[1] - np.exp(-8)) < 1e-12, "perindopril: aromatic-ring Gaussian is exp(-8)")
    # sitagliptin: similarity sits at its own worst point; logP/TPSA/formula targets are met
    v = C.component_vector("sitagliptin_mpo", ref["sitagliptin_mpo"])
    check(abs(v[0] - np.exp(-50)) < 1e-30 and np.allclose(v[1:], 1),
          "sitagliptin reference: sim Gaussian(0,.1) at 1 is exp(-50); logP, TPSA, isomer all 1")
    # moving off the logP target by exactly 1 unit scores exp(-0.5*(1/0.2)^2)
    mol = Chem.MolFromSmiles(ref["sitagliptin_mpo"])
    lp0 = Descriptors.MolLogP(mol)
    lp1 = Descriptors.MolLogP(Chem.MolFromSmiles("CC(C)(C)c1ccccc1"))
    v2 = C.component_vector("sitagliptin_mpo", "CC(C)(C)c1ccccc1")
    check(abs(v2[1] - np.exp(-0.5 * ((lp1 - lp0) / 0.2) ** 2)) < 1e-12, "logP target is computed from the reference at runtime")
    # half-Gaussians are flat on their good side and fall off on the other
    v_hi = C.component_vector("ranolazine_mpo", "C" * 1 + "C" * 24)   # logP above 7
    check(v_hi[1] == 1.0 and abs(v_hi[2] - np.exp(-0.5)) < 1e-15,
          "ranolazine: logP MaxGaussian(7,1) is 1 above 7; zero fluorines scores Gaussian(1,1)=exp(-.5)")
    v_lo = C.component_vector("ranolazine_mpo", "OCC(O)CO")           # logP << 7 -> score exp(-.5*(7-logP)^2), tiny
    check(v_lo[1] < 1e-6, "ranolazine: low logP is penalised (Max side)")
    v_f = C.component_vector("fexofenadine_mpo", "C(F)(F)F")
    check(0.0 <= v_f[2] <= 1.0 and abs(v_f[2] - 1.0) < 1e-12, "fexofenadine: logP MinGaussian(4,1) is 1 below 4")
    # clipped similarity modifier: AP sim of a molecule with itself is 1/0.8 -> clipped to 1
    check(abs(C.component_vector("fexofenadine_mpo", "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4")[0] - 1) < 1e-12,
          "fexofenadine reference: clipped AP similarity is 1")
    # scalar is the plain geometric mean
    comp = C.component_vector("osimertinib_mpo", "CC(=O)Oc1ccccc1C(=O)O")
    check(abs(C.scalar("osimertinib_mpo", "CC(=O)Oc1ccccc1C(=O)O") - np.prod(comp) ** 0.25) < 1e-15,
          "scalar is the unweighted geometric mean")
    # invalid SMILES must not be silently scored
    try:
        C.component_vector("median1", "not a smiles"); raised = False
    except ValueError:
        raised = True
    check(raised, "an invalid SMILES raises rather than scoring")
    try:
        C.component_vector("nope", "CC"); raised = False
    except KeyError:
        raised = True
    check(raised, "an unknown oracle raises")


def test_probe_set_can_fail_the_gate():
    print("\n[25] probe sets exercise their components")
    import phase2_components as C
    from rdkit import Chem
    for oracle in C.COMPONENTS:
        ps = C.probe_set(oracle)
        check(ps == C.probe_set(oracle), f"{oracle} probe set is deterministic")
        check(len(ps) == len(set(ps)) and len(ps) >= 50, f"{oracle} probe set has >=50 unique molecules ({len(ps)})")
        check(all(Chem.MolFromSmiles(s) is not None for s in ps), f"{oracle} probe molecules are all valid")
        vals = np.array([C.component_vector(oracle, s) for s in ps])
        sc = np.array([C.scalar(oracle, s) for s in ps])
        g = P.gate_decision(sc, sc, vals)
        print(f"      {oracle}: min_range={g['min_range']:.3f} frac_nonzero={g['frac_nonzero']:.3f}")
        check(g["probe_exercises"], f"{oracle} probe set exercises every component and is not mostly zero")
        check(not P.gate_decision(sc, sc + 1e-3, vals)["reproduces"], f"{oracle} probe would catch a 1e-3 error")
    refs = C.probe_set("median1")
    check(Chem.CanonSmiles("CC(C)C1CCC(C)CC1O") in refs and Chem.CanonSmiles("CC1(C)C2CCC1(C)C(=O)C2") in refs, "median1 probe set contains both references")


class _FakeTDC:
    """Stands in for a TDC Oracle: a callable taking a list of SMILES."""
    def __init__(self, fn): self.fn = fn
    def __call__(self, smiles): return [self.fn(s) for s in smiles]


def test_run_gate_with_fake_tdc():
    print("\n[26] run_gate against a stand-in oracle")
    import phase2_components as C
    import sys
    check("tdc" not in sys.modules, "importing phase2_components does not import TDC")
    good = C.run_gate("median1", _FakeTDC(lambda s: C.scalar("median1", s)))
    check(good["status"] == "ok" and good["max_deviation"] == 0.0 and good["n_probe"] == len(C.probe_set("median1")),
          "an exact oracle passes")
    # the exact trap in the brief: a different scalarisation under the same name
    arith = _FakeTDC(lambda s: float(np.mean(C.component_vector("median1", s))))
    check(C.run_gate("median1", arith)["status"] == "deviates", "an arithmetic-mean oracle is caught")
    # a TDC that returns 0 for everything would "agree" with a zero probe; ours is not zero here
    check(C.run_gate("median1", _FakeTDC(lambda s: 0.0))["status"] == "deviates", "an all-zero oracle is caught")
    for o in C.COMPONENTS:   # the arithmetic-mean trap must be caught on EVERY oracle's probe set
        ar = _FakeTDC(lambda s, o=o: float(np.mean(C.component_vector(o, s))))
        check(C.run_gate(o, ar)["status"] == "deviates", f"{o}: an arithmetic-mean impostor is caught")
    nan = C.run_gate("median1", _FakeTDC(lambda s: float("nan")))
    check(nan["status"] == "deviates", "an oracle returning NaN is a deviation")


# --- hand-computed decomposition values and the mutation test (review finding) ------------------
def _independent_expected():
    """Expected component vectors computed WITHOUT phase2_components: similarities from RDKit's
    fingerprint generators (a different API from the legacy one the module uses) and modifiers
    written out from their formulas, with parameters restated from GuacaMol's standard_benchmarks.
    Every case sits where the parameter bites -- not on a plateau and not at self-similarity."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdFingerprintGenerator as G, DataStructs
    gens = {"ECFP4": G.GetMorganGenerator(radius=2), "ECFP6": G.GetMorganGenerator(radius=3),
            "FCFP4": G.GetMorganGenerator(radius=2, atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen()),
            "AP": G.GetAtomPairGenerator(maxDistance=10)}   # legacy maxLength=10 == maxDistance=10

    def sim(kind, ref, smi):
        f = lambda x: gens[kind].GetSparseCountFingerprint(Chem.MolFromSmiles(x))
        return DataStructs.TanimotoSimilarity(f(smi), f(ref))

    gauss = lambda x, mu, sg: float(np.exp(-0.5 * ((x - mu) / sg) ** 2))
    ming = lambda x, mu, sg: gauss(max(x, mu), mu, sg)
    maxg = lambda x, mu, sg: gauss(min(x, mu), mu, sg)
    clip = lambda x, up: float(min(max(x / up, 0.0), 1.0))
    lp = lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x))
    tp = lambda x: Descriptors.TPSA(Chem.MolFromSmiles(x))
    nF = lambda x: sum(a.GetSymbol() == "F" for a in Chem.MolFromSmiles(x).GetAtoms())
    ref = {"menthol": "CC(C)C1CCC(C)CC1O", "camphor": "CC1(C)C2CCC1(C)C(=O)C2",
           "tadalafil": "O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C",
           "sildenafil": "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
           "osi": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34",
           "ran": "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2",
           "fex": "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"}
    cases = []
    m = "CC(C)C1CCC(C)CC1N"
    cases.append(("median1", m, [sim("ECFP4", ref["menthol"], m), sim("ECFP4", ref["camphor"], m)]))
    m = ref["tadalafil"] + "C"
    cases.append(("median2", m, [sim("ECFP6", ref["tadalafil"], m), sim("ECFP6", ref["sildenafil"], m)]))
    # osimertinib: close neighbour (ECFP6 > 0.85, so the MinGaussian bites), a mid-similarity
    # molecule (FCFP4, ECFP4 and ECFP6 all differ), and a TPSA/logP probe.
    for m in ("COc1cc(N(C)CCN(C)C)c(NC(=O)C=O)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1",
              "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)ccc1",
              "c1ccc(cc1)Nc1ccc(cc1)C(=O)NC(N)=O"):
        cases.append(("osimertinib_mpo", m, [clip(sim("FCFP4", ref["osi"], m), 0.8),
                                             ming(sim("ECFP6", ref["osi"], m), 0.85, 0.1),
                                             maxg(tp(m), 100, 10), ming(lp(m), 1, 1)]))
    for m in ("COc1ccccc1OCC(O)CN2CCNCC2", "COc1ccccc1OCC(O)CN2CCN(CC(=O)N)CC2",
              "CCCCc1ccc(cc1)C(=O)NCC(F)(F)F", "CCCCCCCCCCc1ccc(cc1)C(=O)NC(N)=O"):
        cases.append(("ranolazine_mpo", m, [clip(sim("AP", ref["ran"], m), 0.7), maxg(lp(m), 7, 1),
                                            gauss(nF(m), 1, 1.0), maxg(tp(m), 95, 20)]))
    for m in ("CCCCCCCCCCc1ccc(cc1)C(=O)NC(N)=O", "COc1ccccc1OCC(O)CN2CCN(CC(=O)N)CC2"):
        cases.append(("fexofenadine_mpo", m, [clip(sim("AP", ref["fex"], m), 0.8),
                                              maxg(tp(m), 90, 10), ming(lp(m), 4, 1)]))
    return cases


def _decomposition_failures(C):
    bad = []
    for oracle, smi, want in _independent_expected():
        got = C.component_vector(oracle, smi)
        if not np.allclose(got, want, rtol=1e-9, atol=1e-12):
            bad.append((oracle, smi))
    return bad


def test_component_values_at_nontrivial_points():
    print("\n[27] hand-computed component values where each parameter bites")
    import phase2_components as C
    cases = _independent_expected()
    # every case must actually be informative: no component pinned to a 0/1 plateau by accident,
    # except where noted below (the plateau rows are the ones that carry no parameter).
    informative = [v for _, _, vec in cases for v in vec if 1e-9 < v < 0.999]
    check(len(informative) >= 0.8 * sum(len(v) for _, _, v in cases),
          "at least 80% of the asserted values are off-plateau (a parameter change would move them)")
    osi_close = [v for o, m, v in cases if o == "osimertinib_mpo"][0]
    check(osi_close[1] < 0.999, "the osimertinib neighbour sits above ECFP6 0.85, so the MinGaussian bites")
    check(_decomposition_failures(C) == [], "our components equal the independently computed values")


def test_mutations_are_detected():
    print("\n[28] mutation test: every plausible decomposition error must fail the value assertions")
    import phase2_components as C
    orig = {k: getattr(C, k) for k in ("_tanimoto", "_fp", "_gaussian", "_min_gaussian", "_max_gaussian", "_clipped")}
    orig_comps = C.COMPONENTS
    check(_decomposition_failures(C) == [], "baseline: unmutated components pass")

    def swap_kind(ref_key, old, new):
        def t(ref_smiles, kind, modifier=None):
            return orig["_tanimoto"](ref_smiles, new if (ref_smiles == C._REF[ref_key] and kind == old) else kind, modifier)
        return {"_tanimoto": t}

    def retarget(fn, old, new):
        def f(*args):
            return orig[fn](*(new if tuple(args) == old else args))
        return {fn: f}

    def ap_len(n):
        def f(mol, kind):
            if kind == "AP":
                from rdkit import rdBase
                from rdkit.Chem import AllChem
                with rdBase.BlockLogs():
                    return AllChem.GetAtomPairFingerprint(mol, maxLength=n)
            return orig["_fp"](mol, kind)
        return {"_fp": f}

    mutations = {
        "osimertinib ECFP6 -> ECFP4": swap_kind("osimertinib", "ECFP6", "ECFP4"),
        "osimertinib FCFP4 -> ECFP4": swap_kind("osimertinib", "FCFP4", "ECFP4"),
        "osimertinib FCFP4 -> ECFP6": swap_kind("osimertinib", "FCFP4", "ECFP6"),
        "median2 ECFP6 -> ECFP4 (tadalafil)": swap_kind("tadalafil", "ECFP6", "ECFP4"),
        "median2 ECFP6 -> ECFP4 (sildenafil)": swap_kind("sildenafil", "ECFP6", "ECFP4"),
        "median1 ECFP4 -> ECFP6": swap_kind("menthol", "ECFP4", "ECFP6"),
        "AP maxLength 10 -> 30": ap_len(30),
        "AP maxLength 10 -> 9": ap_len(9),
        "ranolazine clipped 0.7 -> 0.8": retarget("_clipped", (0.7,), (0.8,)),
        "fexofenadine clipped 0.8 -> 0.7": retarget("_clipped", (0.8,), (0.7,)),
        "fexofenadine TPSA mu 90 -> 80": retarget("_max_gaussian", (90, 10), (80, 10)),
        "fexofenadine TPSA sigma 10 -> 20": retarget("_max_gaussian", (90, 10), (90, 20)),
        "fexofenadine logP mu 4 -> 5": retarget("_min_gaussian", (4, 1), (5, 1)),
        "osimertinib TPSA mu 100 -> 90": retarget("_max_gaussian", (100, 10), (90, 10)),
        "osimertinib logP sigma 1 -> 2": retarget("_min_gaussian", (1, 1), (1, 2)),
        "osimertinib ECFP6 MinGaussian sigma 0.1 -> 0.2": retarget("_min_gaussian", (0.85, 0.1), (0.85, 0.2)),
        "osimertinib ECFP6 MinGaussian mu 0.85 -> 0.80": retarget("_min_gaussian", (0.85, 0.1), (0.80, 0.1)),
        "ranolazine logP mu 7 -> 6": retarget("_max_gaussian", (7, 1), (6, 1)),
        "ranolazine TPSA mu 95 -> 85": retarget("_max_gaussian", (95, 20), (85, 20)),
        "ranolazine TPSA sigma 20 -> 10": retarget("_max_gaussian", (95, 20), (95, 10)),
        "ranolazine fluorine mu 1 -> 0": retarget("_gaussian", (1, 1.0), (0, 1.0)),
        "Max/Min Gaussian swapped (ranolazine logP)": {"_max_gaussian": lambda mu, sg: orig["_min_gaussian"](mu, sg)},
    }
    try:
        for name, patch in mutations.items():
            for k, v in patch.items():
                setattr(C, k, v)
            try:
                C.COMPONENTS = C._build_components()
                failed = _decomposition_failures(C)
            finally:
                for k, v in orig.items():
                    setattr(C, k, v)
                C.COMPONENTS = orig_comps
            check(len(failed) > 0, f"mutation detected: {name}")
    finally:
        for k, v in orig.items():
            setattr(C, k, v)
        C.COMPONENTS = orig_comps
    check(_decomposition_failures(C) == [], "components restored after the mutations")


def test_power_simulation():
    print("\n[29] power gate: simulation")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import phase2_power as PW
    # a null effect must almost never produce a non-insensitive verdict
    null = PW.simulate(n_seeds=5, effect_sd=0.0, n_rep=40)
    check(sum(null["verdict_counts"].values()) == 40, "every replicate lands in exactly one verdict")
    check(null["power"] < 0.10, f"a planted null has low power ({null['power']})")
    # a large planted effect must be detectable at some seed count
    big = PW.simulate(n_seeds=15, effect_sd=2.0, n_rep=40)
    check(big["power"] > null["power"], f"a 2-SD effect is more detectable than a null ({big['power']})")
    check(big["power"] >= 0.9, f"a 2-SD effect at n=15 is detected >= 90% of the time ({big['power']})")
    check(PW.simulate(n_seeds=5, effect_sd=1.0, n_rep=10, seed=3)
          == PW.simulate(n_seeds=5, effect_sd=1.0, n_rep=10, seed=3), "simulation is seed-deterministic")


def test_choose_seeds_halts_when_nothing_is_reachable():
    print("\n[30] power gate: choose_seeds")
    import phase2_power as PW
    # the unsatisfiable case: target power above 1.0 can never be met
    res = PW.choose_seeds(candidates=(5,), target_power=1.01, n_rep=10)
    check(res["halt"] and res["chosen"] is None and not res["reachable"],
          "an unreachable target halts rather than freezing an unsatisfiable rule")
    res2 = PW.choose_seeds(candidates=(5, 10, 15), target_power=0.0, n_rep=10)
    check(res2["chosen"] == 5 and not res2["halt"], "the smallest sufficient seed count is chosen")
    # floor helper picks the smallest effect reaching target, None when none does
    fake = {5: {0.2: {"power": 0.5, "n_rep": 10, "verdict_counts": {}},
                0.4: {"power": 0.9, "n_rep": 10, "verdict_counts": {"METRIC-DEPENDENT": 3}},
                0.6: {"power": 1.0, "n_rep": 10, "verdict_counts": {"METRIC-DEPENDENT": 9}}},
            10: {0.2: {"power": 0.1, "n_rep": 10, "verdict_counts": {}}}}
    check(PW.detectable_floor(fake, (5, 10), 0.8) == {5: 0.4, 10: None}, "floor is the smallest effect reaching target")
    check(PW.detectable_floor(fake, (5,), 0.8, "dependent") == {5: 0.6}, "the METRIC-DEPENDENT floor is stricter")
    pe = PW.phase1_sized_effect()
    check(pe["effect_sd"] > 0 and 0 < pe["null_median"] < 2,
          f"the Phase-1-sized effect is derived from the manifest ({pe['effect_sd']:.3f} SD)")
    par = PW.run_grid((5,), (0.0, 2.0), 6, workers=2)
    ser = PW.run_grid((5,), (0.0, 2.0), 6, workers=1)
    check(par == ser, "parallel grid equals serial grid")
    # main() must turn halt / no-seed-count into non-zero exits (a halt that exits 0 is the failure)
    import tempfile
    cell = {"power": 1.0, "n_rep": 1, "verdict_counts": {"METRIC-DEPENDENT": 1}}
    tab = {n: {e: cell for e in (0.0, 1.0, 2.0)} for n in (5, 10, 15)}
    orig_cs, orig_rg = PW.choose_seeds, PW.run_grid
    try:
        PW.run_grid = lambda cands, effects, *a, **k: {n: {e: cell for e in effects} for n in cands}
        def run_main(chosen, reachable):
            PW.choose_seeds = lambda *a, **k: {"table": {n: dict(r) for n, r in tab.items()},
                                               "chosen": chosen, "reachable": reachable,
                                               "halt": not reachable, "target_power": 0.8}
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "gate.json"
                code = PW.main(["--n-rep", "1", "--floor-rep", "1", "--workers", "1", "--out", str(out)])
                return code, json.loads(out.read_text())
        c_halt, j_halt = run_main(None, False)
        c_nochoice, j_nochoice = run_main(None, True)
        c_ok, j_ok = run_main(5, True)
    finally:
        PW.choose_seeds, PW.run_grid = orig_cs, orig_rg
    check(c_halt == 1 and j_halt["halt"] and j_halt["exit_code"] == 1, "main() exits 1 on HALT")
    check(c_nochoice == 2 and not j_nochoice["halt"] and j_nochoice["chosen"] is None,
          "main() exits 2 when 2 SD is reachable but no n reaches target at 1 SD")
    check(c_ok == 0 and j_ok["chosen"] == 5, "main() exits 0 only when a seed count is chosen")
    for k in ("table", "floor", "params", "data_source", "phase1_sized_effect", "floor_scan"):
        check(k in j_ok, f"persisted gate artifact carries {k}")
    check(set(j_ok["params"]) >= {"seed", "n_rep", "floor_rep", "n_boot", "candidates"},
          "artifact records the effective parameters")
    pe2 = PW.phase1_sized_effect()
    check({"effect_sd", "effect_sd_p25", "effect_sd_all", "effect_sd_pooled", "per_oracle"} <= set(pe2)
          and len(pe2["per_oracle"]) == 14 and pe2["effect_sd_p25"] <= pe2["effect_sd"],
          "per-oracle effect (median, p25) and both superseded figures are reported")
    check(abs(pe2["effect_sd"] - float(np.median(list(pe2["per_oracle"].values())))) < 1e-12,
          "the headline effect is the median of the per-oracle standardised spreads")
    check(set(res2["table"]) == {5, 10, 15} and all(set(v) == {0.0, 1.0, 2.0} for v in res2["table"].values()),
          "the table covers every candidate x planted effect")


# ---------------------------------------------------------------------------------------------
# Task 6: the build artifacts, the generated library, and the pre-registration
# ---------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
BRANCH_POINT = "5700883"


def _lib():
    import phase2_lib
    return phase2_lib


def _source_cells():
    import build_phase2 as B
    return B.parse_percent((ROOT / "src" / "phase2_source.py").read_text(encoding="utf-8"))


def test_build_artifacts():
    nb = json.loads((ROOT / "Phase 2" / "landscape_phase2.ipynb").read_text())
    check(nb["nbformat"] == 4, "the notebook is nbformat 4")
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    check(len(code) > 10, f"the notebook has a real body ({len(code)} code cells)")
    lib_text = (ROOT / "src" / "phase2_lib.py").read_text()
    L = _lib()
    for name in ("execute_run2", "run_sweep2", "build_sweep2", "load_results2",
                 "CANDIDATE_ORACLES", "PREREG2", "RUNS2_DIR", "prereg2_hash", "PREREG2_SHA256"):
        check(hasattr(L, name), f"the generated library exports {name}")
    check("import phase2_stats as PS" in lib_text, "the library imports the statistics module")
    check("import phase0_lib" in lib_text,
          "the library imports phase0_lib (Phase 1 had it notebook-only: NameError in every runner)")
    check(isinstance(L.RUNS2_DIR, Path) and L.RUNS2_DIR.name == "runs_phase2",
          "RUNS2_DIR is a real path, not None outside a kernel")
    check("LAUNCH_SWEEP2" not in lib_text and "DRY_RUN_FIRST" not in lib_text,
          "driver switches are not in the library")
    nbtext = "\n".join("".join(c["source"]) for c in code)
    check("DRY_RUN_FIRST = True" in nbtext and "LAUNCH_SWEEP2 = False" in nbtext,
          "the notebook ships with DRY_RUN_FIRST = True and LAUNCH_SWEEP2 = False")
    check("phase2_power_gate.json" in nbtext, "the notebook references the committed power-gate evidence")


def test_library_cells_are_not_notebook_only():
    """Phase 1's four build defects were all @notebook-only mistakes. Every constant, import and
    runner the sweep CLI needs must sit in a LIBRARY cell."""
    import re
    cells = _source_cells()
    lib_src = "\n".join(src for kind, src, nb_only in cells if kind == "code" and not nb_only)
    nb_src = "\n".join(src for kind, src, nb_only in cells if kind == "code" and nb_only)
    for pat in (r"^CANDIDATE_ORACLES\s*=", r"^PREREG2\s*=", r"^PREREG2_SHA256\s*=",
                r"^RUNS2_DIR\s*=", r"^import phase2_stats as PS", r"^import phase2_components as PC",
                r"^\s+import phase0_lib", r"^def execute_run2", r"^def build_sweep2",
                r"^def run_sweep2", r"^def load_results2", r"^def sklearn_tripwire",
                r"^def phase1_overlap_gate", r"^def reproduction_gate", r"^def population_metrics"):
        check(re.search(pat, lib_src, re.M) is not None, f"library cell defines {pat!r}")
    for pat in (r"^RUNS2_DIR\s*=", r"^CANDIDATE_ORACLES\s*=", r"^PREREG2\s*=", r"^import "):
        check(re.search(pat, nb_src, re.M) is None, f"no notebook-only cell matches {pat!r}")
    check(re.search(r"RUNS2_DIR\s*=\s*None", lib_src) is None, "RUNS2_DIR is never None in the library")
    check(any(nb for _, _, nb in cells), "some cells are tagged @notebook-only (drivers)")


def test_gates_run_in_the_required_order():
    cells = _source_cells()
    srcs = [src for kind, src, _ in cells if kind == "code"]

    def first(token):
        return next(i for i, s in enumerate(srcs) if token in s)
    order = [first("sklearn_tripwire(p0()"), first("check_candidates_against_pmo(pmo_names_via_phase1())"),
             first("reproduction_gate(CANDIDATE_ORACLES"), first("LAUNCH_SWEEP2 = False"),
             first("phase1_overlap_gate(pd.read_csv")]
    check(order == sorted(order) and len(set(order)) == 5,
          f"tripwire, candidate cross-check, reproduction gate, sweep, overlap gate in order {order}")


def test_prior_phases_untouched():
    out = subprocess.run(["git", "diff", "--name-only", BRANCH_POINT, "--",
                          "src/phase0_lib.py", "src/phase0_source.py", "src/phase1_source.py",
                          "src/phase1_stats.py", "src/phase1_lib.py", "Phase 1",
                          ":(exclude)Phase 1/phase1_run_manifest.csv"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    check(out == "", f"Phase 0/1 sources are unmodified since {BRANCH_POINT} (changed: {out!r})")
    # the gate JSON and the Phase 1 manifest were ADDED after the branch point (committed since),
    # so their guard is the last commit, not 5700883
    out = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", "Phase 2/phase2_power_gate.json",
                          "Phase 1/phase1_run_manifest.csv"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    check(out == "", "the committed power-gate evidence and Phase 1 manifest are not modified")


def test_prereg_is_frozen_and_hashed():
    L = _lib()
    P2 = L.PREREG2
    check(P2["version"] == "3.0.0", "the pre-registration is version 3.0.0")
    check(P2["budget"] == 1000, "the budget is 1000 calls")
    check(P2["hv_reference_point"] == "origin", "the HV reference point is the origin")
    check(P2["min_surviving_oracles"] == 5, "the minimum oracle count is 5")
    check(set(P2["aggregation"].values()) == {"geometric"} and len(P2["aggregation"]) == 9,
          "aggregation is a frozen per-oracle dict")
    check(P2["seeds"] == 15 and isinstance(P2["seeds"], int) and P2["seed_ids"] == list(range(15)),
          "the seed count is 15, by ruling")
    check(P2["seed_rule_output"] == 5, "the overridden rule output (5) is recorded in the open")
    n_runs = len(P2["oracles"]) * len(P2["backbones"]) * P2["seeds"]
    check(n_runs == 405, f"the grid is 9 x 3 x 15 = 405 runs (got {n_runs})")
    check(P2["setting"] == "explore_mid", "a single setting")
    check(P2["metrics_sha256"] == P.metrics_hash(), "the metric hash is the computed one")
    check(L.prereg2_hash() == L.PREREG2_SHA256, "the in-code rule matches the frozen hash")
    disk = json.loads((ROOT / "Phase 2" / "preregistration_phase2.json").read_text())
    check(disk["_sha256"] == L.PREREG2_SHA256, "the committed file carries the frozen hash")
    check(L.prereg2_hash(disk) == disk["_sha256"], "the committed file's contents hash to its own stamp")
    check(L.prereg2_status() == "verified", "code, constant and file agree")
    # discriminating: any edit to the rule changes the hash
    tampered = dict(P2); tampered["seeds"] = 5
    check(L.prereg2_hash(tampered) != L.PREREG2_SHA256, "changing the seed count changes the hash")
    tampered = dict(P2); tampered["c_threshold"] = 0.5
    check(L.prereg2_hash(tampered) != L.PREREG2_SHA256, "changing a criterion changes the hash")
    saved = L.PREREG2["seeds"]
    L.PREREG2["seeds"] = 5
    try:
        raised = False
        try:
            L.assert_prereg_intact()
        except L.GateHalt:
            raised = True
        check(raised and L.prereg2_status() == "mismatch", "a mutated rule is refused (mechanism working)")
    finally:
        L.PREREG2["seeds"] = saved
    try:
        L.freeze_prereg2()
        check(False, "freezing over an existing file must refuse")
    except L.GateHalt:
        check(True, "freeze_prereg2 refuses to overwrite the frozen file")


def test_candidate_oracles_are_derived_and_cross_checked():
    import phase2_components as PCm
    L = _lib()
    check(L.CANDIDATE_ORACLES == sorted(PCm.COMPONENTS), "candidates are derived from the components")
    check(L.CANDIDATE_ORACLES == sorted(["amlodipine_mpo", "fexofenadine_mpo", "osimertinib_mpo",
                                          "perindopril_mpo", "ranolazine_mpo", "sitagliptin_mpo",
                                          "zaleplon_mpo", "median1", "median2"]),
          "the derived list equals design 5.3's nine, an independent source")
    check(L.check_candidates_against_pmo(L.CANDIDATE_ORACLES + ["gsk3b"]), "a superset of PMO names passes")
    try:
        L.check_candidates_against_pmo([o for o in L.CANDIDATE_ORACLES if o != "median2"])
        check(False, "a candidate missing from PMO must raise")
    except L.GateHalt as e:
        check("median2" in str(e), "a candidate missing from the PMO listing raises, naming it")
    try:
        L.check_candidates_against_pmo(["hparams_default", "hparams_tune"])
        check(False, "YAML stems must not satisfy the cross-check")
    except L.GateHalt:
        check(True, "Phase 1's defect-4 output (YAML stems) fails the cross-check")


def test_sklearn_tripwire():
    L = _lib()
    try:
        L.sklearn_tripwire(lambda s: 0.0)
        check(False, "an all-zero drd2 must halt")
    except L.GateHalt as e:
        check("all zeros" in str(e), "an all-zero drd2 halts")
    sc = L.sklearn_tripwire(lambda s: 0.0 if s == "CCO" else 0.3)
    check(len(sc) == 4 and any(x > 0 for x in sc), "one non-zero probe is enough")
    check(len(L.PROBE_SMILES) == 4, "four probe SMILES")


def _manifest_frame():
    return pd.read_csv(ROOT / "Phase 1" / "phase1_run_manifest.csv")


def test_phase1_overlap_gate():
    import tempfile
    L = _lib()
    m = _manifest_frame()
    mid = m[(m["setting"] == "explore_mid") & m["oracle"].isin(L.CANDIDATE_ORACLES)]
    p2 = mid[["oracle", "backbone", "seed", "status", "auc_top10"]].copy()
    r = L.phase1_overlap_gate(m, p2)
    check(r["status"] == "passed" and r["n_compared"] == 63,
          f"the real Phase 1 manifest overlaps Phase 2's grid in 63 cells: 7 oracles x 3 x 3; "
          f"ranolazine/sitagliptin were Phase 0 oracles (got {r['n_compared']})")
    check(r["n_expected"] == 63, "the gate computes the expected overlap itself: 63")
    short = p2.iloc[1:]
    with tempfile.TemporaryDirectory() as td0:
        pth = Path(td0) / "o.json"
        try:
            L.phase1_overlap_gate(m, short, path=pth)
            check(False, "62 of 63 cells must not pass")
        except L.GateHalt as e:
            check("62 of the 63" in str(e), "a partial merge halts, saying how many of how many")
        sv = json.loads(pth.read_text())
        check(sv["status"] == "incomplete" and sv["n_compared"] == 62 and sv["n_expected"] == 63,
              "the incomplete overlap is persisted")
    only3 = p2[p2["oracle"] == "median1"]
    try:
        L.phase1_overlap_gate(m, only3)
        check(False, "9 of 63 cells must not pass")
    except L.GateHalt:
        check(True, "a handful of matching cells is not credited as a pass")
    six = [o for o in L.CANDIDATE_ORACLES if o not in ("ranolazine_mpo", "sitagliptin_mpo", "median2")]
    p6 = p2[p2["oracle"].isin(six)]
    r6 = L.phase1_overlap_gate(m, p6, expected_oracles=six)
    check(r6["status"] == "passed" and r6["n_compared"] == r6["n_expected"] == 54,
          "with a gate-excluded oracle the expectation shrinks to the survivors and still passes")
    try:
        L.phase1_overlap_gate(m, p6)
        check(False, "default expectation covers all candidates")
    except L.GateHalt:
        check(True, "without naming survivors the same frame is incomplete")
    fl = p2.copy(); fl["seed"] = fl["seed"].astype(float)
    check(L.phase1_overlap_gate(m, fl)["n_compared"] == 63, "a float seed column still aligns")
    check(r["max_deviation"] == 0.0, "identical records deviate by exactly 0")
    check(L.OVERLAP_TOL == 1e-6 and L.PREREG2["overlap_gate"]["tolerance"] == 1e-6,
          "the overlap tolerance is 1e-6 absolute per cell, in code and in the pre-registration")
    col = p2.columns.get_loc("auc_top10")
    q = p2.copy(); q.iloc[0, col] += 5e-7
    r = L.phase1_overlap_gate(m, q)
    check(r["status"] == "passed" and 4e-7 < r["max_deviation"] < 6e-7, "a 5e-7 deviation is inside the tolerance")
    q = p2.copy(); q.iloc[7, col] += 5e-6
    try:
        L.phase1_overlap_gate(m, q)
        check(False, "a 5e-6 deviation must halt")
    except L.GateHalt as e:
        check("NOT reproduced" in str(e), "a single run 5e-6 off halts the gate")
    # a systematic shift, individually inside tolerance, is invisible to a max but not to the mean
    q = p2.copy(); q["auc_top10"] = q["auc_top10"] + 5e-7
    r = L.phase1_overlap_gate(m, q)
    check(r["status"] == "passed" and abs(r["mean_signed_difference"] + 5e-7) < 1e-12,
          "mean signed difference keeps its sign: phase 2 higher by 5e-7 gives -5e-7")
    q["auc_top10"] = p2["auc_top10"] - 5e-7
    check(abs(L.phase1_overlap_gate(m, q)["mean_signed_difference"] - 5e-7) < 1e-12,
          "and phase 2 lower gives +5e-7")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sub" / "overlap.json"
        L.phase1_overlap_gate(m, p2, path=path)
        saved = json.loads(path.read_text())
        check(saved["status"] == "passed" and saved["n_compared"] == 63 and saved["max_deviation"] == 0.0
              and saved["mean_signed_difference"] == 0.0 and saved["tolerance"] == 1e-6,
              "a passing gate persists n_compared, max_deviation, mean_signed_difference and tolerance")
        q = p2.copy(); q.iloc[3, col] += 5e-6
        try:
            L.phase1_overlap_gate(m, q, path=path)
        except L.GateHalt:
            pass
        saved = json.loads(path.read_text())
        check(saved["status"] == "failed" and saved["max_deviation"] > 1e-6,
              "a failing gate persists its failure before raising")
        L.phase1_overlap_gate(m.iloc[0:0], p2, path=path)
        check(json.loads(path.read_text())["status"] == "could_not_run", "could-not-run is persisted too")
    # the setting filter: other settings' rows for the same key must not matter
    lowhigh = m[m["setting"] != "explore_mid"].copy()
    lowhigh["auc_top10"] += 0.5
    mixed = pd.concat([m[m["setting"] == "explore_mid"], lowhigh])
    check(L.phase1_overlap_gate(mixed, p2)["status"] == "passed",
          "Phase 1 rows from other settings are filtered out before the comparison")
    check(set(L.phase1_explore_mid(m)["setting"]) == {"explore_mid"}, "phase1_explore_mid keeps one setting")
    for label, a, b in (("Phase 1 absent", None, p2), ("Phase 1 empty", m.iloc[0:0], p2),
                        ("Phase 2 absent", m, None), ("Phase 2 empty", m, p2.iloc[0:0])):
        r = L.phase1_overlap_gate(a, b)
        check(r["status"] == "could_not_run" and r["n_compared"] == 0,
              f"{label}: 'could not run', never 'passed'")
    disjoint = p2.copy(); disjoint["seed"] += 100
    r = L.phase1_overlap_gate(m, disjoint)
    check(r["status"] == "could_not_run", "no shared cell is 'could not run', never 'passed'")
    bad = p2.copy(); bad["status"] = "error"
    check(L.phase1_overlap_gate(m, bad)["status"] == "could_not_run", "failed Phase 2 runs are not compared")


def test_reproduction_gate_halts_below_minimum():
    import phase2_components as PCm
    L = _lib()
    exact = lambda o: (lambda smiles: [PCm.scalar(o, s) for s in smiles])
    s = L.reproduction_gate(L.CANDIDATE_ORACLES, exact)
    check(len(s["surviving"]) == 9 and s["enough"], "an exact TDC stand-in clears all nine oracles")
    broken = set(L.CANDIDATE_ORACLES[:5])
    skewed = lambda o: (lambda smiles: [PCm.scalar(o, x) + (0.1 if o in broken else 0.0) for x in smiles])
    try:
        L.reproduction_gate(L.CANDIDATE_ORACLES, skewed)
        check(False, "four survivors must halt")
    except L.GateHalt as e:
        check("only 4" in str(e), "below MIN_ORACLES the gate halts and says how many survived")
    one = set(L.CANDIDATE_ORACLES[:1])
    s = L.reproduction_gate(L.CANDIDATE_ORACLES,
                            lambda o: (lambda smiles: [PCm.scalar(o, x) + (0.1 if o in one else 0.0) for x in smiles]))
    check(len(s["surviving"]) == 8 and list(s["excluded"]) == sorted(one),
          "one deviating oracle is excluded by name, the rest survive")


def test_gate_file_governs_the_sweep():
    import tempfile
    L = _lib()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        good = td / "gate.json"
        L.save_gate({"surviving": L.CANDIDATE_ORACLES, "excluded": {}, "enough": True}, good)
        check(L.surviving_oracles(good) == L.CANDIDATE_ORACLES, "the saved gate round-trips")
        try:
            L.surviving_oracles(td / "missing.json")
            check(False, "no gate file must halt")
        except L.GateHalt:
            check(True, "a missing reproduction-gate file halts")
        few = td / "few.json"
        L.save_gate({"surviving": L.CANDIDATE_ORACLES[:4], "excluded": {}, "enough": False}, few)
        try:
            L.surviving_oracles(few)
            check(False, "fewer than 5 must halt")
        except L.GateHalt:
            check(True, "fewer than MIN_ORACLES survivors halts the sweep")
        old = (L.GATE2_PATH, L.RUNS2_DIR)
        L.GATE2_PATH, L.RUNS2_DIR = few, td / "runs"
        try:
            L.run_sweep2(L.build_sweep2(), n_workers=1)
            check(False, "sweep must refuse")
        except L.GateHalt:
            check(True, "run_sweep2 refuses to start below MIN_ORACLES")
        L.GATE2_PATH = td / "partial.json"
        L.save_gate({"surviving": L.CANDIDATE_ORACLES[:6], "excluded": {}, "enough": True}, L.GATE2_PATH)
        try:
            L.run_sweep2(L.build_sweep2(), n_workers=1)
            check(False, "an excluded oracle must be refused")
        except L.GateHalt as e:
            check("did not survive" in str(e), "run_sweep2 refuses jobs for an oracle that failed the gate")
        L.GATE2_PATH, L.RUNS2_DIR = old
        saved = L.PREREG2["budget"]
        L.PREREG2["budget"] = 999
        try:
            L.run_sweep2([])
            check(False, "mismatched prereg must refuse")
        except L.GateHalt:
            check(True, "run_sweep2 refuses to start on a pre-registration mismatch")
        finally:
            L.PREREG2["budget"] = saved
    jobs = L.build_sweep2()
    check(len(jobs) == 405 and len(set(jobs)) == 405, "the default sweep is 405 distinct jobs")
    check({j[2] for j in jobs} == set(range(15)) and {j[1] for j in jobs} == set(L.PREREG2["backbones"]),
          "the sweep covers seeds 0-14 and all three backbones")
    check(L.build_sweep2(["median1"], ["graph_ga"], [0, 1]) == [("median1", "graph_ga", 0), ("median1", "graph_ga", 1)],
          "filters restrict the sweep")


class _FakeOracle:
    def __init__(self, buf):
        self.mol_buffer = buf


class _FakeP0:
    """Stands in for phase0_lib: the buffer is {smiles: [score, call_index]} as BudgetedOracle's."""
    POP = ["CC(C)C1CCC(C)CC1O", "CC1(C)C2CCC1(C)C(=O)C2", "CCCCCCCC", "c1ccccc1", "CCO"]

    def make_oracle(self, name, budget=1000, freq_log=10):
        # insertion order deliberately differs from call_index order
        buf = {s: [0.1 * (i + 1), len(self.POP) - i] for i, s in enumerate(self.POP)}
        return _FakeOracle(buf)

    def run_backbone(self, backbone, orc, setting, seed, verbose=False):
        return {"auc_top10": 0.25, "top10_smiles": list(orc.mol_buffer)[:2]}


def test_runs_persist_the_full_buffer():
    """The Phase 1 defect: smiles keys were discarded. Assert the persisted record's SHAPE."""
    import tempfile
    L = _lib()
    fake = _FakeP0()
    with tempfile.TemporaryDirectory() as td:
        old = (L._P0, L.RUNS2_DIR)
        L._P0, L.RUNS2_DIR = fake, Path(td) / "runs"
        try:
            r = L.execute_run2("median1", "graph_ga", 3)
            rec = json.loads(L.run_path2("median1", "graph_ga", 3).read_text())
            buf = rec["full_buffer"]
            check(len(buf) == len(fake.POP), "every molecule in the buffer is persisted, not just the top 10")
            check(all(isinstance(t, list) and len(t) == 3 and isinstance(t[0], str)
                      and isinstance(t[1], float) and isinstance(t[2], int) for t in buf),
                  "each element is a (smiles, score, call_index) triple")
            check({t[0] for t in buf} == set(fake.POP), "the SMILES keys survive to disk")
            check([t[2] for t in buf] == sorted(t[2] for t in buf), "triples are in call order")
            check(buf[0][0] == fake.POP[-1] and buf[-1][0] == fake.POP[0],
                  "ordering follows call_index, not dict insertion order")
            check(rec["status"] == "ok" and rec["phase"] == 2 and rec["setting"] == "explore_mid"
                  and rec["oracle"] == "median1" and rec["seed"] == 3
                  and rec["prereg_sha256"] == L.PREREG2_SHA256, "the record carries its provenance")
            check(r["full_buffer"] == buf, "the returned record equals the persisted one")
            again = L.execute_run2("median1", "graph_ga", 3)
            check(again["full_buffer"] == buf, "a completed record is returned untouched (resumable)")
            df = L.load_results2()
            check(len(df) == 1 and "full_buffer" not in df.columns, "the loader drops the buffer from the frame")
            check(set(L.PREREG2["metrics"]) <= set(df.columns) and df["status"].iloc[0] == "ok",
                  "the loader computes every metric from the persisted buffer")
            check(df["auc_top10"].iloc[0] == 0.25, "auc_top10 comes from the run record")
            # a run whose population cannot be scored is named and excluded, not invented
            bad = json.loads(L.run_path2("median1", "graph_ga", 3).read_text())
            bad["seed"] = 4
            bad["full_buffer"] = [["not a smiles", 0.5, 1]]
            L.run_path2("median1", "graph_ga", 4).write_text(json.dumps(bad))
            df = L.load_results2()
            check(sorted(df["status"]) == ["metric_error", "ok"], "an uncomputable run is marked metric_error")
        finally:
            L._P0, L.RUNS2_DIR = old


def _brute_front(pts):
    keep = []
    for i, p in enumerate(pts):
        dominated = any((q >= p).all() and (q > p).any() for j, q in enumerate(pts) if j != i)
        keep.append(not dominated)
    return np.array(keep)


def test_population_metrics_hand_computed():
    import phase2_components as PCm
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    L = _lib()
    smi = ["CC(C)C1CCC(C)CC1O", "CC1(C)C2CCC1(C)C(=O)C2", "CCCCCCCC", "c1ccccc1", "CCO",
           "CC(C)C1CCC(C)CC1=O", "CC1CCC(CC1)C(C)C", "CC(C)C1CCCCC1O", "CC1(C)C2CCC1(C)C(O)C2",
           "CCCCO", "CC(C)O", "C1CCCCC1"]
    comps = np.array([PCm.component_vector("median1", s) for s in smi])
    mask = _brute_front(comps)
    check(0 < mask.sum() < len(smi), "the test population has a non-trivial front")
    # the front molecule `smi[0]` gets the LOWEST score so scalarization loss is non-zero
    scores = np.array([0.0] + [0.9 - 0.05 * i for i in range(1, len(smi))])
    buf = [[s, float(sc), i + 1] for i, (s, sc) in enumerate(zip(smi, scores))]
    m = L.population_metrics("median1", buf, 0.31)
    check(set(m) == set(P.METRICS), "all seven metrics are produced")
    check(m["front_size"] == float(mask.sum()), "front_size counts non-dominated molecules")
    check(abs(m["front_coverage"] - mask.sum() / len(smi)) < 1e-12, "front_coverage = front_size / population")
    front = comps[mask]
    xs = front[np.argsort(-front[:, 0])]
    hv, ymax = 0.0, 0.0
    for x, y in xs:                      # independent 2-D staircase
        hv += max(y - ymax, 0.0) * x
        ymax = max(ymax, y)
    check(abs(m["hypervolume"] - hv) < 1e-12 and 0.0 < hv <= 1.0, f"hypervolume matches a staircase ({hv:.4f})")
    thr = np.sort(scores)[::-1][9]
    exp_loss = float(np.mean(scores[mask] < thr))
    check(0.0 < exp_loss < 1.0 and abs(m["scalarization_loss"] - exp_loss) < 1e-12,
          f"scalarization_loss is the front fraction below the top-10 threshold ({exp_loss:.3f})")
    check(m["auc_top10"] == 0.31, "auc_top10 is passed through")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smi]
    refs = [gen.GetFingerprint(Chem.MolFromSmiles(PCm._REFS["median1"][k])) for k in (0, 1)]
    exp_nov = np.mean([max(DataStructs.TanimotoSimilarity(f, r) for r in refs) < 0.4 for f in fp])
    check(0.0 < exp_nov < 1.0 and abs(m["novelty"] - exp_nov) < 1e-12, f"novelty matches ({exp_nov:.3f})")
    exp_div = np.mean([1 - DataStructs.TanimotoSimilarity(fp[i], fp[j])
                       for i in range(len(fp)) for j in range(i)])
    check(abs(m["diversity"] - exp_div) < 1e-12, f"diversity is the mean pairwise distance ({exp_div:.3f})")
    # the top-100 cut matters only beyond 100 molecules; with fewer, all are used
    check(L.TOP_N == 100, "top-N is 100")
    try:
        L.population_metrics("median1", [], 0.0)
        check(False, "an empty population must raise")
    except ValueError:
        check(True, "an empty population raises rather than inventing metrics")


def test_notebook_parses_and_sweep_script_is_importable():
    import ast
    for rel in ("src/phase2_lib.py", "scripts/run_sweep_phase2.py", "src/build_phase2.py"):
        ast.parse((ROOT / rel).read_text())
        check(True, f"{rel} parses")
    sweep = (ROOT / "scripts" / "run_sweep_phase2.py").read_text()
    check("--workers" in sweep and "surviving_oracles" in sweep and "--force" in sweep,
          "the sweep CLI takes --workers and defaults to the gate's survivors")
    nb = json.loads((ROOT / "Phase 2" / "landscape_phase2.ipynb").read_text())
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    check(True, "every notebook code cell parses")


def _run_backbone_factory(fail_first=False):
    class P0(_FakeP0):
        calls = 0
        drd2_value = 0.4

        def get_tdc_oracle(self, name):
            return lambda s: self.drd2_value

        def run_backbone(self, backbone, orc, setting, seed, verbose=False):
            P0.calls += 1
            if fail_first and P0.calls == 1:
                raise RuntimeError("CUDA out of memory (simulated)")
            return {"auc_top10": 0.25}
    return P0()


def test_resume_only_trusts_complete_ok_records():
    import tempfile
    L = _lib()
    with tempfile.TemporaryDirectory() as td:
        old = (L._P0, L.RUNS2_DIR)
        L.RUNS2_DIR = Path(td)
        try:
            L._P0 = _run_backbone_factory(fail_first=True)
            r = L.execute_run2("median1", "graph_ga", 0)
            check(r["status"] == "error" and not L.record_complete(L.run_path2("median1", "graph_ga", 0)),
                  "an error record is not complete")
            r = L.execute_run2("median1", "graph_ga", 0)
            check(r["status"] == "ok" and L.record_complete(L.run_path2("median1", "graph_ga", 0)),
                  "an error record is retried on resume without --force")
            first = L.run_path2("median1", "graph_ga", 0).read_text()
            L._P0.calls = 0
            L.execute_run2("median1", "graph_ga", 0)
            check(L._P0.calls == 0 and L.run_path2("median1", "graph_ga", 0).read_text() == first,
                  "a complete ok record at full budget is not re-run")
            L.execute_run2("median1", "graph_ga", 1, budget=200)
            check(not L.record_complete(L.run_path2("median1", "graph_ga", 1)),
                  "a budget-200 record is not accepted as a real run")
            r = L.execute_run2("median1", "graph_ga", 1)
            check(r["budget"] == 1000, "and is overwritten by the full-budget run")
            L.run_path2("median1", "graph_ga", 2).write_text("{not json")
            check(not L.record_complete(L.run_path2("median1", "graph_ga", 2)), "a corrupt record is not complete")
        finally:
            L._P0, L.RUNS2_DIR = old


def test_dry_run_cannot_contaminate_the_real_directory():
    import tempfile
    L = _lib()
    with tempfile.TemporaryDirectory() as td:
        old = (L._P0, L.RUNS2_DIR)
        real = Path(td) / "real"
        L.RUNS2_DIR = real
        try:
            L._P0 = _run_backbone_factory()
            out = L.dry_run_check([("median1", "graph_ga", 0), ("median1", "reinvent", 0)])
            check(len(out) == 2 and all(r["budget"] == 200 for r in out), "the dry run executes at budget 200")
            check(L.RUNS2_DIR == real and not real.exists(), "the real runs directory is untouched and restored")
            L._P0.run_backbone = lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
            try:
                L.dry_run_check([("median1", "graph_ga", 0)])
            except KeyboardInterrupt:
                pass
            check(L.RUNS2_DIR == real and not real.exists(), "an interrupted dry run still restores the directory")
        finally:
            L._P0, L.RUNS2_DIR = old


def test_sweep_checks_the_environment_before_the_pool():
    import tempfile
    L = _lib()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        old = (L._P0, L.RUNS2_DIR, L.GATE2_PATH)
        L.RUNS2_DIR, L.GATE2_PATH = td / "runs", td / "gate.json"
        L.save_gate({"surviving": L.CANDIDATE_ORACLES, "excluded": {}, "enough": True}, L.GATE2_PATH)
        try:
            fake = _run_backbone_factory()
            fake.drd2_value = 0.0
            L._P0 = fake
            try:
                L.run_sweep2([("median1", "graph_ga", 0)], n_workers=1)
                check(False, "an all-zero drd2 must stop the sweep")
            except L.GateHalt:
                check(True, "the sklearn tripwire halts run_sweep2 in the main thread")
            check(not L.RUNS2_DIR.exists() and fake.calls == 0, "no run was started or written")
            L._P0 = None
            real_p0 = L.p0
            def broken():
                raise ImportError("no torch")
            L.p0 = broken
            try:
                L.run_sweep2([("median1", "graph_ga", 0)], n_workers=1)
                check(False, "a missing phase0_lib must stop the sweep")
            except ImportError:
                check(True, "a missing phase0_lib raises before the pool, not as 12 CRASHED lines")
            finally:
                L.p0 = real_p0
            L._P0 = _run_backbone_factory()
            jobs = [("median1", "graph_ga", 0), ("median1", "reinvent", 0)]
            L.run_sweep2(jobs, n_workers=2)
            check(L.sweep_failures(jobs) == [], "a healthy sweep leaves a complete ok record for every job")
            check(L.sweep_failures(jobs + [("median2", "graph_ga", 0)]) == [("median2", "graph_ga", 0)],
                  "a job with no record is reported as failed")
        finally:
            L._P0, L.RUNS2_DIR, L.GATE2_PATH = old


def test_cli_exits_nonzero_on_failed_jobs():
    import runpy
    import tempfile
    L = _lib()
    script = str(ROOT / "scripts" / "run_sweep_phase2.py")
    saved = (L.surviving_oracles, L.run_sweep2, L.RUNS2_DIR, sys.argv)
    with tempfile.TemporaryDirectory() as td:
        L.surviving_oracles = lambda: ["median1"]
        L.run_sweep2 = lambda *a, **k: None          # a sweep that "finishes" having written nothing
        L.RUNS2_DIR = Path(td)
        sys.argv = [script, "--backbone", "graph_ga", "--seed", "0"]
        try:
            try:
                runpy.run_path(script, run_name="__main__")
                check(False, "a sweep that wrote no records must exit non-zero")
            except SystemExit as e:
                check(e.code == 1, "the CLI exits 1 when jobs have no complete ok record")
            (Path(td) / "median1__graph_ga__explore_mid__seed0.json").write_text(
                json.dumps({"status": "error", "budget": 1000}))
            try:
                runpy.run_path(script, run_name="__main__")
                check(False, "an error record must exit non-zero")
            except SystemExit as e:
                check(e.code == 1, "the CLI exits 1 on an error record")
            (Path(td) / "median1__graph_ga__explore_mid__seed0.json").write_text(
                json.dumps({"status": "ok", "budget": 1000}))
            runpy.run_path(script, run_name="__main__")
            check(True, "the CLI exits 0 when every job has a complete ok record")
        finally:
            L.surviving_oracles, L.run_sweep2, L.RUNS2_DIR, sys.argv = saved


def test_verdict_cell_refuses_without_a_passed_overlap_gate():
    nb = json.loads((ROOT / "Phase 2" / "landscape_phase2.ipynb").read_text())
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "V2 = None" in "".join(c["source"]))
    refuse = cell.index('OVERLAP is None or OVERLAP["status"] != "passed"')
    check(refuse < cell.index("PS.phase2_verdict(") and refuse < cell.index("phase2_verdict.json")
          and '"REFUSED"' in cell[refuse:cell.index("PS.phase2_verdict(")],
          "the verdict cell refuses on a missing or unpassed overlap gate before computing or writing a verdict")
    check("phase2_overlap_gate.json" in "".join("".join(c["source"]) for c in nb["cells"]),
          "the notebook persists the overlap gate under Phase 2/")


PROBE_SHA = {
    "amlodipine_mpo": "a7b85142da575bbfc74e587b115f45d1fb62a434beec67b163032ed4b6f787ca",
    "fexofenadine_mpo": "9d4b92fc87b523c2dbd07a6c7bbfeac967b3d588e22a32169a8e9e247d2cfa7a",
    "median1": "cef7c73663dce0b9cbc5a04519396e0b29adc820a3f257d0ed5e160bf2521f80",
    "median2": "df223c6230750f2adc756304bb0226a3e5061305052ab99d98067b11b9304924",
    "osimertinib_mpo": "02a3d1fba18bfb90ac4f193c2bc99192759437d7db4da76a879becc7e80058af",
    "perindopril_mpo": "6fdd3f762d4ecf532f9b84df77dac58d7c854ee1382b7e7d0f205c1692adfa0a",
    "ranolazine_mpo": "1a891f23fc676e54c516092f45474666cbbaa514af59d984c84b94654b43877a",
    "sitagliptin_mpo": "6b613294da363062c12f91e5c094c6b89eff3c37cf04e721b113ad6ecebeee4d",
    "zaleplon_mpo": "28de1264390592c56a7f83dc5834f365b1ae29002c70f689226bc5baa9a47da5",
}
COMPONENTS_SHA = "c9bd0126888847a0248dd2dbfb943631d3b27300e1bc3f279373881df78c422a"


def test_gate_inputs_are_pinned():
    """The probe set, component table and gate-on-the-gate thresholds are 'frozen' by the spec but
    live outside the frozen JSON. Pin them here so a post-hoc edit cannot pass silently. The
    probe/component hashes depend on the RDKit canonicalisation of this environment; a legitimate
    RDKit upgrade that moves them must be reviewed, not waved through."""
    L = _lib()
    check(P.GATE_MIN_RANGE == 0.5, "GATE_MIN_RANGE is exactly 0.5")
    check(P.GATE_MIN_NONZERO == 0.20, "GATE_MIN_NONZERO is exactly 0.20")
    check(P.GATE_TOL == 1e-6 and P.MIN_ORACLES == 5, "gate tolerance and minimum survivors are pinned")
    ph = L.probe_hashes()
    for o in sorted(PROBE_SHA):
        check(ph[o] == PROBE_SHA[o], f"the frozen probe set for {o} is unchanged")
    check(L.components_hash() == COMPONENTS_SHA, "the component table (names, values on the probes) is unchanged")
    check(L.prereg2_hash() == "42243abd61d9d5e6d6ca7d2d0f935baf3ea46fcc74f9af545eb2f1e5d4b2dd3e",
          "the frozen pre-registration hash did not move")


def test_gate_file_is_bound_to_prereg_and_components():
    import tempfile
    L = _lib()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "gate.json"
        L.save_gate({"surviving": L.CANDIDATE_ORACLES, "excluded": {}, "enough": True}, path)
        g = json.loads(path.read_text())
        check(g["prereg_sha256"] == L.PREREG2_SHA256 and g["components_sha256"] == COMPONENTS_SHA
              and g["gate_min_range"] == 0.5 and g["gate_min_nonzero"] == 0.20,
              "the saved gate is stamped with the prereg hash, component hash and thresholds")
        check(L.surviving_oracles(path) == L.CANDIDATE_ORACLES, "a current gate file is accepted")
        for key, bad, frag in (("components_sha256", "0" * 64, "component table"),
                               ("prereg_sha256", "0" * 64, "pre-registration"),
                               ("gate_min_range", 0.3, "thresholds"),
                               ("gate_min_nonzero", 0.1, "thresholds")):
            h = dict(g); h[key] = bad
            path.write_text(json.dumps(h))
            try:
                L.surviving_oracles(path)
                check(False, f"a stale {key} must halt")
            except L.GateHalt as e:
                check(frag in str(e), f"a gate file with a stale {key} cannot govern a sweep")
        h = dict(g); h.pop("components_sha256")
        path.write_text(json.dumps(h))
        try:
            L.surviving_oracles(path)
            check(False, "an unstamped gate file must halt")
        except L.GateHalt:
            check(True, "an unstamped (pre-fix) gate file cannot govern a sweep")


def test_load_results_filters_stale_records():
    import tempfile
    L = _lib()
    with tempfile.TemporaryDirectory() as td:
        old = (L._P0, L.RUNS2_DIR)
        L.RUNS2_DIR = Path(td)
        try:
            L._P0 = _run_backbone_factory()
            L.execute_run2("median1", "graph_ga", 0)
            L.execute_run2("median1", "graph_ga", 1, budget=200)     # what `--budget 200` leaves behind
            L.execute_run2("median1", "graph_ga", 2)
            p = L.run_path2("median1", "graph_ga", 2)
            d = json.loads(p.read_text()); d["prereg_sha256"] = "f" * 64
            p.write_text(json.dumps(d))                              # a record from another rule
            df = L.load_results2()
            check(sorted(df["seed"]) == [0], "reduced-budget and foreign-prereg records never reach the frame")
        finally:
            L._P0, L.RUNS2_DIR = old
    surv = ["median1", "median2"]
    df = pd.DataFrame({"oracle": ["median1", "median2", "zaleplon_mpo"], "status": ["ok"] * 3})
    check(list(L.restrict_to_survivors(df, surv)["oracle"]) == surv,
          "a record for a gate-excluded oracle is removed before analysis")
    check(L.restrict_to_survivors(df.iloc[0:0], surv).empty, "an empty frame passes through")


def test_exclusion_path_is_reachable_and_counted_correctly():
    import tempfile
    L = _lib()
    seven = L.CANDIDATE_ORACLES[:7]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        old = (L._P0, L.RUNS2_DIR, L.GATE2_PATH)
        L.RUNS2_DIR, L.GATE2_PATH = td / "runs", td / "gate.json"
        L.save_gate({"surviving": seven, "excluded": {"x": "y", "z": "w"}, "enough": True}, L.GATE2_PATH)
        try:
            jobs = L.build_sweep2(L.surviving_oracles())
            check(len(jobs) == 7 * 3 * 15 == 315, "a 7-survivor sweep has 315 jobs, which is its expected total")
            L._P0 = _run_backbone_factory()
            L.run_sweep2(jobs[:2], n_workers=1)
            check(True, "survivor-derived jobs pass the stray-oracle check (the exclusion path runs)")
        finally:
            L._P0, L.RUNS2_DIR, L.GATE2_PATH = old
    nb = json.loads((ROOT / "Phase 2" / "landscape_phase2.ipynb").read_text())
    text = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    check("SWEEP2 = build_sweep2(SURVIVORS2)" in text and "SURVIVORS2 = surviving_oracles()" in text
          and "_n_exp = len(SWEEP2)" in text and "expected_oracles=SURVIVORS2" in text,
          "the notebook derives SWEEP2, the expected count and the overlap expectation from the survivors")


def test_builder_fails_loudly_and_artifacts_match_source():
    import build_phase2 as B
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # every cell parses alone; make ONLY the concatenated library fail to parse
        import ast
        real_parse = ast.parse

        def parse(text, *a, **k):
            if isinstance(text, str) and text.startswith('"""Phase 2 library'):
                raise SyntaxError("simulated concatenation failure")
            return real_parse(text, *a, **k)
        saved = (B.SRC, B.OUT, B.LIB, B.SWEEP)
        B.SRC, B.OUT, B.LIB, B.SWEEP = saved[0], td / "nb.ipynb", td / "lib.py", td / "sweep.py"
        ast.parse = parse
        try:
            rc = B.main()
        finally:
            ast.parse = real_parse
            B.SRC, B.OUT, B.LIB, B.SWEEP = saved
        check(rc == 1, "the builder returns 1 when the library cannot be written")
        check(not (td / "lib.py").exists() and not (td / "sweep.py").exists() and not (td / "nb.ipynb").exists(),
              "and writes nothing at all")
        B.SRC, B.OUT, B.LIB, B.SWEEP = saved[0], td / "nb.ipynb", td / "lib.py", td / "sweep.py"
        try:
            rc = B.main()
        finally:
            B.SRC, B.OUT, B.LIB, B.SWEEP = saved
        check(rc == 0, "a regeneration of the real source succeeds")
        for gen, committed in ((td / "lib.py", ROOT / "src" / "phase2_lib.py"),
                               (td / "nb.ipynb", ROOT / "Phase 2" / "landscape_phase2.ipynb"),
                               (td / "sweep.py", ROOT / "scripts" / "run_sweep_phase2.py")):
            check(gen.read_text() == committed.read_text(),
                  f"{committed.name} is byte-identical to a regeneration from phase2_source.py")


def test_power_gate_used_the_manifest_not_the_fallback():
    g = json.loads((ROOT / "Phase 2" / "phase2_power_gate.json").read_text())
    check(g["data_source"] == "manifest", f"the committed gate ran on the manifest (got {g['data_source']!r})")


def main():
    print("=" * 74)
    print("PHASE 2 STATISTICS GATE")
    print("=" * 74)
    for t in (test_hypervolume, test_pareto_front, test_scale_and_nan, test_common_unit,
              test_bootstrap_is_shared_across_backbones, test_pure_noise_has_no_stable_winner,
              test_exact_tie, test_criterion_0_covers_only_the_primary_pair,
              test_point_estimate_and_bootstrap_share_one_core,
              test_near_tie_with_nonzero_sd_is_a_tie, test_sparse_backbone_raises_no_warning,
              test_criterion_A, test_criterion_A_agreement_fails, test_metric_robust_end_to_end,
              test_bh_fdr,
              test_criterion_B_excludes_degenerate_before_correction, test_criterion_B_empty_bootstrap_is_degenerate, test_criterion_C_threshold,
              test_criterion_G2, test_verdict_branches, test_planted_disagreement_end_to_end,
              test_noise_gives_benchmark_insensitive_end_to_end,
              test_gate_logic, test_gate_summary_refuses_below_minimum,
              test_components_are_in_unit_range, test_component_arithmetic,
              test_probe_set_can_fail_the_gate, test_run_gate_with_fake_tdc,
              test_component_values_at_nontrivial_points, test_mutations_are_detected,
              test_power_simulation, test_choose_seeds_halts_when_nothing_is_reachable,
              test_build_artifacts, test_library_cells_are_not_notebook_only,
              test_gates_run_in_the_required_order, test_prior_phases_untouched,
              test_prereg_is_frozen_and_hashed, test_candidate_oracles_are_derived_and_cross_checked,
              test_sklearn_tripwire, test_phase1_overlap_gate,
              test_reproduction_gate_halts_below_minimum, test_gate_file_governs_the_sweep,
              test_runs_persist_the_full_buffer, test_population_metrics_hand_computed,
              test_notebook_parses_and_sweep_script_is_importable,
              test_resume_only_trusts_complete_ok_records,
              test_dry_run_cannot_contaminate_the_real_directory,
              test_sweep_checks_the_environment_before_the_pool,
              test_cli_exits_nonzero_on_failed_jobs,
              test_verdict_cell_refuses_without_a_passed_overlap_gate,
              test_gate_inputs_are_pinned, test_gate_file_is_bound_to_prereg_and_components,
              test_load_results_filters_stale_records,
              test_exclusion_path_is_reachable_and_counted_correctly,
              test_builder_fails_loudly_and_artifacts_match_source,
              test_power_gate_used_the_manifest_not_the_fallback):
        t()
    print("\n" + "=" * 74)
    print(f"{N} assertions passed")
    print("=" * 74)


if __name__ == "__main__":
    main()
