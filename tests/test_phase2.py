#!/usr/bin/env python
"""Phase 2 statistics gate.

    python tests/test_phase2.py

Runs on numpy/scipy/pandas alone -- no torch, no rdkit, no TDC, no GPU, no oracle
downloads -- because every claim it checks lives in src/phase2_stats.py by design.
"""
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


def main():
    print("=" * 74)
    print("PHASE 2 STATISTICS GATE")
    print("=" * 74)
    for t in (test_hypervolume, test_pareto_front, test_scale_and_nan, test_common_unit,
              test_bootstrap_is_shared_across_backbones, test_pure_noise_has_no_stable_winner,
              test_exact_tie, test_criterion_0_covers_only_the_primary_pair,
              test_point_estimate_and_bootstrap_share_one_core,
              test_near_tie_with_nonzero_sd_is_a_tie, test_sparse_backbone_raises_no_warning):
        t()
    print("\n" + "=" * 74)
    print(f"{N} assertions passed")
    print("=" * 74)


if __name__ == "__main__":
    main()
