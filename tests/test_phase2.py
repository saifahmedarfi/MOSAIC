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


def main():
    print("=" * 74)
    print("PHASE 2 STATISTICS GATE")
    print("=" * 74)
    for t in (test_hypervolume, test_pareto_front, test_scale_and_nan, test_common_unit):
        t()
    print("\n" + "=" * 74)
    print(f"{N} assertions passed")
    print("=" * 74)


if __name__ == "__main__":
    main()
