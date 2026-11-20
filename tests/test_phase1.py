#!/usr/bin/env python
"""Phase 1 statistics gate.

    python tests/test_phase1.py

Runs on numpy/scipy/pandas alone -- no torch, no rdkit, no TDC, no GPU, no oracle
downloads -- because every claim it checks lives in src/phase1_stats.py by design.

MUST pass before any Phase 1 sweep. Phase 0 shipped two degenerate features and screened
its diagnostic on the wrong backbone; both are the kind of thing a gate like this catches.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import phase1_stats as P  # noqa: E402

N = 0


def check(cond, label):
    global N
    N += 1
    if not cond:
        raise AssertionError(f"FAILED: {label}")
    print(f"  ok  {label}")


def frame(rows):
    return pd.DataFrame(rows)


def synth(winners, backbone="augmented_memory", seeds=(0, 1, 2), noise=0.0,
          clusters=None, n_oracles=2, rng=None):
    """Synthetic sweep frame where `winners[cluster]` is the genuinely best setting."""
    rng = rng or np.random.default_rng(0)
    rows = []
    for cl, best in winners.items():
        for k in range(n_oracles):
            o = f"{cl}_{k}"
            for st in P.SETTINGS:
                for s in seeds:
                    base = 0.50 if st == best else 0.40
                    rows.append({"oracle": o, "cluster": clusters or cl,
                                 "backbone": backbone, "setting": st, "seed": s,
                                 "auc_top10": base + noise * rng.standard_normal()})
    return frame(rows)


# ----------------------------------------------------------------- Task 1: features
def test_features():
    print("\n[1] feature redesign")
    f = P.quantile_features([0.3] * 50)
    check(all(np.isfinite(v) for v in f.values()),
          "constant reward vector yields no NaN (Phase 0 shipped 2 NaN features)")
    check(f["reward_skew"] == 0.0 and f["reward_kurtosis"] == 0.0,
          "moments of a constant vector are 0, not NaN")

    check(abs(P.gini(np.ones(100))) < 1e-9, "gini of a uniform vector is ~0")
    spike = np.zeros(100)
    spike[0] = 1.0
    check(P.gini(spike) > 0.98, "gini of a single spike is ~1")
    check(P.gini(np.zeros(10)) == 1.0, "all-zero reward is maximally concentrated")

    check("frac_above_0.7" not in P.FEATURES and "frac_above_0.9" not in P.FEATURES,
          "Phase 0's degenerate absolute thresholds are gone")
    check(all(f"q{int(q*100)}_gap" in P.FEATURES for q in P.QUANTILES),
          "quantile gap features present for every breakpoint")

    h1 = P.features_hash(("a", "b"))
    check(h1 != P.features_hash(("b", "a")), "features_hash is order-sensitive")
    check(h1 != P.features_hash(("a", "b", "c")), "features_hash covers membership")
    check(len(P.features_hash()) == 64, "features_hash is a sha256 hex digest")

    # Reward vectors of wildly different scale must not collapse the gap features.
    small = P.quantile_features(np.linspace(0, 0.01, 200))
    big = P.quantile_features(np.linspace(0, 100.0, 200))
    check(abs(small["q50_gap"] - big["q50_gap"]) < 1e-6,
          "quantile gaps are scale-free")


# -------------------------------------------------------------- Task 2: the flip test
def test_flip():
    print("\n[2] Criterion A -- the flip test")
    planted = synth({"isomer": "explore_low", "mpo": "explore_mid"}, noise=0.01)
    pairs = P.flip_pairs(planted, "augmented_memory", n_boot=2000)
    check(len(pairs) == 1 and pairs[0]["confirmed"],
          "a planted flip is confirmed")
    check(pairs[0]["status"] == "ok", "planted flip has status ok")

    rng = np.random.default_rng(7)
    confirmed = 0
    for _ in range(40):
        noise_only = synth({"a": "explore_low", "b": "explore_low"},
                           noise=0.05, rng=rng)
        noise_only["auc_top10"] = 0.45 + 0.05 * rng.standard_normal(len(noise_only))
        confirmed += sum(1 for p in P.flip_pairs(noise_only, "augmented_memory",
                                                 n_boot=500, seed=int(rng.integers(1e6)))
                         if p["confirmed"])
    check(confirmed == 0, f"pure noise confirms no flip in 40 trials (got {confirmed})")

    thin = synth({"isomer": "explore_low", "mpo": "explore_mid"},
                 noise=0.01, n_oracles=1)
    tp = P.flip_pairs(thin, "augmented_memory", n_boot=500)
    check(tp and tp[0]["status"] == "insufficient" and not tp[0]["confirmed"],
          "a 1-oracle cluster reports insufficient and is never confirmed")

    tie = frame([{"oracle": f"{cl}_0", "cluster": cl, "backbone": "x",
                  "setting": st, "seed": s, "auc_top10": 0.5}
                 for cl in ("a", "b") for st in P.SETTINGS for s in (0, 1, 2)])
    check(P.winning_setting(tie, "x")["a"] is None,
          "an exact three-way tie yields no winner (Phase 0 took argmax silently)")
    check(P.flip_pairs(tie, "x", n_boot=100) == [],
          "a tie produces no flip candidates")


# ------------------------------------------------------- Task 3: diagnostics vs winner
def test_criterion_b():
    print("\n[3] Criterion B -- primary hypothesis + BH-FDR")
    from scipy.stats import spearmanr
    x = np.array([1., 2, 3, 4, 5, 6, 7, 8])
    y = np.array([2., 1, 4, 3, 6, 5, 8, 7])
    rho, p = P.exact_spearman(x, y)
    check(abs(rho - spearmanr(x, y).statistic) < 1e-12,
          "exact_spearman rho matches scipy at n=8")
    check(0 < p < 1, "exact permutation p is a probability")

    _, p_two = P.exact_spearman(x, y, alternative="two-sided")
    _, p_gt = P.exact_spearman(x, y, alternative="greater")
    check(p_gt < p_two, "one-sided p is smaller than two-sided for a positive rho")

    pv = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.9}
    bh = P.bh_fdr(pv, q=0.05)
    holm_sig = sum(1 for i, (k, v) in enumerate(sorted(pv.items(), key=lambda kv: kv[1]))
                   if v <= 0.05 / (len(pv) - i))
    check(sum(1 for r in bh.values() if r["significant"]) >= holm_sig,
          "BH rejects at least as much as Holm on the same p-values")

    withnan = dict(pv, dead=float("nan"))
    bh2 = P.bh_fdr(withnan, q=0.05)
    check(bh2["dead"]["degenerate"] and bh2["dead"]["rank"] is None,
          "a NaN feature is excluded from the FDR denominator, not counted as a test")
    check(bh2["a"]["threshold"] == bh["a"]["threshold"],
          "excluding a degenerate feature leaves other thresholds unchanged")

    # gini is declared NEGATIVE: a positive correlation must not pass however small p is.
    oracles = [f"o{i}" for i in range(8)]
    diag = frame([{"oracle": o, "cluster": "c", "gini": float(i)}
                  for i, o in enumerate(oracles)])
    rows = []
    for i, o in enumerate(oracles):
        best = P.SETTINGS[min(2, i // 3)]          # gini UP with setting index: positive
        for st in P.SETTINGS:
            for s in (0, 1, 2):
                rows.append({"oracle": o, "cluster": "c", "backbone": "augmented_memory",
                             "setting": st, "seed": s,
                             "auc_top10": 0.5 if st == best else 0.4})
    B = P.criterion_B(frame(rows), diag, features=("gini",))
    check(B["primary_rho"] > 0 and not B["primary_pass"],
          "a positive rho fails a negative-direction primary regardless of p")


# ----------------------------------------------- Task 4: headroom vs the probe's cost
def test_criterion_c():
    print("\n[4] Criterion C -- headroom vs probe cost")
    trace = [(0.42, c) for c in range(1, 1001)]
    check(abs(P.auc_at(trace, 1000) - 0.4179) < 2e-3,
          "constant-0.42 run scores ~0.4179 (the ramp-in artifact, reproduced)")
    check(P.auc_at(trace, 700) <= P.auc_at(trace, 1000) + 1e-12,
          "auc_at is monotone in budget for a constant trace")
    rising = [(c / 1000.0, c) for c in range(1, 1001)]
    check(P.auc_at(rising, 300) < P.auc_at(rising, 1000),
          "a rising trace scores higher at a longer budget")
    check(P.auc_at([], 1000) == 0.0, "an empty trace scores 0, not NaN")

    flat = synth({"a": "explore_low", "b": "explore_low"}, noise=0.0)
    flat["auc_top10"] = 0.45
    H = P.cv_headroom(flat, "augmented_memory", n_boot=500)
    check(abs(H["headroom"]) < 1e-9,
          "setting-independent data has zero cross-validated headroom")

    rng = np.random.default_rng(3)
    noisy = synth({"a": "explore_low", "b": "explore_mid"}, noise=0.08, rng=rng)
    noisy["auc_top10"] = 0.45 + 0.08 * rng.standard_normal(len(noisy))
    cv = P.cv_headroom(noisy, "augmented_memory", n_boot=500)
    data = P._oracle_setting_seed(noisy, "augmented_memory")
    naive_cond = np.mean([max(np.mean(list(v.values())) for v in d.values())
                          for d in data.values()])
    fixed = max(np.mean([np.mean(list(d[st].values())) for d in data.values()])
                for st in P.SETTINGS)
    check(cv["headroom"] < naive_cond - fixed,
          "cross-validated headroom is below the naive winner's-curse estimate")

    two = synth({"a": "explore_low", "b": "explore_mid"}, seeds=(0, 1))
    check(P.cv_headroom(two, "augmented_memory", n_boot=100)["status"] == "insufficient",
          "2 seeds reports insufficient rather than fitting and scoring on one seed")

    no_trace = synth({"a": "explore_low", "b": "explore_mid"})
    pc = P.probe_cost(no_trace, "augmented_memory")
    check(pc["status"] == "unavailable" and pc["approximate"],
          "a record without score_trace is flagged, never silently treated as exact")

    with_trace = no_trace.copy()
    with_trace["score_trace"] = [trace] * len(with_trace)
    pc2 = P.probe_cost(with_trace, "augmented_memory")
    check(pc2["status"] == "ok" and pc2["cost"] >= 0,
          "probe cost is computable and non-negative when score_trace is present")

    C = P.criterion_C(with_trace, "augmented_memory", n_boot=300)
    check(C["status"] == "ok" and "net" in C, "criterion_C returns a net figure")
    check(C["viable"] == bool(C["headroom"]["lo"] - C["probe"]["cost"] > 0),
          "viability is the CI lower bound clearing the probe cost")


# --------------------------------------------------- Task 5: Criterion D and the verdict
def test_criterion_d_and_verdict():
    print("\n[5] Criterion D and the verdict")
    rows = []
    for cl, arm in (("isomer", ("graph_ga", "explore_low")),
                    ("mpo", ("augmented_memory", "explore_mid"))):
        for k in range(2):
            for bb in ("reinvent", "augmented_memory", "graph_ga"):
                for st in P.SETTINGS:
                    for s in (0, 1, 2):
                        rows.append({"oracle": f"{cl}_{k}", "cluster": cl,
                                     "backbone": bb, "setting": st, "seed": s,
                                     "auc_top10": 0.6 if (bb, st) == arm else 0.4})
    D = P.criterion_D(frame(rows), n_boot=200)
    check(D["winners"]["isomer"] == "graph_ga:explore_low",
          "criterion_D's action space is (backbone, setting) pairs, not settings")
    check(D["n_distinct_winning_arms"] == 2 and D["flips"],
          "a method-level flip across clusters is detected")

    ok = {"passes": True}
    no = {"passes": False}
    check(P.phase1_verdict(no, ok, {"viable": True}, {})["verdict"] == "NULL",
          "A failing forces NULL")
    check(P.phase1_verdict(ok, ok, {"viable": False}, {})["verdict"]
          == "CHARACTERISATION-ONLY",
          "the economics veto: A and B pass, C not viable -> characterisation only")
    check(P.phase1_verdict(ok, no, {"viable": True}, {})["verdict"]
          == "CHARACTERISATION-ONLY",
          "B failing with viable economics is still characterisation only")
    check(P.phase1_verdict(ok, ok, {"viable": True}, {})["verdict"]
          == "CONTROLLER-VIABLE",
          "CONTROLLER-VIABLE requires A, B and C together")
    check(P.phase1_verdict(ok, ok, {"viable": True}, {})["features_sha256"]
          == P.features_hash(),
          "the verdict records the computed feature hash")


# ------------------------------------------------ Task 6: build and prereg integrity
def test_build_artifacts():
    print("\n[6] build and pre-registration integrity")
    import ast
    import json

    root = Path(__file__).resolve().parent.parent
    src = root / "src" / "phase1_source.py"
    nb = root / "Phase 1" / "landscape_phase1.ipynb"
    lib = root / "src" / "phase1_lib.py"
    sweep = root / "scripts" / "run_sweep_phase1.py"

    check(src.exists(), "src/phase1_source.py (source of truth) exists")
    check(nb.exists(), "Phase 1/landscape_phase1.ipynb was built")
    check(lib.exists(), "src/phase1_lib.py was generated")
    check(sweep.exists(), "scripts/run_sweep_phase1.py was generated")

    doc = json.loads(nb.read_text())
    check(doc["nbformat"] == 4 and len(doc["cells"]) > 20,
          f"notebook is nbformat 4 with {len(doc['cells'])} cells")
    code = [c for c in doc["cells"] if c["cell_type"] == "code"]
    bad = []
    for i, c in enumerate(code):
        try:
            ast.parse("".join(c["source"]))
        except SyntaxError:
            bad.append(i)
    check(not bad, f"all {len(code)} notebook code cells parse")
    check(all(c["outputs"] == [] for c in code),
          "the committed notebook carries no stale outputs")

    sys.path.insert(0, str(root / "src"))
    import build_phase1 as B
    cells = B.parse_percent(src.read_text(encoding="utf-8"))
    check(B.check_cells(cells) == 0, "every source cell parses in isolation")
    nb_only = [c for c in cells if c[0] == "code" and c[2]]
    check(len(nb_only) >= 5,
          f"{len(nb_only)} driver cells are @notebook-only (excluded from the library)")
    libtext = lib.read_text()
    check("LAUNCH_SWEEP1" not in libtext,
          "the library cannot launch a sweep on import")
    check("DRY_RUN_FIRST" not in libtext,
          "the library cannot run a dry check on import")
    check("def execute_run1" in libtext and "def run_sweep1" in libtext,
          "the library exports the runner the sweep script imports")
    check("import phase1_stats as PS" in libtext,
          "the library takes its statistics from phase1_stats")

    # phase0 must be untouched: it is the artifact Phase 0's verdict was produced under.
    import subprocess
    diff = subprocess.run(["git", "diff", "--name-only", "HEAD", "--",
                           "src/phase0_lib.py", "src/phase0_source.py"],
                          cwd=root, capture_output=True, text=True).stdout.strip()
    check(diff == "", "src/phase0_lib.py and phase0_source.py are unmodified")

    # The prereg hash must respond to the COMPUTED feature tuple, Phase 0's integrity gap.
    import hashlib
    payload = {"diagnostic_features_sha256": P.features_hash(P.FEATURES)}
    h_before = hashlib.sha256(P.canonical_json(payload).encode()).hexdigest()
    payload2 = {"diagnostic_features_sha256": P.features_hash(P.FEATURES[:-1])}
    h_after = hashlib.sha256(P.canonical_json(payload2).encode()).hexdigest()
    check(h_before != h_after,
          "dropping a computed feature changes the prereg hash (Phase 0's gap, closed)")

    tags = src.read_text(encoding="utf-8")
    check("unassigned" in tags and "CLUSTER_RULES" in tags,
          "cluster tags are rule-based with an explicit 'unassigned' escape")
    check("hand-type" in tags or "hand-typed" in tags or "do NOT hand-type" in tags,
          "the oracle list is derived, never hand-typed")


def main():
    print("=" * 74)
    print("PHASE 1 STATISTICS GATE")
    print("=" * 74)
    for t in (test_features, test_flip, test_criterion_b,
              test_criterion_c, test_criterion_d_and_verdict, test_build_artifacts):
        t()
    print("\n" + "=" * 74)
    print(f"{N} assertions passed")
    print("=" * 74)


if __name__ == "__main__":
    main()
