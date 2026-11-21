"""Phase 1 analysis: criteria A-D, the frozen verdict rule, figures and supporting tables.

HAND-WRITTEN -- this file is NOT a product of src/build_phase1.py. It is the exact
driver that produced results/phase1_verdict.json and every number in
PHASE1_RESULTS.md, archived here so the analysis is reproducible from the
repository alone.

Every statistical call is the one the notebook's pre-registered cells make, with the same
arguments read out of preregistration_phase1.json. No threshold, feature list, alpha, seed
set or oracle set is touched here.
"""
import json, sys, time, collections
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import phase1_lib as L
import phase1_stats as PS
import phase0_lib as P0

PREREG1 = L.PREREG1
OUT = P0.RESULTS_DIR
FIG = P0.FIG_DIR
FIG.mkdir(parents=True, exist_ok=True)
T = {}

print("=" * 78)
print("PRE-REGISTRATION GATE")
print("=" * 78)
CURRENT_HASH1 = L.prereg1_hash()
print("prereg status :", L.PREREG1_STATUS)
print("current hash  :", CURRENT_HASH1)
print("features hash :", PS.features_hash(), "(prereg:",
      PREREG1.get("diagnostic_features_sha256"), ")")
assert L.PREREG1_STATUS != "mismatch", "REFUSING: prereg hash mismatch"
assert PS.features_hash() == PREREG1["diagnostic_features_sha256"], \
    "REFUSING: diagnostic feature list differs from the frozen one"
T["prereg"] = {"status": L.PREREG1_STATUS, "sha256": CURRENT_HASH1,
               "features_sha256": PS.features_hash(),
               "gated_by": PREREG1.get("gated_by")}

RESULTS1 = L.load_results1()
print("\nloaded %d Phase 1 records" % len(RESULTS1))


def analysis_frame(df):
    """Drop failed runs and unassigned oracles before any cluster-level test."""
    if df.empty:
        return df
    d = df[df.get("status", "ok") == "ok"].copy()
    return d[d["cluster"] != "unassigned"]


d = analysis_frame(RESULTS1)
print("analysis frame: %d runs, %d oracles, clusters %s"
      % (len(d), d["oracle"].nunique(), sorted(d["cluster"].unique())))
T["inventory"] = {
    "n_records": int(len(RESULTS1)),
    "n_ok": int((RESULTS1["status"] == "ok").sum()),
    "n_analysis": int(len(d)),
    "confirmation_oracles": list(L.CONFIRMATION),
    "discovery_oracles": list(L.DISCOVERY),
    "all_pmo": list(L.ALL_PMO),
    "oracle_tags": {o: L.ORACLE_TAGS[o] for o in L.ALL_PMO},
    "cluster_counts_confirmation": {
        c: int(sum(1 for o in L.CONFIRMATION if L.ORACLE_TAGS[o] == c))
        for c in sorted({L.ORACLE_TAGS[o] for o in L.CONFIRMATION})},
    "cluster_counts_analysis": {k: int(v) for k, v in
                                d.groupby("cluster")["oracle"].nunique().items()},
}

# ----------------------------------------------------------------- Criterion A
print("\n" + "=" * 78)
print("CRITERION A -- does the flip survive on held-out oracles?")
print("=" * 78)
A1 = PS.criterion_A(d, PREREG1["backbones"],
                    n_boot=PREREG1["criterion_A_flip"]["n_bootstrap"])
for _bb, _pairs in A1["per_backbone"].items():
    print("\n%s  --  %d confirmed" % (_bb, A1["n_confirmed"][_bb]))
    print("  winners: %s" % (PS.winning_setting(d, _bb),))
    for _p in _pairs:
        print("  %s(%s) vs %s(%s): D=%+.4f CI=[%+.4f,%+.4f] repl=%.2f -> %s"
              % (_p["c1"], _p["e1"], _p["c2"], _p["e2"], _p["D"], _p["lo"], _p["hi"],
                 _p["replication"], "CONFIRMED" if _p["confirmed"] else _p["status"]))
print("\nCriterion A: %s (best backbone %s, needs >= %d pairs)"
      % ("PASS" if A1["passes"] else "FAIL", A1["best_backbone"], A1["min_pairs"]))
_tc = int(sum(1 for c, n in T["inventory"]["cluster_counts_analysis"].items() if n >= 2))
T["A"] = {"passes": bool(A1["passes"]), "min_pairs": A1["min_pairs"],
          "best_backbone": A1["best_backbone"],
          "n_confirmed": {k: int(v) for k, v in A1["n_confirmed"].items()},
          "per_backbone": A1["per_backbone"],
          "winners": {bb: PS.winning_setting(d, bb) for bb in PREREG1["backbones"]},
          "n_testable_clusters": _tc,
          "max_possible_pairs": int(_tc * (_tc - 1) // 2)}

# ----------------------------------------------------------------- Criterion B
print("\n" + "=" * 78)
print("CRITERION B -- does a cheap diagnostic track the winning setting?")
print("=" * 78)
DIAG1 = L.compute_all_diagnostics1(L.CONFIRMATION)
B1 = PS.criterion_B(d, DIAG1,
                    backbone=PREREG1["criterion_B_diagnostic"]["primary"]["backbone"])
print("n = %s oracles, backbone %s" % (B1["n_oracles"], B1["backbone"]))
print("\nPRIMARY  %s: rho = %+.3f, p = %.4f (one-sided %s, alpha = %s) -> %s"
      % (B1["primary"], B1["primary_rho"], B1["primary_p"], PS.PRIMARY_DIRECTION,
         PS.PRIMARY_ALPHA, "PASS" if B1["primary_pass"] else "FAIL"))
print("\nsecondary (Benjamini-Hochberg, q = %s):" % PS.SECONDARY_Q)
for _f, _r in sorted(B1["per_feature"].items(), key=lambda kv: kv[1].get("p", 1.0)):
    if _f == B1["primary"]:
        continue
    print("  %-22s rho %+.3f  p %.4f  %s"
          % (_f, _r.get("rho", float("nan")), _r.get("p", 1.0),
             "SIGNIFICANT" if _r.get("significant") else ""))
if B1["degenerate_features"]:
    print("\ndegenerate (excluded before correction): %s"
          % ", ".join(B1["degenerate_features"]))
print("\nCriterion B: %s" % ("PASS" if B1["passes"] else "FAIL"))
T["B"] = {k: B1[k] for k in B1}
T["diagnostics"] = DIAG1.to_dict(orient="records")

# ----------------------------------------------------------------- Criterion C
print("\n" + "=" * 78)
print("CRITERION C -- is the flip worth the probe? (the economics veto)")
print("=" * 78)
C1 = {bb: PS.criterion_C(d, bb,
                         probe_calls=PREREG1["criterion_C_economics"]["probe_calls"],
                         budget=PREREG1["budget"])
      for bb in PREREG1["backbones"]}
for _bb, _c in C1.items():
    if _c["status"] != "ok":
        print("%-18s insufficient data (%s)"
              % (_bb, _c["probe"].get("reason", _c["headroom"].get("status"))))
        continue
    _h, _p = _c["headroom"], _c["probe"]
    print("\n%s" % _bb)
    print("  best fixed setting     %s = %.4f" % (_h["best_fixed_setting"], _h["best_fixed_auc"]))
    print("  conditional (CV)       %.4f" % _h["conditional_auc"])
    print("  headroom               %+.4f  CI [%+.4f, %+.4f]" % (_h["headroom"], _h["lo"], _h["hi"]))
    print("  probe cost (300 calls) %+.4f  (AUC@1000 %.4f - AUC@700 %.4f)"
          % (_p["cost"], _p["auc_full"], _p["auc_short"]))
    print("  NET                    %+.4f  CI [%+.4f, %+.4f]  -> %s"
          % (_c["net"], _c["net_lo"], _c["net_hi"], "VIABLE" if _c["viable"] else "NOT VIABLE"))
_best_c = max((b for b in C1 if C1[b]["status"] == "ok"),
              key=lambda b: C1[b]["net"], default=None)
print("\nCriterion C: %s%s" % ("PASS" if _best_c and C1[_best_c]["viable"] else "FAIL",
                               (" (best %s)" % _best_c) if _best_c else ""))
T["C"] = C1
T["C_best_backbone"] = _best_c

# ----------------------------------------------------------------- Criterion D
print("\n" + "=" * 78)
print("CRITERION D -- selection across methods")
print("=" * 78)
D1 = PS.criterion_D(d)
for _g, _w in D1["winners"].items():
    _top = sorted(D1["per_group"][_g].items(), key=lambda kv: -kv[1])[:3]
    print("%-14s winner %-30s top3 %s"
          % (_g, str(_w), ", ".join("%s=%.3f" % (k, v) for k, v in _top)))
print("\n%d distinct winning arms -> %s"
      % (D1["n_distinct_winning_arms"],
         "METHOD-LEVEL FLIP" if D1["flips"] else "no method-level flip"))
T["D"] = D1

# ----------------------------------------------------------------- the verdict
print("\n" + "=" * 78)
V1 = PS.phase1_verdict(A1, B1, C1[_best_c] if _best_c else {"viable": False}, D1)
_n_ok = int((RESULTS1.get("status", "ok") == "ok").sum())
_n_exp = len(L.build_sweep1(L.CONFIRMATION))
V1.update(n_runs_ok=_n_ok, n_runs_expected=_n_exp,
          prereg_sha256=CURRENT_HASH1,
          decided_at=time.strftime("%Y-%m-%d %H:%M:%S"))
print("PHASE 1 VERDICT: %s" % V1["verdict"])
print("=" * 78)
print(V1["consequence"])
print("\nA %s  B %s  C %s  D %s" % (V1["criterion_A_passes"], V1["criterion_B_passes"],
                                    V1["criterion_C_viable"], V1["criterion_D_flips"]))
print("runs %d/%d (%.0f%% complete)" % (_n_ok, _n_exp, 100 * _n_ok / max(1, _n_exp)))
if _n_ok < _n_exp:
    print("INCOMPLETE SWEEP -- verdict is provisional.")

# The structural fact that makes A's FAIL non-informative, recorded ALONGSIDE the frozen
# verdict, never in place of it. The rule is applied exactly as frozen.
_testable = {bb: [p for p in ps if p.get("status") == "ok"]
             for bb, ps in A1["per_backbone"].items()}
V1["criterion_A_structural_note"] = {
    "testable_clusters_in_confirmation_set": T["A"]["n_testable_clusters"],
    "max_possible_confirmed_pairs": T["A"]["max_possible_pairs"],
    "min_pairs_required": A1["min_pairs"],
    "n_testable_pairs_per_backbone": {bb: len(ps) for bb, ps in _testable.items()},
    "testable_pairs": _testable,
    "note": ("Two facts, both recorded, neither used to alter the rule. (1) STRUCTURAL: "
             "the confirmation set admits at most %d cluster pair(s) with >=2 oracles "
             "(bioactivity holds 1 held-out oracle, so every pair involving it is "
             "'insufficient'), while the frozen threshold requires %d confirmed pairs -- "
             "so the threshold could not have been met by any data. (2) EMPIRICAL: the "
             "pair that WAS testable (mpo vs similarity) was tested and did not confirm "
             "on any backbone -- its bootstrap CI spans zero and its replication rate is "
             "far below the pre-registered 0.95. Criterion A therefore also fails on the "
             "evidence, so no amendment of min_pairs could have changed this verdict."
             % (T["A"]["max_possible_pairs"], A1["min_pairs"])),
}
V1["criteria_detail"] = {
    "A": {"n_confirmed": T["A"]["n_confirmed"], "winners": T["A"]["winners"],
          "pairs": T["A"]["per_backbone"]},
    "B": {"backbone": B1["backbone"], "n_oracles": B1["n_oracles"],
          "primary": B1["primary"], "primary_rho": B1["primary_rho"],
          "primary_p": B1["primary_p"], "primary_pass": bool(B1["primary_pass"]),
          "degenerate_features": B1["degenerate_features"],
          "per_feature": B1["per_feature"]},
    "C": {bb: ({"status": c["status"]} if c["status"] != "ok" else
               {"status": "ok", "headroom": c["headroom"], "probe": c["probe"],
                "net": c["net"], "net_lo": c["net_lo"], "net_hi": c["net_hi"],
                "viable": bool(c["viable"])}) for bb, c in C1.items()},
    "D": {"winners": D1["winners"], "n_distinct_winning_arms": D1["n_distinct_winning_arms"],
          "distinct_arms": D1["distinct_arms"], "flips": bool(D1["flips"])},
}
(OUT / "phase1_verdict.json").write_text(json.dumps(V1, indent=2, default=str),
                                         encoding="utf-8")
print("\nwritten -> %s" % (OUT / "phase1_verdict.json"))
T["verdict"] = V1

# ----------------------------------------------------------------- supporting tables
print("\n" + "=" * 78)
print("SUPPORTING TABLES")
print("=" * 78)
cell = d.groupby(["backbone", "cluster", "setting"])["auc_top10"].agg(["mean", "std", "count"])
print("\n--- mean AUC-top-10 by backbone x cluster x setting (analysis frame) ---")
print(cell.round(4).to_string())
T["cell_table"] = json.loads(cell.reset_index().to_json(orient="records"))

full = L.load_results1()
per_oracle = full.groupby(["oracle", "cluster", "backbone"])["auc_top10"].mean().unstack()
print("\n--- mean AUC-top-10 per oracle (3 seeds x 3 settings per backbone cell) ---")
print(per_oracle.round(4).to_string())
T["per_oracle"] = json.loads(per_oracle.reset_index().to_json(orient="records"))

po_set = full.pivot_table(index=["oracle", "cluster"], columns=["backbone", "setting"],
                          values="auc_top10", aggfunc="mean")
T["per_oracle_setting"] = json.loads(po_set.reset_index().to_json(orient="records"))

bb_overall = full.groupby("backbone")["auc_top10"].agg(["mean", "std", "min", "max", "count"])
print("\n--- overall by backbone (all 405 runs, unassigned included) ---")
print(bb_overall.round(4).to_string())
T["backbone_overall"] = json.loads(bb_overall.reset_index().to_json(orient="records"))

st_overall = full.groupby("setting")["auc_top10"].agg(["mean", "std", "count"])
print("\n--- overall by setting ---")
print(st_overall.round(4).to_string())
T["setting_overall"] = json.loads(st_overall.reset_index().to_json(orient="records"))

bs = full.groupby(["backbone", "setting"])["auc_top10"].mean().unstack()
print("\n--- backbone x setting (arm means) ---")
print(bs.round(4).to_string())
T["arm_means"] = json.loads(bs.reset_index().to_json(orient="records"))

print("\n--- winning setting per (oracle, backbone), full 15-oracle set ---")
wins = {}
for (o, bb), g in full.groupby(["oracle", "backbone"]):
    m = g.groupby("setting")["auc_top10"].mean().sort_values(ascending=False)
    wins["%s|%s" % (o, bb)] = m.index[0] if (len(m) == 1 or m.iloc[0] != m.iloc[1]) else None
print(dict(collections.Counter(wins.values())))
T["per_oracle_winners"] = wins
T["winner_counts"] = {str(k): int(v) for k, v in collections.Counter(wins.values()).items()}

# seed-level dispersion: how much of the knob effect is noise
sd = full.groupby(["oracle", "backbone", "setting"])["auc_top10"].std()
print("\nwithin-cell seed sd: median %.4f  mean %.4f  max %.4f"
      % (sd.median(), sd.mean(), sd.max()))
spread = (full.groupby(["oracle", "backbone"])["auc_top10"].max()
          - full.groupby(["oracle", "backbone"])["auc_top10"].min())
print("per-(oracle,backbone) setting spread: median %.4f" % spread.median())
T["noise"] = {"within_cell_seed_sd_median": float(sd.median()),
              "within_cell_seed_sd_mean": float(sd.mean()),
              "within_cell_seed_sd_max": float(sd.max()),
              "setting_spread_median": float(spread.median())}

# ----------------------------------------------------------------- figures
print("\n" + "=" * 78)
print("FIGURES")
print("=" * 78)
CL_COLORS = {"similarity": "#2563eb", "mpo": "#d97706", "bioactivity": "#16a34a",
             "isomer": "#dc2626", "unassigned": "#94a3b8"}
BB_COLORS = {"reinvent": "#2563eb", "augmented_memory": "#d97706", "graph_ga": "#16a34a"}


def fig1():
    bbs = [b for b in PREREG1["backbones"] if (d["backbone"] == b).any()]
    fig, axes = plt.subplots(1, len(bbs), figsize=(5 * len(bbs), 4), sharey=True)
    for ax, bb in zip(np.atleast_1d(axes), bbs):
        for cl in sorted(d["cluster"].unique()):
            ys = [PS._cell(d, bb, cl, s) for s in PS.SETTINGS]
            ax.plot(range(3), ys, marker="o", label=cl, color=CL_COLORS.get(cl))
        ax.set_xticks(range(3))
        ax.set_xticklabels([s.replace("explore_", "") for s in PS.SETTINGS])
        ax.set_title(bb)
        ax.set_xlabel("exploration setting")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("mean AUC-top-10")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle("Criterion A: non-parallel lines are the flip", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "phase1_setting_by_cluster.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig2():
    ok = [(b, c) for b, c in C1.items() if c["status"] == "ok"]
    if not ok:
        print("  skip economics figure: no backbone had sufficient data")
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for i, (bb, c) in enumerate(ok):
        h, p = c["headroom"], c["probe"]
        ax.errorbar(i, h["headroom"],
                    yerr=[[h["headroom"] - h["lo"]], [h["hi"] - h["headroom"]]],
                    marker="o", capsize=4, color="#2563eb",
                    label="headroom (held-out-seed CV)" if i == 0 else None)
        ax.scatter(i, p["cost"], marker="_", s=400, color="#dc2626",
                   label="probe cost (300 calls)" if i == 0 else None)
        ax.annotate("net %+.3f" % c["net"], (i, max(h["hi"], p["cost"])),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels([b for b, _ in ok])
    ax.set_ylabel("AUC-top-10")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Criterion C: headroom must clear the probe's cost")
    fig.tight_layout()
    fig.savefig(FIG / "phase1_economics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig3():
    feats = [f for f in ["gini", "reward_mean", "reward_std", "dynamic_range",
                         "ruggedness_corr", "basin_width", "neighbor_gain", "q90_gap"]
             if f in DIAG1.columns]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, f in zip(axes.ravel(), feats):
        for cl in sorted(DIAG1["cluster"].unique()):
            v = DIAG1.loc[DIAG1["cluster"] == cl, f].dropna()
            ax.scatter([cl] * len(v), v, s=30, color=CL_COLORS.get(cl), alpha=0.85)
        ax.set_title(f, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Phase 1 landscape diagnostics by cluster "
                 "(300 budgeted calls x 3 probe seeds x 15 confirmation oracles)")
    fig.tight_layout()
    fig.savefig(FIG / "phase1_diagnostics_by_cluster.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig4():
    bb = PREREG1["criterion_B_diagnostic"]["primary"]["backbone"]
    sub = full[full["backbone"] == bb]
    piv = sub.pivot_table(index="oracle", columns="setting", values="auc_top10",
                          aggfunc="mean").reindex(columns=list(PS.SETTINGS))
    tags = {o: L.ORACLE_TAGS[o] for o in piv.index}
    piv = piv.loc[sorted(piv.index, key=lambda o: (tags[o], o))]
    x = np.arange(len(piv))
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, s in enumerate(PS.SETTINGS):
        ax.bar(x + (i - 1) * w, piv[s].values, w, label=s.replace("explore_", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(%s)" % (o, tags[o]) for o in piv.index],
                       rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("mean AUC-top-10 (3 seeds)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, title="setting")
    ax.set_title("Per-oracle mean AUC-top-10 -- %s, 15 confirmation oracles" % bb)
    fig.tight_layout()
    fig.savefig(FIG / ("phase1_per_oracle_%s.png" % bb), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig5():
    bb = B1["backbone"]
    prim = B1["primary"]
    sub = full[full["backbone"] == bb]
    idx = {}
    for o, g in sub.groupby("oracle"):
        m = g.groupby("setting")["auc_top10"].mean()
        idx[o] = int(np.argmax([m.get(s, -np.inf) for s in PS.SETTINGS]))
    dd = DIAG1.set_index("oracle")
    xs, ys, cs, ns = [], [], [], []
    for o, i in idx.items():
        if o in dd.index and np.isfinite(dd.loc[o, prim]):
            xs.append(float(dd.loc[o, prim]))
            ys.append(i)
            cs.append(CL_COLORS.get(dd.loc[o, "cluster"], "#94a3b8"))
            ns.append(o)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(xs, ys, c=cs, s=55)
    for xi, yi, n in zip(xs, ys, ns):
        ax.annotate(n, (xi, yi), fontsize=6.5, textcoords="offset points", xytext=(4, 4))
    ax.set_yticks(range(3))
    ax.set_yticklabels([s.replace("explore_", "") for s in PS.SETTINGS])
    ax.set_xlabel("%s (diagnostic probe, 300 calls)" % prim)
    ax.set_ylabel("winning setting")
    ax.set_title("Criterion B primary: %s vs winning setting (%s)\nrho = %+.3f, p = %.4f"
                 % (prim, bb, B1["primary_rho"], B1["primary_p"]))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "phase1_criterionB_primary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig6():
    hist = pd.read_csv(OUT / "phase1_training_history.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    grid = np.arange(0, 1001, 25)
    for ax, bb in zip(axes, PREREG1["backbones"]):
        h = hist[hist["backbone"] == bb]
        for s in PS.SETTINGS:
            curves = []
            for _, g in h[h["setting"] == s].groupby(["oracle", "seed"]):
                g = g.sort_values("n_oracle_calls")
                if len(g) < 2:
                    continue
                curves.append(np.interp(grid, g["n_oracle_calls"], g["top10_mean"]))
            if curves:
                m = np.mean(curves, axis=0)
                sdv = np.std(curves, axis=0)
                ax.plot(grid, m, label=s.replace("explore_", ""))
                ax.fill_between(grid, m - sdv, m + sdv, alpha=0.15)
        ax.axvline(700, ls="--", lw=0.9, color="#dc2626")
        ax.annotate("probe-cost\nboundary (700)", (690, 0.03), fontsize=7,
                    color="#dc2626", ha="right")
        med = int(h.groupby(["oracle", "setting", "seed"])["step"].max().median())
        ax.set_title("%s  (median %d update steps)" % (bb, med))
        ax.set_xlabel("oracle calls")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("running top-10 mean score")
    axes[-1].legend(fontsize=8, title="setting")
    fig.suptitle("Optimisation trajectories across the 405 Phase 1 runs "
                 "(mean +/- 1 sd over oracles x seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "phase1_training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig7():
    mani = pd.read_csv(OUT / "phase1_run_manifest.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    _data = [full.loc[full["backbone"] == b, "auc_top10"].values
             for b in PREREG1["backbones"]]
    try:                                 # matplotlib >= 3.9
        axes[0].boxplot(_data, tick_labels=PREREG1["backbones"], showmeans=True)
    except TypeError:                    # matplotlib < 3.9
        axes[0].boxplot(_data, labels=PREREG1["backbones"], showmeans=True)
    axes[0].set_ylabel("AUC-top-10")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_title("AUC spread across all 405 runs, by backbone")
    for b in PREREG1["backbones"]:
        m = mani[mani["backbone"] == b]
        axes[1].scatter(m["wall_s"], m["auc_top10"], s=16, alpha=0.7, label=b,
                        color=BB_COLORS.get(b))
    axes[1].set_xscale("log")
    axes[1].set_xlabel("wall clock per run (s, log scale)")
    axes[1].set_ylabel("AUC-top-10")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Cost vs outcome (12 workers sharing one GPU)")
    fig.tight_layout()
    fig.savefig(FIG / "phase1_auc_spread.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig8():
    """Training dynamics that explain the knob: entropy and scaffold diversity."""
    hist = pd.read_csv(OUT / "phase1_training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    grid = np.arange(0, 1001, 25)
    for col, ax, lab in [("policy_entropy", axes[0], "policy entropy"),
                         ("scaffold_entropy", axes[1], "scaffold entropy")]:
        for s in PS.SETTINGS:
            curves = []
            for _, g in hist[(hist["setting"] == s)
                             & (hist["backbone"] != "graph_ga")].groupby(
                                 ["oracle", "backbone", "seed"]):
                g = g.sort_values("n_oracle_calls").dropna(subset=[col])
                if len(g) < 2:
                    continue
                curves.append(np.interp(grid, g["n_oracle_calls"], g[col]))
            if curves:
                ax.plot(grid, np.mean(curves, axis=0), label=s.replace("explore_", ""))
        ax.set_xlabel("oracle calls")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="setting")
        ax.set_title("%s vs budget (LM backbones)" % lab)
    fig.tight_layout()
    fig.savefig(FIG / "phase1_training_dynamics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
    try:
        fn()
        print("  ok   %s" % fn.__name__)
    except Exception as e:
        import traceback
        print("  FAIL %s: %s" % (fn.__name__, e))
        traceback.print_exc()

(OUT / "phase1_tables.json").write_text(
    json.dumps(T, indent=2, default=str), encoding="utf-8")
print("\ntables -> %s" % (OUT / "phase1_tables.json"))
print("\nVERDICT1_OK")
