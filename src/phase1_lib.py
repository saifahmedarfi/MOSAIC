"""Phase 1 library -- GENERATED from src/phase1_source.py. Do not edit.

Regenerate with:  python src/build_phase1.py

Importing this module defines the Phase 1 pre-registration, cluster tags, diagnostics,
runner and analyses. Driver cells are excluded, so an import has no side effects beyond
what phase0_lib does on import (vendoring mol_opt and loading the prior, both cached after
Phase 0's first run).
"""


import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("LCE_ROOT", Path.cwd())).resolve()
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))
os.environ["LCE_ROOT"] = str(ROOT)

import phase1_stats as PS

print(f"project root     {ROOT}")
print(f"feature set      {len(PS.FEATURES)} features, sha256 {PS.features_hash()[:16]}...")
print(f"primary feature  {PS.PRIMARY_FEATURE} ({PS.PRIMARY_DIRECTION}, alpha={PS.PRIMARY_ALPHA})")


PREREG1 = {
    "version": "2.0.0",
    "phase": 1,
    "gated_by": {"phase": 0, "verdict": "PARTIAL",
                 "prereg_sha256": "fca4982097668bd980197cf9f067c474d500434357331975b0d93c82a0ee329c"},
    "frozen_on": time.strftime("%Y-%m-%d"),
    "budget": 1000,
    "seeds": [0, 1, 2],
    "settings": list(PS.SETTINGS),
    "backbones": ["reinvent", "augmented_memory", "graph_ga"],
    "primary_metric": "auc_top10",
    "diagnostic_budget": 300,
    "diagnostic_features": list(PS.FEATURES),
    "diagnostic_features_sha256": PS.features_hash(),

    "criterion_A_flip": {
        "statement": "At least two landscape clusters have different winning exploration "
                     "settings, within one backbone, and the difference generalises across "
                     "oracles of the same type.",
        "test": "Interaction contrast D = [AUC(c1,e1)-AUC(c1,e2)] + [AUC(c2,e2)-AUC(c2,e1)]. "
                "Hierarchical bootstrap (10,000 draws: oracles within cluster, then seeds "
                "within oracle). A pair is confirmed iff (i) the 95% CI of D excludes 0 AND "
                "(ii) the flip pattern re-selects in >=95% of draws.",
        "rationale": "Requirement (ii) is new in Phase 1. Phase 0 selected each winner by "
                     "argmax on the full data and then bootstrapped D with that choice held "
                     "fixed, which makes D positive almost by construction; measured on pure "
                     "noise that procedure confirms ~4 flips per 40 trials, while adding (ii) "
                     "returns 0 per 40. Requirement (i) is retained and reported for "
                     "comparability with Phase 0.",
        "threshold": "at least 2 confirmed pairs within a single backbone",
        "n_bootstrap": 10000, "ci": 0.95, "replication_rate": PS.REPLICATION_RATE,
    },

    "criterion_B_diagnostic": {
        "statement": "A pre-specified cheap diagnostic predicts the winning exploration "
                     "setting across oracles.",
        "primary": {"feature": PS.PRIMARY_FEATURE, "direction": PS.PRIMARY_DIRECTION,
                    "alpha": PS.PRIMARY_ALPHA, "multiplicity_penalty": None,
                    "backbone": "augmented_memory"},
        "secondary": {"features": [f for f in PS.FEATURES if f != PS.PRIMARY_FEATURE],
                      "correction": "benjamini-hochberg", "q": PS.SECONDARY_Q},
        "oracle_set": "confirmation (15 held-out oracles) only",
        "test": "Spearman rho against the winning-setting index (0=low,1=mid,2=high). "
                "Exact permutation p by full enumeration while n<=8, Monte Carlo (100,000 "
                "draws, fixed seed) above that, since 15! is not enumerable.",
        "rationale": "gini is a single pre-specified directional hypothesis carried over from "
                     "Phase 0's screen (rho=-0.655, p=0.0893, mechanistically sensible sign), "
                     "so it is tested at alpha=0.05 with no multiplicity penalty -- legitimate "
                     "because Phase 0 is PRIOR evidence, not this phase's data. That is also "
                     "why it must be confirmed on the 15 oracles that did not generate it. "
                     "BH replaces Holm for the rest: the features are heavily correlated "
                     "measures of reward concentration and Holm is needlessly conservative.",
    },

    "criterion_C_economics": {
        "statement": "Landscape-conditional setting choice gains more AUC than the diagnostic "
                     "probe costs.",
        "test": "headroom = AUC(conditional) - AUC(best single fixed setting), with the "
                "conditional setting selected on 2 seeds and scored on the held-out 3rd, "
                "rotated over all 3 folds. probe_cost = AUC@1000 - AUC@700 for the best fixed "
                "setting. Viable iff the bootstrap CI of (headroom - probe_cost) excludes 0 "
                "and is positive.",
        "rationale": "Phase 0 never asked this. Its naive headroom takes the max of three "
                     "noisy 3-seed means -- a winner's curse that inflates the estimate -- and "
                     "even so reached only +0.0050 to +0.0074 AUC, against a probe costing 30% "
                     "of the budget. Held-out-seed selection removes the bias. If this fails, "
                     "the controller is not viable at this budget and no amount of A, B or D "
                     "changes that.",
        "probe_calls": 300, "n_bootstrap": 10000,
    },

    "criterion_D_method_selection": {
        "statement": "The best (backbone, setting) ARM differs across landscape clusters.",
        "test": "Per cluster, rank all 9 arms by mean AUC; a flip is >=2 distinct winning arms "
                "across clusters, with the same hierarchical bootstrap as Criterion A.",
        "rationale": "Phase 0's method-level margins were an order of magnitude larger than "
                     "its within-method knob effects: graph_ga had the HIGHEST overall mean "
                     "AUC (0.4378 vs 0.4131 vs 0.3901) and showed no landscape dependence at "
                     "all. If anything is worth selecting per landscape it is the method.",
    },

    "verdict_rule": "NULL if A fails. Else CONTROLLER-VIABLE iff B's primary passes AND C is "
                    "viable. Else CHARACTERISATION-ONLY.",
    "verdict_consequences": PS.VERDICTS,

    "prohibited": [
        "Testing Criterion B on a backbone chosen after seeing which backbone flipped.",
        "Assigning cluster tags to the 15 confirmation oracles after seeing any Phase 1 result.",
        "Confirming the gini hypothesis on data that includes the 8 discovery oracles.",
        "Adding or dropping diagnostic features after seeing results.",
        "Reaching a CONTROLLER-VIABLE conclusion without Criterion C clearing the probe cost.",
        "Modifying src/phase0_lib.py or src/phase0_source.py -- Phase 0's verdict was produced "
        "under them.",
        "Running more seeds and reporting only the favourable subset. If more are run, ALL are "
        "reported.",
    ],
}

PREREG1_PATH = ROOT / "Phase 1" / "preregistration_phase1.json"


def prereg1_hash(obj=None) -> str:
    """sha256 over the canonical rule. Includes the COMPUTED feature hash, not a copy."""
    import hashlib
    payload = dict(obj or PREREG1)
    payload.pop("_sha256", None)
    return hashlib.sha256(PS.canonical_json(payload).encode()).hexdigest()


CURRENT_HASH1 = prereg1_hash()

if PREREG1_PATH.exists():
    _disk = json.loads(PREREG1_PATH.read_text())
    PREREG1_STATUS = "verified" if _disk.get("_sha256") == CURRENT_HASH1 else "mismatch"
    if PREREG1_STATUS == "mismatch":
        print("!! PRE-REGISTRATION MISMATCH")
        print(f"   on disk  {_disk.get('_sha256', '')[:32]}...")
        print(f"   in code  {CURRENT_HASH1[:32]}...")
        print("   Part 9 will REFUSE to produce a verdict until this is resolved.")
    else:
        print(f"pre-registration verified, sha256 {CURRENT_HASH1[:32]}...")
else:
    PREREG1_PATH.parent.mkdir(parents=True, exist_ok=True)
    _payload = dict(PREREG1)
    _payload["_sha256"] = CURRENT_HASH1
    PREREG1_PATH.write_text(json.dumps(_payload, indent=2, sort_keys=True))
    PREREG1_STATUS = "frozen"
    print(f"Pre-registration FROZEN -> {PREREG1_PATH}")
    print(f"  sha256 = {CURRENT_HASH1}")
    print("\n  COMMIT THIS FILE NOW, before running any experiment.")


CLUSTER_RULES = (
    ("isomer", ("isomers_",)),
    ("similarity", ("_similarity", "_rediscovery")),
    ("mpo", ("_mpo",)),
    ("bioactivity", ("drd2", "gsk3b", "jnk3")),
)


def tag_oracle(name: str) -> str:
    """Map a PMO oracle name onto a Phase 0 cluster, or 'unassigned'.

    The rules are structural -- they key off PMO's own naming convention, which encodes the
    task family -- so they are checkable by anyone and were fixed before any Phase 1 run.
    """
    low = name.lower()
    for cluster, patterns in CLUSTER_RULES:
        for pat in patterns:
            if pat in low or low == pat:
                return cluster
    return "unassigned"


def pmo_oracle_names() -> list:
    """The PMO suite, read from the installed TDC rather than hardcoded."""
    try:
        from tdc.utils import retrieve_all_benchmarks
        names = retrieve_all_benchmarks("PMO")
        if names:
            return sorted({str(n) for n in np.ravel(names)})
    except Exception as exc:
        print(f"  TDC benchmark listing unavailable ({type(exc).__name__}); "
              f"falling back to the vendored mol_opt task list")
    main = P0.VENDOR_DIR / "mol_opt" / "main"
    if main.is_dir():
        found = sorted({p.stem for p in main.rglob("*.yaml")})
        if found:
            return found
    raise RuntimeError(
        "Could not enumerate the PMO oracle suite from TDC or from the vendored mol_opt. "
        "Resolve this before freezing the tags -- do NOT hand-type the list."
    )


DIAG1_PATH = None  # set in the driver cell below


def compute_diagnostics1(oracle_name: str, seed: int = 0, n_prior: int = 200,
                         n_edits: int = 100) -> dict:
    """Phase 1 diagnostics for one oracle: Phase 0's probe, Phase 1's features."""
    base = P0.compute_diagnostics(oracle_name, n_prior=n_prior,
                                  n_edits_total=n_edits, seed=seed)
    rewards = base.get("_rewards") or base.get("prior_rewards") or []
    out = {"oracle": oracle_name, "cluster": tag_oracle(oracle_name)}
    if len(rewards):
        out.update(PS.quantile_features(rewards))
    else:
        # Phase 0's probe does not return the raw reward vector on every code path; fall
        # back to the shape features it does report, and leave quantile gaps undefined
        # rather than inventing them.
        for k in ("reward_mean", "reward_std", "reward_skew", "reward_kurtosis",
                  "gini", "dynamic_range"):
            out[k] = float(base.get(k, float("nan")))
        for q in PS.QUANTILES:
            out[f"q{int(q * 100)}_gap"] = float("nan")
    for k in ("ruggedness_corr", "ruggedness_absdelta", "basin_width", "neighbor_gain"):
        out[k] = float(base.get(k, float("nan")))
    return out


def compute_all_diagnostics1(oracles, seeds=(0, 1, 2), force: bool = False):
    """Diagnostics for every oracle, averaged over probe seeds. Cached on disk."""
    path = P0.RESULTS_DIR / "diagnostics_phase1.json"
    if path.exists() and not force:
        print(f"loading cached diagnostics <- {path}")
        return pd.DataFrame(json.loads(path.read_text()))
    rows = []
    for nm in oracles:
        t0 = time.time()
        per_seed = [compute_diagnostics1(nm, seed=s) for s in seeds]
        agg = {"oracle": nm, "cluster": tag_oracle(nm)}
        for f in PS.FEATURES:
            vals = [d[f] for d in per_seed if np.isfinite(d.get(f, np.nan))]
            agg[f] = float(np.mean(vals)) if vals else float("nan")
        rows.append(agg)
        print(f"  {nm:30s} {time.time() - t0:5.1f}s  gini {agg['gini']:.3f}")
    df = pd.DataFrame(rows)
    df.to_json(path, orient="records")
    print(f"\nsaved -> {path}")
    return df


RUNS1_DIR = None  # set in the driver cell


def run_key1(oracle: str, backbone: str, setting: str, seed: int) -> str:
    return f"{oracle}__{backbone}__{setting}__seed{seed}"


def run_path1(*args) -> Path:
    return RUNS1_DIR / f"{run_key1(*args)}.json"


def execute_run1(oracle_name: str, backbone: str, setting: str, seed: int,
                 budget: int = 1000, force: bool = False, verbose: bool = False) -> dict:
    """One Phase 1 run. Resumable: a completed record is returned untouched."""
    p = run_path1(oracle_name, backbone, setting, seed)
    if p.exists() and not force:
        return json.loads(p.read_text())

    orc = P0.make_oracle(oracle_name, budget=budget, freq_log=10)
    t0 = time.time()
    try:
        res = P0.run_backbone(backbone, orc, setting, seed, verbose=verbose)
        res["status"] = "ok"
        # (score, call_index) in call order -- what makes auc_at exact at any truncation.
        res["score_trace"] = [[float(v[0]), int(v[1])]
                              for v in sorted(orc.mol_buffer.values(),
                                              key=lambda t: t[1])]
    except Exception as exc:
        import traceback
        res = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()[-2000:], "auc_top10": float("nan")}
    res.update({"oracle": oracle_name, "cluster": tag_oracle(oracle_name),
                "backbone": backbone, "setting": setting, "seed": seed,
                "budget": budget, "elapsed_s": round(time.time() - t0, 1),
                "phase": 1, "prereg_sha256": CURRENT_HASH1})
    p.write_text(json.dumps(res))
    return res


def build_sweep1(oracles, backbones=None, settings=None, seeds=None) -> list:
    backbones = backbones or PREREG1["backbones"]
    settings = settings or PREREG1["settings"]
    seeds = seeds or PREREG1["seeds"]
    return [(o, b, s, sd) for o in oracles for b in backbones
            for s in settings for sd in seeds]


def run_sweep1(jobs, budget: int = 1000, n_workers: int = None, force: bool = False):
    """Resumable sweep. ThreadPoolExecutor, NOT a process pool.

    Phase 0 established why: threads share one CUDA context (~500 MB paid once), so 12
    workers held ~4.9 GB of 12 GB. A process pool would mean 12 contexts and OOM.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 4)
    todo = [j for j in jobs if force or not run_path1(*j).exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run, {n_workers} workers, budget {budget}")
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(execute_run1, *j, budget=budget, force=force): j for j in todo}
        for f in as_completed(futs):
            j, done = futs[f], done + 1
            try:
                r = f.result()
                print(f"[{done}/{len(todo)}] {run_key1(*j):56s} "
                      f"AUC {r.get('auc_top10', float('nan')):.4f} {r['status']}")
            except Exception as exc:
                print(f"[{done}/{len(todo)}] {run_key1(*j):56s} CRASHED {exc}")
    print(f"\nsweep finished in {(time.time() - t0) / 60:.1f} min")
    return load_results1()


def load_results1(include_phase0: bool = False):
    """Phase 1 records. `include_phase0=True` unions Phase 0's for the secondary report."""
    rows = []
    for p in sorted(RUNS1_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        for k in ("history", "top10_smiles", "traceback"):
            d.pop(k, None)
        rows.append(d)
    if include_phase0:
        for p in sorted(P0.RUNS_DIR.glob("*.json")):
            d = json.loads(p.read_text())
            for k in ("history", "top10_smiles", "traceback"):
                d.pop(k, None)
            d["phase"] = 0
            d.setdefault("score_trace", None)
            rows.append(d)
    return pd.DataFrame(rows)
