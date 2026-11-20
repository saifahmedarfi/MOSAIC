"""Phase 2 library -- GENERATED from src/phase2_source.py. Do not edit.

Regenerate with:  python src/build_phase2.py

Importing this module defines the Phase 2 pre-registration, candidate oracles, gates,
runner and metric computation. Driver cells are excluded, so an import launches nothing and
downloads nothing: phase0_lib is loaded lazily by p0() on first use, so the library imports on
a machine without torch or TDC.
"""


import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    ROOT = Path(os.environ["LCE_ROOT"]).resolve() if "LCE_ROOT" in os.environ \
        else Path(__file__).resolve().parent.parent
except NameError:                      # notebook kernel: no __file__
    ROOT = Path.cwd().resolve()
if (ROOT / "src").is_dir() and str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import phase2_components as PC
import phase2_stats as PS

PHASE2_DIR = ROOT / "Phase 2"
RUNS2_DIR = ROOT / "results" / "runs_phase2"
PREREG2_PATH = PHASE2_DIR / "preregistration_phase2.json"
GATE2_PATH = PHASE2_DIR / "phase2_reproduction_gate.json"
POWER_GATE_PATH = PHASE2_DIR / "phase2_power_gate.json"
PHASE1_MANIFEST = ROOT / "Phase 1" / "phase1_run_manifest.csv"

# The Phase 0 library is heavy (torch, TDC, a mol_opt clone) and creates directories on import.
# It is therefore loaded on first use through p0(), not at import. The import statement is
# here, in a library cell, so it exists in the generated library -- Phase 1 had it
# @notebook-only and every runner raised NameError.
_P0 = None


def p0():
    """The phase0_lib module, imported once on first use."""
    global _P0
    if _P0 is None:
        os.environ.setdefault("LCE_ROOT", str(ROOT))
        import phase0_lib as _mod  # noqa: E402
        _P0 = _mod
    return _P0


class GateHalt(RuntimeError):
    """A gate failed. The build stops here rather than degrading."""


print(f"project root     {ROOT}")
print(f"metric set       {len(PS.METRICS)} metrics, sha256 {PS.metrics_hash()[:16]}...")
print(f"primary pair     {PS.PRIMARY_PAIR}")


PREREG2 = {
    "version": "3.0.0",
    "phase": 2,
    "gated_by": {"phase": 1, "verdict": "NULL",
                 "prereg_sha256": "bad00ca8186a58d74c1bcaa5615f648d4fd4c0b2aac78f44d52386c50a9d2094"},
    "frozen_on": "2026-11-20",
    "budget": 1000,
    "setting": "explore_mid",
    "backbones": ["reinvent", "augmented_memory", "graph_ga"],
    "oracles": sorted(PC.COMPONENTS),
    # Seed count: 15 by controller ruling, which OVERRIDES the 5 that choose_seeds returned.
    # n=15 is never worse than n=5 at any simulated effect and is decisively better in the
    # 0.4-0.6 SD range (power 0.37-0.80 at n=5 vs 0.92-1.00 at n=15; floor 0.6 -> 0.4 SD).
    # See design 5.3.2 and Phase 2/phase2_power_gate.json. choose_seeds was NOT amended.
    "seeds": 15,
    "seed_ids": list(range(15)),
    "seed_rule_output": 5,
    "seed_ruling": "15, by controller ruling on dominance (design 5.3.2); overrides "
                   "choose_seeds, which returned 5 and was not amended",
    "power_gate_evidence": "Phase 2/phase2_power_gate.json",
    "metrics": list(PS.METRICS),
    "metrics_sha256": PS.metrics_hash(),
    "primary_pair": list(PS.PRIMARY_PAIR),
    "secondary_metrics": list(PS.SECONDARY_METRICS),
    "hv_reference_point": "origin",
    "aggregation": dict(PC.AGGREGATION),
    "min_surviving_oracles": PS.MIN_ORACLES,
    "gate_tolerance": PS.GATE_TOL,
    "replication_rate": PS.REPLICATION_RATE,
    "c_threshold": PS.C_THRESHOLD,
    "bh_q": PS.BH_Q,
    "n_boot": PS.N_BOOT,
    "alpha": 0.05,
    "overlap_gate": {"cells": "explore_mid, seeds 0-2, every candidate present in Phase 1's "
                              "manifest: 7 oracles x 3 backbones x 3 seeds = 63 cells "
                              "(ranolazine_mpo and sitagliptin_mpo were Phase 0 oracles and "
                              "are not in Phase 1's manifest)",
                     "metric": "auc_top10", "tolerance": 1e-9, "on_deviation": "halt",
                     "on_absent_records": "record 'could not run', never 'passed'"},
    "persistence": "full molecule buffer as (smiles, score, call_index) triples; component "
                   "vectors recomputed from SMILES, never stored",
    "verdict_consequences": PS.VERDICTS,
    "prohibited": [
        "Adjusting a criterion, alpha, tolerance, reference point, metric list, oracle set, "
        "seed count or budget to change a verdict.",
        "Extending the seed count after seeing any Phase 2 result (optional stopping).",
        "Lowering C_THRESHOLD so that METRIC-DEPENDENT becomes reachable.",
        "Modifying src/phase0_lib.py, src/phase0_source.py or src/phase1_source.py.",
        "Running more seeds and reporting only the favourable subset. If more are run, ALL "
        "are reported.",
    ],
}

# The hash of PREREG2 as frozen. Changing PREREG2 without re-freezing makes prereg2_hash()
# disagree with this constant, and every run and the verdict refuse to proceed.
PREREG2_SHA256 = "e8bb792e4b7e9ebfc62b5f061a0d2b3c067ce7da728b3453e819655bcd40bf3a"


def prereg2_hash(obj=None) -> str:
    """sha256 over the canonical rule (the in-code dict, never a copy)."""
    payload = dict(obj or PREREG2)
    payload.pop("_sha256", None)
    return hashlib.sha256(PS.canonical_json(payload).encode()).hexdigest()


def prereg2_status() -> str:
    """'verified' | 'mismatch' | 'unfrozen'. Compares the code, the frozen constant and the
    committed file; any disagreement is 'mismatch'."""
    code = prereg2_hash()
    if code != PREREG2_SHA256:
        return "mismatch"
    if not PREREG2_PATH.exists():
        return "unfrozen"
    disk = json.loads(PREREG2_PATH.read_text())
    return "verified" if disk.get("_sha256") == code else "mismatch"


def assert_prereg_intact():
    st = prereg2_status()
    if st == "mismatch":
        raise GateHalt(
            f"pre-registration mismatch: code {prereg2_hash()[:16]}..., frozen constant "
            f"{PREREG2_SHA256[:16]}.... Refusing to proceed.")
    return st


def freeze_prereg2() -> Path:
    """Write Phase 2/preregistration_phase2.json from the code. Refuses to overwrite."""
    if PREREG2_PATH.exists():
        raise GateHalt(f"{PREREG2_PATH} exists; the pre-registration is frozen, not regenerated")
    payload = dict(PREREG2)
    payload["_sha256"] = prereg2_hash()
    PREREG2_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREREG2_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return PREREG2_PATH


PROBE_SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1CCC[C@H]1c1cccnc1"]


def sklearn_tripwire(drd2, probes=PROBE_SMILES):
    """Score the probes through `drd2`; halt unless at least one is non-zero."""
    scores = [float(drd2(s)) for s in probes]
    if not any(s > 0 for s in scores):
        raise GateHalt(
            "drd2 returned all zeros -- scikit-learn has regressed to >=1.3 and every "
            "bioactivity number would be void. Reinstall scikit-learn==1.2.2. "
            f"Probe scores: {scores}")
    return scores


CANDIDATE_ORACLES = sorted(PC.COMPONENTS)

OVERLAP_TOL = 1e-9


def check_candidates_against_pmo(pmo_names, candidates=None):
    """Halt unless every candidate is in the PMO suite listing."""
    candidates = CANDIDATE_ORACLES if candidates is None else candidates
    missing = sorted(set(candidates) - set(pmo_names))
    if missing:
        raise GateHalt(f"candidate oracle(s) not in the PMO suite listing: {missing}")
    return True


def pmo_names_via_phase1():
    """phase1's pmo_oracle_names() (TDC listing, else the vendored mol_opt task stems).
    phase1_lib's fallback branch references a module-level P0 its library never defines, so it
    is supplied here rather than editing phase1_lib."""
    import phase1_lib as L1
    L1.P0 = p0()
    return L1.pmo_oracle_names()


def phase1_explore_mid(df):
    """Phase 1 rows restricted to the setting Phase 2 runs, ok status only."""
    d = df[(df["setting"] == PREREG2["setting"]) & (df["status"] == "ok")]
    return d[d["oracle"].isin(CANDIDATE_ORACLES)].copy()


def phase1_overlap_gate(p1, p2, tol=OVERLAP_TOL):
    """Compare auc_top10 run-for-run on (oracle, backbone, seed) cells present in both.

    Returns {"status": "passed" | "could_not_run", "n_compared", "max_deviation", ...}; raises
    GateHalt on any deviation above `tol`. `could_not_run` when nothing overlaps -- that is
    never reported as a pass."""
    if p1 is None or len(p1) == 0 or p2 is None or len(p2) == 0:
        return {"status": "could_not_run", "n_compared": 0, "max_deviation": None,
                "reason": "Phase 1 or Phase 2 run records are absent"}
    a = phase1_explore_mid(p1)
    b = p2[p2["status"] == "ok"]
    key = ["oracle", "backbone", "seed"]
    m = a[key + ["auc_top10"]].merge(b[key + ["auc_top10"]], on=key, suffixes=("_p1", "_p2"))
    if len(m) == 0:
        return {"status": "could_not_run", "n_compared": 0, "max_deviation": None,
                "reason": "no (oracle, backbone, seed) cell exists in both phases"}
    dev = (m["auc_top10_p1"] - m["auc_top10_p2"]).abs()
    worst = float(dev.max())
    if not np.isfinite(worst) or worst > tol:
        bad = m.loc[dev.sort_values(ascending=False).index[:5], key + ["auc_top10_p1", "auc_top10_p2"]]
        raise GateHalt(f"Phase 1 overlap NOT reproduced: max |auc_top10 diff| = {worst:.3g} "
                       f"> {tol:g} over {len(m)} cells. Worst:\n{bad.to_string(index=False)}")
    return {"status": "passed", "n_compared": int(len(m)), "max_deviation": worst}


def reproduction_gate(oracles, get_tdc):
    """Run `phase2_components.run_gate` on every oracle. `get_tdc(name)` returns the TDC
    scorer. Halts below MIN_ORACLES; otherwise returns the summary with per-oracle detail."""
    per = {o: PC.run_gate(o, get_tdc(o)) for o in oracles}
    summary = PS.gate_summary(per)
    summary["per_oracle"] = per
    if not summary["enough"]:
        raise GateHalt(f"only {len(summary['surviving'])} oracle(s) survive the reproduction "
                       f"gate (need >= {PS.MIN_ORACLES}): excluded {summary['excluded']}")
    return summary


def save_gate(summary, path=None):
    path = GATE2_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    return path


def surviving_oracles(path=None):
    """The oracles the committed reproduction gate cleared. Raises if the gate has not run."""
    path = GATE2_PATH if path is None else path
    if not path.exists():
        raise GateHalt(f"{path} not found: run the notebook's Part 3 (reproduction gate) first")
    g = json.loads(path.read_text())
    if not g.get("enough") or len(g["surviving"]) < PS.MIN_ORACLES:
        raise GateHalt(f"reproduction gate recorded {len(g.get('surviving', []))} survivors; "
                       f"need >= {PS.MIN_ORACLES}")
    return list(g["surviving"])


def run_key2(oracle: str, backbone: str, seed: int) -> str:
    return f"{oracle}__{backbone}__{PREREG2['setting']}__seed{seed}"


def run_path2(*args) -> Path:
    return RUNS2_DIR / f"{run_key2(*args)}.json"


def buffer_triples(mol_buffer) -> list:
    """BudgetedOracle.mol_buffer ({smiles: [score, call_index]}) as a list of
    [smiles, score, call_index] in call order."""
    return [[str(smi), float(v[0]), int(v[1])]
            for smi, v in sorted(mol_buffer.items(), key=lambda kv: kv[1][1])]


def execute_run2(oracle_name: str, backbone: str, seed: int, budget: int = None,
                 force: bool = False, verbose: bool = False) -> dict:
    """One Phase 2 run. Resumable: a completed record is returned untouched."""
    budget = PREREG2["budget"] if budget is None else budget
    p = run_path2(oracle_name, backbone, seed)
    if p.exists() and not force:
        return json.loads(p.read_text())

    P0 = p0()
    orc = P0.make_oracle(oracle_name, budget=budget, freq_log=10)
    t0 = time.time()
    try:
        res = P0.run_backbone(backbone, orc, PREREG2["setting"], seed, verbose=verbose)
        res["status"] = "ok"
        res["full_buffer"] = buffer_triples(orc.mol_buffer)
    except Exception as exc:
        import traceback
        res = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()[-2000:], "auc_top10": float("nan")}
    res.update({"oracle": oracle_name, "backbone": backbone, "setting": PREREG2["setting"],
                "seed": seed, "budget": budget, "elapsed_s": round(time.time() - t0, 1),
                "phase": 2, "prereg_sha256": PREREG2_SHA256})
    RUNS2_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res))
    return res


def build_sweep2(oracles=None, backbones=None, seeds=None) -> list:
    oracles = CANDIDATE_ORACLES if oracles is None else oracles
    backbones = PREREG2["backbones"] if backbones is None else backbones
    seeds = PREREG2["seed_ids"] if seeds is None else seeds
    return [(o, b, sd) for o in oracles for b in backbones for sd in seeds]


def run_sweep2(jobs, budget: int = None, n_workers: int = None, force: bool = False):
    """Resumable sweep. ThreadPoolExecutor, NOT a process pool: threads share one CUDA context
    (~500 MB paid once; Phase 0 measured ~4.9 GB at 12 workers), a process pool would OOM.
    Refuses to start on a pre-registration mismatch or a failed reproduction gate."""
    assert_prereg_intact()
    allowed = set(surviving_oracles())
    stray = sorted({j[0] for j in jobs} - allowed)
    if stray:
        raise GateHalt(f"job oracle(s) did not survive the reproduction gate: {stray}")
    budget = PREREG2["budget"] if budget is None else budget
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 4)
    todo = [j for j in jobs if force or not run_path2(*j).exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run, {n_workers} workers, budget {budget}")
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(execute_run2, *j, budget=budget, force=force): j for j in todo}
        for f in as_completed(futs):
            j, done = futs[f], done + 1
            try:
                r = f.result()
                print(f"[{done}/{len(todo)}] {run_key2(*j):56s} "
                      f"AUC {r.get('auc_top10', float('nan')):.4f} {r['status']}")
            except Exception as exc:
                print(f"[{done}/{len(todo)}] {run_key2(*j):56s} CRASHED {exc}")
    print(f"\nsweep finished in {(time.time() - t0) / 60:.1f} min")
    return load_results2()


TOP_N = 100
NOVELTY_CUTOFF = 0.4


def _fingerprints(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"invalid SMILES in persisted buffer: {s!r}")
        out.append(gen.GetFingerprint(m))
    return out


def population_metrics(oracle: str, full_buffer, auc_top10: float) -> dict:
    """The seven design-5.2 metrics for one run, from its (smiles, score, call_index) triples."""
    from rdkit import DataStructs
    if len(full_buffer) == 0:
        raise ValueError("empty population")
    smiles = [t[0] for t in full_buffer]
    scores = np.array([t[1] for t in full_buffer], dtype=float)
    comps = np.array([PC.component_vector(oracle, s) for s in smiles])

    front = PS.pareto_front(comps)
    on_front = {tuple(r) for r in front}
    mask = np.array([tuple(r) in on_front for r in comps])
    front_size = int(mask.sum())

    order = np.argsort(-scores, kind="stable")
    top = order[:min(TOP_N, len(order))]
    thr = float(scores[order[min(10, len(order)) - 1]])
    sc_loss = float(np.mean(scores[mask] < thr))

    fps = _fingerprints([smiles[i] for i in top])
    refs = _fingerprints([r for r in _ref_smiles(oracle)])
    nov = float(np.mean([max(DataStructs.BulkTanimotoSimilarity(f, refs)) < NOVELTY_CUTOFF
                         for f in fps]))
    if len(fps) > 1:
        dists = [1.0 - s for i in range(1, len(fps))
                 for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])]
        div = float(np.mean(dists))
    else:
        div = float("nan")
    return {"auc_top10": float(auc_top10),
            "hypervolume": float(PS.hypervolume(front)),
            "front_size": float(front_size),
            "front_coverage": front_size / len(smiles),
            "novelty": nov, "diversity": div, "scalarization_loss": sc_loss}


def _ref_smiles(oracle):
    return PC._REFS[oracle]


def load_results2(runs_dir=None, with_metrics: bool = True):
    """Phase 2 records as one frame (one row per run, population dropped). With
    `with_metrics`, the seven metrics are computed from each record's persisted buffer."""
    runs_dir = RUNS2_DIR if runs_dir is None else Path(runs_dir)
    rows, failed = [], []
    for p in sorted(runs_dir.glob("*.json")) if runs_dir.is_dir() else []:
        d = json.loads(p.read_text())
        buf = d.pop("full_buffer", None)
        for k in ("history", "top10_smiles", "traceback"):
            d.pop(k, None)
        if with_metrics and d.get("status") == "ok":
            try:
                d.update(population_metrics(d["oracle"], buf, d["auc_top10"]))
            except Exception as exc:
                d["status"] = "metric_error"
                d["error"] = f"{type(exc).__name__}: {exc}"
                failed.append((p.name, d["error"]))
        rows.append(d)
    if failed:
        print(f"!! {len(failed)} run(s) have no computable metrics and are excluded:")
        for n, e in failed:
            print(f"   {n}: {e}")
    return pd.DataFrame(rows)
