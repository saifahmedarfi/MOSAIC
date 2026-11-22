"""Phase 3 library -- GENERATED from src/phase3_source.py. Do not edit.

Regenerate with:  python src/build_phase3.py

Importing this module defines the Phase 3 pre-registration, the three arms, the gates, the
runner and the metric computation. Driver cells are excluded, so an import launches nothing
and downloads nothing: phase0_lib is loaded lazily by p0() on first use, so the library
imports on a machine without torch or TDC.
"""


from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("LCE_ROOT", Path.cwd())).resolve()
if not (ROOT / "src").is_dir():                      # notebook launched from Phase 3/
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import phase2_components as PC                       # noqa: E402  rdkit only, no torch
import phase2_stats as PS                            # noqa: E402  pure numpy
import phase3_pareto as P3P                          # noqa: E402  NSGA-II + the GA loop
import phase3_stats as P3St                          # noqa: E402  the criteria

PHASE3_DIR = ROOT / "Phase 3"
RUNS3_DIR = ROOT / "results" / "runs_phase3"
PREREG3_PATH = PHASE3_DIR / "preregistration_phase3.json"
PILOT_PATH = PHASE3_DIR / "phase3_pilot.json"
GATE3_PATH = PHASE3_DIR / "phase3_reproduction_gate.json"

print(f"project root     {ROOT}")
print(f"metric set       {len(P3St.METRICS)} metrics, sha256 {P3St.metrics_hash3()[:16]}...")
print(f"arms             {P3St.ARMS}")
print(f"python  {platform.python_version()}   ({platform.system()} {platform.release()})")
print(f"numpy   {np.__version__}")
print(f"pandas  {pd.__version__}")


_P0 = None


def p0():
    """phase0_lib, imported lazily and once.

    Lazy so that importing phase3_lib on a machine without torch or TDC still defines
    everything. phase0_lib runs `ZINC_POOL = get_zinc_pool()` at module level, which
    downloads ZINC-250k via TDC when `data/zinc_pool.txt` is absent.
    """
    global _P0
    if _P0 is None:
        import phase0_lib as P0
        _P0 = P0
    return _P0


PREREG3 = {
    "phase": 3,
    "version": "1.0.0",
    "frozen_on": None,
    "hypothesis": (
        "NSGA-II selection over the verified component decomposition produces a better "
        "Pareto front than scalarized selection at equal oracle budget, without "
        "surrendering auc_top10."
    ),
    "gated_on": {
        "phase": 2,
        "verdict": "REFUSED",
        "measurement_used": (
            "Phase 2's hypervolume z-spread of 0.843 across three backbones against "
            "auc_top10's 0.133, and the reordering of graph_ga from middle to last."
        ),
        "measurement_NOT_used": (
            "scalarization_loss. It measures the fraction of Pareto-optimal molecules the "
            "scalar top-10 cut excludes, tracks front cardinality (r = 0.577), and rises "
            "when an optimizer enlarges the front. See spec section 1.2."
        ),
    },
    "oracles": sorted(PC.COMPONENTS),
    "arms": list(P3St.ARMS),
    "backbone": "graph_ga",
    "setting": "explore_mid",
    "budget": 1000,
    "metrics": list(P3St.METRICS),
    "metrics_sha256": P3St.metrics_hash3(),
    "primary_metric_P": "hypervolume",
    "primary_metric_Q": "auc_top10",
    "replication_rate": P3St.REPLICATION_RATE,
    "n_boot": PS.N_BOOT,
    "f_bound": P3St.F_BOUND,
    "eps_q": 0.14,
    "eps_q_rule": (
        "FIXED and pre-registered, never pilot-derived. The design's 95% resolution limit: "
        "the bootstrap SE of an arm difference is ~sqrt(2/(n_seeds*n_oracles)) = 0.086 at 30 "
        "seeds with 9 oracles. Criterion Q asserts the AUC loss is indistinguishable from "
        "zero at the finest resolution available."
    ),
    "delta_p": None,
    "seeds": None,
    "seed_ids": None,
    "seed_ruling": None,
    "seed_bounds": [20, 30],
    "pilot_oracles": ["median1", "fexofenadine_mpo", "osimertinib_mpo"],
    "held_out_oracles": ["amlodipine_mpo", "median2", "perindopril_mpo", "ranolazine_mpo",
                         "sitagliptin_mpo", "zaleplon_mpo"],
    "zero_component_rule": (
        "Both the mating pool and the survival step restrict to molecules with all "
        "components > 0 before non-dominated sorting. Every one of these oracles aggregates "
        "by geometric mean, so a zero component means an oracle score of exactly 0.0, and "
        "the control arm already excludes such molecules automatically."
    ),
    "component_budget_rule": (
        "The treatment arm may compute component vectors ONLY for molecules already present "
        "in oracle.mol_buffer. No component-based pre-screening of unscored candidates."
    ),
    "verdicts": dict(P3St.VERDICTS3),
    "declared_modal_outcome": "PARETO-TRADES",
}

PREREG3_SHA256 = None           # set by freeze_prereg3() and pasted back here when frozen


def prereg3_hash(obj=None) -> str:
    """sha256 over the canonical rule (the in-code dict, never a copy)."""
    payload = dict(obj or PREREG3)
    payload.pop("_sha256", None)
    return hashlib.sha256(PS.canonical_json(payload).encode()).hexdigest()


def prereg3_status() -> str:
    """'unfrozen' | 'verified' | 'mismatch'.

    'unfrozen' while delta_p or seeds is null -- Stage A has not run, so there is no rule to
    verify yet. After freezing, compares the in-code dict, the frozen constant and the
    committed file; any disagreement is 'mismatch'.
    """
    if PREREG3["delta_p"] is None or PREREG3["seeds"] is None:
        return "unfrozen"
    if PREREG3_SHA256 is None or prereg3_hash() != PREREG3_SHA256:
        return "mismatch"
    if not PREREG3_PATH.exists():
        return "mismatch"
    on_disk = json.loads(PREREG3_PATH.read_text())
    return "verified" if on_disk.get("_sha256") == PREREG3_SHA256 else "mismatch"


def assert_prereg_intact():
    s = prereg3_status()
    if s == "unfrozen":
        raise RuntimeError(
            "pre-registration is UNFROZEN: delta_p and/or seeds are null. Run Stage A "
            "(Part 5), paste its delta_p / seeds / seed_ruling into PREREG3, call "
            "freeze_prereg3(), rebuild, and COMMIT the frozen file before any Stage B run."
        )
    if s != "verified":
        raise RuntimeError(f"pre-registration {s}: refusing to proceed")


def freeze_prereg3() -> Path:
    """Write the frozen rule and its hash. Refuses to overwrite an existing frozen file."""
    if PREREG3["delta_p"] is None or PREREG3["seeds"] is None:
        raise RuntimeError("cannot freeze: delta_p and seeds must come from Stage A first")
    if PREREG3_PATH.exists():
        raise RuntimeError(f"{PREREG3_PATH.name} already exists -- frozen files are never "
                           f"regenerated. A mismatch is the mechanism working.")
    payload = dict(PREREG3)
    payload["frozen_on"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    h = prereg3_hash(payload)
    payload["_sha256"] = h
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    PREREG3_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"frozen -> {PREREG3_PATH.name}")
    print(f"  sha256 {h}")
    print(f"  paste this into PREREG3_SHA256 in src/phase3_source.py, then rebuild and COMMIT")
    return PREREG3_PATH


PROBE_SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1CCC[C@H]1c1cccnc1"]
MIN_ORACLES3 = PS.MIN_ORACLES


def sklearn_tripwire(drd2, probes=PROBE_SMILES):
    """Assert TDC's drd2 is live. Returns the probe scores."""
    scores = [float(drd2(s)) for s in probes]
    print(f"drd2 probe scores: {[round(s, 4) for s in scores]}")
    if not any(s > 0.0 for s in scores):
        raise RuntimeError(
            "drd2 returned 0.0 for every probe. scikit-learn is almost certainly >= 1.3, "
            "on which TDC's pickle loads without error and scores nothing. Pin 1.2.2."
        )
    print("tripwire OK -- drd2 is live")
    return scores


def reproduction_gate3():
    """Re-run Phase 2's component reproduction gate and persist the result."""
    per = {o: PC.run_gate(o, p0().get_tdc_oracle(o)) for o in sorted(PC.COMPONENTS)}
    summary = PS.gate_summary(per)
    summary["components_sha256"] = _components_hash()
    GATE3_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE3_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _components_hash() -> str:
    """sha256 over the component table's structure, so a silent edit is detectable."""
    table = {o: [getattr(c, "name", repr(c)) for c in comps]
             for o, comps in sorted(PC.COMPONENTS.items())}
    return hashlib.sha256(PS.canonical_json(table).encode()).hexdigest()


def surviving_oracles3(path=None):
    """Oracles that cleared the reproduction gate. Halts below the floor."""
    p = Path(path or GATE3_PATH)
    if not p.exists():
        raise RuntimeError(f"{p.name} missing: run Part 3 before any sweep")
    g = json.loads(p.read_text())
    alive = sorted(o for o, r in g["per_oracle"].items() if r.get("reproduces"))
    if len(alive) < MIN_ORACLES3:
        raise RuntimeError(f"only {len(alive)} oracles reproduce, floor is {MIN_ORACLES3}")
    return alive


def run_key3(oracle: str, arm: str, seed: int) -> str:
    return f"{oracle}__{arm}__{PREREG3['setting']}__seed{seed}"


def run_path3(*args) -> Path:
    return RUNS3_DIR / f"{run_key3(*args)}.json"


def buffer_triples(mol_buffer) -> list:
    """[(smiles, score, call_index)] for every molecule the run paid for."""
    return [[s, float(v[0]), int(v[1])] for s, v in mol_buffer.items()]


def record_complete(path) -> bool:
    """Complete means status ok AND the full budget -- a reduced-budget dry run is not a result."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        r = json.loads(p.read_text())
    except Exception:
        return False
    return r.get("status") == "ok" and r.get("budget") == PREREG3["budget"]


def execute_run3(oracle_name: str, arm: str, seed: int, budget: int = None,
                 runs_dir=None, verbose: bool = False) -> dict:
    """One (oracle, arm, seed) cell. Writes a record and returns it."""
    budget = budget or PREREG3["budget"]
    out_dir = Path(runs_dir or RUNS3_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    P0 = p0()
    cfg = P0.EXPLORATION_LADDER["graph_ga"][PREREG3["setting"]]
    orc = P0.make_oracle(oracle_name, budget=budget, freq_log=10)
    t0 = time.time()
    try:
        if arm == "anchor":
            res = P0.run_ga_optimizer(orc, cfg, seed=seed, verbose=verbose)
            res["arm"] = "anchor"
            res["selection"] = "scalar"
            res["n_fallback_generations"] = 0
        elif arm == "control":
            res = P3P.run_ga_arm(orc, cfg, oracle_name, "scalar", seed=seed, verbose=verbose)
        elif arm == "treatment":
            res = P3P.run_ga_arm(orc, cfg, oracle_name, "pareto", seed=seed, verbose=verbose)
        else:
            raise ValueError(f"unknown arm {arm!r}; expected one of {P3St.ARMS}")
        status = "ok"
        err = None
    except Exception as e:                       # a crashed cell is recorded, never silent
        res, status, err = {}, "error", f"{type(e).__name__}: {e}"
    rec = dict(res)
    rec.update({
        "oracle": oracle_name, "arm": arm, "seed": int(seed), "setting": PREREG3["setting"],
        "budget": int(budget), "status": status, "error": err,
        "elapsed_s": round(time.time() - t0, 1), "phase": 3,
        "prereg_sha256": PREREG3_SHA256,
        "full_buffer": buffer_triples(orc.mol_buffer) if status == "ok" else [],
    })
    rec.pop("top10_smiles", None)
    run_path3(oracle_name, arm, seed).write_text(json.dumps(rec))
    return rec


def build_sweep3(oracles=None, arms=None, seeds=None) -> list:
    """The Stage B grid as a list of (oracle, arm, seed)."""
    oracles = oracles or surviving_oracles3()
    arms = arms or list(P3St.ARMS)
    if seeds is None:
        if PREREG3["seed_ids"] is None:
            raise RuntimeError("seed_ids is null: Stage A has not run, so there is no grid")
        seeds = PREREG3["seed_ids"]
    return [(o, a, s) for o in oracles for a in arms for s in seeds]


def run_sweep3(jobs, budget: int = None, n_workers: int = None, force: bool = False):
    """Run the grid in a thread pool. Resumable; refuses on an unfrozen pre-registration."""
    assert_prereg_intact()
    allowed = set(surviving_oracles3())
    bad = sorted({o for o, _, _ in jobs} - allowed)
    if bad:
        raise RuntimeError(f"oracles {bad} did not clear the reproduction gate")

    todo = [j for j in jobs if force or not record_complete(run_path3(*j))]
    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 4)
    print(f"{len(jobs)} jobs, {len(todo)} to run, {n_workers} workers, "
          f"budget {budget or PREREG3['budget']}")
    if not todo:
        return

    p0()                                    # warm the main thread: a missing torch or TDC
    sklearn_tripwire(p0().get_tdc_oracle("drd2"))   # must fail HERE, not 540x inside workers

    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(execute_run3, o, a, s, budget): (o, a, s) for o, a, s in todo}
        for f in as_completed(futs):
            o, a, s = futs[f]
            done += 1
            try:
                r = f.result()
                auc = r.get("auc_top10")
                tag = f"AUC {auc:.4f} {r['status']}" if auc is not None else r["status"]
            except Exception as e:
                tag = f"CRASHED {type(e).__name__}: {e}"
            print(f"[{done}/{len(todo)}] {run_key3(o, a, s):<58} {tag}")


def sweep_failures3(jobs) -> list:
    return [j for j in jobs if not record_complete(run_path3(*j))]


def population_metrics3(oracle: str, full_buffer, auc_top10: float) -> dict:
    """The three Phase 3 metrics for one run, from its (smiles, score, call_index) triples.

    Mirrors Phase 2's definitions for auc_top10, hypervolume and front_size so the numbers
    stay commensurable. front_coverage is deliberately NOT computed: it is front_size / n and
    would be perfectly collinear (Phase 2 defect 6).
    """
    if not len(full_buffer):
        raise ValueError("empty population")
    smiles = [t[0] for t in full_buffer]
    comps = np.array([PC.component_vector(oracle, s) for s in smiles])
    front = PS.pareto_front(comps)
    on_front = {tuple(r) for r in front}
    front_size = int(sum(tuple(r) in on_front for r in comps))
    return {"auc_top10": float(auc_top10),
            "hypervolume": float(PS.hypervolume(front)),
            "front_size": float(front_size)}


def load_results3(runs_dir=None) -> list:
    """Every Phase 3 record produced under the FROZEN rule. A record from another rule is
    dropped rather than silently mixed in."""
    d = Path(runs_dir or RUNS3_DIR)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("phase") != 3:
            continue
        if PREREG3_SHA256 is not None and r.get("prereg_sha256") != PREREG3_SHA256:
            continue
        out.append(r)
    return out


def analysis_frame3(records) -> pd.DataFrame:
    """One row per run, metrics computed from the persisted buffer. `ok` rows only."""
    rows = []
    for r in records:
        if r.get("status") != "ok":
            continue
        base = {k: r.get(k) for k in ("oracle", "arm", "seed", "setting", "budget", "status",
                                      "prereg_sha256", "elapsed_s", "steps",
                                      "n_fallback_generations", "selection")}
        if r.get("full_buffer"):
            base.update(population_metrics3(r["oracle"], r["full_buffer"], r["auc_top10"]))
        else:
            base.update({m: np.nan for m in P3St.METRICS})
        rows.append(base)
    df = pd.DataFrame(rows)
    if len(df):
        collinear = P3St.check_collinear(df, P3St.METRICS)
        if collinear:
            raise RuntimeError(f"metrics are collinear: {collinear} -- the metric list is "
                               f"hashed into the pre-registration and must not contain a "
                               f"rescaling of another metric")
    return df
