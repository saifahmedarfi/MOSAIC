# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: lce
#     language: python
#     name: lce
# ---

# %% [markdown]
# # Phase 3 — Pareto-Front Selection vs Scalarized Selection
#
# **Hypothesis.** Hypervolume is where this project's variance lives: Phase 2 separated the
# three backbones across **0.843 z** on hypervolume against **0.133 z** on `auc_top10`, a
# factor of 6.3, and reordered them. Phase 3 asks whether *selecting* on the front rather
# than on the scalar produces a better one, and what that costs on the scalarized metric.
#
# **Three arms, because `phase0_lib.py` is frozen.** The Pareto operator cannot be patched
# into `run_ga_optimizer`, so it lives in a reimplemented loop — which creates a
# "my reimplementation vs the vendored one" confound. Hence:
#
# | arm | what it is | why |
# |---|---|---|
# | `anchor` | `phase0_lib.run_ga_optimizer`, untouched | fidelity reference |
# | `control` | reimplemented loop, **scalar** selection | the real baseline |
# | `treatment` | reimplemented loop, **NSGA-II** selection | the hypothesis |
#
# `treatment` vs `control` carries the verdict. `anchor` vs `control` is a halting gate.
#
# **Spec:** `docs/superpowers/specs/2026-11-22-phase3-pareto-optimization-design.md`.
# Read §1.2 before quoting any motivation: Phase 2's `scalarization_loss` does **not** mean
# "half the Pareto quality is discarded", and appears nowhere in this phase.
#
# **No GPU.** All three arms are `graph_ga`, which touches no torch or CUDA, and none of the
# nine oracles is ML-based — `drd2`, `jnk3` and `gsk3b` are not in the set. This is CPU work.
#
# **Run order.** Part 5 is Stage A (the pilot). The pre-registration is deliberately
# **unfrozen** until Stage A fills `delta_p` and `seeds`; Part 6's sweep refuses to run until
# it is frozen and committed. That ordering is mechanical, not procedural.

# %% [markdown]
# ## Part 1 — Environment, root, and what this phase may not touch

# %%
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


# %%
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


# %% [markdown]
# ## Part 2 — The pre-registration
#
# **Ships unfrozen on purpose.** `delta_p` and `seeds` are `null` until Stage A fills them,
# and `run_sweep3` refuses to run while they are. No Stage B record can physically exist
# before the rule is frozen — the ordering is enforced by the code, not by timestamps.
#
# `eps_q` is **not** a pilot output. §6.2 fixes it, because a non-inferiority tolerance must
# not be a function of the loss it judges.

# %%
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


# %%
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


# %% @notebook-only
print(f"prereg status    {prereg3_status()}")
print(f"prereg hash      {prereg3_hash()[:16]}...  (changes when Stage A fills delta_p/seeds)")
print(f"eps_q            {PREREG3['eps_q']}  (fixed, not pilot-derived)")
print(f"seed bounds      {PREREG3['seed_bounds']}")
print(f"modal outcome    {PREREG3['declared_modal_outcome']}  (declared in advance)")

# %% [markdown]
# ## Part 3 — The gates
#
# Two inherited from Phase 2, run before anything else:
#
# 1. **The `scikit-learn` tripwire.** On scikit-learn >= 1.3, TDC's `drd2` and `jnk3`
#    pickles load *without raising* and return 0.0 for every molecule. Phase 3 uses neither
#    oracle, so this is not load-bearing here — it is kept because a silent-zero oracle is the
#    kind of failure you check for even when you believe it cannot reach you.
# 2. **The reproduction gate.** Each oracle's components are reimplemented in
#    `phase2_components` and scored against TDC's own scalar. Phase 3 re-runs it and uses the
#    survivors. The tolerance is **inherited from `phase2_stats.GATE_TOL`** and is never
#    restated here: Phase 3 declares no cell-level tolerance of its own (spec §3.3).

# %%
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


# %% @notebook-only
sklearn_tripwire(p0().get_tdc_oracle("drd2"))
GATE3 = reproduction_gate3()
print(f"\nreproduction gate: {GATE3['n_ok']}/{len(PC.COMPONENTS)} oracles reproduce "
      f"(floor {MIN_ORACLES3})")
for o, r in sorted(GATE3["per_oracle"].items()):
    print(f"  {o:20s} max dev {r['max_deviation']:.2e}  reproduces {r['reproduces']}")
print(f"\ncomponents sha256 {GATE3['components_sha256'][:16]}...")

# %% [markdown]
# ## Part 4 — The three arms and the runner
#
# `anchor` calls the vendored loop. `control` and `treatment` call the reimplemented loop
# with the two selection rules. The RNG split is deliberate and load-bearing:
#
# - the **scalar** path is handed `np.random` (the module), so it draws from the same global
#   legacy stream `phase0_lib._make_mating_pool` uses and is bit-identical to it;
# - the **NSGA-II** path is handed `np.random.default_rng(seed)`, because its tournament
#   needs `.integers`, which the module does not have.

# %%
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


# %% [markdown]
# ## Part 5 — Stage A, the pilot
#
# A real three-arm sweep on 3 of the 9 oracles at 5 seeds: **45 runs**, chosen to span the
# component-dimension range (`median1` 2, `fexofenadine_mpo` 3, `osimertinib_mpo` 4).
#
# It produces `delta_p` and the seed count — **not** `eps_q`. It is exploratory, reaches no
# verdict, and with only 3 oracles the shared-oracle bootstrap resamples 3 clusters, so its
# intervals are wide by construction. It is a threshold-setting device, not an estimate.
#
# Run it from the CLI so its log is a file:
#
# ```
# python scripts/phase3_pilot.py --workers 12
# ```
#
# Then paste `delta_p`, `seeds`, `seed_ids` and `seed_ruling` into `PREREG3`, call
# `freeze_prereg3()`, rebuild, and **commit the frozen file before running Part 6**.

# %% @notebook-only
if PILOT_PATH.exists():
    _pilot = json.loads(PILOT_PATH.read_text())
    print(f"Stage A: resolvable={_pilot['resolvable']}  exploratory={_pilot['exploratory']}")
    print(f"  delta_p   {_pilot['delta_p']}")
    print(f"  eps_q     {PREREG3['eps_q']} (fixed, not from the pilot)")
    print(f"  seeds     {_pilot['seeds']}")
    print(f"  ruling    {_pilot['seed_ruling']}")
else:
    print(f"Stage A has not run. Expected {PILOT_PATH.name}.")
    print("Run:  python scripts/phase3_pilot.py --workers 12")

# %% [markdown]
# ## Part 6 — Stage B, the sweep
#
# 9 oracles x 3 arms x seeds, `explore_mid`, budget 1000. Resumable; completed runs are
# skipped. **Never pass `--force` to a full sweep.**
#
# ```
# python scripts/run_sweep_phase3.py --workers 12
# ```
#
# `graph_ga` is CPU-bound Python + RDKit, so expect sub-linear thread scaling under the GIL.
# Time one run and one 12-wide batch before trusting any wall-clock estimate.

# %% @notebook-only
LAUNCH_SWEEP3 = False       # True launches Stage B in this kernel; prefer the CLI

if LAUNCH_SWEEP3:
    _jobs = build_sweep3()
    run_sweep3(_jobs)
    _failed = sweep_failures3(_jobs)
    print(f"\n{len(_jobs) - len(_failed)}/{len(_jobs)} complete")
else:
    print("Stage B not launched from the notebook. Use scripts/run_sweep_phase3.py.")

# %% [markdown]
# ## Part 7 — Loading results and the analysis frame

# %%
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


# %% @notebook-only
RECORDS3 = load_results3()
DF3 = analysis_frame3(RECORDS3)
print(f"loaded {len(RECORDS3)} Phase 3 records")
print(f"analysis frame: {len(DF3)} rows")
if len(DF3):
    print(DF3.groupby("arm")[list(P3St.METRICS)].mean().round(4))

# %% [markdown]
# ## Part 8 — Criterion F, the fidelity gate (halting)
#
# The 95% CI of `score(anchor) - score(control)` must lie entirely within +/- `f_bound`, for
# **both** primary metrics. It is an equivalence test in the TOST sense, so more seeds make it
# **easier** to pass. There is no "CI contains 0" clause: that accepts a point null and gets
# harder with more seeds, which is the wrong direction for a fidelity check.
#
# A failure halts the phase and yields `REFUSED`. A reimplementation that does not reproduce
# vendored `graph_ga` makes the treatment/control contrast meaningless, and reporting it
# anyway would be worse than reporting nothing.

# %% @notebook-only
if len(DF3):
    CF = P3St.criterion_F(DF3)
    print(f"Criterion F: passes={CF['passes']}  width={CF['width']:.4f}  "
          f"bound=+/-{P3St.F_BOUND}")
    for m, r in CF["per_metric"].items():
        print(f"  {m:14s} CI [{r['lo']:+.4f}, {r['hi']:+.4f}]  inside={r['inside']}")
    if not CF["passes"]:
        raise RuntimeError("Criterion F FAILED: the reimplementation does not reproduce "
                           "vendored graph_ga distributionally. Verdict is REFUSED; the "
                           "treatment/control contrast is not interpretable.")
else:
    print("no rows yet -- Criterion F needs Stage B")

# %% [markdown]
# ## Part 9 — Criteria P and Q, and the verdict
#
# Both put the margin **inside** the 95% quantile, and both count un-evaluable draws against
# the criterion: the denominator is `n_boot`, not the surviving draws. A draw that cannot be
# evaluated does not satisfy the criterion.
#
# - **P** — `>=95%` of draws satisfy `score_hv(treatment) - score_hv(control) >= delta_p`
# - **Q** — `>=95%` of draws satisfy `score_auc(control) - score_auc(treatment) <= eps_q`
#
# Q is **co-primary**, not secondary. An optimizer that wins on the metric it optimizes has
# shown nothing; the defensible claim is front quality bought *without* surrendering
# scalarized performance.
#
# Both are also evaluated on the **6 held-out oracles** Stage A did not touch. The all-9
# result is primary; a disagreement between them is reported as a finding, not reconciled.

# %% @notebook-only
if len(DF3) and PREREG3["delta_p"] is not None:
    CP = P3St.criterion_P(DF3, PREREG3["delta_p"])
    CQ = P3St.criterion_Q(DF3, PREREG3["eps_q"])
    V = P3St.phase3_verdict(CP, CQ, CF, df=DF3)

    print(f"Criterion P: diff={CP['diff']:+.4f}  rate={CP['rate']:.4f}  "
          f"delta_p={PREREG3['delta_p']:.4f}  passes={CP['passes']}")
    print(f"Criterion Q: diff={CQ['diff']:+.4f}  rate={CQ['rate']:.4f}  "
          f"eps_q={PREREG3['eps_q']:.4f}  passes={CQ['passes']}")
    print(f"  draws scored: P {CP['draws_scored']}/{CP['n_boot']}, "
          f"Q {CQ['draws_scored']}/{CQ['n_boot']}")
    print(f"  oracles (both arms present): P {CP['n_oracles']}, Q {CQ['n_oracles']}")

    _held = DF3[DF3["oracle"].isin(PREREG3["held_out_oracles"])]
    if len(_held):
        CPh = P3St.criterion_P(_held, PREREG3["delta_p"])
        CQh = P3St.criterion_Q(_held, PREREG3["eps_q"])
        print(f"\nheld-out 6: P passes={CPh['passes']} (diff {CPh['diff']:+.4f}), "
              f"Q passes={CQh['passes']} (diff {CQh['diff']:+.4f})")

    print(f"\nVERDICT: {V['verdict']}")
    print(f"  {V['consequence']}")
    print(f"  arms checked: {V.get('arms_checked')}  missing: {V.get('missing_arms')}  "
          f"unexpected: {V.get('unexpected_arms')}")
    print(f"\ndeclared modal outcome was {PREREG3['declared_modal_outcome']}")
else:
    print("verdict needs Stage B rows and a frozen delta_p")

# %% [markdown]
# ## Part 10 — Figures

# %% @notebook-only
if len(DF3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, m in zip(axes, P3St.METRICS):
        for i, a in enumerate(P3St.ARMS):
            v = DF3[DF3["arm"] == a][m].dropna()
            ax.scatter(np.full(len(v), i) + np.random.default_rng(0).normal(0, 0.06, len(v)),
                       v, s=10, alpha=0.5)
            ax.scatter([i], [v.mean()], marker="_", s=600, c="k", zorder=3)
        ax.set_xticks(range(len(P3St.ARMS)))
        ax.set_xticklabels(P3St.ARMS, rotation=15)
        ax.set_title(m)
        ax.grid(alpha=0.3)
    fig.suptitle("Phase 3 -- per-run metrics by arm (bar = mean)")
    fig.tight_layout()
    fig.savefig(PHASE3_DIR / "phase3_arms_by_metric.png", dpi=150)
    print(f"wrote {PHASE3_DIR / 'phase3_arms_by_metric.png'}")

# %% [markdown]
# ## Part 11 — What this phase can and cannot conclude
#
# Carry these into any write-up. They are not boilerplate; each one bounds a claim.
#
# 1. **L1 — oracle reuse.** All 9 oracles appeared in Phases 0-2. No held-out set exists in
#    PMO, and the "held-out 6" above are held out from *Stage A*, not from the project.
# 2. **L2 — privileged information.** The treatment arm sees the component decomposition.
#    The result is about *known* MPO objectives, and must be titled that way.
# 3. **L3 — five selection events.** `graph_ga` spends budget 1000 in 5-6 generations, so the
#    operator fires about five times. A `PARETO-NULL` cannot be distinguished from "too few
#    generations to tell". Accepted deliberately to keep the configuration commensurable with
#    Phases 0-2.
# 4. **L4 — one optimizer family.** Nothing here generalises to the RL/LM backbones.
# 5. **L5 — partial scalarization on 2 of 9 oracles.** For `sitagliptin_mpo` and
#    `zaleplon_mpo` one "component" is `_isomer(...)`, itself a geometric mean. It is treated
#    as ONE Pareto dimension, because decomposing it would break the reproduction gate.
# 6. **L6 — runs are not individually reproducible.** `set_all_seeds` seeds process-global
#    RNGs and the sweep runs many-wide. Distributional results are sound; cell-level
#    provenance is not. See the Phase 0 and Phase 1 addenda.
# 7. **L7 — the premise is a Phase 2 observation from a phase that returned `REFUSED`.**
# 8. **L8 — the novelty check is still not done.** A literature search, not compute. It gates
#    framing, not execution.
# 9. **L9 — what the prereg hash does not cover.** It covers the decision rule, not the code
#    computing its inputs. `phase3_pareto.py` and `phase2_components.py` sit outside it; the
#    reproduction and fidelity gates stand in their place.
