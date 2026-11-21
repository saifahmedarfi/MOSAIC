# %% [markdown]
# # Phase 2 — Metric sensitivity
#
# **Gated by Phase 1's verdict: `NULL`.** Phase 1 found no landscape-conditional exploration
# effect. Phase 2 asks a different question of the same three backbones: *does the ranking of
# methods depend on which metric scores it?* Specifically, does `auc_top10` (PMO's scalar
# integral) and `hypervolume` of the component-space Pareto front crown different backbones,
# and if so by more than one pooled seed SD?
#
# No new optimizer and no new oracle: nine multi-property oracles, the three Phase 1 backbones,
# `explore_mid` only. **405 runs** = 9 oracles x 3 backbones x **15 seeds**, about 3 h at 12
# workers.
#
# The design is `docs/superpowers/specs/2026-11-20-phase2-metric-sensitivity-design.md`. This
# file is the single editable source; the notebook, `src/phase2_lib.py` and
# `scripts/run_sweep_phase2.py` are generated from it by `python src/build_phase2.py`.
#
# **This notebook was built on a machine without TDC, the vendored `mol_opt` clone, `results/`
# or a GPU. It has not been executed end to end.** Everything that needs them is a gate that
# halts rather than degrades.

# %%
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

# %% [markdown]
# ---
# ## Part 0 — Pre-registration v3
#
# **The seed count is 15, by ruling, and the ruling overrides the gate's own rule in the open.**
# `scripts/phase2_power.py::choose_seeds` selects the smallest n reaching 80% power at a 1-SD
# disagreement, and returned **5**. That rule is deliberately *not* amended. The controller ruled
# 15 on dominance (design section 5.3.2): n=15 is never worse than n=5 at any simulated effect,
# and is decisively better at 0.4-0.6 SD, where the noise-corrected Phase 1 effect most likely
# sits (power 0.37-0.80 at n=5 against 0.92-1.00 at n=15; the detectable floor drops from
# 0.6 SD to 0.4 SD). The evidence is `Phase 2/phase2_power_gate.json`, committed and not
# regenerated here.
#
# Realistic outcome space, declared in advance (design 5.3.3): `COSMETIC` is the most likely
# non-null verdict, and `METRIC-DEPENDENT` needs about 1.2 SD at every n.

# %%
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
                     # 1e-6 absolute per cell, set BEFORE any Phase 2 run existed (no
                     # results/runs_phase2/, the sweep launch switch off). The first freeze used 1e-9,
                     # which would halt on GPU floating-point dust: reduction order is
                     # nondeterministic and Phase 1 did not force determinism. The mean signed
                     # difference is reported too, because a systematic shift is what would
                     # actually invalidate the cross-phase comparison and a max cannot see it.
                     "metric": "auc_top10", "tolerance": 1e-6, "tolerance_kind": "absolute, per cell",
                     "also_reported": "mean signed difference (phase1 - phase2)",
                     "on_deviation": "halt",
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
PREREG2_SHA256 = "42243abd61d9d5e6d6ca7d2d0f935baf3ea46fcc74f9af545eb2f1e5d4b2dd3e"


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

# %% @notebook-only
PREREG2_STATUS = prereg2_status()
print(f"pre-registration status: {PREREG2_STATUS}")
if PREREG2_STATUS == "mismatch":
    print("!! PRE-REGISTRATION MISMATCH -- Part 10 will REFUSE to produce a verdict.")
elif PREREG2_STATUS == "unfrozen":
    freeze_prereg2()
    print(f"FROZEN -> {PREREG2_PATH}\n  COMMIT THIS FILE NOW, before any run exists.")
print(f"seeds {PREREG2['seeds']} x {len(PREREG2['backbones'])} backbones x "
      f"{len(PREREG2['oracles'])} oracles = "
      f"{PREREG2['seeds'] * len(PREREG2['backbones']) * len(PREREG2['oracles'])} runs")
print(f"power-gate evidence: {POWER_GATE_PATH} (committed; reported, not re-run)")
if POWER_GATE_PATH.exists():
    _pg = json.loads(POWER_GATE_PATH.read_text())
    print(f"  gate halted: {_pg['halt']}; rule chose n={_pg['chosen']}; "
          f"ruling fixed n={PREREG2['seeds']}")

# %% [markdown]
# ---
# ## Part 1 — Environment, and the tripwire that matters
#
# On scikit-learn >= 1.3, TDC's `drd2` oracle loads with no error and returns `0.0` for every
# molecule, because a tree-node dtype change silently breaks the pre-1.3 pickles. Phase 1's
# `valsartan_smarts` is the proof that this failure class is invisible: all 27 of its runs
# scored exactly 0.0 and nothing raised. The tripwire runs **before** anything else.

# %%
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

# %% @notebook-only
_scores = sklearn_tripwire(p0().get_tdc_oracle("drd2"))
print(f"drd2 probe scores: {[round(s, 4) for s in _scores]}")
print("tripwire OK -- drd2 is live")

# %% [markdown]
# ---
# ## Part 2 — Candidate oracles, and the Phase 1 overlap reproducibility gate
#
# The candidate list is **derived**, not typed: it is the set of oracles `phase2_components`
# knows how to decompose, cross-checked against the PMO suite read from TDC. Two independent
# sources must agree.
#
# **Overlap gate (design 8, L8).** Phase 1 ran `explore_mid` on seven of the nine candidates x 3
# backbones x seeds 0-2, so **63** cells coincide with Phase 2's grid. `ranolazine_mpo` and
# `sitagliptin_mpo` were Phase 0 oracles that Phase 1 reused rather than re-ran, so its manifest
# has no rows for them; the other 18 cells could only be checked against Phase 0's own records,
# which are not wired in here. Same oracle, backbone,
# setting and seed must give the same `auc_top10` to 1e-6 absolute per cell (GPU reduction order is nondeterministic, so 1e-9 would halt on
# floating-point dust); if not, Phase 2's numbers are not
# comparable to Phase 1's and the cross-phase argument collapses. A deviation halts. Absent
# records are recorded as "could not run", never "passed".
#
# **The Phase 1 frame is filtered to `explore_mid` before it reaches `phase2_stats`.** That
# module defines a cell as (oracle, backbone) and ignores `setting`, which is right for a
# single-setting grid and silently wrong on three settings (setting variance inflates the seed
# SD).

# %%
CANDIDATE_ORACLES = sorted(PC.COMPONENTS)

OVERLAP_TOL = PREREG2["overlap_gate"]["tolerance"]
OVERLAP_PATH = PHASE2_DIR / "phase2_overlap_gate.json"


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


def _persist_overlap(result, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(result, indent=2, sort_keys=True, default=float))
    return result


def phase1_overlap_gate(p1, p2, tol=OVERLAP_TOL, path=None):
    """Compare auc_top10 run-for-run on (oracle, backbone, seed) cells present in both.

    Returns {"status": "passed" | "could_not_run", "n_compared", "max_deviation",
    "mean_signed_difference", ...}; raises GateHalt on any per-cell |deviation| above `tol`.
    `mean_signed_difference` is mean(phase1 - phase2), sign kept: a systematic shift is what
    would invalidate the cross-phase comparison and an absolute max cannot see one.
    `could_not_run` when nothing overlaps -- never reported as a pass. With `path`, the result
    (including a failure) is written there before any raise."""
    if p1 is None or len(p1) == 0 or p2 is None or len(p2) == 0:
        return _persist_overlap({"status": "could_not_run", "n_compared": 0, "max_deviation": None,
                                 "mean_signed_difference": None, "tolerance": tol,
                                 "reason": "Phase 1 or Phase 2 run records are absent"}, path)
    a = phase1_explore_mid(p1)
    b = p2[p2["status"] == "ok"]
    key = ["oracle", "backbone", "seed"]
    m = a[key + ["auc_top10"]].merge(b[key + ["auc_top10"]], on=key, suffixes=("_p1", "_p2"))
    if len(m) == 0:
        return _persist_overlap({"status": "could_not_run", "n_compared": 0, "max_deviation": None,
                                 "mean_signed_difference": None, "tolerance": tol,
                                 "reason": "no (oracle, backbone, seed) cell exists in both phases"}, path)
    signed = m["auc_top10_p1"] - m["auc_top10_p2"]
    dev = signed.abs()
    worst = float(dev.max())
    res = {"n_compared": int(len(m)), "max_deviation": worst,
           "mean_signed_difference": float(signed.mean()), "tolerance": tol}
    if not np.isfinite(worst) or worst > tol:
        res["status"] = "failed"
        _persist_overlap(res, path)
        bad = m.loc[dev.sort_values(ascending=False).index[:5], key + ["auc_top10_p1", "auc_top10_p2"]]
        raise GateHalt(f"Phase 1 overlap NOT reproduced: max |auc_top10 diff| = {worst:.3g} "
                       f"> {tol:g} over {len(m)} cells (mean signed diff "
                       f"{res['mean_signed_difference']:+.3g}). Worst:\n{bad.to_string(index=False)}")
    res["status"] = "passed"
    return _persist_overlap(res, path)

# %% @notebook-only
check_candidates_against_pmo(pmo_names_via_phase1())
print(f"candidates ({len(CANDIDATE_ORACLES)}), derived from phase2_components and "
      f"confirmed in the PMO suite:")
for _o in CANDIDATE_ORACLES:
    print(f"  {_o:20s} {len(PC.COMPONENTS[_o])} components")

# %% [markdown]
# ---
# ## Part 3 — The reproduction gate
#
# Each oracle's components are reimplemented in `src/phase2_components.py`. For each candidate
# the frozen probe set is scored through our components and through TDC's oracle, and
#
# ```
# max | gmean(our_components(smi)) - tdc_oracle(smi) | <= 1e-6
# ```
#
# on a probe set that can fail (it must exercise every component and score non-zero on at least
# a fifth of molecules). Oracles that do not reproduce are excluded by name with the deviation
# recorded. **Below `MIN_ORACLES` = 5 survivors the phase halts.** This is the first point
# where TDC is actually required.

# %%
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

# %% @notebook-only
GATE2 = reproduction_gate(CANDIDATE_ORACLES, p0().get_tdc_oracle)
for _o, _g in GATE2["per_oracle"].items():
    print(f"  {_o:20s} {_g['status']:15s} max|diff| {_g['max_deviation']:.2e}  "
          f"probe n={_g['n_probe']}  nonzero {_g['frac_nonzero']:.2f}  "
          f"min range {_g['min_range']:.2f}")
print(f"\nsurviving {len(GATE2['surviving'])}: {GATE2['surviving']}")
print(f"excluded  {GATE2['excluded']}")
print(f"gate written -> {save_gate(GATE2)}  (commit it before the sweep)")

# %% [markdown]
# ---
# ## Part 4 — The runner
#
# **Phase 2 writes to `results/runs_phase2/`** -- never `runs/` or `runs_phase1/`, whose
# records sit under other pre-registrations.
#
# **The full molecule buffer is persisted.** Phase 1's `execute_run1` built `score_trace` from
# `orc.mol_buffer` but kept only `(score, call_index)`, discarding the SMILES keys, so its 405
# runs hold 4,050 molecules instead of 405,000 and no population-level re-analysis is possible.
# Here every run keeps `(smiles, score, call_index)` for all of its molecules. Component
# vectors are not stored; they are recomputed from SMILES.

# %%
def run_key2(oracle: str, backbone: str, seed: int) -> str:
    return f"{oracle}__{backbone}__{PREREG2['setting']}__seed{seed}"


def run_path2(*args) -> Path:
    return RUNS2_DIR / f"{run_key2(*args)}.json"


def buffer_triples(mol_buffer) -> list:
    """BudgetedOracle.mol_buffer ({smiles: [score, call_index]}) as a list of
    [smiles, score, call_index] in call order."""
    return [[str(smi), float(v[0]), int(v[1])]
            for smi, v in sorted(mol_buffer.items(), key=lambda kv: kv[1][1])]


def record_complete(path) -> bool:
    """A record counts as done only if it is status 'ok' AND at the full pre-registered budget.
    An error record (e.g. a transient OOM) is retried on resume, and a reduced-budget dry-run
    record can never be mistaken for a real run."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
    except ValueError:
        return False
    return d.get("status") == "ok" and d.get("budget") == PREREG2["budget"]


def execute_run2(oracle_name: str, backbone: str, seed: int, budget: int = None,
                 force: bool = False, verbose: bool = False) -> dict:
    """One Phase 2 run. Resumable: a complete record (see record_complete) is returned untouched."""
    budget = PREREG2["budget"] if budget is None else budget
    p = run_path2(oracle_name, backbone, seed)
    if not force and record_complete(p):
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
    # Load phase0_lib and run the drd2 tripwire HERE, in the main thread, before any worker
    # exists. Otherwise a missing torch/TDC, a wrong LCE_ROOT or a regressed scikit-learn only
    # surfaces inside the pool, every job prints CRASHED and the sweep still "finishes".
    sklearn_tripwire(p0().get_tdc_oracle("drd2"))
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 4)
    todo = [j for j in jobs if force or not record_complete(run_path2(*j))]
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


def sweep_failures(jobs) -> list:
    """Jobs with no complete record on disk: crashed (no record) or errored / wrong budget."""
    return [j for j in jobs if not record_complete(run_path2(*j))]


def dry_run_check(jobs, budget: int = 200):
    """Run `jobs` at a reduced budget into a TEMPORARY runs directory, so a dry-run record can
    never land in (or be mistaken for a record in) the real RUNS2_DIR, even if interrupted."""
    import tempfile
    global RUNS2_DIR
    real = RUNS2_DIR
    out = []
    with tempfile.TemporaryDirectory() as td:
        RUNS2_DIR = Path(td)
        try:
            for j in jobs:
                out.append(execute_run2(*j, budget=budget, force=True))
        finally:
            RUNS2_DIR = real
    return out

# %% [markdown]
# ---
# ## Part 5 — Metrics from the persisted populations
#
# All free: no oracle calls. Definitions are design 5.2. The population is every molecule the
# run scored. Component vectors come from `phase2_components`; the recorded scalar score is what
# the optimizer saw. Fingerprints are ECFP4 bit vectors (Morgan radius 2, 2048 bits).
#
# A run whose metrics cannot be computed is marked `metric_error` and **named in the loader's
# output**; it is never given an invented value, and `phase2_stats` excludes it as not-ok.

# %%
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

# %% @notebook-only
RUNS2_DIR.mkdir(parents=True, exist_ok=True)
SWEEP2 = build_sweep2()
print(f"Phase 2 sweep: {len(SWEEP2)} runs ({len(CANDIDATE_ORACLES)} oracles x "
      f"{len(PREREG2['backbones'])} backbones x {PREREG2['seeds']} seeds), "
      f"{len(SWEEP2) * 301.9 / 12 / 3600:.1f} h at 12 workers (Phase 1: 301.9 s/run)")

# %% @notebook-only
# A 2-run dry check at reduced budget. The real sweep goes through
# scripts/run_sweep_phase2.py -- a notebook kernel holding a multi-hour job is a liability.
DRY_RUN_FIRST = True

if DRY_RUN_FIRST:
    print("dry check (2 runs at reduced budget, temporary directory) ...")
    for _r in dry_run_check([(CANDIDATE_ORACLES[0], "reinvent", 0),
                             (CANDIDATE_ORACLES[0], "graph_ga", 0)]):
        print(f"  {_r['oracle']}/{_r['backbone']} AUC {_r.get('auc_top10', float('nan')):.4f} "
              f"{_r['status']}  buffer {len(_r.get('full_buffer', []))} molecules")
    print("\npipeline OK\n")

# %% @notebook-only
LAUNCH_SWEEP2 = False     # True launches ~3 h of compute in this kernel

if LAUNCH_SWEEP2:
    RESULTS2 = run_sweep2(SWEEP2)
else:
    RESULTS2 = load_results2()
    print(f"loaded {len(RESULTS2)} completed Phase 2 runs "
          f"(set LAUNCH_SWEEP2=True, or run scripts/run_sweep_phase2.py)")

# %% [markdown]
# ### Phase 1 overlap gate (run once Phase 2 records exist)
#
# Halts on any per-cell `auc_top10` deviation above 1e-6 and persists the result, including the
# mean signed difference, to `Phase 2/phase2_overlap_gate.json`. The Phase 1 manifest is committed at
# `Phase 1/phase1_run_manifest.csv`.

# %% @notebook-only
OVERLAP = None
if RESULTS2.empty:
    OVERLAP = {"status": "could_not_run", "reason": "no Phase 2 records yet"}
elif not PHASE1_MANIFEST.exists():
    OVERLAP = {"status": "could_not_run", "reason": f"{PHASE1_MANIFEST} absent"}
else:
    OVERLAP = phase1_overlap_gate(pd.read_csv(PHASE1_MANIFEST), RESULTS2, path=OVERLAP_PATH)
print(f"overlap gate: {OVERLAP}")
if OVERLAP["status"] == "could_not_run":
    print("NOT VERIFIED -- 'could not run' is not 'passed'; Part 10 will REFUSE a verdict.")
else:
    print(f"persisted -> {OVERLAP_PATH}  (commit it)")

# %% [markdown]
# ---
# ## Part 6 — Criterion 0: is there anything to rank?
#
# Over the primary pair only: does each metric have an argmax backbone that replicates in at
# least 95% of hierarchical bootstrap draws (oracles resampled once per draw, shared across
# backbones, winner re-selected inside each draw)?

# %% @notebook-only
def analysis_frame2(df):
    """ok runs only. Phase 2 has one setting, so no setting filter can leak variance."""
    return df[df["status"] == "ok"].copy() if len(df) else df


C0 = CA = CB = CC = CG2 = None
if not RESULTS2.empty:
    _d = analysis_frame2(RESULTS2)
    C0 = PS.criterion_0(_d)
    print(f"Criterion 0 state: {C0['state']}  ({C0.get('n_stable')} of 2 stable)")
    print(json.dumps({k: v for k, v in C0.items() if k not in ("state", "n_stable")},
                     indent=1, default=float)[:1500])

# %% [markdown]
# ---
# ## Part 7 — Criterion A: do the two metrics crown different backbones?

# %% @notebook-only
if C0 is not None:
    CA = PS.criterion_A(_d)
    print(json.dumps(CA, indent=1, default=float)[:1500])
    print(f"\nCriterion A: {'PASS' if CA['passes'] else 'FAIL'}")

# %% [markdown]
# ---
# ## Part 8 — Criterion B (secondary metrics, reported) and Criterion C (the honesty veto)
#
# C passes iff the hypervolume winner beats the AUC winner, on hypervolume, by more than one
# pooled seed SD. `C_THRESHOLD` is not lowered; a real winner change smaller than seed noise is
# `COSMETIC`, and that was the most likely non-null verdict before any run (design 5.3.3).

# %% @notebook-only
if C0 is not None:
    CB = PS.criterion_B(_d)
    for _m, _r in CB["per_metric"].items():
        print(f"  {_m:18s} {_r}")
    if CB["degenerate_metrics"]:
        print(f"degenerate (excluded before correction): {CB['degenerate_metrics']}")
    CC = PS.criterion_C(_d, C0, CA)
    print(f"\nCriterion C: gap {CC['gap']:.3f} SD vs threshold {CC['threshold']} -> "
          f"{'PASS' if CC['passes'] else 'FAIL'}")

# %% [markdown]
# ---
# ## Part 9 — Criterion G2: scalarization loss (descriptive)
#
# The fraction of Pareto-optimal molecules whose scalar score falls below the run's top-10
# threshold. Enters no rule; it is the number that decides whether a Pareto optimizer is worth
# building.

# %% @notebook-only
if C0 is not None:
    CG2 = PS.criterion_G2(_d)
    print(f"mean scalarization_loss {CG2['mean']:.4f}  95% CI [{CG2['lo']:.4f}, {CG2['hi']:.4f}]")

# %% [markdown]
# ---
# ## Part 10 — The verdict
#
# Applies the rule frozen in Part 0. **Refuses** on a pre-registration mismatch, on a metric
# list whose hash differs from the frozen one, and on a failed overlap gate.

# %% @notebook-only
V2 = None
if prereg2_status() == "mismatch" or PS.metrics_hash() != PREREG2["metrics_sha256"]:
    print("REFUSING TO PRODUCE A VERDICT -- pre-registration or metric-list hash mismatch.")
    V2 = {"verdict": "REFUSED", "reason": "hash mismatch"}
elif RESULTS2.empty or C0 is None:
    print("No Phase 2 results yet.")
    V2 = {"verdict": "NO_DATA"}
elif OVERLAP is None or OVERLAP["status"] != "passed":
    # Records exist and the Phase 1 manifest is committed, so "could not run" means something
    # is broken, and the cross-phase comparison is unsupported. Same treatment as a hash mismatch.
    print("REFUSING TO PRODUCE A VERDICT -- the Phase 1 overlap gate did not pass: "
          f"{None if OVERLAP is None else OVERLAP['status']}.")
    V2 = {"verdict": "REFUSED", "reason": "Phase 1 overlap gate did not pass"}
else:
    _n_ok = int((RESULTS2["status"] == "ok").sum())
    _n_exp = len(SWEEP2)
    V2 = PS.phase2_verdict(C0, CA, CB, CC, CG2)
    V2.update(n_runs_ok=_n_ok, n_runs_expected=_n_exp, prereg_sha256=PREREG2_SHA256,
              overlap_gate=OVERLAP, decided_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 74)
    print(f"PHASE 2 VERDICT: {V2['verdict']}")
    print("=" * 74)
    print(V2["consequence"])
    print(f"\nruns {_n_ok}/{_n_exp}")
    if _n_ok < _n_exp:
        print("INCOMPLETE SWEEP -- verdict is provisional.")
    (PHASE2_DIR / "phase2_verdict.json").write_text(json.dumps(V2, indent=2, default=str))
    print(f"\nwritten -> {PHASE2_DIR / 'phase2_verdict.json'}  (commit it)")

# %% [markdown]
# ---
# ## Part 11 — Figures

# %% @notebook-only
def plot_scores_by_metric(df, save: bool = True):
    import matplotlib.pyplot as plt
    d = analysis_frame2(df)
    bbs = sorted(d["backbone"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, m in zip(axes, PS.PRIMARY_PAIR):
        sc = PS.metric_score(d, m)
        ax.bar(range(len(bbs)), [sc.get(b, np.nan) for b in bbs], color="#2563eb")
        ax.set_xticks(range(len(bbs)))
        ax.set_xticklabels(bbs, rotation=15)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(m)
        ax.set_ylabel("score (pooled seed SDs)")
        ax.grid(alpha=0.3)
    fig.suptitle("Primary pair: backbone score in units of seed noise")
    fig.tight_layout()
    if save:
        fig.savefig(PHASE2_DIR / "phase2_primary_pair.png", dpi=160, bbox_inches="tight")
    return fig


if not RESULTS2.empty:
    plot_scores_by_metric(RESULTS2)
    print(f"figure -> {PHASE2_DIR}")

# %% [markdown]
# ---
# ## Part 12 — What this phase does *not* decide
#
# 1. **The novelty check (L4).** Whether metric-sensitivity analysis for molecular optimization
#    is already published is unverified. It is a literature search and it gates the framing of
#    the write-up.
# 2. **Oracle reuse (L1).** All nine candidates appeared in Phases 0-1; no clean held-out set
#    exists inside PMO. External validation needs oracles from outside it.
# 3. **Whether a Pareto optimizer is worth building.** `scalarization_loss` (Part 9) is the
#    measurement that informs that, not a decision.
# 4. **Which verdict.** `BENCHMARK-INSENSITIVE` is the modal outcome and `COSMETIC` the most
#    likely non-null one (design L6). Both are real claims about the benchmark.
