# %% [markdown]
# # Phase 1 — Characterisation
#
# **Gated by Phase 0's verdict: `PARTIAL`.** That verdict's frozen consequence is binding:
#
# > The landscape dependence is real but our diagnostics do not capture it. Proceed to Phase 1
# > for **C1 (characterisation) only**; **redesign the diagnostic feature set before attempting
# > the controller**. **Downgrade venue expectation** to a benchmark/characterisation
# > contribution.
#
# So this notebook builds **no controller**. It answers four questions and stops:
#
# | | Question | Why it is here |
# |---|---|---|
# | **A** | Does the flip survive at 23 oracles? | Phase 0 confirmed 2 pairs, in 1 of 3 backbones, both sharing the `isomer` cluster. Thin. |
# | **B** | Does a *redesigned* feature set predict the winner? | Phase 0's screen failed, but `gini` (ρ = −0.655) was underpowered at n=8, not absent. |
# | **C** | **Is the flip worth more than the probe it costs?** | Never asked in Phase 0. On Phase 0's own numbers the answer looks like *no*. |
# | **D** | Does the best *method* flip, not just the best knob? | Phase 0's method-level margins were ~10× its knob effects. |
#
# **C can veto everything.** A perfect cluster-conditional chooser bought +0.0074 AUC
# (reinvent), +0.0050 (augmented_memory) and +0.0000 (graph_ga) in Phase 0, while the probe
# costs 300 of 1000 oracle calls. If that holds here, the controller is not viable at this
# budget and that is the headline result whatever A, B and D say.
#
# ---
#
# ### What runs, and what is reused
#
# Phase 0's 8 oracles **generated** the `gini` hypothesis, so they cannot confirm it.
#
# | Set | Oracles | Role |
# |---|---|---|
# | Discovery | the 8 from Phase 0 | hypothesis source; runs reused, **not** re-run |
# | Confirmation | the 15 remaining PMO oracles | carries every confirmatory test |
#
# **405 new runs** (15 × 3 backbones × 3 settings × 3 seeds) at 1000 calls — about 2.7 h at
# 12 workers on one RTX 4070 Super, from Phase 0's measured 286.5 s/run.

# %%
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

# %% [markdown]
# ---
# ## Part 0 — Pre-registration v2
#
# Phase 0's pre-registration had three defects, all disclosed in its own results document.
# This version fixes them, and the fixes are the reason a v2 exists at all:
#
# 1. **"the LM backbone" was ambiguous.** Both REINVENT and Augmented Memory are the LM
#    backbone; the implementation picked `reinvent`, which confirmed **zero** flips, while the
#    2 confirmed pairs came from `augmented_memory`. The diagnostic was screened against the
#    arm whose rankings were never shown to flip. **Here the backbone is named.**
# 2. **The hash did not cover what it claimed.** The prereg said the feature list was "frozen
#    in Part 8 and hashed with this document", but hashed a *copy* inside the prereg dict. The
#    two could diverge silently. **Here the hash is taken from the computed tuple.**
# 3. **Two features were mathematically dead.** `frac_above_0.7` and `frac_above_0.9` were
#    constant (no oracle exceeded 0.5), returned ρ = NaN, and still consumed Holm budget.
#    **Here thresholds are quantiles of the observed reward, which cannot go degenerate.**

# %%
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

# %% [markdown]
# ---
# ## Part 1 — Environment, and the tripwire that matters
#
# Phase 1 reuses Phase 0's backbones, oracles, budget semantics and metric by importing
# `phase0_lib`. That import is deliberately **not** at the top of this notebook: it clones
# `mol_opt` and loads the ChEMBL prior, so it needs network and a GPU-capable environment.
# Everything above this point runs anywhere.
#
# **The `scikit-learn` pin is load-bearing.** On sklearn 1.9.1, TDC's `drd2` oracle loads with
# no error at all and then returns `0.0` for every molecule — the entire bioactivity cluster
# would score zero and nothing would raise. Phase 0 caught it; the four probe values below are
# the regression tripwire, and they are asserted **before** the sweep, not after.

# %% @notebook-only
PROBE_SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1CCC[C@H]1c1cccnc1"]
EXPECTED_DRD2_NONZERO = True

import phase0_lib as P0  # noqa: E402

print(f"phase0_lib loaded: {len(P0.ALL_ORACLES)} discovery oracles, "
      f"prereg status {P0.PREREG_STATUS}")

_drd2 = P0.get_tdc_oracle("drd2")
_scores = [float(_drd2(s)) for s in PROBE_SMILES]
print(f"drd2 probe scores: {[round(s, 4) for s in _scores]}")
assert any(s > 0 for s in _scores), (
    "drd2 returned all zeros -- the scikit-learn pin has regressed to >=1.3 and EVERY "
    "bioactivity number would be void. Reinstall scikit-learn==1.2.2 before continuing."
)
print("tripwire OK -- drd2 is live")

# %% [markdown]
# ---
# ## Part 2 — Oracle inventory and cluster tags
#
# The 15 confirmation oracles need cluster tags, and **how** they are assigned is the one place
# a reviewer can credibly allege the flip was manufactured. So the tags come from PMO's own
# published oracle categories, read out of the installed TDC metadata rather than typed in from
# memory, and each assignment records its source. An oracle whose category does not map onto one
# of Phase 0's four clusters is tagged `unassigned` and **excluded from every cluster-level
# test** rather than placed by judgement.

# %%
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

# %% @notebook-only
ALL_PMO = pmo_oracle_names()
DISCOVERY = sorted(P0.ALL_ORACLES)
CONFIRMATION = sorted(o for o in ALL_PMO if o not in set(DISCOVERY))
ORACLE_TAGS = {o: tag_oracle(o) for o in ALL_PMO}

print(f"PMO suite        {len(ALL_PMO)} oracles")
print(f"discovery        {len(DISCOVERY)}  (Phase 0, runs reused)")
print(f"confirmation     {len(CONFIRMATION)}  (new runs)")
assert not (set(DISCOVERY) & set(CONFIRMATION)), "discovery and confirmation must be disjoint"

_by_cluster = {}
for o in CONFIRMATION:
    _by_cluster.setdefault(ORACLE_TAGS[o], []).append(o)
for c in sorted(_by_cluster):
    print(f"  {c:12s} {len(_by_cluster[c]):2d}  {', '.join(_by_cluster[c])}")
_un = _by_cluster.get("unassigned", [])
if _un:
    print(f"\n{len(_un)} oracle(s) unassigned and excluded from cluster-level tests: "
          f"{', '.join(_un)}")

# %% [markdown]
# ---
# ## Part 3 — The redesigned diagnostic probe
#
# Same protocol as Phase 0 — 300 budgeted calls, 200 prior samples plus 100 local edits around
# the top 5 hits, from a fixed policy that is **none of the competitors** so Criterion B cannot
# reduce to "this method's early behaviour predicts its own later behaviour".
#
# What changed is the feature set: absolute thresholds become **quantile gaps**, and the two
# dead features are gone. See `src/phase1_stats.py`.

# %%
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

# %% [markdown]
# ---
# ## Part 4 — The runner
#
# **Phase 1 writes to `results/runs_phase1/`, never `results/runs/`.** Phase 0's 216 records
# live under a different pre-registration hash, and `phase0_lib.load_results()` globs its whole
# directory — mixing them would silently blend two rules into one analysis.
#
# Two differences from Phase 0's runner, both deliberate:
#
# 1. **`score_trace` is persisted** — the `(score, call_index)` pairs — so AUC is exactly
#    computable at *any* truncation. Phase 0 stored only `top10_smiles` and a coarse history
#    grid, which is why Criterion C's probe cost cannot be measured exactly on its runs.
# 2. **Its own cluster lookup.** `phase0_lib.execute_run` does `ORACLE_TO_CLUSTER[oracle]`,
#    and that dict is derived from Phase 0's 8-oracle pre-registration, so it raises `KeyError`
#    on every confirmation oracle. `phase0_lib` is not edited — it is wrapped.

# %%
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

# %% @notebook-only
RUNS1_DIR = P0.RESULTS_DIR / "runs_phase1"
RUNS1_DIR.mkdir(parents=True, exist_ok=True)
DIAG1_PATH = P0.RESULTS_DIR / "diagnostics_phase1.json"

SWEEP1 = build_sweep1(CONFIRMATION)
print(f"Phase 1 sweep: {len(SWEEP1)} runs "
      f"({len(CONFIRMATION)} oracles x {len(PREREG1['backbones'])} backbones x "
      f"{len(PREREG1['settings'])} settings x {len(PREREG1['seeds'])} seeds)")
print(f"estimated wall clock at 12 workers: "
      f"{len(SWEEP1) * 286.5 / 12 / 3600:.1f} h  (Phase 0 measured 286.5 s/run)")

# %% @notebook-only
# A 2-run dry check at reduced budget, exactly as Phase 0 did. The real sweep goes through
# scripts/run_sweep_phase1.py -- a notebook kernel holding a multi-hour job is a liability.
DRY_RUN_FIRST = True

if DRY_RUN_FIRST and CONFIRMATION:
    print("dry check (2 runs at reduced budget) ...")
    for _j in [(CONFIRMATION[0], "reinvent", "explore_mid", 0),
               (CONFIRMATION[0], "graph_ga", "explore_mid", 0)]:
        _p = run_path1(*_j)
        _bak = _p.read_text() if _p.exists() else None
        _r = execute_run1(*_j, budget=200, force=True)
        print(f"  {run_key1(*_j):56s} AUC {_r.get('auc_top10', float('nan')):.4f} "
              f"{_r['status']}")
        if _bak is not None:
            _p.write_text(_bak)
        else:
            _p.unlink(missing_ok=True)
    print("\npipeline OK\n")

# %% @notebook-only
LAUNCH_SWEEP1 = False     # True launches ~2.7 h of compute in this kernel

if LAUNCH_SWEEP1:
    RESULTS1 = run_sweep1(SWEEP1, budget=PREREG1["budget"])
else:
    RESULTS1 = load_results1()
    print(f"loaded {len(RESULTS1)} completed Phase 1 runs "
          f"(set LAUNCH_SWEEP1=True, or run scripts/run_sweep_phase1.py)")

# %% [markdown]
# ---
# ## Part 5 — Criterion A: does the flip survive at scale?
#
# Confirmation oracles only, `unassigned` excluded. Both quantities are reported: Phase 0's
# CI-excludes-zero test, and the replication rate that Phase 1 additionally requires.

# %% @notebook-only
def analysis_frame(df):
    """Drop failed runs and unassigned oracles before any cluster-level test."""
    if df.empty:
        return df
    d = df[df.get("status", "ok") == "ok"].copy()
    return d[d["cluster"] != "unassigned"]


A1 = None
if not RESULTS1.empty:
    _d = analysis_frame(RESULTS1)
    A1 = PS.criterion_A(_d, PREREG1["backbones"],
                        n_boot=PREREG1["criterion_A_flip"]["n_bootstrap"])
    for _bb, _pairs in A1["per_backbone"].items():
        print(f"\n{_bb}  --  {A1['n_confirmed'][_bb]} confirmed")
        print(f"  winners: {PS.winning_setting(_d, _bb)}")
        for _p in _pairs:
            print(f"  {_p['c1']}({_p['e1']}) vs {_p['c2']}({_p['e2']}): "
                  f"D={_p['D']:+.4f} CI=[{_p['lo']:+.4f},{_p['hi']:+.4f}] "
                  f"repl={_p['replication']:.2f} -> "
                  f"{'CONFIRMED' if _p['confirmed'] else _p['status']}")
    print(f"\nCriterion A: {'PASS' if A1['passes'] else 'FAIL'} "
          f"(best backbone {A1['best_backbone']}, needs >= {A1['min_pairs']} pairs)")
else:
    print("no Phase 1 results yet -- run the sweep first")

# %% [markdown]
# ---
# ## Part 6 — Criterion B: one pre-specified hypothesis, then FDR
#
# `gini`, directional negative, α = 0.05, on `augmented_memory`, over the **15 confirmation
# oracles only**. Everything else is secondary under Benjamini–Hochberg.

# %% @notebook-only
B1 = DIAG1 = None
if not RESULTS1.empty:
    DIAG1 = compute_all_diagnostics1(CONFIRMATION)
    _d = analysis_frame(RESULTS1)
    B1 = PS.criterion_B(_d, DIAG1,
                        backbone=PREREG1["criterion_B_diagnostic"]["primary"]["backbone"])
    print(f"n = {B1['n_oracles']} oracles, backbone {B1['backbone']}")
    print(f"\nPRIMARY  {B1['primary']}: rho = {B1['primary_rho']:+.3f}, "
          f"p = {B1['primary_p']:.4f} (one-sided {PS.PRIMARY_DIRECTION}, "
          f"alpha = {PS.PRIMARY_ALPHA}) -> "
          f"{'PASS' if B1['primary_pass'] else 'FAIL'}")
    print("\nsecondary (Benjamini-Hochberg, q = 0.05):")
    for _f, _r in sorted(B1["per_feature"].items(),
                         key=lambda kv: kv[1].get("p", 1.0)):
        if _f == B1["primary"]:
            continue
        _rho = _r.get("rho", float("nan"))
        print(f"  {_f:22s} rho {_rho:+.3f}  p {_r.get('p', 1.0):.4f}  "
              f"{'SIGNIFICANT' if _r.get('significant') else ''}")
    if B1["degenerate_features"]:
        print(f"\ndegenerate (excluded before correction): "
              f"{', '.join(B1['degenerate_features'])}")
    print(f"\nCriterion B: {'PASS' if B1['passes'] else 'FAIL'}")

# %% [markdown]
# ---
# ## Part 7 — Criterion C: is the flip worth the probe?
#
# **This is the part that can end the project.** Phase 0's numbers say a perfect chooser buys
# 1–2% AUC at cluster granularity while the probe costs 30% of the budget. Here the headroom
# is measured with held-out-seed selection, so the winner's curse is gone, and the probe cost
# is measured in the metric's own units from the persisted traces.

# %% @notebook-only
C1 = None
if not RESULTS1.empty:
    _d = analysis_frame(RESULTS1)
    C1 = {bb: PS.criterion_C(_d, bb,
                             probe_calls=PREREG1["criterion_C_economics"]["probe_calls"],
                             budget=PREREG1["budget"])
          for bb in PREREG1["backbones"]}
    for _bb, _c in C1.items():
        if _c["status"] != "ok":
            print(f"{_bb:18s} insufficient data "
                  f"({_c['probe'].get('reason', _c['headroom'].get('status'))})")
            continue
        _h, _p = _c["headroom"], _c["probe"]
        print(f"\n{_bb}")
        print(f"  best fixed setting    {_h['best_fixed_setting']} = {_h['best_fixed_auc']:.4f}")
        print(f"  conditional (CV)      {_h['conditional_auc']:.4f}")
        print(f"  headroom              {_h['headroom']:+.4f}  "
              f"CI [{_h['lo']:+.4f}, {_h['hi']:+.4f}]")
        print(f"  probe cost (300 calls){_p['cost']:+.4f}  "
              f"(AUC@1000 {_p['auc_full']:.4f} - AUC@700 {_p['auc_short']:.4f})")
        print(f"  NET                   {_c['net']:+.4f}  "
              f"CI [{_c['net_lo']:+.4f}, {_c['net_hi']:+.4f}]  -> "
              f"{'VIABLE' if _c['viable'] else 'NOT VIABLE'}")
    _best = max((b for b in C1 if C1[b]["status"] == "ok"),
                key=lambda b: C1[b]["net"], default=None)
    print(f"\nCriterion C: {'PASS' if _best and C1[_best]['viable'] else 'FAIL'}"
          f"{f' (best {_best})' if _best else ''}")

# %% [markdown]
# ---
# ## Part 8 — Criterion D: selection across methods
#
# The action space is the `(backbone, setting)` arm. Phase 0's method-level margins were an
# order of magnitude larger than its knob effects, and `graph_ga` — the non-neural control —
# had the highest overall mean AUC while showing no landscape dependence at all.

# %% @notebook-only
D1 = None
if not RESULTS1.empty:
    D1 = PS.criterion_D(analysis_frame(RESULTS1))
    for _g, _w in D1["winners"].items():
        _top = sorted(D1["per_group"][_g].items(), key=lambda kv: -kv[1])[:3]
        print(f"{_g:14s} winner {str(_w):30s} "
              f"top3 {', '.join(f'{k}={v:.3f}' for k, v in _top)}")
    print(f"\n{D1['n_distinct_winning_arms']} distinct winning arms -> "
          f"{'METHOD-LEVEL FLIP' if D1['flips'] else 'no method-level flip'}")

# %% [markdown]
# ---
# ## Part 9 — The verdict
#
# Applies the rule frozen in Part 0. Refuses to run on a modified pre-registration.

# %% @notebook-only
V1 = None
if PREREG1_STATUS == "mismatch":
    print("REFUSING TO PRODUCE A VERDICT -- pre-registration hash mismatch.")
    V1 = {"verdict": "REFUSED", "reason": "prereg hash mismatch"}
elif RESULTS1.empty:
    print("No Phase 1 results yet.")
    V1 = {"verdict": "NO_DATA"}
else:
    _n_ok = int((RESULTS1.get("status", "ok") == "ok").sum())
    _best_c = max((b for b in C1 if C1[b]["status"] == "ok"),
                  key=lambda b: C1[b]["net"], default=None)
    V1 = PS.phase1_verdict(A1, B1,
                           C1[_best_c] if _best_c else {"viable": False}, D1)
    V1.update(n_runs_ok=_n_ok, n_runs_expected=len(SWEEP1),
              prereg_sha256=CURRENT_HASH1,
              decided_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 74)
    print(f"PHASE 1 VERDICT: {V1['verdict']}")
    print("=" * 74)
    print(V1["consequence"])
    print(f"\nA {V1['criterion_A_passes']}  B {V1['criterion_B_passes']}  "
          f"C {V1['criterion_C_viable']}  D {V1['criterion_D_flips']}")
    print(f"runs {_n_ok}/{len(SWEEP1)}  "
          f"({100 * _n_ok / max(1, len(SWEEP1)):.0f}% complete)")
    if _n_ok < len(SWEEP1):
        print("INCOMPLETE SWEEP -- verdict is provisional.")
    (P0.RESULTS_DIR / "phase1_verdict.json").write_text(
        json.dumps(V1, indent=2, default=str))
    print(f"\nwritten -> {P0.RESULTS_DIR / 'phase1_verdict.json'}  (commit it)")

# %% [markdown]
# ---
# ## Part 10 — Figures
#
# Two: the headline non-parallel-lines plot per backbone, and the economics plot that is the
# actual decision — headroom against probe cost, with the break-even line.

# %% @notebook-only
def plot_setting_by_cluster1(df, save: bool = True):
    import matplotlib.pyplot as plt
    d = analysis_frame(df)
    bbs = [b for b in PREREG1["backbones"] if (d["backbone"] == b).any()]
    fig, axes = plt.subplots(1, len(bbs), figsize=(5 * len(bbs), 4), sharey=True)
    for ax, bb in zip(np.atleast_1d(axes), bbs):
        for cl in sorted(d["cluster"].unique()):
            ys = [PS._cell(d, bb, cl, s) for s in PS.SETTINGS]
            ax.plot(range(3), ys, marker="o", label=cl)
        ax.set_xticks(range(3))
        ax.set_xticklabels([s.replace("explore_", "") for s in PS.SETTINGS])
        ax.set_title(bb)
        ax.set_xlabel("exploration setting")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("mean AUC-top-10")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle("Criterion A: non-parallel lines are the flip", y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(P0.FIG_DIR / "phase1_setting_by_cluster.png", dpi=160,
                    bbox_inches="tight")
    return fig


def plot_economics(C, save: bool = True):
    import matplotlib.pyplot as plt
    ok = [(b, c) for b, c in C.items() if c["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (bb, c) in enumerate(ok):
        h, p = c["headroom"], c["probe"]
        ax.errorbar(i, h["headroom"],
                    yerr=[[h["headroom"] - h["lo"]], [h["hi"] - h["headroom"]]],
                    marker="o", capsize=4, label="headroom (CV)" if i == 0 else None,
                    color="#2563eb")
        ax.scatter(i, p["cost"], marker="_", s=400, color="#dc2626",
                   label="probe cost" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels([b for b, _ in ok])
    ax.set_ylabel("AUC-top-10")
    ax.set_title("Criterion C: headroom must clear the probe's cost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(P0.FIG_DIR / "phase1_economics.png", dpi=160, bbox_inches="tight")
    return fig

# %% @notebook-only
if not RESULTS1.empty:
    plot_setting_by_cluster1(RESULTS1)
    if C1:
        plot_economics(C1)
    print(f"figures -> {P0.FIG_DIR}")

# %% [markdown]
# ---
# ## Part 11 — What this phase does *not* decide
#
# Still open, still not coding tasks, still not decided here:
#
# 1. **The step-0 novelty check.** Landscape-aware / per-instance algorithm selection has a long
#    history in the broader optimization literature (portfolio solvers for SAT and TSP). Whether
#    the molecular-optimization version is already published is unknown and unverified. It gates
#    everything and it is a literature search.
# 2. **The online-bandit baseline.** EXP3 or successive halving over the settings as arms. Any
#    reviewer's first question is *"why predict the landscape when you could adapt online?"* —
#    and Criterion C makes that question sharper, not softer, since a bandit pays no probe cost.
# 3. **The submission timeline.**
#
# If Criterion C fails, item 2 stops being a baseline and becomes the paper.
