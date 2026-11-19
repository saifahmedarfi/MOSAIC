# %% [markdown]
# # Landscape-Conditional Exploration for Sample-Efficient Molecular Optimization
# ## Phase 0 — Pre-Registered Falsification
#
# **Saif Ahmed** · Department of CSE (AI & ML), North South University · Target venue: **ICML 2027** (main track)
#
# ---
#
# ### What this notebook is
#
# This is the **Phase 0 falsification** from the research proposal: the one-week experiment
# whose only job is to **cheaply kill the premise** if it is wrong, before any further compute
# is spent. It is not a results notebook. It is a decision procedure.
#
# The central hypothesis:
#
# > **H0.** Across the modern cohort there is no dominant exploration setting: the best setting
# > `e` *flips* with landscape geometry, a cheap early diagnostic predicts the flip, and a
# > landscape-conditional controller beats the best fixed `e` at equal budget.
#
# Phase 0 tests only the first two clauses. The controller (Phase 2) is not built until Phase 0
# returns **GO**.
#
# ### Structure
#
# | Part | Contents |
# |---|---|
# | 0 | **Pre-registration** — the decision rule, frozen and hashed *before* any result exists |
# | 1 | Environment, determinism, resource report |
# | 2 | Vendoring published baselines (mol_opt) |
# | 3 | Oracles + exact PMO budget and AUC-top-10 semantics |
# | 4 | Chemistry utilities (vocabulary, tokenizer, augmentation, scaffolds) |
# | 5 | Backbone A — REINVENT / Augmented Memory (SMILES language model) |
# | 6 | Backbone B — Graph GA (non-neural control) |
# | 7 | The exploration-setting ladder `E` |
# | 8 | Landscape diagnostics |
# | 9 | Experiment runner (resumable) |
# | 10 | Analysis + mechanical GO/PIVOT verdict |
#
# ### A note on the venue
#
# Target is **ICML 2027 main track**, as the proposal states. That shapes what this notebook
# optimises for: an ICML reviewer will not be impressed by a benchmark sweep, and will press
# hardest on two things — whether the claim is **instance-adaptive algorithm selection** done
# rigorously, or merely hyperparameter tuning with extra steps; and whether the statistics
# supporting a "the ranking flips" claim actually survive contact with multiple comparisons.
#
# Both pressures are answered structurally rather than rhetorically. The decision rule is
# frozen and hashed before any result exists (Part 0), the rank-flip test is an interaction
# contrast with a hierarchical bootstrap rather than a pair of t-tests (Part 10), and the
# diagnostic probe is deliberately **method-agnostic** so that "the diagnostic predicts the
# winner" cannot collapse into "REINVENT predicts REINVENT" (Part 8). Part 12 sets out the
# full positioning argument and the objections to prepare for.

# %% [markdown]
# ---
# # Part 0 — Pre-Registration
#
# **This cell must be run, and its output committed, before any experiment is launched.**
#
# The proposal commits to a pre-registered decision rule "fixed before any result is inspected."
# A promise in a PDF is not a mechanism. This is the mechanism: the rule is written as
# executable code, serialized, and hashed. The analysis in Part 10 **refuses to run** unless the
# hash on disk matches the rule it is about to apply. If you edit the rule after seeing results,
# the hash changes and the notebook tells you so — loudly, in the output, where a reviewer
# would see it.
#
# Read the rule carefully now, because this is the last moment at which you are allowed to
# change it.

# %%
PREREGISTRATION = {
    "title": "Landscape-Conditional Exploration — Phase 0 falsification",
    "version": "1.1.0",
    "frozen_on": "2026-11-19",

    # ---------------------------------------------------------------- scope
    "oracles": {
        # cluster -> oracles. Two per cluster, spanning the PMO landscape types
        # named in the proposal (Sec. 7.1).
        "similarity":  ["albuterol_similarity", "celecoxib_rediscovery"],
        "isomer":      ["isomers_c7h8n2o2", "isomers_c9h10n2o2pf2cl"],
        "mpo":         ["sitagliptin_mpo", "ranolazine_mpo"],
        "bioactivity": ["drd2", "jnk3"],
    },
    "backbones": ["reinvent", "augmented_memory", "graph_ga"],
    "settings": ["explore_low", "explore_mid", "explore_high"],
    "seeds": [0, 1, 2],
    "budget": 1000,            # true oracle calls per run (PMO semantics)
    "diagnostic_budget": 300,  # charged against the same budget in Phase 2+

    # ------------------------------------------------------- primary metric
    "primary_metric": "auc_top10",
    "metric_definition": (
        "PMO AUC-top-10: trapezoidal integral of mean top-10 reward against the "
        "number of unique canonical SMILES evaluated, normalised by max_oracle_calls. "
        "Reimplemented to match mol_opt/main/optimizer.py::top_auc exactly."
    ),

    # ------------------------------------------- THE DECISION RULE (frozen)
    # H0 survives Phase 0 iff BOTH criteria below are met.
    "criterion_A_rank_flip": {
        "statement": (
            "There exist at least two landscape clusters whose winning exploration "
            "setting differs, and the difference is larger than seed variance."
        ),
        "test": (
            "For each backbone and each ordered pair of clusters (c1, c2) with "
            "argmax_e AUC(c1,e) = e1 != e2 = argmax_e AUC(c2,e): form the "
            "interaction contrast D = [AUC(c1,e1) - AUC(c1,e2)] + [AUC(c2,e2) - AUC(c2,e1)]. "
            "HIERARCHICAL bootstrap (10,000 resamples): resample oracles within each "
            "cluster WITH replacement, then seeds within each resampled oracle. "
            "The pair counts as a confirmed flip iff the 95% percentile CI of D excludes 0. "
            "A seed-only bootstrap is also reported as an optimistic bound, but the "
            "hierarchical interval is the one that gates the decision."
        ),
        "threshold": "at least 2 confirmed flipped cluster pairs, within a single backbone",
        "n_bootstrap": 10000,
        "ci": 0.95,
        "rationale": (
            "An interaction contrast, not two separate comparisons. A flip means the "
            "RANKING reverses; D>0 is exactly that statement and is invariant to any "
            "oracle-level shift that affects all settings equally. "
            "The bootstrap is hierarchical because the claim is about landscape CLUSTERS: "
            "resampling seeds alone treats the 2 oracles per cluster as the entire "
            "population and understates uncertainty about generalising to other oracles "
            "of the same type. With only 2 oracles per cluster this test has low power "
            "BY CONSTRUCTION -- that is the honest cost of a cheap Phase 0, and it biases "
            "toward PIVOT rather than toward a false GO."
        ),
    },
    "criterion_B_diagnostic": {
        "statement": (
            "At least one cheap landscape diagnostic correlates with the winning "
            "exploration setting across oracles."
        ),
        "test": (
            "Spearman rho between each diagnostic feature and the winning-setting index "
            "(0=low, 1=mid, 2=high) across the 8 oracles, for the LM backbone. "
            "Exact permutation p-value (all 8! = 40,320 permutations enumerated). "
            "Holm-Bonferroni correction across the diagnostic features tested."
        ),
        "alpha": 0.10,
        "correction": "holm",
        "rationale": (
            "alpha=0.10 and not 0.05: Phase 0 is a SCREEN with n=8 oracles, and is "
            "explicitly underpowered. Criterion B asks 'is there a signal worth chasing', "
            "not 'is there a publishable effect'. Phase 1 (23 oracles) is the confirmatory "
            "test and will use alpha=0.05. Declaring this split here is what keeps "
            "Phase 1 confirmatory rather than a second bite at the same apple."
        ),
    },

    # -------------------------------------------------------------- verdict
    "verdict_rule": {
        "GO":        "criterion A passes AND criterion B passes",
        "PARTIAL":   "criterion A passes AND criterion B fails",
        "PIVOT":     "criterion A fails",
    },
    "verdict_consequences": {
        "GO": "Proceed to Phase 1 (full 23-oracle characterisation) as written.",
        "PARTIAL": (
            "The landscape dependence is real but our diagnostics do not capture it. "
            "Proceed to Phase 1 for C1 (characterisation) only; redesign the diagnostic "
            "feature set before attempting the controller. Downgrade venue expectation "
            "to a benchmark/characterisation contribution."
        ),
        "PIVOT": (
            "The premise is dead: exploration settings do not flip with landscape. "
            "Do NOT proceed to Phase 1. Switch to the proxy-gap sibling direction. "
            "Write up the null result as a short note -- it is informative, because the "
            "field has assumed this dependence since PMO without testing it directly."
        ),
    },

    # ----------------------------------------------------- what we will NOT do
    "prohibited": [
        "Adding or dropping oracles after seeing results.",
        "Changing the exploration-setting ladder after seeing results.",
        "Adding diagnostic features after seeing results (the feature list is frozen "
        "in Part 8 and hashed with this document).",
        "Reporting a cluster pair as a flip without the bootstrap CI.",
        "Switching the primary metric.",
        "Running more seeds and reporting only the favourable subset. If more seeds "
        "are run, ALL are reported.",
    ],
}

# The diagnostic feature list is part of the pre-registration and is hashed with it.
DIAGNOSTIC_FEATURES = [
    "reward_mean", "reward_std", "reward_skew", "reward_kurtosis",
    "frac_above_0.5", "frac_above_0.7", "frac_above_0.9",
    "gini", "dynamic_range",
    "ruggedness_corr", "ruggedness_absdelta",
    "basin_width", "neighbor_gain",
]
PREREGISTRATION["diagnostic_features"] = DIAGNOSTIC_FEATURES

# %%
import hashlib
import json
import os
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    """Walk up for the directory that holds src/, so the notebook resolves the
    same root wherever it is opened from.

    Jupyter sets the kernel's cwd to the notebook's own directory. The notebook
    lives in a subdirectory ("Phase 0/"), while scripts/run_sweep.py derives the
    root from __file__ and exports LCE_ROOT. Keying off cwd alone would make the
    notebook write results/ and preregistration.json inside that subdirectory
    while the cluster driver wrote them at the root -- two result sets, and a
    Part 10 verdict that sees none of the sweep's runs.
    """
    for cand in (start, *start.parents):
        if (cand / "src" / "phase0_source.py").exists() or \
           (cand / "src" / "phase0_lib.py").exists():
            return cand
    return start


if "LCE_ROOT" in os.environ:
    PROJECT_ROOT = Path(os.environ["LCE_ROOT"]).resolve()
else:
    PROJECT_ROOT = _find_project_root(Path.cwd().resolve())
os.environ["LCE_ROOT"] = str(PROJECT_ROOT)   # so child processes agree
RESULTS_DIR = PROJECT_ROOT / "results"
VENDOR_DIR = PROJECT_ROOT / "vendor"
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
for _d in (RESULTS_DIR, VENDOR_DIR, DATA_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PREREG_PATH = PROJECT_ROOT / "preregistration.json"


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def prereg_hash(obj) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


CURRENT_HASH = prereg_hash(PREREGISTRATION)

if PREREG_PATH.exists():
    _stored = json.loads(PREREG_PATH.read_text())
    _stored_hash = _stored.get("_sha256")
    if _stored_hash == CURRENT_HASH:
        print(f"Pre-registration verified unchanged.\n  sha256 = {CURRENT_HASH}")
        PREREG_STATUS = "verified"
    else:
        print("=" * 78)
        print("!! PRE-REGISTRATION MISMATCH !!")
        print("=" * 78)
        print(f"  on disk : {_stored_hash}")
        print(f"  in cell : {CURRENT_HASH}")
        print()
        print("The decision rule in this notebook differs from the one frozen on disk.")
        print("If results have already been generated, this change is NOT pre-registered")
        print("and must be disclosed as a post-hoc deviation in any write-up.")
        print()
        print("To deliberately re-freeze (only legitimate before any run):")
        print(f"    {PREREG_PATH}.unlink()  # then re-run this cell")
        print("=" * 78)
        PREREG_STATUS = "mismatch"
else:
    _payload = dict(PREREGISTRATION)
    _payload["_sha256"] = CURRENT_HASH
    PREREG_PATH.write_text(json.dumps(_payload, indent=2, sort_keys=True))
    print(f"Pre-registration FROZEN -> {PREREG_PATH}")
    print(f"  sha256 = {CURRENT_HASH}")
    print()
    print("Commit this file to git NOW, before running any experiment.")
    PREREG_STATUS = "frozen"

ORACLE_TO_CLUSTER = {
    o: c for c, os_ in PREREGISTRATION["oracles"].items() for o in os_
}
ALL_ORACLES = list(ORACLE_TO_CLUSTER)
print(f"\n{len(ALL_ORACLES)} oracles across {len(PREREGISTRATION['oracles'])} clusters:")
for _c, _os in PREREGISTRATION["oracles"].items():
    print(f"  {_c:12s} {', '.join(_os)}")

# %% [markdown]
# ---
# # Part 1 — Environment
#
# ### Install
#
# Run the cell below **once per machine**. It is idempotent. On a cluster, prefer creating the
# conda environment from `environment.yml` (written out at the end of this notebook) and
# skipping this cell entirely — notebook-driven installs are convenient for a first run and a
# liability for a reproducible one.
#
# `PyTDC` pulls the PMO oracles. The DRD2 / JNK3 / GSK3β oracles download pickled scikit-learn
# models on first use, so the machine needs network access **once**; Part 3 prefetches them so
# a later offline compute node does not fail mid-run.

# %% @notebook-only
# Set to True on first run on a new machine.
DO_INSTALL = False

if DO_INSTALL:
    import subprocess
    import sys

    PKGS = [
        "torch",                 # pin to your cluster's CUDA build if needed
        "rdkit>=2023.9",
        "PyTDC>=0.4.1",
        "numpy>=1.24,<3",
        "scipy>=1.10",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "joblib>=1.3",
        "matplotlib>=3.7",
        "tqdm>=4.65",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PKGS])
    print("Install complete. RESTART THE KERNEL before continuing.")

# %%
import math
import platform
import random
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("torch missing -- the LM backbone will not run. Set DO_INSTALL=True above.")

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("rdkit missing -- nothing will run. Set DO_INSTALL=True above.")

print(f"python  {platform.python_version()}   ({platform.system()} {platform.release()})")
if HAS_TORCH:
    print(f"torch   {torch.__version__}")
if HAS_RDKIT:
    import rdkit
    print(f"rdkit   {rdkit.__version__}")
print(f"numpy   {np.__version__}")

# %% [markdown]
# ### Devices and the VRAM question
#
# The cell below measures what this project actually costs on your GPU rather than guessing.
# The headline, worked out analytically and confirmed empirically in Part 5:
#
# The only GPU resident is the REINVENT RNN — `Embedding(|V|,128)` + three `GRUCell` layers of
# width 512 + `Linear(512,|V|)`, which is **4.18 M parameters ≈ 16.7 MB** in fp32. Add gradients
# and Adam moments for the agent, a frozen prior copy, ~330 MB of activations for a batch of 64
# unrolled over 140 timesteps, and CUDA's own context, and a single training run peaks at
# roughly **1.0–1.5 GB**.
#
# Every oracle in the PMO suite is CPU-only (RDKit fingerprints and descriptors, molecular
# formula matching, a scikit-learn SVM for DRD2, random forests for JNK3 and GSK3β), and
# Graph GA is CPU-only too.
#
# **Consequence:** a 12 GB RTX 4070 Super holds 6–8 concurrent runs; an 11 GB 2080 Ti holds 5–6.
# Neither is the constraint. **Oracle throughput on the CPU is the constraint**, so set
# `N_WORKERS` from your core count, not your VRAM.

# %%
def describe_devices() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cuda": False, "gpus": [], "cpu_count": os.cpu_count()}
    if HAS_TORCH and torch.cuda.is_available():
        info["cuda"] = True
        info["cuda_version"] = torch.version.cuda
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {"index": i, "name": p.name,
                 "total_gb": round(p.total_memory / 1024**3, 2),
                 "capability": f"{p.major}.{p.minor}"}
            )
    return info


DEVICE_INFO = describe_devices()
DEVICE = torch.device("cuda" if DEVICE_INFO["cuda"] else "cpu") if HAS_TORCH else None

print(f"CPU cores : {DEVICE_INFO['cpu_count']}")
if DEVICE_INFO["cuda"]:
    for g in DEVICE_INFO["gpus"]:
        print(f"GPU {g['index']}     : {g['name']}  {g['total_gb']} GB  (sm_{g['capability'].replace('.','')})")
else:
    print("GPU       : none visible -- LM backbone will run on CPU (slow but correct)")


def estimate_lm_vram(vocab_size: int = 72, batch_size: int = 64,
                     max_len: int = 140, verbose: bool = True) -> Dict[str, float]:
    """Analytic VRAM estimate for one REINVENT training run."""
    emb = vocab_size * 128
    gru1 = 3 * 512 * 128 + 3 * 512 * 512 + 2 * 3 * 512
    gru23 = 2 * (3 * 512 * 512 + 3 * 512 * 512 + 2 * 3 * 512)
    lin = 512 * vocab_size + vocab_size
    params = emb + gru1 + gru23 + lin

    mb = lambda n: n * 4 / 1024**2
    # ~6 saved tensors per GRUCell per step (gates + candidates) is the dominant term
    act_mb = mb(max_len * batch_size * 512 * 3 * 6)
    out = {
        "params_M": params / 1e6,
        "agent_params_mb": mb(params),
        "prior_params_mb": mb(params),
        "grads_mb": mb(params),
        "adam_states_mb": mb(2 * params),
        "activations_mb": act_mb,
        "cuda_context_mb": 500.0,
    }
    out["total_mb"] = sum(v for k, v in out.items() if k.endswith("_mb"))
    out["total_gb"] = out["total_mb"] / 1024
    if verbose:
        print(f"\nAnalytic VRAM, one REINVENT run (batch={batch_size}, len={max_len}, |V|={vocab_size}):")
        for k, v in out.items():
            if k.endswith("_mb"):
                print(f"  {k:22s} {v:9.1f} MB")
        print(f"  {'TOTAL':22s} {out['total_mb']:9.1f} MB  = {out['total_gb']:.2f} GB")
        for g in DEVICE_INFO["gpus"]:
            n_fit = int(g["total_gb"] * 1024 * 0.85 // out["total_mb"])
            print(f"  -> {g['name']} ({g['total_gb']} GB): ~{n_fit} concurrent runs")
    return out


# %% @notebook-only
VRAM_ESTIMATE = estimate_lm_vram()

# %% [markdown]
# ### Determinism
#
# Three seeds is a thin basis for a variance claim, so every source of randomness that we *can*
# control, we do. What remains uncontrolled — RDKit's conformer-free descriptor calculations are
# deterministic, but `joblib` worker scheduling in Graph GA is not — is documented rather than
# hidden. Graph GA reproducibility is achieved by seeding each worker from the run seed, not by
# forcing single-threaded execution.

# %%
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_all_seeds(0)
print("seeded")

# %% [markdown]
# ---
# # Part 2 — Vendoring the Published Baselines
#
# We reimplement the REINVENT training loop (Part 5) because we need to instrument it, but we
# **do not** reimplement two things:
#
# 1. **`Prior.ckpt` and `Voc`** — the ChEMBL-pretrained prior. Retraining our own prior would
#    make every number incomparable to published PMO results for no scientific gain.
# 2. **`crossover.py` / `mutate.py`** — the GB-GA graph operators. These are several hundred
#    lines of reaction SMARTS; reimplementing them from memory is how you get a silent bug that
#    invalidates an entire backbone.
#
# Both are pulled from `wenhao-gao/mol_opt` and **checksummed**, so a future reader can verify
# they got the same artifacts we did. The RNN *architecture* is reimplemented in Part 5 with
# `state_dict` keys matching `MultiGRU` exactly, so `Prior.ckpt` loads straight into it.

# %%
MOL_OPT_REPO = "https://github.com/wenhao-gao/mol_opt.git"
MOL_OPT_DIR = VENDOR_DIR / "mol_opt"

VENDORED_FILES = {
    "Prior.ckpt":   "main/reinvent/data/Prior.ckpt",
    "Voc":          "main/reinvent/data/Voc",
    "crossover.py": "main/graph_ga/crossover.py",
    "mutate.py":    "main/graph_ga/mutate.py",
}


def vendor_mol_opt(force: bool = False) -> Dict[str, str]:
    """Clone mol_opt (shallow) and checksum the files we depend on."""
    import subprocess

    if force and MOL_OPT_DIR.exists():
        import shutil
        shutil.rmtree(MOL_OPT_DIR)

    if not MOL_OPT_DIR.exists():
        print(f"cloning {MOL_OPT_REPO} (shallow) ...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", MOL_OPT_REPO, str(MOL_OPT_DIR)],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        print("  done")
    else:
        print(f"using existing clone at {MOL_OPT_DIR}")

    sums = {}
    missing = []
    for alias, rel in VENDORED_FILES.items():
        p = MOL_OPT_DIR / rel
        if not p.exists():
            missing.append(rel)
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        sums[alias] = h.hexdigest()
        print(f"  {alias:14s} {p.stat().st_size/1024:9.1f} KB  sha256={h.hexdigest()[:16]}...")

    if missing:
        raise FileNotFoundError(
            f"mol_opt layout changed; missing {missing}. "
            "Inspect the clone and update VENDORED_FILES."
        )

    (VENDOR_DIR / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))
    return sums


VENDOR_CHECKSUMS = vendor_mol_opt()
PRIOR_CKPT = MOL_OPT_DIR / VENDORED_FILES["Prior.ckpt"]
VOC_FILE = MOL_OPT_DIR / VENDORED_FILES["Voc"]

# %%
# Make the GB-GA operators importable. They are self-contained RDKit modules;
# we add their directory to sys.path rather than restructuring the vendored tree.
_GA_DIR = str((MOL_OPT_DIR / "main" / "graph_ga").resolve())
if _GA_DIR not in sys.path:
    sys.path.insert(0, _GA_DIR)

import crossover as co   # noqa: E402  (vendored GB-GA)
import mutate as mu      # noqa: E402  (vendored GB-GA)

# %% @notebook-only
_m1 = Chem.MolFromSmiles("CCOc1ccccc1N")
_m2 = Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")
random.seed(0)
_child = co.crossover(_m1, _m2)
print(f"crossover smoke test : {Chem.MolToSmiles(_child) if _child else 'None (retry, stochastic)'}")
_mut = mu.mutate(_m1, 1.0)
print(f"mutate smoke test    : {Chem.MolToSmiles(_mut) if _mut else 'None (retry, stochastic)'}")

# %% [markdown]
# ---
# # Part 3 — Oracles, Budget, and the Metric
#
# This part is where a sloppy reimplementation quietly destroys comparability with published
# numbers, so it is worth being pedantic.
#
# ### PMO budget semantics (from `mol_opt/main/optimizer.py`)
#
# Three rules, all of which matter:
#
# 1. **Canonicalise, then dedupe.** A molecule is identified by its canonical SMILES. Proposing
#    the same molecule twice costs **one** oracle call, not two. The cached score is returned.
# 2. **Invalid SMILES are free.** They score 0 and do not consume budget.
# 3. **`finish` fires when the buffer reaches `max_oracle_calls`,** i.e. on *unique valid*
#    molecules.
#
# Rule 1 is the one people get wrong, and it is not a technicality: it means a mode-collapsed
# generator that emits the same molecule 500 times burns almost no budget. Any exploration
# metric computed against *proposals* rather than against *buffer growth* will therefore be
# systematically wrong, and wrong in a direction that flatters collapsed policies — exactly the
# policies this project is about. We measure against buffer growth.
#
# ### AUC-top-10
#
# Reimplemented line-for-line from `optimizer.py::top_auc`: a trapezoidal integral of mean
# top-10 reward against buffer index, sampled every `freq_log` calls, normalised by
# `max_oracle_calls`. The tail term matters — a run that finishes early is credited with its
# final top-10 value for the remaining budget, which is what makes early convergence look good
# and is precisely the bias we are studying.
#
# Two quirks of the published metric, inherited deliberately and verified against a
# transcription of the original in `tests/test_core.py` (agreement to 1e-12):
#
# 1. **Ramp-in.** The integral starts from `prev = 0`, asserting that mean top-10 reward is 0 at
#    zero oracle calls. A run with a *constant* reward of 0.42 therefore scores 0.4179, short by
#    `freq_log × final / (2 × budget)`. The deficit is near-identical across methods so it does
#    not bias our comparisons, but it means **absolute AUC values are not interpretable as mean
#    rewards** and must not be described as such in the paper.
# 2. **Sub-`freq_log` blindness.** With `freq_log = 10`, anything happening inside the first 10
#    calls is invisible to the integral.
#
# We keep both, because changing either would silently break comparability with every published
# PMO number.

# %%
@dataclass
class OracleCall:
    """One entry in the oracle buffer."""
    smiles: str
    score: float
    index: int          # 1-based order of first evaluation


class BudgetedOracle:
    """PMO-faithful oracle wrapper: budget, canonical dedupe, call trace.

    Mirrors mol_opt/main/optimizer.py::Oracle. The additions over the original are
    purely observational (`trace`, `proposals_seen`) and do not change scoring.
    """

    def __init__(self, name: str, scoring_fn: Callable[[str], float],
                 max_oracle_calls: int = 1000, freq_log: int = 10):
        self.name = name
        self._fn = scoring_fn
        self.max_oracle_calls = max_oracle_calls
        self.freq_log = freq_log
        self.mol_buffer: Dict[str, List[float]] = {}   # smi -> [score, index]
        self.trace: List[Tuple[int, float]] = []       # (buffer_size, score)
        self.proposals_seen = 0                        # incl. duplicates & invalid
        self.invalid_seen = 0
        self.duplicate_seen = 0

    # ----------------------------------------------------------------- core
    def score_smi(self, smi: str) -> float:
        self.proposals_seen += 1
        # Deliberate deviation from mol_opt, which uses `>` and so permits one call
        # beyond the budget. `>=` caps the buffer at exactly max_oracle_calls, which
        # matters because top_auc normalises by that number. Effect on AUC is ~1e-3.
        if len(self.mol_buffer) >= self.max_oracle_calls:
            return 0.0
        if smi is None or smi == "":
            self.invalid_seen += 1
            return 0.0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            self.invalid_seen += 1
            return 0.0
        canon = Chem.MolToSmiles(mol)
        if canon in self.mol_buffer:
            self.duplicate_seen += 1
            return self.mol_buffer[canon][0]
        score = float(self._fn(canon))
        self.mol_buffer[canon] = [score, len(self.mol_buffer) + 1]
        self.trace.append((len(self.mol_buffer), score))
        return score

    def __call__(self, smiles) -> Any:
        if isinstance(smiles, str):
            return self.score_smi(smiles)
        return [self.score_smi(s) for s in smiles]

    def __len__(self) -> int:
        return len(self.mol_buffer)

    @property
    def finish(self) -> bool:
        return len(self.mol_buffer) >= self.max_oracle_calls

    # ------------------------------------------------------------- metrics
    def top_auc(self, top_n: int = 10) -> float:
        """Exact reimplementation of mol_opt optimizer.py::top_auc."""
        buffer = self.mol_buffer
        if not buffer:
            return 0.0
        total, prev, called = 0.0, 0.0, 0
        ordered = sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False)
        for idx in range(self.freq_log, min(len(buffer), self.max_oracle_calls), self.freq_log):
            temp = sorted(ordered[:idx], key=lambda kv: kv[1][0], reverse=True)[:top_n]
            now = float(np.mean([it[1][0] for it in temp]))
            total += self.freq_log * (now + prev) / 2
            prev = now
            called = idx
        temp = sorted(ordered, key=lambda kv: kv[1][0], reverse=True)[:top_n]
        now = float(np.mean([it[1][0] for it in temp]))
        total += (len(buffer) - called) * (now + prev) / 2
        if self.finish and len(buffer) < self.max_oracle_calls:
            total += (self.max_oracle_calls - len(buffer)) * now
        return total / self.max_oracle_calls

    def top_k_mean(self, k: int = 10) -> float:
        scores = sorted((v[0] for v in self.mol_buffer.values()), reverse=True)[:k]
        return float(np.mean(scores)) if scores else 0.0

    def top_k_smiles(self, k: int = 10) -> List[str]:
        return [s for s, _ in sorted(self.mol_buffer.items(),
                                     key=lambda kv: kv[1][0], reverse=True)[:k]]

    def summary(self) -> Dict[str, Any]:
        return {
            "oracle": self.name,
            "auc_top10": self.top_auc(10),
            "auc_top1": self.top_auc(1),
            "auc_top100": self.top_auc(100),
            "top10_mean": self.top_k_mean(10),
            "top1": self.top_k_mean(1),
            "n_calls": len(self.mol_buffer),
            "n_proposals": self.proposals_seen,
            "n_invalid": self.invalid_seen,
            "n_duplicate": self.duplicate_seen,
            # fraction of proposals that bought new information -- our cheapest
            # collapse indicator, and the reason we track proposals at all
            "yield_rate": len(self.mol_buffer) / max(1, self.proposals_seen),
        }

# %% [markdown]
# ### Diversity and novelty (secondary metrics)
#
# AUC-top-10 cannot distinguish "found a good scaffold" from "found a good scaffold and nothing
# else". Since the whole project is about exploration, we instrument diversity directly:
# internal diversity (mean pairwise Tanimoto distance over the top-k), scaffold count, and
# novelty against the ZINC starting pool.

# %%
_FP_CACHE: Dict[str, Any] = {}


def morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048):
    if smi in _FP_CACHE:
        return _FP_CACHE[smi]
    mol = Chem.MolFromSmiles(smi)
    fp = None if mol is None else AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    if len(_FP_CACHE) < 200_000:
        _FP_CACHE[smi] = fp
    return fp


def internal_diversity(smiles: Sequence[str]) -> float:
    """Mean pairwise Tanimoto DISTANCE. 0 = identical set, 1 = maximally diverse."""
    fps = [f for f in (morgan_fp(s) for s in smiles) if f is not None]
    if len(fps) < 2:
        return 0.0
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return float(1.0 - np.mean(sims)) if sims else 0.0


def n_scaffolds(smiles: Sequence[str]) -> int:
    scafs = set()
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            scafs.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:
            pass
    return len(scafs)


def circles_count(smiles: Sequence[str], threshold: float = 0.75) -> int:
    """#Circles (Xie et al.): greedy count of mutually dissimilar molecules.

    A better diversity measure than mean pairwise distance, which saturates.
    Greedy approximation; deterministic given input order.
    """
    fps = [f for f in (morgan_fp(s) for s in smiles) if f is not None]
    centers = []
    for fp in fps:
        if not centers or max(DataStructs.BulkTanimotoSimilarity(fp, centers)) < threshold:
            centers.append(fp)
    return len(centers)

# %% [markdown]
# ### Binding the TDC oracles
#
# `tdc.Oracle` downloads pickled scikit-learn models for the bioactivity oracles on first use.
# The prefetch below forces those downloads now, so a compute node without egress does not fail
# three hours into a sweep. It also times each oracle: **oracle latency sets the wall clock for
# this entire project**, and knowing which oracles are slow tells you how to pack the sweep.

# %%
from tdc import Oracle as TDCOracle  # noqa: E402

_TDC_CACHE: Dict[str, Any] = {}


def get_tdc_oracle(name: str):
    if name not in _TDC_CACHE:
        _TDC_CACHE[name] = TDCOracle(name=name)
    return _TDC_CACHE[name]


def make_oracle(name: str, budget: int = 1000, freq_log: int = 10) -> BudgetedOracle:
    tdc = get_tdc_oracle(name)

    def _score(smi: str) -> float:
        try:
            v = tdc(smi)
            return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0.0
        except Exception:
            return 0.0

    return BudgetedOracle(name, _score, max_oracle_calls=budget, freq_log=freq_log)


def prefetch_oracles(names: Sequence[str] = tuple(ALL_ORACLES)) -> "pd.DataFrame":
    import pandas as pd
    probe = [
        "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)NCC(O)c1ccc(O)c(CO)c1",                       # salbutamol
        "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",                   # diclofenac
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",                     # caffeine
    ]
    rows = []
    for nm in names:
        t0 = time.time()
        orc = get_tdc_oracle(nm)
        t_load = time.time() - t0
        t0 = time.time()
        vals = [orc(s) for s in probe]
        t_call = (time.time() - t0) / len(probe)
        rows.append({
            "oracle": nm, "cluster": ORACLE_TO_CLUSTER[nm],
            "load_s": round(t_load, 2),
            "ms_per_call": round(t_call * 1000, 2),
            "est_s_per_1k": round(t_call * 1000, 1),
            "probe_range": f"{min(vals):.3f}-{max(vals):.3f}",
        })
        print(f"  {nm:26s} {t_call*1000:7.2f} ms/call   {t_call*1000:7.1f} s per 1k calls")
    return pd.DataFrame(rows)


import pandas as pd  # noqa: E402

# %% @notebook-only
print("Prefetching + timing oracles (first run downloads models) ...")
ORACLE_TIMING = prefetch_oracles()
print()
print(ORACLE_TIMING.to_string(index=False))
_tot = ORACLE_TIMING["est_s_per_1k"].sum()
print(f"\nSerial cost of one 1k-call run on each of the 8 oracles: {_tot/60:.1f} min of pure oracle time.")
print("Multiply by 3 settings x 3 seeds x 3 backbones for the full Phase-0 sweep.")

# %% [markdown]
# ---
# # Part 4 — Chemistry Utilities
#
# SMILES is a formal language with a context-free-ish grammar, and the generator is an
# autoregressive LM over its token sequence. This part sets up that language: the vocabulary
# (loaded from the vendored `Voc` so the pretrained prior loads), the tokenizer, and SMILES
# randomization — the augmentation that Augmented Memory depends on.
#
# ### Tokenization
#
# REINVENT's tokenizer, reproduced exactly: bracket atoms `[...]` are single tokens; two-character
# halogens are folded to single characters (`Cl`→`L`, `Br`→`R`) before splitting so that every
# remaining character is its own token. This is a **character-level LM with a hand-built
# multi-character exception list**. Modern subword tokenizers for SMILES (SmilesPE,
# Atom-in-SMILES) are known to change what the model can express, so this is a real design
# choice, not a neutral one. We keep REINVENT's tokenizer because changing it would invalidate
# the pretrained prior and break comparability with every published PMO number. It is held
# **fixed across all exploration settings**, so it cannot confound the rank-flip test — which
# is the argument to make if a reviewer raises it.
#
# ### SMILES randomization
#
# The same molecule has many valid SMILES strings depending on which atom you start the
# traversal from. `MolToSmiles(mol, doRandom=True, canonical=False)` samples one. Augmented
# Memory exploits this: it re-expresses molecules whose scores are **already paid for** and
# takes additional gradient steps on them. That is the whole trick — more supervised signal per
# oracle call, which is why it is sample-efficient, and also why it sharpens the policy and
# raises the collapse risk this project studies.

# %%
class Vocabulary:
    """REINVENT vocabulary. Loaded from the vendored Voc file so Prior.ckpt is loadable."""

    def __init__(self, init_from_file: Optional[str] = None, max_length: int = 140):
        self.special_tokens = ["EOS", "GO"]
        self.additional_chars: set = set()
        self.chars = list(self.special_tokens)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file:
            self.init_from_file(init_from_file)

    def add_characters(self, chars) -> None:
        for ch in chars:
            self.additional_chars.add(ch)
        char_list = sorted(self.additional_chars)
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_file(self, path) -> None:
        with open(path, "r") as f:
            chars = f.read().split()
        self.add_characters(chars)

    def encode(self, char_list) -> np.ndarray:
        arr = np.zeros(len(char_list), dtype=np.float32)
        for i, ch in enumerate(char_list):
            arr[i] = self.vocab[ch]
        return arr

    def decode(self, matrix) -> str:
        chars = []
        for i in matrix:
            if i == self.vocab["EOS"]:
                break
            chars.append(self.reversed_vocab[int(i)])
        smiles = "".join(chars)
        return smiles.replace("L", "Cl").replace("R", "Br")

    def tokenize(self, smiles: str) -> List[str]:
        regex = r"(\[[^\[\]]{1,6}\])"
        smiles = replace_halogen(smiles)
        tokenized: List[str] = []
        for char in re.split(regex, smiles):
            if char.startswith("["):
                tokenized.append(char)
            else:
                tokenized.extend(list(char))
        tokenized.append("EOS")
        return tokenized

    def __len__(self) -> int:
        return len(self.chars)


import re  # noqa: E402


def replace_halogen(string: str) -> str:
    """Cl -> L, Br -> R so each remaining character is one token."""
    br = re.compile("Br")
    cl = re.compile("Cl")
    return cl.sub("L", br.sub("R", string))


VOC = Vocabulary(init_from_file=str(VOC_FILE))
# %% @notebook-only
print(f"vocabulary size : {VOC.vocab_size}")
print(f"  GO  index     : {VOC.vocab['GO']}")
print(f"  EOS index     : {VOC.vocab['EOS']}")
print(f"  sample tokens : {VOC.chars[:18]} ...")

_demo = "CC(C)NCC(O)c1ccc(O)c(CO)c1"
print(f"\ntokenize({_demo})\n  -> {VOC.tokenize(_demo)}")

# %%
def randomize_smiles(smiles: str, n: int = 1) -> List[str]:
    """Sample `n` non-canonical SMILES for the same molecule (for augmentation)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    out = []
    for _ in range(n):
        try:
            out.append(Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=False))
        except Exception:
            continue
    return out


def canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def murcko_scaffold(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


# %% @notebook-only
_aug = randomize_smiles(_demo, 4)
print("SMILES randomization (same molecule, 4 surface forms):")
for a in _aug:
    print(f"  {a}")
print(f"all identical after canonicalization: {len({canonical(a) for a in _aug}) == 1}")

# %% [markdown]
# ---
# # Part 5 — Backbone A: the SMILES Language Model (REINVENT / Augmented Memory)
#
# The primary backbone: a 4.2 M-parameter autoregressive LM over SMILES tokens, pretrained on
# ChEMBL, then fine-tuned with policy gradient against the oracle reward. It is primary because
# it is the strongest cohort member under a 10k budget and the one whose exploration knob (σ)
# is most directly interpretable as an exploration–exploitation dial.
#
# ### The architecture
#
# `MultiGRU`, reproduced with `state_dict` keys matching `mol_opt` exactly — `embedding`,
# `gru_1`, `gru_2`, `gru_3`, `linear` — so the vendored `Prior.ckpt` loads without remapping.
# Three stacked `GRUCell`s, hidden 512, embedding 128. Note these are *cells*, unrolled manually
# in Python, not `nn.GRU`. That is slower than cuDNN's fused kernel, but it is what the
# checkpoint was trained with and what makes per-step temperature control straightforward.
#
# ### The objective (DAP)
#
# REINVENT's Difference between Augmented and Posterior likelihood:
#
# $$\mathcal{L}(\theta) = \Big[\underbrace{\log P_{\text{prior}}(x)+\sigma\, R(x)}_{\text{augmented}} - \log P_\theta(x)\Big]^2$$
#
# **σ is the exploration knob and the sign is counter-intuitive, so be careful:** σ scales how
# far the target likelihood is pushed above the prior for a rewarded molecule. **Large σ ⇒
# strong pull toward high-reward regions ⇒ *less* exploration.** Our `explore_high` setting
# therefore has the *smallest* σ. Getting this backwards would invert every conclusion in the
# paper, so Part 7 asserts the ordering in code.
#
# Both `likelihood()` and `sample()` return **log-likelihoods** (negative numbers), despite
# REINVENT's historical naming of the helper as `NLLLoss`. The two must be on the same scale for
# the DAP loss to mean anything; a unit test below checks it.

# %%
def to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(DEVICE)


class MultiGRU(nn.Module):
    """Three-layer GRU with embedding and output projection.

    state_dict keys match mol_opt/main/reinvent/model.py::MultiGRU so that the
    published Prior.ckpt loads directly.
    """

    def __init__(self, voc_size: int):
        super().__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.gru_1 = nn.GRUCell(128, 512)
        self.gru_2 = nn.GRUCell(512, 512)
        self.gru_3 = nn.GRUCell(512, 512)
        self.linear = nn.Linear(512, voc_size)

    def forward(self, x, h):
        x = self.embedding(x)
        h_out = torch.zeros_like(h)
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out

    def init_h(self, batch_size: int):
        return torch.zeros(3, batch_size, 512, device=DEVICE)


class SmilesLM:
    """Autoregressive SMILES language model: sampling and scoring."""

    def __init__(self, voc: Vocabulary):
        self.voc = voc
        self.rnn = MultiGRU(voc.vocab_size).to(DEVICE)

    # -------------------------------------------------------------- scoring
    def likelihood(self, target):
        """Sum of token log-probabilities for each sequence. Shape (B,)."""
        batch_size, seq_length = target.size()
        start = torch.full((batch_size, 1), self.voc.vocab["GO"],
                           dtype=torch.long, device=DEVICE)
        x = torch.cat((start, target[:, :-1]), dim=1)
        h = self.rnn.init_h(batch_size)
        log_probs = torch.zeros(batch_size, device=DEVICE)
        entropy = torch.zeros(batch_size, device=DEVICE)
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            log_prob = F.log_softmax(logits, dim=1)
            prob = F.softmax(logits, dim=1)
            log_probs += log_prob.gather(1, target[:, step].unsqueeze(1)).squeeze(1)
            entropy += -torch.sum(log_prob * prob, dim=1)
        return log_probs, entropy

    # ------------------------------------------------------------- sampling
    def sample(self, batch_size: int, max_length: int = 140, temperature: float = 1.0):
        """Ancestral sampling with temperature. Returns (seqs, log_probs, entropy)."""
        start = torch.full((batch_size,), self.voc.vocab["GO"],
                           dtype=torch.long, device=DEVICE)
        h = self.rnn.init_h(batch_size)
        x = start
        sequences, log_probs = [], torch.zeros(batch_size, device=DEVICE)
        entropy = torch.zeros(batch_size, device=DEVICE)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

        for _ in range(max_length):
            logits, h = self.rnn(x, h)
            scaled = logits / temperature
            prob = F.softmax(scaled, dim=1)
            # credit assignment uses the UNSCALED distribution: temperature changes
            # the behaviour policy, not the policy we are computing gradients for
            log_prob = F.log_softmax(logits, dim=1)
            x = torch.multinomial(prob, num_samples=1).view(-1)
            sequences.append(x.view(-1, 1))
            log_probs += (1 - finished.float()) * log_prob.gather(1, x.unsqueeze(1)).squeeze(1)
            true_prob = log_prob.exp()
            entropy += (1 - finished.float()) * -torch.sum(log_prob * true_prob, dim=1)
            finished = finished | (x == self.voc.vocab["EOS"])
            if finished.all():
                break

        return torch.cat(sequences, dim=1), log_probs, entropy

    # -------------------------------------------------------------- weights
    def load_prior(self, path) -> None:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        self.rnn.load_state_dict(state)

    def clone_from(self, other: "SmilesLM") -> None:
        self.rnn.load_state_dict(other.rnn.state_dict())


def seqs_to_smiles(seqs, voc: Vocabulary) -> List[str]:
    return [voc.decode(seq.cpu().numpy()) for seq in seqs]


def unique_rows(arr):
    """Indices of first occurrences of unique rows (REINVENT's `unique`)."""
    arr_np = arr.cpu().numpy()
    view = np.ascontiguousarray(arr_np).view(
        np.dtype((np.void, arr_np.dtype.itemsize * arr_np.shape[1]))
    )
    _, idxs = np.unique(view, return_index=True)
    return torch.LongTensor(np.sort(idxs)).to(arr.device)

# %%
# ---- load the pretrained prior and verify ------------------------------------
PRIOR = SmilesLM(VOC)
PRIOR.load_prior(PRIOR_CKPT)
for p in PRIOR.rnn.parameters():
    p.requires_grad = False
PRIOR.rnn.eval()

n_params = sum(p.numel() for p in PRIOR.rnn.parameters())
print(f"prior loaded: {n_params:,} parameters ({n_params*4/1024**2:.1f} MB fp32) on {DEVICE}")

# %% @notebook-only
set_all_seeds(0)
with torch.no_grad():
    _seqs, _lp, _ent = PRIOR.sample(64)
_smi = seqs_to_smiles(_seqs, VOC)
_valid = [s for s in _smi if Chem.MolFromSmiles(s) is not None]
print(f"\nsanity: sampled 64, {len(_valid)} chemically valid ({100*len(_valid)/64:.0f}%)")
print("  (a healthy ChEMBL prior gives ~90-97%)")
for s in _valid[:5]:
    print(f"    {s}")

# ---- consistency check: sample() and likelihood() must agree -----------------
with torch.no_grad():
    _lp2, _ = PRIOR.likelihood(_seqs)
_mask_ok = torch.allclose(_lp, _lp2, atol=1e-3)
print(f"\nsample() log-prob == likelihood() log-prob : {_mask_ok}")
if not _mask_ok:
    _d = (_lp - _lp2).abs()
    print(f"  max abs deviation {_d.max().item():.4f} "
          f"(expected: >0 only from post-EOS padding; verify before trusting DAP)")

# %% [markdown]
# ### Empirical VRAM measurement
#
# The analytic estimate in Part 1 is a calculation; this is a measurement. It runs one real
# training step and reports `torch.cuda.max_memory_allocated`.

# %%
def measure_vram(batch_size: int = 64, temperature: float = 1.0) -> Optional[float]:
    if not (HAS_TORCH and torch.cuda.is_available()):
        print("no CUDA device -- skipping empirical VRAM measurement")
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    agent = SmilesLM(VOC)
    agent.clone_from(PRIOR)
    opt = torch.optim.Adam(agent.rnn.parameters(), lr=5e-4)

    seqs, agent_lp, _ = agent.sample(batch_size, temperature=temperature)
    prior_lp, _ = PRIOR.likelihood(seqs)
    fake_reward = torch.rand(seqs.size(0), device=DEVICE)
    augmented = prior_lp + 500.0 * fake_reward
    loss = torch.pow(augmented - agent_lp, 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
    print(f"MEASURED peak allocated : {peak_mb:8.1f} MB   (batch={batch_size})")
    print(f"MEASURED peak reserved  : {reserved_mb:8.1f} MB")
    print(f"  + CUDA context overhead, not counted above: ~300-600 MB")
    total = reserved_mb + 500
    print(f"  => realistic per-run footprint ~{total/1024:.2f} GB")
    for g in DEVICE_INFO["gpus"]:
        print(f"  => {g['name']} ({g['total_gb']} GB): ~{int(g['total_gb']*1024*0.85//total)} concurrent runs")
    del agent, opt
    torch.cuda.empty_cache()
    return peak_mb


# %% @notebook-only
MEASURED_VRAM = measure_vram()

# %% [markdown]
# ### Experience replay, augmentation, and Selective Memory Purge
#
# Three mechanisms, which together are what separates REINVENT (2017) from Augmented Memory
# (2024). Each is an exploration knob.
#
# **Experience replay.** A buffer of the top-`max_size` molecules by reward. Each step, a
# score-weighted sample is drawn and extra gradient steps are taken on it. Replaying *more*
# ⇒ more exploitation.
#
# **SMILES augmentation (Augmented Memory).** For molecules already in the buffer — whose
# oracle cost is sunk — re-express them as randomized SMILES and take `augmentation_rounds`
# further updates. This is the sample-efficiency trick: more gradient signal per oracle call.
# It is also a sharpening operator, so more rounds ⇒ more exploitation.
#
# **Selective Memory Purge.** The anti-collapse mechanism. When one Bemis–Murcko scaffold comes
# to dominate the buffer, the replayed signal becomes self-reinforcing and the policy collapses
# onto it. SMP caps how many buffer entries may share a scaffold, keeping the best and evicting
# the rest.
#
# > **Fidelity note.** The replay buffer and augmentation follow the published descriptions
# > closely. Our SMP is a **scaffold-cap adaptation**, not a line-for-line port of Guo &
# > Schwaller's purge criterion. It is labelled as such here and must be labelled as such in
# > the paper. Since SMP is held *fixed* within each exploration setting and only toggled
# > between settings, this affects the absolute numbers, not the rank-flip test that Phase 0
# > turns on.

# %%
class ExperienceBuffer:
    """Score-prioritised replay with optional scaffold-capped purging."""

    def __init__(self, voc: Vocabulary, max_size: int = 100,
                 smp_enabled: bool = False, smp_max_per_scaffold: int = 8):
        self.voc = voc
        self.max_size = max_size
        self.memory: List[Tuple[str, float, float]] = []   # (smiles, score, prior_ll)
        self.smp_enabled = smp_enabled
        self.smp_max_per_scaffold = smp_max_per_scaffold
        self.n_purged = 0
        self._scaffold_cache: Dict[str, Optional[str]] = {}

    def _scaffold(self, smi: str) -> Optional[str]:
        if smi not in self._scaffold_cache:
            self._scaffold_cache[smi] = murcko_scaffold(smi)
        return self._scaffold_cache[smi]

    def add_experience(self, experience: Sequence[Tuple[str, float, float]]) -> None:
        self.memory.extend(experience)
        if len(self.memory) <= self.max_size and not self.smp_enabled:
            return
        # dedupe by SMILES, keep best
        best: Dict[str, Tuple[str, float, float]] = {}
        for e in self.memory:
            if e[0] not in best or e[1] > best[e[0]][1]:
                best[e[0]] = e
        items = sorted(best.values(), key=lambda x: x[1], reverse=True)

        if self.smp_enabled:
            kept, per_scaffold = [], {}
            for e in items:
                sc = self._scaffold(e[0])
                key = sc if sc is not None else f"__none__{e[0]}"
                c = per_scaffold.get(key, 0)
                if c < self.smp_max_per_scaffold:
                    kept.append(e)
                    per_scaffold[key] = c + 1
                else:
                    self.n_purged += 1
            items = kept

        self.memory = items[:self.max_size]

    def sample(self, n: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
        if len(self.memory) < n:
            raise IndexError(f"buffer has {len(self.memory)} < {n}")
        scores = np.array([x[1] for x in self.memory], dtype=np.float64) + 1e-10
        idxs = np.random.choice(len(self.memory), size=n, replace=False, p=scores / scores.sum())
        sampled = [self.memory[i] for i in idxs]
        return (
            [x[0] for x in sampled],
            np.array([x[1] for x in sampled], dtype=np.float32),
            np.array([x[2] for x in sampled], dtype=np.float32),
        )

    def scaffold_entropy(self) -> float:
        """Shannon entropy over buffer scaffolds -- our collapse monitor."""
        if not self.memory:
            return 0.0
        counts: Dict[str, int] = {}
        for smi, _, _ in self.memory:
            sc = self._scaffold(smi) or smi
            counts[sc] = counts.get(sc, 0) + 1
        p = np.array(list(counts.values()), dtype=np.float64)
        p = p / p.sum()
        return float(-(p * np.log(p + 1e-12)).sum())

    def __len__(self) -> int:
        return len(self.memory)


def smiles_to_tensor(smiles: Sequence[str], voc: Vocabulary):
    """Encode SMILES to a padded LongTensor. Returns (tensor, kept_indices)."""
    encoded, kept = [], []
    for i, smi in enumerate(smiles):
        try:
            toks = voc.tokenize(smi)
            if len(toks) > voc.max_length:
                continue
            encoded.append(voc.encode(toks))
            kept.append(i)
        except (KeyError, ValueError):
            continue           # token outside the ChEMBL vocabulary
    if not encoded:
        return None, []
    max_len = max(len(e) for e in encoded)
    out = torch.zeros(len(encoded), max_len, dtype=torch.long)
    for i, e in enumerate(encoded):
        out[i, :len(e)] = torch.from_numpy(e).long()
    return out.to(DEVICE), kept

# %% [markdown]
# ### The optimizer loop
#
# One function covers both REINVENT and Augmented Memory; `augmentation_rounds=0` recovers
# plain REINVENT. Everything the diagnostics or the analysis might want is logged per step —
# entropy, scaffold entropy, buffer yield rate — because re-running the sweep to recover a
# metric we forgot to log is the most expensive mistake available here.

# %%
@dataclass
class LMConfig:
    sigma: float = 500.0
    learning_rate: float = 5e-4
    batch_size: int = 64
    experience_replay: int = 24
    buffer_size: int = 100
    augmentation_rounds: int = 0
    temperature: float = 1.0
    smp_enabled: bool = False
    smp_max_per_scaffold: int = 8
    likelihood_regularizer: float = 5e3   # REINVENT's -1/agent_likelihood term
    max_length: int = 140


def run_lm_optimizer(oracle: BudgetedOracle, cfg: LMConfig, seed: int = 0,
                     log_every: int = 1, verbose: bool = False) -> Dict[str, Any]:
    """REINVENT / Augmented Memory against a budgeted oracle."""
    set_all_seeds(seed)

    agent = SmilesLM(VOC)
    agent.clone_from(PRIOR)
    agent.rnn.train()
    optimizer = torch.optim.Adam(agent.rnn.parameters(), lr=cfg.learning_rate)
    buffer = ExperienceBuffer(VOC, max_size=cfg.buffer_size,
                              smp_enabled=cfg.smp_enabled,
                              smp_max_per_scaffold=cfg.smp_max_per_scaffold)

    history: List[Dict[str, Any]] = []
    step = 0
    t_start = time.time()

    while not oracle.finish:
        step += 1

        # ---- 1. sample from the current policy -------------------------------
        seqs, agent_ll, entropy = agent.sample(cfg.batch_size, cfg.max_length, cfg.temperature)
        uidx = unique_rows(seqs)
        seqs, agent_ll, entropy = seqs[uidx], agent_ll[uidx], entropy[uidx]
        smiles = seqs_to_smiles(seqs, VOC)

        # ---- 2. pay the oracle ----------------------------------------------
        scores = np.array(oracle(smiles), dtype=np.float32)

        # ---- 3. DAP loss -----------------------------------------------------
        with torch.no_grad():
            prior_ll, _ = PRIOR.likelihood(seqs)
        augmented_ll = prior_ll + cfg.sigma * to_tensor(scores)
        loss = torch.pow(augmented_ll - agent_ll, 2)

        # ---- 4. experience replay -------------------------------------------
        if cfg.experience_replay and len(buffer) > cfg.experience_replay:
            e_smiles, e_scores, e_prior_ll = buffer.sample(cfg.experience_replay)
            e_seqs, kept = smiles_to_tensor(e_smiles, VOC)
            if e_seqs is not None and len(kept) > 0:
                e_agent_ll, _ = agent.likelihood(e_seqs)
                e_aug_ll = to_tensor(e_prior_ll[kept]) + cfg.sigma * to_tensor(e_scores[kept])
                loss = torch.cat((loss, torch.pow(e_aug_ll - e_agent_ll, 2)))
                agent_ll = torch.cat((agent_ll, e_agent_ll))

        loss = loss.mean()
        loss = loss + cfg.likelihood_regularizer * -(1 / agent_ll).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- 5. record the batch in the replay buffer ------------------------
        new_experience = list(zip(smiles, scores, prior_ll.detach().cpu().numpy()))
        buffer.add_experience(new_experience)

        # ---- 6. Augmented Memory: extra passes on ALREADY-PAID molecules -----
        #        no new oracle calls are made here; that is the entire point
        for _ in range(cfg.augmentation_rounds):
            if len(buffer) < max(4, cfg.experience_replay):
                break
            n_take = min(cfg.experience_replay or 24, len(buffer))
            a_smiles, a_scores, _ = buffer.sample(n_take)
            randomized, keep_scores = [], []
            for smi, sc in zip(a_smiles, a_scores):
                r = randomize_smiles(smi, 1)
                if r:
                    randomized.append(r[0])
                    keep_scores.append(sc)
            if not randomized:
                break
            a_seqs, kept = smiles_to_tensor(randomized, VOC)
            if a_seqs is None or not kept:
                break
            a_agent_ll, _ = agent.likelihood(a_seqs)
            with torch.no_grad():
                a_prior_ll, _ = PRIOR.likelihood(a_seqs)
            a_sc = to_tensor(np.array(keep_scores, dtype=np.float32)[kept])
            a_loss = torch.pow(a_prior_ll + cfg.sigma * a_sc - a_agent_ll, 2).mean()
            a_loss = a_loss + cfg.likelihood_regularizer * -(1 / a_agent_ll).mean()
            optimizer.zero_grad()
            a_loss.backward()
            optimizer.step()

        # ---- 7. log ----------------------------------------------------------
        if step % log_every == 0:
            history.append({
                "step": step,
                "n_oracle_calls": len(oracle),
                "batch_mean_score": float(scores.mean()) if len(scores) else 0.0,
                "batch_max_score": float(scores.max()) if len(scores) else 0.0,
                "top10_mean": oracle.top_k_mean(10),
                "policy_entropy": float(entropy.mean().item()),
                "buffer_size": len(buffer),
                "scaffold_entropy": buffer.scaffold_entropy(),
                "loss": float(loss.item()),
                "valid_frac": len([s for s in smiles if Chem.MolFromSmiles(s)]) / max(1, len(smiles)),
            })
            if verbose and step % 10 == 0:
                h = history[-1]
                print(f"    step {step:4d} | calls {h['n_oracle_calls']:5d} | "
                      f"top10 {h['top10_mean']:.3f} | H(pi) {h['policy_entropy']:.2f} | "
                      f"H(scaf) {h['scaffold_entropy']:.2f}")

        if step > 10000:
            break   # safety valve: a degenerate policy emitting only duplicates

    top10 = oracle.top_k_smiles(10)
    result = oracle.summary()
    result.update({
        "backbone": "lm",
        "steps": step,
        "wall_s": round(time.time() - t_start, 1),
        "final_scaffold_entropy": buffer.scaffold_entropy(),
        "n_purged": buffer.n_purged,
        "top10_diversity": internal_diversity(top10),
        "top10_n_scaffolds": n_scaffolds(top10),
        "top100_circles": circles_count(oracle.top_k_smiles(100)),
        "top10_smiles": top10,
        "history": history,
    })
    del agent, optimizer
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

# %% @notebook-only
# ---- smoke test: 150 calls on a cheap oracle ---------------------------------
print("LM backbone smoke test (150 oracle calls on albuterol_similarity) ...")
_orc = make_oracle("albuterol_similarity", budget=150, freq_log=10)
_res = run_lm_optimizer(_orc, LMConfig(sigma=500, experience_replay=16,
                                       augmentation_rounds=1), seed=0, verbose=True)
print(f"\n  AUC-top10        {_res['auc_top10']:.4f}")
print(f"  top-10 mean      {_res['top10_mean']:.4f}")
print(f"  oracle calls     {_res['n_calls']} from {_res['n_proposals']} proposals "
      f"(yield {_res['yield_rate']:.2f})")
print(f"  top-10 diversity {_res['top10_diversity']:.3f}  "
      f"({_res['top10_n_scaffolds']} distinct scaffolds)")
print(f"  wall clock       {_res['wall_s']}s")

# %% [markdown]
# ---
# # Part 6 — Backbone B: Graph GA (the non-neural control)
#
# Graph GA is here for a specific argumentative purpose. If the rank flip appears **only** in
# the language model, the honest reading is that we have found something about neural decoding
# dynamics, not about oracle landscapes. If it appears in a population-based graph search with
# no learned policy at all, the effect is a property of the **objective**, which is the claim
# the paper actually wants to make.
#
# This control does real work for an ICML submission: it is what converts "we tuned a
# hyperparameter per instance" into "landscape geometry governs exploration across algorithm
# families". A result that replicates across a gradient-based policy and a population-based
# graph search is a statement about the **problem**, which is the kind of claim ICML rewards.
#
# Implementation follows `mol_opt/main/graph_ga/run.py`: fitness-proportionate mating pool,
# vendored GB-GA crossover and mutation, truncation selection. The initial population is drawn
# from ZINC via TDC and is **identical across settings at a given seed**, so the ladder is not
# confounded by its starting point.

# %%
from tdc.generation import MolGen  # noqa: E402

_ZINC_CACHE: Optional[List[str]] = None


def get_zinc_pool(n: int = 20000) -> List[str]:
    """ZINC-250k SMILES, used as the GA start pool and as diagnostic probes."""
    global _ZINC_CACHE
    if _ZINC_CACHE is None:
        cache = DATA_DIR / "zinc_pool.txt"
        if cache.exists():
            _ZINC_CACHE = cache.read_text().split()
        else:
            print("downloading ZINC-250k via TDC ...")
            df = MolGen(name="ZINC", path=str(DATA_DIR)).get_data()
            smiles = df["smiles"].tolist()
            rng = np.random.default_rng(12345)       # fixed: the pool is part of the protocol
            idx = rng.choice(len(smiles), size=min(50000, len(smiles)), replace=False)
            _ZINC_CACHE = [smiles[i] for i in idx]
            cache.write_text("\n".join(_ZINC_CACHE))
            print(f"  cached {len(_ZINC_CACHE)} SMILES -> {cache}")
    return _ZINC_CACHE[:n]


ZINC_POOL = get_zinc_pool()
print(f"ZINC pool: {len(ZINC_POOL)} molecules")
print(f"  e.g. {ZINC_POOL[0]}")

# %%
@dataclass
class GAConfig:
    population_size: int = 100
    offspring_size: int = 200
    mutation_rate: float = 0.01
    elitism: float = 1.0        # fraction of parents surviving into selection
    n_jobs: int = 1


def _make_mating_pool(population_mol, population_scores, offspring_size: int):
    MINIMUM = 1e-10
    scores = np.array(population_scores, dtype=np.float64) + MINIMUM
    total = scores.sum()
    if total <= 0 or not np.isfinite(total):
        probs = np.ones(len(scores)) / len(scores)
    else:
        probs = scores / total
    idx = np.random.choice(len(population_mol), p=probs, size=offspring_size, replace=True)
    return [population_mol[i] for i in idx]


def _reproduce(mating_pool, mutation_rate: float):
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


def run_ga_optimizer(oracle: BudgetedOracle, cfg: GAConfig, seed: int = 0,
                     log_every: int = 1, verbose: bool = False) -> Dict[str, Any]:
    set_all_seeds(seed)
    t_start = time.time()

    # identical start population across settings at a given seed
    rng = np.random.default_rng(seed)
    start_idx = rng.choice(len(ZINC_POOL), size=cfg.population_size, replace=False)
    population_smiles = [ZINC_POOL[i] for i in start_idx]
    population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
    population_mol = [m for m in population_mol if m is not None]
    population_scores = oracle([Chem.MolToSmiles(m) for m in population_mol])

    history: List[Dict[str, Any]] = []
    gen = 0

    while not oracle.finish:
        gen += 1
        mating_pool = _make_mating_pool(population_mol, population_scores, cfg.offspring_size)
        offspring = [_reproduce(mating_pool, cfg.mutation_rate) for _ in range(cfg.offspring_size)]
        offspring = [m for m in offspring if m is not None]
        if not offspring:
            if gen > 50:
                break
            continue

        new_smiles = []
        for m in offspring:
            try:
                new_smiles.append(Chem.MolToSmiles(m))
            except Exception:
                continue

        combined_mol = population_mol + [Chem.MolFromSmiles(s) for s in new_smiles]
        combined_smi = [Chem.MolToSmiles(m) for m in combined_mol if m is not None]
        combined_mol = [m for m in combined_mol if m is not None]
        combined_scores = oracle(combined_smi)

        order = np.argsort(combined_scores)[::-1][:cfg.population_size]
        population_mol = [combined_mol[i] for i in order]
        population_scores = [combined_scores[i] for i in order]

        if gen % log_every == 0:
            pop_smi = [Chem.MolToSmiles(m) for m in population_mol]
            history.append({
                "step": gen,
                "n_oracle_calls": len(oracle),
                "batch_mean_score": float(np.mean(combined_scores)) if combined_scores else 0.0,
                "batch_max_score": float(np.max(combined_scores)) if combined_scores else 0.0,
                "top10_mean": oracle.top_k_mean(10),
                "population_mean": float(np.mean(population_scores)),
                "population_diversity": internal_diversity(pop_smi),
                "scaffold_entropy": float(np.log(max(1, n_scaffolds(pop_smi)))),
            })
            if verbose and gen % 5 == 0:
                h = history[-1]
                print(f"    gen {gen:4d} | calls {h['n_oracle_calls']:5d} | "
                      f"top10 {h['top10_mean']:.3f} | pop div {h['population_diversity']:.3f}")

        if gen > 2000:
            break

    top10 = oracle.top_k_smiles(10)
    result = oracle.summary()
    result.update({
        "backbone": "graph_ga",
        "steps": gen,
        "wall_s": round(time.time() - t_start, 1),
        "final_scaffold_entropy": history[-1]["scaffold_entropy"] if history else 0.0,
        "n_purged": 0,
        "top10_diversity": internal_diversity(top10),
        "top10_n_scaffolds": n_scaffolds(top10),
        "top100_circles": circles_count(oracle.top_k_smiles(100)),
        "top10_smiles": top10,
        "history": history,
    })
    return result

# %% @notebook-only
print("Graph GA smoke test (150 oracle calls) ...")
_orc = make_oracle("albuterol_similarity", budget=150, freq_log=10)
_res = run_ga_optimizer(_orc, GAConfig(population_size=50, offspring_size=50,
                                       mutation_rate=0.01), seed=0, verbose=True)
print(f"\n  AUC-top10        {_res['auc_top10']:.4f}")
print(f"  top-10 mean      {_res['top10_mean']:.4f}")
print(f"  top-10 diversity {_res['top10_diversity']:.3f}")
print(f"  wall clock       {_res['wall_s']}s")

# %% [markdown]
# ---
# # Part 7 — The Exploration-Setting Ladder `E`
#
# ### Design decision: one knob per backbone, not a grid
#
# The proposal lists several knobs per backbone (σ, augmentation rounds, replay admission,
# diversity filters; population size, mutation rate, elitism). Phase 0 varies **one primary
# knob per backbone** and holds everything else at PMO defaults.
#
# This is deliberate. The Phase-0 question is narrow — *does the rank order of exploration
# settings flip across landscapes?* — and a multi-knob ladder makes a flip uninterpretable:
# you cannot tell whether σ flipped, the replay count flipped, or they interacted. A clean
# single-knob ladder gives an unconfounded answer to the only question that gates the project.
# The full grid belongs in Phase 1, where characterisation, not falsification, is the goal.
#
# | Backbone | Primary knob | low expl. | mid (PMO default) | high expl. |
# |---|---|---|---|---|
# | `reinvent` | σ | 1000 | **500** | 128 |
# | `augmented_memory` | σ | 1000 | **500** | 128 |
# | `graph_ga` | mutation rate | 0.001 | **0.01** | 0.05 |
#
# **The σ ordering is inverted relative to its numeric value** — larger σ pulls the policy
# harder toward observed high-reward molecules, which is *less* exploration. This is the single
# easiest way to invert every conclusion in the paper, so it is asserted in code below rather
# than trusted to a reader's attention.
#
# `augmented_memory` differs from `reinvent` only in `augmentation_rounds=2` and `smp_enabled`,
# matching the published configuration. That makes the two rows a built-in replication: if the
# flip pattern is the same in both, it is a property of the DAP objective, not of augmentation.

# %%
EXPLORATION_LADDER: Dict[str, Dict[str, Any]] = {
    "reinvent": {
        "explore_low":  LMConfig(sigma=1000.0, experience_replay=24, augmentation_rounds=0,
                                 temperature=1.0, smp_enabled=False),
        "explore_mid":  LMConfig(sigma=500.0,  experience_replay=24, augmentation_rounds=0,
                                 temperature=1.0, smp_enabled=False),
        "explore_high": LMConfig(sigma=128.0,  experience_replay=24, augmentation_rounds=0,
                                 temperature=1.0, smp_enabled=False),
    },
    "augmented_memory": {
        "explore_low":  LMConfig(sigma=1000.0, experience_replay=24, augmentation_rounds=2,
                                 temperature=1.0, smp_enabled=True),
        "explore_mid":  LMConfig(sigma=500.0,  experience_replay=24, augmentation_rounds=2,
                                 temperature=1.0, smp_enabled=True),
        "explore_high": LMConfig(sigma=128.0,  experience_replay=24, augmentation_rounds=2,
                                 temperature=1.0, smp_enabled=True),
    },
    "graph_ga": {
        "explore_low":  GAConfig(population_size=100, offspring_size=200, mutation_rate=0.001),
        "explore_mid":  GAConfig(population_size=100, offspring_size=200, mutation_rate=0.01),
        "explore_high": GAConfig(population_size=100, offspring_size=200, mutation_rate=0.05),
    },
}

SETTING_ORDER = ["explore_low", "explore_mid", "explore_high"]
SETTING_INDEX = {s: i for i, s in enumerate(SETTING_ORDER)}

# ---- assert the exploration ordering is monotone in the intended direction ----
for _bb in ("reinvent", "augmented_memory"):
    _sig = [EXPLORATION_LADDER[_bb][s].sigma for s in SETTING_ORDER]
    assert _sig[0] > _sig[1] > _sig[2], (
        f"{_bb}: sigma must DECREASE as exploration increases, got {_sig}"
    )
_mr = [EXPLORATION_LADDER["graph_ga"][s].mutation_rate for s in SETTING_ORDER]
assert _mr[0] < _mr[1] < _mr[2], f"graph_ga: mutation rate must increase, got {_mr}"
print("exploration ladder orderings verified\n")

for bb, settings in EXPLORATION_LADDER.items():
    print(f"{bb}:")
    for s in SETTING_ORDER:
        c = settings[s]
        knob = f"sigma={c.sigma:>6.0f}" if isinstance(c, LMConfig) else f"mut_rate={c.mutation_rate:<6}"
        print(f"  {s:14s} {knob}")


def run_backbone(backbone: str, oracle: BudgetedOracle, setting: str,
                 seed: int, verbose: bool = False) -> Dict[str, Any]:
    cfg = EXPLORATION_LADDER[backbone][setting]
    if backbone == "graph_ga":
        return run_ga_optimizer(oracle, cfg, seed=seed, verbose=verbose)
    return run_lm_optimizer(oracle, cfg, seed=seed, verbose=verbose)

# %% [markdown]
# ---
# # Part 8 — Landscape Diagnostics
#
# The diagnostic is the load-bearing contribution. If it is not genuinely predictive, there is
# no controller and the paper collapses to a characterisation. Two design constraints follow,
# and both are easy to violate without noticing:
#
# **1. The probe must be method-agnostic.** If we computed the diagnostic from a REINVENT run,
# "the diagnostic predicts which setting wins" could reduce to "REINVENT's own early behaviour
# predicts REINVENT's later behaviour" — circular, and useless for choosing a method at
# deployment. So the probe uses a **fixed policy that is not any of the competitors**: samples
# from the untrained ChEMBL prior plus local edits around whatever it finds. It would be
# identical if we were choosing between entirely different optimizers.
#
# **2. The probe must be budget-honest.** All 300 probe calls are charged to the same oracle
# budget. In Phase 2 the controller gets 1000 total calls — 300 diagnostic plus 700
# optimization — against baselines that get all 1000 for optimization. Anything else is
# comparing against a handicapped baseline.
#
# ### Probe design (300 calls)
#
# | Stage | Calls | Purpose |
# |---|---|---|
# | Prior samples | 200 | the reward distribution over generic drug-like space |
# | Local edits around the top 5 hits | 100 | ruggedness and basin width |
#
# ### The features
#
# *Reward-distribution shape* — mean, sd, skew, kurtosis, fraction above {0.5, 0.7, 0.9}, Gini,
# dynamic range. Separates **needle-in-haystack** (near-zero everywhere, heavy right tail) from
# **graded** (broad spread, informative gradient). The prediction: needles reward exploration,
# graded landscapes reward exploitation.
#
# *Ruggedness* — Spearman correlation between parent and single-edit-neighbour scores, plus mean
# |Δscore| per edit. Low correlation means local search carries no information, so there is
# nothing for a sharpened policy to exploit.
#
# *Basin width* — fraction of a hit's neighbours retaining ≥90% of its score. Wide basins are
# safe to exploit; narrow ones punish it.
#
# *Neighbour gain* — mean improvement of the best neighbour over its parent. Directly measures
# whether hill-climbing pays on this landscape.
#
# > **This feature list is frozen.** It is hashed into the pre-registration in Part 0. Adding a
# > feature after seeing results is specifically prohibited there.

# %%
from scipy import stats as sps  # noqa: E402


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1).dot(x) / (n * x.sum()))


def local_edits(smiles: str, n: int = 20, mutation_rate: float = 0.5,
                max_tries: int = 100) -> List[str]:
    """Small structural perturbations: the discrete analogue of a step size."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    out, seen, tries = [], {Chem.MolToSmiles(mol)}, 0
    while len(out) < n and tries < max_tries:
        tries += 1
        try:
            child = mu.mutate(mol, mutation_rate)
        except Exception:
            continue
        if child is None:
            continue
        try:
            smi = Chem.MolToSmiles(child)
        except Exception:
            continue
        if smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
    return out


def compute_diagnostics(oracle_name: str, n_prior: int = 200, n_edits_total: int = 100,
                        n_hits: int = 5, seed: int = 0,
                        shared_oracle: Optional[BudgetedOracle] = None) -> Dict[str, float]:
    """Method-agnostic landscape probe. Spends exactly n_prior + n_edits_total calls."""
    set_all_seeds(seed)
    orc = shared_oracle or make_oracle(oracle_name,
                                       budget=n_prior + n_edits_total + 50, freq_log=10)

    # ---- stage 1: unconditional prior samples --------------------------------
    probe_smiles, probe_scores = [], []
    attempts = 0
    while len(probe_smiles) < n_prior and attempts < 20:
        attempts += 1
        with torch.no_grad():
            seqs, _, _ = PRIOR.sample(min(128, (n_prior - len(probe_smiles)) * 2))
        for s in seqs_to_smiles(seqs, VOC):
            if len(probe_smiles) >= n_prior:
                break
            c = canonical(s)
            if c is None or c in probe_smiles:
                continue
            probe_smiles.append(c)
            probe_scores.append(orc.score_smi(c))

    scores = np.array(probe_scores, dtype=np.float64)
    if len(scores) == 0:
        return {f: float("nan") for f in DIAGNOSTIC_FEATURES}

    feats: Dict[str, float] = {
        "reward_mean": float(scores.mean()),
        "reward_std": float(scores.std()),
        "reward_skew": float(sps.skew(scores)) if scores.std() > 1e-9 else 0.0,
        "reward_kurtosis": float(sps.kurtosis(scores)) if scores.std() > 1e-9 else 0.0,
        "frac_above_0.5": float((scores > 0.5).mean()),
        "frac_above_0.7": float((scores > 0.7).mean()),
        "frac_above_0.9": float((scores > 0.9).mean()),
        "gini": _gini(scores),
        "dynamic_range": float(scores.max() - np.median(scores)),
    }

    # ---- stage 2: local edits around the best probes -------------------------
    order = np.argsort(scores)[::-1][:n_hits]
    per_hit = max(1, n_edits_total // max(1, len(order)))
    parent_vals, neighbor_vals, deltas, basin_hits, basin_total, gains = [], [], [], 0, 0, []

    for i in order:
        parent_smi, parent_score = probe_smiles[i], scores[i]
        neighbors = local_edits(parent_smi, n=per_hit)
        if not neighbors:
            continue
        n_scores = [orc.score_smi(s) for s in neighbors]
        for ns in n_scores:
            parent_vals.append(parent_score)
            neighbor_vals.append(ns)
            deltas.append(abs(ns - parent_score))
            basin_total += 1
            if parent_score > 1e-6 and ns >= 0.9 * parent_score:
                basin_hits += 1
        gains.append(max(n_scores) - parent_score)

    if len(parent_vals) >= 5 and np.std(parent_vals) > 1e-9 and np.std(neighbor_vals) > 1e-9:
        rho = sps.spearmanr(parent_vals, neighbor_vals).correlation
        feats["ruggedness_corr"] = float(rho) if np.isfinite(rho) else 0.0
    else:
        feats["ruggedness_corr"] = 0.0
    feats["ruggedness_absdelta"] = float(np.mean(deltas)) if deltas else 0.0
    feats["basin_width"] = float(basin_hits / basin_total) if basin_total else 0.0
    feats["neighbor_gain"] = float(np.mean(gains)) if gains else 0.0

    assert set(feats) == set(DIAGNOSTIC_FEATURES), (
        f"feature drift vs pre-registration: "
        f"extra={set(feats)-set(DIAGNOSTIC_FEATURES)}, "
        f"missing={set(DIAGNOSTIC_FEATURES)-set(feats)}"
    )
    feats["_probe_calls_used"] = len(orc)
    return feats

# %%
DIAG_PATH = RESULTS_DIR / "diagnostics.json"


def compute_all_diagnostics(oracles: Sequence[str] = tuple(ALL_ORACLES),
                            seeds: Sequence[int] = (0, 1, 2),
                            force: bool = False) -> "pd.DataFrame":
    """Diagnostics for every oracle, averaged over probe seeds. Cached."""
    if DIAG_PATH.exists() and not force:
        print(f"loading cached diagnostics <- {DIAG_PATH}")
        return pd.DataFrame(json.loads(DIAG_PATH.read_text()))

    rows = []
    for nm in oracles:
        print(f"  probing {nm:26s} ", end="", flush=True)
        t0 = time.time()
        per_seed = [compute_diagnostics(nm, seed=s) for s in seeds]
        agg = {"oracle": nm, "cluster": ORACLE_TO_CLUSTER[nm]}
        for f in DIAGNOSTIC_FEATURES:
            vals = [d[f] for d in per_seed if np.isfinite(d.get(f, np.nan))]
            agg[f] = float(np.mean(vals)) if vals else float("nan")
            agg[f + "__sd"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        rows.append(agg)
        print(f"{time.time()-t0:5.1f}s   mean reward {agg['reward_mean']:.3f}, "
              f"frac>0.5 {agg['frac_above_0.5']:.3f}, rugged {agg['ruggedness_corr']:+.2f}")

    df = pd.DataFrame(rows)
    DIAG_PATH.write_text(df.to_json(orient="records"))
    print(f"\nsaved -> {DIAG_PATH}")
    return df


# %% @notebook-only
print("Computing landscape diagnostics (300 budgeted calls x 3 seeds x 8 oracles) ...")
DIAGNOSTICS = compute_all_diagnostics()
print()
_show = ["oracle", "cluster", "reward_mean", "frac_above_0.5", "gini",
         "ruggedness_corr", "basin_width", "neighbor_gain"]
print(DIAGNOSTICS[_show].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

# %% [markdown]
# ### Reading the diagnostic table before any optimization has run
#
# Look at the table above now, before the sweep. If the diagnostics are capturing anything
# real, oracles within a cluster should resemble each other more than oracles across clusters.
# The isomer oracles in particular should look unlike everything else: they are near-binary
# formula-matching objectives, so expect low `reward_mean`, high `gini`, and poor
# `ruggedness_corr` — a single atom substitution either fixes the molecular formula or does not.
#
# If instead the table looks like noise, Criterion B is already in trouble and the honest move
# is to say so here rather than to discover it in Part 10.

# %%
def plot_diagnostic_separation(df: "pd.DataFrame", save: bool = True):
    import matplotlib.pyplot as plt
    feats = ["reward_mean", "frac_above_0.5", "gini", "dynamic_range",
             "ruggedness_corr", "basin_width", "neighbor_gain", "reward_kurtosis"]
    clusters = sorted(df["cluster"].unique())
    cmap = dict(zip(clusters, ["#2563eb", "#dc2626", "#16a34a", "#ca8a04"]))

    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, f in zip(axes.ravel(), feats):
        for c in clusters:
            sub = df[df["cluster"] == c]
            ax.scatter(range(len(sub)), sub[f], label=c, color=cmap[c], s=90,
                       edgecolor="white", linewidth=1.4, zorder=3)
        ax.set_title(f, fontsize=11, fontweight="600")
        ax.set_xticks([])
        ax.grid(alpha=0.25, zorder=0)
    axes.ravel()[0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle("Landscape diagnostics by oracle cluster  (300 budgeted calls per oracle)",
                 fontsize=13, fontweight="600")
    fig.tight_layout()
    if save:
        fig.savefig(FIG_DIR / "diagnostics_by_cluster.png", dpi=160, bbox_inches="tight")
    return fig


# %% @notebook-only
_ = plot_diagnostic_separation(DIAGNOSTICS)

# %% [markdown]
# ---
# # Part 9 — Experiment Runner
#
# The Phase-0 sweep is **8 oracles × 3 backbones × 3 settings × 3 seeds = 216 runs** of 1000
# oracle calls each.
#
# ### Resumability is not a convenience here
#
# Each run writes its own JSON keyed by `(oracle, backbone, setting, seed)`, and completed runs
# are skipped. A sweep this size on a shared cluster *will* be interrupted — preemption, a
# walltime limit, a dropped SSH session — and losing hours of oracle calls to a dropped
# connection is avoidable with ten lines of caching.
#
# ### Sizing it for your hardware
#
# From the oracle timings in Part 3, the LM backbone runs at roughly 1–3 min per 1000 calls and
# Graph GA at 1–2 min. Serially that is 4–8 hours. With `N_WORKERS=8` on a 16-core box, under
# an hour.
#
# **Set `N_WORKERS` from CPU cores, not VRAM.** Each LM worker holds ~1.2 GB on the GPU, so
# even 8 workers use under 10 GB — your 4070 Super has room, and so does one 2080 Ti. The
# oracles are what saturate: DRD2 and JNK3 call into scikit-learn models on every molecule, and
# that is pure CPU. A reasonable rule is `N_WORKERS = min(cores - 2, floor(0.85 * VRAM_GB / 1.3))`.
#
# On the two-2080Ti box, set `CUDA_VISIBLE_DEVICES` per worker group to spread the LM runs
# across both cards; the code honours whatever device it is given.

# %%
RUNS_DIR = RESULTS_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def run_key(oracle: str, backbone: str, setting: str, seed: int) -> str:
    return f"{oracle}__{backbone}__{setting}__seed{seed}"


def run_path(*args) -> Path:
    return RUNS_DIR / f"{run_key(*args)}.json"


def execute_run(oracle_name: str, backbone: str, setting: str, seed: int,
                budget: int = 1000, force: bool = False,
                verbose: bool = False) -> Dict[str, Any]:
    p = run_path(oracle_name, backbone, setting, seed)
    if p.exists() and not force:
        return json.loads(p.read_text())

    orc = make_oracle(oracle_name, budget=budget, freq_log=10)
    t0 = time.time()
    try:
        res = run_backbone(backbone, orc, setting, seed, verbose=verbose)
        res["status"] = "ok"
    except Exception as exc:                       # a crashed run must not kill the sweep
        import traceback
        res = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
            "auc_top10": float("nan"),
        }
    res.update({
        "oracle": oracle_name, "cluster": ORACLE_TO_CLUSTER[oracle_name],
        "backbone": backbone, "setting": setting, "seed": seed,
        "budget": budget, "elapsed_s": round(time.time() - t0, 1),
        "prereg_sha256": CURRENT_HASH,
    })
    p.write_text(json.dumps(res))
    return res


def build_sweep(oracles=None, backbones=None, settings=None, seeds=None) -> List[Tuple]:
    oracles = oracles or ALL_ORACLES
    backbones = backbones or PREREGISTRATION["backbones"]
    settings = settings or SETTING_ORDER
    seeds = seeds or PREREGISTRATION["seeds"]
    return [(o, b, s, sd) for o in oracles for b in backbones
            for s in settings for sd in seeds]


SWEEP = build_sweep()
_done = sum(1 for j in SWEEP if run_path(*j).exists())
print(f"Phase-0 sweep: {len(SWEEP)} runs  ({_done} already complete, {len(SWEEP)-_done} to go)")
print(f"  {len(ALL_ORACLES)} oracles x {len(PREREGISTRATION['backbones'])} backbones "
      f"x {len(SETTING_ORDER)} settings x {len(PREREGISTRATION['seeds'])} seeds")

# %%
def recommended_workers() -> int:
    cores = os.cpu_count() or 4
    by_cpu = max(1, cores - 2)
    if DEVICE_INFO["cuda"]:
        vram_gb = sum(g["total_gb"] for g in DEVICE_INFO["gpus"])
        by_vram = max(1, int(0.85 * vram_gb / 1.3))
        n = min(by_cpu, by_vram)
        print(f"CPU allows {by_cpu}, VRAM ({vram_gb:.0f} GB) allows {by_vram} "
              f"-> {'CPU' if by_cpu < by_vram else 'VRAM'}-bound at {n}")
        return n
    print(f"CPU-only: {by_cpu} workers")
    return by_cpu


N_WORKERS = recommended_workers()

# %%
def run_sweep(jobs: Sequence[Tuple] = None, budget: int = 1000,
              n_workers: int = None, force: bool = False) -> "pd.DataFrame":
    """Execute the sweep. Resumable; safe to interrupt and re-run."""
    jobs = list(jobs if jobs is not None else SWEEP)
    n_workers = n_workers or N_WORKERS
    todo = [j for j in jobs if force or not run_path(*j).exists()]
    print(f"{len(jobs)} jobs, {len(jobs)-len(todo)} cached, {len(todo)} to execute "
          f"on {n_workers} worker(s)\n")

    t_sweep = time.time()
    if n_workers <= 1 or not todo:
        for i, j in enumerate(todo, 1):
            t0 = time.time()
            r = execute_run(*j, budget=budget, force=force)
            print(f"[{i:3d}/{len(todo)}] {run_key(*j):58s} "
                  f"AUC {r.get('auc_top10', float('nan')):.4f}  {time.time()-t0:5.1f}s")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # Threads, not processes: RDKit and the sklearn oracles release the GIL in
        # their C extensions, and threads let all workers share one CUDA context and
        # one copy of the loaded oracle models. Processes would multiply both.
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(execute_run, *j, budget=budget, force=force): j for j in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                j = futs[fut]
                try:
                    r = fut.result()
                    print(f"[{i:3d}/{len(todo)}] {run_key(*j):58s} "
                          f"AUC {r.get('auc_top10', float('nan')):.4f}")
                except Exception as e:
                    print(f"[{i:3d}/{len(todo)}] {run_key(*j):58s} FAILED {e}")

    print(f"\nsweep wall clock: {(time.time()-t_sweep)/60:.1f} min")
    return load_results()


def load_results() -> "pd.DataFrame":
    rows = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        d.pop("history", None)
        d.pop("top10_smiles", None)
        d.pop("traceback", None)
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "status" not in df.columns:
        df["status"] = "ok"
    if "prereg_sha256" in df.columns:
        bad = df[df["prereg_sha256"] != CURRENT_HASH]
        if len(bad):
            print(f"WARNING: {len(bad)} runs were produced under a DIFFERENT "
                  f"pre-registration hash. Disclose or re-run them.")
    return df

# %% [markdown]
# ### Launch
#
# Start with a 2-run dry check to confirm the pipeline end to end, then launch the sweep.
# On a cluster, run the exported `scripts/run_sweep.py` under your scheduler instead — a
# notebook kernel holding a multi-hour job is a liability.

# %% @notebook-only
DRY_RUN_FIRST = True

if DRY_RUN_FIRST:
    print("dry check (2 runs at reduced budget) ...")
    for j in [("albuterol_similarity", "reinvent", "explore_mid", 0),
              ("isomers_c7h8n2o2", "graph_ga", "explore_mid", 0)]:
        _p = run_path(*j)
        _bak = _p.read_text() if _p.exists() else None
        r = execute_run(*j, budget=200, force=True)
        print(f"  {run_key(*j):58s} AUC {r['auc_top10']:.4f}  status={r['status']}")
        if _bak is not None:
            _p.write_text(_bak)       # restore the real 1000-call result
        else:
            _p.unlink(missing_ok=True)
    print("\npipeline OK\n")

# %% @notebook-only
# Set to True to launch the full Phase-0 sweep (~1h at 8 workers, ~6h serial).
LAUNCH_SWEEP = False

if LAUNCH_SWEEP:
    RESULTS = run_sweep(budget=PREREGISTRATION["budget"])
else:
    RESULTS = load_results()
    print(f"loaded {len(RESULTS)} completed runs "
          f"(set LAUNCH_SWEEP=True to execute the rest)")

# %% [markdown]
# ---
# # Part 10 — Analysis and the Mechanical Verdict
#
# Everything below applies the rule frozen in Part 0. The code does not choose a test after
# looking at the data; it executes a test that was chosen before the data existed. The verdict
# function refuses to run if the pre-registration hash has changed.
#
# ### Criterion A — the interaction contrast
#
# A "rank flip" is an **interaction**, so it must be tested as one. For clusters $c_1, c_2$ with
# different winners $e_1 \ne e_2$:
#
# $$D = \big[\mathrm{AUC}(c_1,e_1) - \mathrm{AUC}(c_1,e_2)\big] + \big[\mathrm{AUC}(c_2,e_2) - \mathrm{AUC}(c_2,e_1)\big]$$
#
# $D > 0$ says exactly "each cluster prefers its own winner". Testing the two differences
# separately would be the wrong test — it would ask whether each cluster has a preference, not
# whether the preferences *disagree*. $D$ is also invariant to any per-oracle shift affecting
# all settings equally, which matters because oracle difficulty varies enormously (the isomer
# oracles live near 0, the similarity oracles near 0.6) and that variation is not signal.
#
# CI by stratified bootstrap over seeds within each oracle, 10,000 resamples.
#
# ### Criterion B — exact permutation Spearman
#
# With $n=8$ oracles, asymptotic $p$-values for Spearman's $\rho$ are unreliable. All $8! =
# 40{,}320$ permutations are enumerated, giving an exact null. Holm–Bonferroni across the 13
# frozen features. $\alpha = 0.10$, declared in Part 0 as a screening threshold with Phase 1 as
# the confirmatory test at $\alpha = 0.05$.

# %%
def aggregate_results(df: "pd.DataFrame") -> "pd.DataFrame":
    ok = df[df.get("status", "ok") == "ok"].copy()
    if ok.empty:
        return ok
    g = (ok.groupby(["backbone", "cluster", "oracle", "setting"])["auc_top10"]
           .agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"mean": "auc_mean", "std": "auc_sd", "count": "n_seeds"}))
    return g


def winning_setting(df: "pd.DataFrame", backbone: str, by: str = "cluster") -> Dict[str, str]:
    ok = df[(df.get("status", "ok") == "ok") & (df["backbone"] == backbone)]
    if ok.empty:
        return {}
    m = ok.groupby([by, "setting"])["auc_top10"].mean().reset_index()
    return {k: sub.loc[sub["auc_top10"].idxmax(), "setting"]
            for k, sub in m.groupby(by)}


def bootstrap_interaction(df: "pd.DataFrame", backbone: str,
                          c1: str, e1: str, c2: str, e2: str,
                          n_boot: int = 10000, ci: float = 0.95,
                          rng_seed: int = 20261119,
                          resample_oracles: bool = True) -> Dict[str, float]:
    """Bootstrap CI for D = (c1:e1-e2) + (c2:e2-e1).

    resample_oracles=True (the pre-registered test) resamples oracles within a
    cluster and then seeds within each oracle, so the interval covers
    generalisation to other oracles of the same landscape type.
    resample_oracles=False resamples seeds only and is reported as an optimistic
    bound; it answers the narrower question "is the effect real for THESE oracles".
    """
    rng = np.random.default_rng(rng_seed)
    ok = df[(df.get("status", "ok") == "ok") & (df["backbone"] == backbone)]

    def cell(cluster, setting):
        sub = ok[(ok["cluster"] == cluster) & (ok["setting"] == setting)]
        return {o: g["auc_top10"].to_numpy() for o, g in sub.groupby("oracle")}

    cells = {k: cell(*k) for k in [(c1, e1), (c1, e2), (c2, e2), (c2, e1)]}
    if any(not v for v in cells.values()):
        return {"D": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_boot": float("nan"), "confirmed": False}

    # Oracle lists per cluster, shared by the two settings of that cluster so a
    # resample draws the SAME oracles for e1 and e2 -- otherwise the contrast would
    # compare different molecules and the pairing that makes D meaningful is lost.
    oracles_of = {c1: sorted(cells[(c1, e1)]), c2: sorted(cells[(c2, e2)])}

    def stat(resample: bool):
        if resample and resample_oracles:
            picks = {c: list(rng.choice(oracles_of[c], size=len(oracles_of[c]), replace=True))
                     for c in (c1, c2)}
        else:
            picks = {c: list(oracles_of[c]) for c in (c1, c2)}

        parts = []
        for (cl, st), sign in [((c1, e1), +1), ((c1, e2), -1),
                               ((c2, e2), +1), ((c2, e1), -1)]:
            vals = []
            for o in picks[cl]:
                arr = cells[(cl, st)].get(o)
                if arr is None or len(arr) == 0:
                    continue
                v = rng.choice(arr, size=len(arr), replace=True) if resample else arr
                vals.append(v.mean())
            parts.append(sign * (np.mean(vals) if vals else 0.0))
        return float(sum(parts))

    point = stat(False)
    boots = np.array([stat(True) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    p_boot = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"D": point, "lo": float(lo), "hi": float(hi),
            "p_boot": float(min(1.0, p_boot)), "confirmed": bool(lo > 0),
            "resample_oracles": resample_oracles}


def criterion_A(df: "pd.DataFrame", verbose: bool = True) -> Dict[str, Any]:
    spec = PREREGISTRATION["criterion_A_rank_flip"]
    out: Dict[str, Any] = {"per_backbone": {}, "passes": False, "max_confirmed": 0}

    for backbone in PREREGISTRATION["backbones"]:
        winners = winning_setting(df, backbone, by="cluster")
        if len(winners) < 2:
            continue
        pairs, confirmed = [], 0
        clusters = sorted(winners)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1, c2 = clusters[i], clusters[j]
                e1, e2 = winners[c1], winners[c2]
                if e1 == e2:
                    continue
                res = bootstrap_interaction(df, backbone, c1, e1, c2, e2,
                                            n_boot=spec["n_bootstrap"], ci=spec["ci"],
                                            resample_oracles=True)
                opt = bootstrap_interaction(df, backbone, c1, e1, c2, e2,
                                            n_boot=spec["n_bootstrap"], ci=spec["ci"],
                                            resample_oracles=False)
                res.update({"c1": c1, "e1": e1, "c2": c2, "e2": e2,
                            "seed_only_lo": opt["lo"], "seed_only_hi": opt["hi"],
                            "seed_only_confirmed": opt["confirmed"]})
                pairs.append(res)
                confirmed += int(res["confirmed"])
        out["per_backbone"][backbone] = {
            "winners": winners, "pairs": pairs, "n_confirmed": confirmed
        }
        out["max_confirmed"] = max(out["max_confirmed"], confirmed)

        if verbose:
            print(f"\n{backbone}")
            print(f"  winning setting per cluster: "
                  f"{ {c: w.replace('explore_','') for c, w in winners.items()} }")
            if not pairs:
                print("  no cluster pair has differing winners -> no flip candidates")
            for r in pairs:
                mark = "CONFIRMED" if r["confirmed"] else "not confirmed"
                print(f"  {r['c1']}({r['e1'].replace('explore_','')}) vs "
                      f"{r['c2']}({r['e2'].replace('explore_','')}): D={r['D']:+.4f}")
                print(f"      hierarchical 95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]  "
                      f"<- decides: {mark}")
                print(f"      seed-only    95% CI [{r['seed_only_lo']:+.4f}, "
                      f"{r['seed_only_hi']:+.4f}]  (optimistic bound"
                      f"{', would confirm' if r['seed_only_confirmed'] else ''})")
            print(f"  -> {confirmed} confirmed flipped pair(s)")

    out["passes"] = out["max_confirmed"] >= 2
    if verbose:
        print(f"\nCRITERION A: {'PASS' if out['passes'] else 'FAIL'} "
              f"(needs >=2 confirmed pairs within one backbone; "
              f"best backbone has {out['max_confirmed']})")
    return out

# %%
from itertools import permutations  # noqa: E402


def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Spearman rho and an EXACT two-sided permutation p-value (n<=8)."""
    n = len(x)
    rho = sps.spearmanr(x, y).correlation
    if not np.isfinite(rho):
        return float("nan"), 1.0
    if n > 8:                                     # fall back for larger n
        return float(rho), float(sps.spearmanr(x, y).pvalue)
    # Spearman on fixed rank vectors is Pearson on those ranks, so centre and
    # normalise once and the permutation null is 40,320 dot products.
    rx = sps.rankdata(x).astype(np.float64)
    ry = sps.rankdata(y).astype(np.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    nx, ny = np.linalg.norm(rx), np.linalg.norm(ry)
    if nx < 1e-12 or ny < 1e-12:
        return float(rho), 1.0
    rx, ry = rx / nx, ry / ny
    obs = abs(float(rx @ ry))
    perms = np.array(list(permutations(range(n))), dtype=np.intp)
    null = np.abs(ry[perms] @ rx)
    return float(rho), float((null >= obs - 1e-12).mean())


def holm(pvals: Dict[str, float], alpha: float) -> Dict[str, Dict[str, Any]]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, rejected_so_far = {}, True
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        rej = rejected_so_far and (p <= thresh)
        rejected_so_far = rej
        out[k] = {"p": p, "holm_threshold": thresh, "significant": rej}
    return out


def criterion_B(df: "pd.DataFrame", diagnostics: "pd.DataFrame",
                backbone: str = "reinvent", verbose: bool = True) -> Dict[str, Any]:
    spec = PREREGISTRATION["criterion_B_diagnostic"]
    winners = winning_setting(df, backbone, by="oracle")
    if len(winners) < 4:
        if verbose:
            print(f"CRITERION B: insufficient data ({len(winners)} oracles with results)")
        return {"passes": False, "reason": "insufficient data", "features": {}}

    rows = diagnostics[diagnostics["oracle"].isin(winners)].copy()
    rows["win_idx"] = rows["oracle"].map(lambda o: SETTING_INDEX[winners[o]])
    y = rows["win_idx"].to_numpy(dtype=float)

    if len(np.unique(y)) < 2:
        if verbose:
            print(f"CRITERION B: FAIL -- the same setting wins on every oracle "
                  f"({SETTING_ORDER[int(y[0])]}); nothing for a diagnostic to predict.")
        return {"passes": False, "reason": "constant winner", "features": {},
                "winners": winners}

    raw: Dict[str, float] = {}
    rhos: Dict[str, float] = {}
    for f in DIAGNOSTIC_FEATURES:
        x = rows[f].to_numpy(dtype=float)
        if not np.all(np.isfinite(x)) or np.std(x) < 1e-12:
            rhos[f], raw[f] = float("nan"), 1.0
            continue
        r, p = exact_spearman_p(x, y)
        rhos[f], raw[f] = r, p

    corrected = holm(raw, spec["alpha"])
    passes = any(v["significant"] for v in corrected.values())

    if verbose:
        print(f"\nCriterion B (backbone = {backbone}, n = {len(y)} oracles)")
        print(f"  winner per oracle: "
              f"{ {o: w.replace('explore_','') for o, w in winners.items()} }")
        print(f"\n  {'feature':24s} {'rho':>7s} {'p_exact':>9s} {'holm_thr':>9s}  sig")
        for f in sorted(DIAGNOSTIC_FEATURES, key=lambda k: raw[k]):
            c = corrected[f]
            print(f"  {f:24s} {rhos[f]:+7.3f} {c['p']:9.4f} {c['holm_threshold']:9.4f}  "
                  f"{'YES' if c['significant'] else '.'}")
        print(f"\nCRITERION B: {'PASS' if passes else 'FAIL'} "
              f"(alpha={spec['alpha']}, Holm across {len(DIAGNOSTIC_FEATURES)} features)")

    return {"passes": passes, "features": {f: {"rho": rhos[f], **corrected[f]}
                                           for f in DIAGNOSTIC_FEATURES},
            "winners": winners, "n_oracles": int(len(y))}

# %%
def phase0_verdict(df: "pd.DataFrame", diagnostics: "pd.DataFrame") -> Dict[str, Any]:
    """Apply the frozen decision rule. Refuses to run on a modified pre-registration."""
    print("=" * 78)
    print("PHASE 0 VERDICT")
    print("=" * 78)

    if PREREG_STATUS == "mismatch":
        print("\nREFUSING TO PRODUCE A VERDICT.")
        print("The pre-registration in this notebook does not match the one on disk.")
        print("Restore the frozen rule, or disclose the change as a post-hoc deviation")
        print("and re-run this cell with the deviation documented.")
        return {"verdict": "REFUSED", "reason": "prereg hash mismatch"}

    if df.empty:
        print("\nNo results yet. Set LAUNCH_SWEEP=True in Part 9.")
        return {"verdict": "NO_DATA"}

    n_ok = int((df.get("status", "ok") == "ok").sum())
    n_err = len(df) - n_ok
    expected = len(SWEEP)
    print(f"\nruns: {n_ok} ok, {n_err} failed, {expected} expected "
          f"({100*n_ok/expected:.0f}% complete)")
    if n_ok < expected:
        print("INCOMPLETE SWEEP -- verdict below is provisional.")

    print("\n" + "-" * 78)
    print("CRITERION A: does the winning exploration setting flip across clusters?")
    print("-" * 78)
    A = criterion_A(df)

    print("\n" + "-" * 78)
    print("CRITERION B: does any cheap diagnostic track the winner?")
    print("-" * 78)
    B = criterion_B(df, diagnostics, backbone="reinvent")

    if A["passes"] and B["passes"]:
        verdict = "GO"
    elif A["passes"]:
        verdict = "PARTIAL"
    else:
        verdict = "PIVOT"

    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print("=" * 78)
    print(PREREGISTRATION["verdict_consequences"][verdict])
    print(f"\n(decision rule sha256 {CURRENT_HASH[:32]}...)")

    out = {"verdict": verdict, "criterion_A": A, "criterion_B": B,
           "prereg_sha256": CURRENT_HASH, "n_runs_ok": n_ok,
           "n_runs_expected": expected,
           "decided_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RESULTS_DIR / "phase0_verdict.json").write_text(
        json.dumps(out, indent=2, default=str))
    return out


# %% @notebook-only
VERDICT = phase0_verdict(RESULTS, DIAGNOSTICS) if not RESULTS.empty else {"verdict": "NO_DATA"}

# %% [markdown]
# ### Figures

# %%
def plot_setting_by_cluster(df: "pd.DataFrame", save: bool = True):
    """The headline figure: AUC by exploration setting, per cluster, per backbone.

    If H0 is true, the lines are NOT parallel -- the ordering changes between panels.
    Parallel lines are the null.
    """
    import matplotlib.pyplot as plt
    ok = df[df.get("status", "ok") == "ok"]
    if ok.empty:
        print("no results to plot")
        return None
    backbones = [b for b in PREREGISTRATION["backbones"] if b in set(ok["backbone"])]
    clusters = sorted(ok["cluster"].unique())
    colors = dict(zip(clusters, ["#2563eb", "#dc2626", "#16a34a", "#ca8a04"]))

    fig, axes = plt.subplots(1, len(backbones), figsize=(5.2 * len(backbones), 4.6),
                             squeeze=False)
    x = np.arange(len(SETTING_ORDER))
    for ax, bb in zip(axes[0], backbones):
        sub = ok[ok["backbone"] == bb]
        for c in clusters:
            cs = sub[sub["cluster"] == c]
            means = [cs[cs["setting"] == s]["auc_top10"].mean() for s in SETTING_ORDER]
            sems = [cs[cs["setting"] == s]["auc_top10"].sem() for s in SETTING_ORDER]
            ax.errorbar(x, means, yerr=sems, marker="o", capsize=4, lw=2.2,
                        ms=8, label=c, color=colors[c])
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("explore_", "") for s in SETTING_ORDER])
        ax.set_xlabel("exploration setting")
        ax.set_ylabel("AUC top-10" if bb == backbones[0] else "")
        ax.set_title(bb, fontsize=12, fontweight="600")
        ax.grid(alpha=0.25)
    axes[0][-1].legend(fontsize=9, title="landscape", framealpha=0.9)
    fig.suptitle("H1: does the winning exploration setting change with landscape?\n"
                 "non-parallel lines = rank flip = H0 survives",
                 fontsize=12.5, fontweight="600")
    fig.tight_layout()
    if save:
        fig.savefig(FIG_DIR / "h1_setting_by_cluster.png", dpi=160, bbox_inches="tight")
    return fig


def plot_per_oracle_heatmap(df: "pd.DataFrame", backbone: str = "reinvent", save: bool = True):
    import matplotlib.pyplot as plt
    ok = df[(df.get("status", "ok") == "ok") & (df["backbone"] == backbone)]
    if ok.empty:
        return None
    piv = ok.groupby(["oracle", "setting"])["auc_top10"].mean().unstack()[SETTING_ORDER]
    # z-score within oracle: we care about the ORDERING, not oracle difficulty
    z = piv.sub(piv.mean(axis=1), axis=0).div(piv.std(axis=1).replace(0, np.nan), axis=0)
    order = sorted(z.index, key=lambda o: (ORACLE_TO_CLUSTER[o], o))
    z = z.loc[order]
    piv = piv.loc[order]          # keep the annotation values aligned with z

    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(z) + 2.2))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
    ax.set_xticks(range(len(SETTING_ORDER)))
    ax.set_xticklabels([s.replace("explore_", "") for s in SETTING_ORDER])
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels([f"{o}  [{ORACLE_TO_CLUSTER[o][:4]}]" for o in z.index], fontsize=9)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(v) > 0.9 else "black")
    ax.set_title(f"{backbone}: AUC top-10 per oracle\n"
                 "(colour = z-scored within oracle; text = raw AUC)",
                 fontsize=11, fontweight="600")
    fig.colorbar(im, ax=ax, label="z within oracle", fraction=0.04)
    fig.tight_layout()
    if save:
        fig.savefig(FIG_DIR / f"per_oracle_{backbone}.png", dpi=160, bbox_inches="tight")
    return fig


# %% @notebook-only
if not RESULTS.empty:
    _ = plot_setting_by_cluster(RESULTS)
    _ = plot_per_oracle_heatmap(RESULTS, "reinvent")

# %% [markdown]
# ---
# # Part 11 — Exports
#
# Written out so the sweep can run under a cluster scheduler rather than a notebook kernel, and
# so the environment is reproducible from a file instead of from an install cell.

# %% @notebook-only
ENVIRONMENT_YML = """name: lce
channels: [conda-forge, pytorch, nvidia]
dependencies:
  - python=3.10
  - pip
  - numpy>=1.24,<3
  - scipy>=1.10
  - pandas>=2.0
  - scikit-learn>=1.3
  - matplotlib>=3.7
  - joblib>=1.3
  - tqdm
  - rdkit>=2023.09
  - pytorch>=2.0
  - pytorch-cuda=11.8      # match your cluster's driver
  - pip:
      - PyTDC>=0.4.1
"""

RUN_SWEEP_PY = '''#!/usr/bin/env python
"""Phase-0 sweep runner for the cluster scheduler.

    python scripts/run_sweep.py --workers 8
    python scripts/run_sweep.py --backbone reinvent --workers 6
    CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep.py --backbone reinvent &
    CUDA_VISIBLE_DEVICES=1 python scripts/run_sweep.py --backbone augmented_memory &

Resumable: completed runs are skipped, so re-running after preemption is safe.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("LCE_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from phase0_lib import (ALL_ORACLES, PREREGISTRATION, SETTING_ORDER,   # noqa: E402
                        build_sweep, run_sweep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--budget", type=int, default=PREREGISTRATION["budget"])
    ap.add_argument("--oracle", nargs="*", default=None)
    ap.add_argument("--backbone", nargs="*", default=None)
    ap.add_argument("--setting", nargs="*", default=None)
    ap.add_argument("--seed", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    jobs = build_sweep(a.oracle or ALL_ORACLES,
                       a.backbone or PREREGISTRATION["backbones"],
                       a.setting or SETTING_ORDER,
                       a.seed or PREREGISTRATION["seeds"])
    print(f"{len(jobs)} jobs, budget {a.budget}")
    run_sweep(jobs, budget=a.budget, n_workers=a.workers, force=a.force)


if __name__ == "__main__":
    main()
'''

SLURM_SH = """#!/bin/bash
#SBATCH --job-name=lce-phase0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/phase0_%j.out

# VRAM note: one LM run peaks near 1.2 GB, so a single 11 GB 2080 Ti or a 12 GB
# 4070 Super holds every worker this job will start. --cpus-per-task is the knob
# that actually governs throughput, because the PMO oracles are CPU-bound.

set -euo pipefail
module load cuda/11.8 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lce

mkdir -p logs
python scripts/run_sweep.py --workers "${SLURM_CPUS_PER_TASK:-8}"
"""

(PROJECT_ROOT / "environment.yml").write_text(ENVIRONMENT_YML)
(PROJECT_ROOT / "scripts").mkdir(exist_ok=True)
(PROJECT_ROOT / "scripts" / "run_sweep.py").write_text(RUN_SWEEP_PY)
(PROJECT_ROOT / "scripts" / "slurm_phase0.sh").write_text(SLURM_SH)
os.chmod(PROJECT_ROOT / "scripts" / "run_sweep.py", 0o755)

GITIGNORE = """results/runs/
data/
vendor/
figures/
logs/
__pycache__/
*.pyc
.ipynb_checkpoints/
.claude/
.ropeproject/
*.egg-info/
.venv/
"""
(PROJECT_ROOT / ".gitignore").write_text(GITIGNORE)

print("wrote:")
for p in ["environment.yml", "scripts/run_sweep.py", "scripts/slurm_phase0.sh",
          ".gitignore", "preregistration.json"]:
    print(f"  {PROJECT_ROOT / p}")
print("\nNOTE: results/runs/, data/, vendor/ and figures/ are gitignored, but")
print("      preregistration.json and results/phase0_verdict.json are NOT --")
print("      commit both. The frozen rule and the verdict are the record.")

# %% [markdown]
# ---
# # Part 12 — Positioning for ICML 2027
#
# The experiment is sound. Whether it becomes an ICML paper depends on an argument that has to
# be made deliberately, because the default reading of this work is unflattering.
#
# ### The objection that decides the paper
#
# > *"This is per-instance hyperparameter selection. Algorithm selection and configuration is a
# > solved research area — SMAC, Hyperband, bandit-based online selection, meta-learning over
# > problem features. You picked one hyperparameter, built a featurisation, and fit a shallow
# > predictor. What is new?"*
#
# This will be raised. It is the single most likely rejection reason, more likely than any
# concern about chemistry, and the proposal does not currently have a prepared answer. Three
# defensible responses, in order of strength:
#
# **1. The budget regime is the novelty, not the mechanism.** Classical algorithm selection
# assumes you can afford to *try* configurations — Hyperband needs many partial runs, bandits
# need repeated pulls to estimate arms. Here the total budget is 10,000 oracle calls and a
# single configuration consumes all of it. You get **one** shot, and must choose from ~300
# probes, before any optimiser has run. That is a genuinely different problem: zero-shot
# configuration from a cheap landscape probe, not online selection among arms you can sample.
# Make this the framing of the introduction.
#
# **2. The diagnostic is a contribution independent of the controller.** If
# reward-distribution geometry measured from a few hundred samples predicts which exploration
# regime wins, that is a statement about optimisation landscapes that outlives our specific
# controller. Fitness-landscape analysis has a long history in evolutionary computation but has
# never been connected to the modern sample-efficient generative cohort.
#
# **3. The negative result is publishable too.** If landscape-conditioning does *not* beat
# generic adaptivity (Phase 2d), that bounds the value of instance-conditioning in this
# setting — which the field has assumed without testing since PMO.
#
# ### What must be added to Phase 2 because of this objection
#
# The proposal's Phase-2 baselines are: (a) best fixed, (b) default, (c) random, (d) generic
# adaptive (EEPO-style). **Baseline (d) is not sufficient for ICML.** Add:
#
# > **(e) An online bandit baseline.** Split the budget into rounds, treat each exploration
# > setting as an arm, and run a standard algorithm (EXP3 or successive halving) that reallocates
# > budget across settings as rewards come in. This is the strongest, most obvious thing a
# > reviewer will ask why you did not compare against.
#
# This matters because (e) tests a real alternative hypothesis: *maybe you do not need to
# predict the landscape at all, you can just adapt online*. If the bandit wins, the diagnostic
# is unnecessary and the paper is substantially weaker — which is exactly why it has to be run
# rather than argued around. Budget it into Phase 2 now.
#
# ### What ICML will like
#
# - **Pre-registration with an enforced hash.** Rare in ML papers and increasingly valued.
#   Say so explicitly in the paper; it is cheap credibility and it is already built.
# - **An interaction contrast with a hierarchical bootstrap** instead of the usual pile of
#   uncorrected pairwise t-tests. Reviewers who check statistics will notice.
# - **Replication across algorithm families** (gradient policy and population search).
# - **A declared screen/confirmation split** (Phase 0 at α=0.10, Phase 1 at α=0.05), which
#   pre-empts the "you ran 23 oracles and reported the significant ones" critique.
#
# ### What ICML will dislike
#
# **1. "Empirical and theorem-free."** The proposal says this plainly, and it is fine — ICML
# accepts empirical papers — but only when the empirical contribution is *decisive*. Decisive
# here means a clear effect size with tight intervals and a controller that beats every baseline
# including (e). A marginal effect with wide intervals and no theory is a reject.
#
# **2. The venue mismatch in the other direction.** ICML cares little for molecules per se.
# Frame every claim in terms of **optimisation under budget**; let chemistry be the instantiation.
# The abstract should be legible to someone who works on black-box optimisation and has never
# heard of a Murcko scaffold.
#
# **3. Single-objective, in-silico, proxy oracles.** The proposal's limitations section is
# honest about this. Keep it that way; do not let a reviewer be the one to point it out.
#
# ### The timeline is the practical risk, and it is severe
#
# The proposal's own dates are **abstract 16 Jan 2027, paper 22 Jan 2027** — verify these
# against the official CFP, since they were written in advance. Today is **19 November 2026**,
# which is **~58 days** to the abstract deadline. The proposal's research plan is **8 weeks
# (56 days)** from Phase 0 through writing.
#
# That is **zero slack**. Every week of Phase 0 slippage comes directly out of the writing
# window, and a paper written in the last three days will read like one. Two realistic options:
#
# | Option | What it means |
# |---|---|
# | **Compress** | Run Phase 0 and Phase 1 in parallel (the sweep is embarrassingly parallel and your hardware is not the bottleneck). Cut Phase 3 to the backbone-transfer experiment only. Start the related-work section during the Phase-0 sweep, when you are blocked on compute anyway. |
# | **Re-target** | Aim at NeurIPS 2027 (≈May) and use the extra months for the docking oracle and a proper Phase 3. The work is strictly better, and the proposal already names NeurIPS as the fallback. |
#
# **Recommendation: decide this before launching the sweep, not after.** If ICML 2027 is the
# real target, the compressed plan starts now and Phase 3 is scoped down today. Aiming at ICML
# and discovering in January that the controller needs another two weeks is the worst of both.
#
# ### Honest assessment of where the risk actually sits
#
# Not in the venue, and not in Criterion A. Criterion A — that the best setting flips across
# landscapes — is quite likely to pass; PMO's published results already suggest it. But
# Criterion A alone is a characterisation, and characterisation is a weak ICML submission.
#
# The paper lives or dies on two things: the diagnostic being genuinely predictive (Criterion B,
# where `n=8` in Phase 0 gives weak evidence either way), and the controller beating baseline
# (e) as well as (d). **Do not over-read a Phase-0 pass on Criterion B.** It is a screen.
# Phase 1, with 23 oracles, is the test to be nervous about.

# %% [markdown]
# ---
# ## What to do next
#
# 1. **Commit `preregistration.json` before running anything.** Its value is entirely in being
#    timestamped earlier than the results.
# 2. Run the setup cells and confirm the smoke tests pass — especially the prior's validity rate
#    (expect 90–97%) and the `sample()`/`likelihood()` agreement check.
# 3. Launch the sweep via `scripts/run_sweep.py` under your scheduler, not in the notebook.
# 4. Run Part 10 once, read the verdict, and **follow it** — including if it says PIVOT. The
#    pre-registration exists precisely for the case where the result is unwelcome.
# 5. Run the Step-0 novelty check from the proposal (Sec. 10) in parallel with the sweep.
#    "Landscape-aware / instance-adaptive method selection" is a crowded enough phrase that it
#    is worth knowing now whether someone published this in the last six months.
