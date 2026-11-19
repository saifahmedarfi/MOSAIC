"""Phase 0 library -- GENERATED from src/phase0_source.py. Do not edit.

Regenerate with:  python src/build_notebook.py

Importing this module defines the oracles, backbones, diagnostics and analysis
used by the notebook and by scripts/run_sweep.py. Demonstration and driver cells
are excluded, so an import has no side effects beyond loading the vendored prior
and the ZINC pool (both cached on disk after the first notebook run).
"""


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


import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("LCE_ROOT", Path.cwd())).resolve()
if PROJECT_ROOT.name == "notebooks":          # tolerate being run from a subdir
    PROJECT_ROOT = PROJECT_ROOT.parent
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


# Make the GB-GA operators importable. They are self-contained RDKit modules;
# we add their directory to sys.path rather than restructuring the vendored tree.
_GA_DIR = str((MOL_OPT_DIR / "main" / "graph_ga").resolve())
if _GA_DIR not in sys.path:
    sys.path.insert(0, _GA_DIR)

import crossover as co   # noqa: E402  (vendored GB-GA)
import mutate as mu      # noqa: E402  (vendored GB-GA)


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


# ---- load the pretrained prior and verify ------------------------------------
PRIOR = SmilesLM(VOC)
PRIOR.load_prior(PRIOR_CKPT)
for p in PRIOR.rnn.parameters():
    p.requires_grad = False
PRIOR.rnn.eval()

n_params = sum(p.numel() for p in PRIOR.rnn.parameters())
print(f"prior loaded: {n_params:,} parameters ({n_params*4/1024**2:.1f} MB fp32) on {DEVICE}")


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
