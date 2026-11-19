#!/usr/bin/env python
"""Unit tests for the statistics and metric core of Phase 0.

These are the functions that decide GO vs PIVOT, so they are tested against
independently-computed expected values rather than against themselves.

    python tests/test_core.py

Runs without rdkit or torch: the tested functions are extracted from
src/phase0_source.py by source surgery, so the tests exercise the SAME code
the notebook runs, not a copy that can drift out of sync.
"""
import ast
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "phase0_source.py"

# ---------------------------------------------------------------------------
# Extract the pure-python / numpy functions from the notebook source.
# ---------------------------------------------------------------------------
WANTED = {
    "_gini", "exact_spearman_p", "holm", "bootstrap_interaction",
    "criterion_A", "criterion_B", "winning_setting", "aggregate_results",
    "BudgetedOracle", "OracleCall",
}

src_text = SRC.read_text()
code_cells = []
cur, kind = [], None
for line in src_text.splitlines():
    st = line.strip()
    if st.startswith("# %% [markdown]"):
        if kind == "code":
            code_cells.append("\n".join(cur))
        kind, cur = "md", []
    elif st.startswith("# %%"):
        if kind == "code":
            code_cells.append("\n".join(cur))
        kind, cur = "code", []
    else:
        cur.append(line)
if kind == "code":
    code_cells.append("\n".join(cur))

# The extracted definitions carry real annotations (`Callable[[str], float]`,
# `Optional[int]`), which are evaluated when the def or class body runs. Stubbing
# them as bare `object` makes a subscript raise, so bind the genuine objects.
import dataclasses
import typing

ns = {
    "np": np, "pd": pd, "sps": sps, "permutations": permutations,
    "Chem": None, "math": __import__("math"), "time": __import__("time"),
    "json": __import__("json"),
    "dataclass": dataclasses.dataclass, "field": dataclasses.field,
    "Dict": typing.Dict, "List": typing.List, "Any": typing.Any,
    "Optional": typing.Optional, "Tuple": typing.Tuple,
    "Sequence": typing.Sequence, "Callable": typing.Callable,
}
extracted = set()
for cell in code_cells:
    try:
        tree = ast.parse(cell)
    except SyntaxError:
        continue
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in WANTED and isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            try:
                exec(compile(ast.Module([node], []), "<extract>", "exec"), ns)
                extracted.add(name)
            except Exception as e:
                print(f"  could not extract {name}: {e}")

print(f"extracted {len(extracted)} objects from the notebook source: "
      f"{', '.join(sorted(extracted))}\n")

_gini = ns["_gini"]
exact_spearman_p = ns["exact_spearman_p"]
holm = ns["holm"]
bootstrap_interaction = ns["bootstrap_interaction"]
winning_setting = ns["winning_setting"]

# Globals the extracted functions close over.
ns["PREREGISTRATION"] = {
    "backbones": ["reinvent"],
    "criterion_A_rank_flip": {"n_bootstrap": 2000, "ci": 0.95},
    "criterion_B_diagnostic": {"alpha": 0.10},
}
ns["SETTING_ORDER"] = ["explore_low", "explore_mid", "explore_high"]
ns["SETTING_INDEX"] = {s: i for i, s in enumerate(ns["SETTING_ORDER"])}

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------------------
print("=" * 72)
print("1. top_auc -- must match mol_opt/main/optimizer.py::top_auc")
print("=" * 72)


def reference_top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    """Verbatim transcription of the published implementation."""
    sum_, prev, called = 0, 0, 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum_ += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum_ += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum_ += (max_oracle_calls - len(buffer)) * top_n_now
    return sum_ / max_oracle_calls


class FakeOracle(ns["BudgetedOracle"]):
    """BudgetedOracle with the RDKit canonicalisation path stubbed out."""

    def __init__(self, scores, max_oracle_calls=1000, freq_log=10):
        super().__init__("fake", lambda s: 0.0, max_oracle_calls, freq_log)
        for i, sc in enumerate(scores):
            self.mol_buffer[f"m{i}"] = [float(sc), i + 1]


rng = np.random.default_rng(7)
for trial, (n, budget, freq) in enumerate(
    [(1000, 1000, 10), (537, 1000, 10), (1000, 1000, 100), (73, 100, 10), (1000, 1000, 1)]
):
    scores = rng.random(n)
    orc = FakeOracle(scores, max_oracle_calls=budget, freq_log=freq)
    for k in (1, 10, 100):
        mine = orc.top_auc(k)
        ref = reference_top_auc(orc.mol_buffer, k, orc.finish, freq, budget)
        check(f"top_auc n={n} budget={budget} freq={freq} top{k}",
              abs(mine - ref) < 1e-12, f"{mine:.10f} vs {ref:.10f}")

# A monotone-increasing run must beat a monotone-decreasing one with the same scores.
up = FakeOracle(np.linspace(0, 1, 1000)).top_auc(10)
down = FakeOracle(np.linspace(1, 0, 1000)).top_auc(10)
check("early discovery scores higher than late", down > up, f"down={down:.4f} up={up:.4f}")

# A constant-score run does NOT equal that constant: the published integral starts
# from prev=0, losing freq_log*final/(2*budget). We assert the artifact EXACTLY,
# because silently "fixing" it would break comparability with published PMO numbers.
const = FakeOracle(np.full(1000, 0.42), max_oracle_calls=1000, freq_log=10).top_auc(10)
expected = 0.42 - 10 * 0.42 / (2 * 1000)
check("constant 0.42 run -> 0.4179 (documented ramp-in artifact)",
      abs(const - expected) < 1e-9, f"{const:.6f} == {expected:.6f}")

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("2. budget semantics -- dedupe must not consume budget")
print("=" * 72)


class CountingOracle(ns["BudgetedOracle"]):
    def __init__(self, budget):
        self.n_evals = 0
        super().__init__("count", self._fn_impl, budget, 10)

    def _fn_impl(self, smi):
        self.n_evals += 1
        return 0.5

    def score_smi(self, smi):           # bypass RDKit; treat the string as canonical
        self.proposals_seen += 1
        if len(self.mol_buffer) >= self.max_oracle_calls:
            return 0.0
        if not smi:
            self.invalid_seen += 1
            return 0.0
        if smi in self.mol_buffer:
            self.duplicate_seen += 1
            return self.mol_buffer[smi][0]
        score = float(self._fn(smi))
        self.mol_buffer[smi] = [score, len(self.mol_buffer) + 1]
        return score


o = CountingOracle(100)
o([f"m{i}" for i in range(50)])
o([f"m{i}" for i in range(50)])          # exact repeats
check("50 unique + 50 repeats -> 50 evaluations", o.n_evals == 50, f"{o.n_evals}")
check("buffer holds 50", len(o) == 50, f"{len(o)}")
check("duplicates counted", o.duplicate_seen == 50, f"{o.duplicate_seen}")
check("yield_rate = 0.5", abs(o.summary()["yield_rate"] - 0.5) < 1e-9)

o2 = CountingOracle(100)
o2([f"x{i}" for i in range(500)])
check("budget caps buffer at exactly 100", len(o2) == 100, f"{len(o2)}")
check("finish fires at the cap", o2.finish)
check("no evaluations past the cap", o2.n_evals == 100, f"{o2.n_evals}")

o3 = CountingOracle(100)
o3(["", "", "a"])
check("empty SMILES are free", len(o3) == 1 and o3.invalid_seen == 2)

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("3. exact permutation Spearman")
print("=" * 72)

x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
rho, p = exact_spearman_p(x, y)
check("perfect correlation -> rho=1", abs(rho - 1.0) < 1e-12, f"rho={rho}")
check("perfect correlation -> p=2/8!", abs(p - 2 / 40320) < 1e-9,
      f"p={p:.3e} (both the identity and the reversal hit |rho|=1)")

rho_r, p_r = exact_spearman_p(x, y[::-1])
check("perfect anticorrelation -> rho=-1", abs(rho_r + 1.0) < 1e-12)
check("anticorrelation -> same two-sided p", abs(p_r - p) < 1e-12)

# agreement with scipy's rho on random data, and a calibrated null
rng = np.random.default_rng(3)
max_dev, pvals = 0.0, []
for _ in range(40):
    a, b = rng.random(8), rng.random(8)
    r_mine, p_mine = exact_spearman_p(a, b)
    r_scipy = sps.spearmanr(a, b).correlation
    max_dev = max(max_dev, abs(r_mine - r_scipy))
    pvals.append(p_mine)
check("rho agrees with scipy.spearmanr", max_dev < 1e-12, f"max dev {max_dev:.2e}")
check("exact p-values lie in (0,1]", all(0 < q <= 1 for q in pvals))

# Under the null the p-value should be roughly uniform: ~10% below 0.10.
rng = np.random.default_rng(11)
below = sum(exact_spearman_p(rng.random(8), rng.random(8))[1] <= 0.10 for _ in range(300))
check("null p-values calibrated (~10% <= 0.10)", 15 <= below <= 60,
      f"{below}/300 = {below/3:.0f}%")

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("4. Holm-Bonferroni")
print("=" * 72)

res = holm({"a": 0.001, "b": 0.04, "c": 0.5, "d": 0.9}, alpha=0.10)
check("smallest p tested at alpha/m", abs(res["a"]["holm_threshold"] - 0.10 / 4) < 1e-12)
check("smallest p rejected", res["a"]["significant"])
check("0.04 vs alpha/3=0.0333 -> not rejected", not res["b"]["significant"])
check("Holm stops after the first failure", not res["c"]["significant"]
      and not res["d"]["significant"])

res2 = holm({f"f{i}": 0.5 for i in range(13)}, alpha=0.10)
check("13 null features -> nothing significant",
      not any(v["significant"] for v in res2.values()))

res3 = holm({"a": 0.0001, "b": 0.001, "c": 0.9}, alpha=0.10)
check("step-down admits the second hit", res3["a"]["significant"] and res3["b"]["significant"])

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("5. bootstrap interaction contrast (Criterion A)")
print("=" * 72)


def synth(flip: bool, noise: float, seed: int = 0, n_seeds: int = 3) -> pd.DataFrame:
    """Two clusters, three settings. `flip` controls whether preferences disagree."""
    r = np.random.default_rng(seed)
    rows = []
    for cluster, oracles in [("A", ["a1", "a2"]), ("B", ["b1", "b2"])]:
        base = 0.5 if cluster == "A" else 0.2         # clusters differ in difficulty
        for oracle in oracles:
            for setting in ns["SETTING_ORDER"]:
                if flip:
                    eff = {"explore_low": 0.10, "explore_mid": 0.0, "explore_high": -0.10}
                    if cluster == "B":
                        eff = {k: -v for k, v in eff.items()}
                else:
                    eff = {"explore_low": 0.10, "explore_mid": 0.0, "explore_high": -0.10}
                for sd in range(n_seeds):
                    rows.append({
                        "backbone": "reinvent", "cluster": cluster, "oracle": oracle,
                        "setting": setting, "seed": sd, "status": "ok",
                        "auc_top10": base + eff[setting] + r.normal(0, noise),
                    })
    return pd.DataFrame(rows)


ns["bootstrap_interaction"].__globals__.update(ns)

df_flip = synth(flip=True, noise=0.005, seed=1)
w = winning_setting(df_flip, "reinvent", by="cluster")
check("synthetic flip: A prefers low", w["A"] == "explore_low", str(w))
check("synthetic flip: B prefers high", w["B"] == "explore_high", str(w))

r_flip = bootstrap_interaction(df_flip, "reinvent", "A", "explore_low",
                               "B", "explore_high", n_boot=3000)
check("real flip is detected", r_flip["confirmed"],
      f"D={r_flip['D']:+.4f} CI [{r_flip['lo']:+.4f},{r_flip['hi']:+.4f}]")
check("D ~ 0.4 as constructed", abs(r_flip["D"] - 0.4) < 0.05, f"D={r_flip['D']:.4f}")

df_noflip = synth(flip=False, noise=0.005, seed=2)
w2 = winning_setting(df_noflip, "reinvent", by="cluster")
check("no-flip synthetic: both clusters prefer low",
      w2["A"] == "explore_low" and w2["B"] == "explore_low", str(w2))
# criterion_A only forms a pair when the clusters have DIFFERENT winners, so this
# data must yield zero candidate pairs; a hand-picked non-flip pair would exercise
# a branch the real gate never reaches.
check("identical winners -> no flip candidate is even formed", w2["A"] == w2["B"])

# Hierarchical vs seed-only: the hierarchical interval must be WIDER, because it
# carries oracle-level uncertainty that seed-only resampling throws away.
hier = bootstrap_interaction(df_flip, "reinvent", "A", "explore_low",
                             "B", "explore_high", n_boot=3000, resample_oracles=True)
seed_only = bootstrap_interaction(df_flip, "reinvent", "A", "explore_low",
                                  "B", "explore_high", n_boot=3000, resample_oracles=False)
check("hierarchical CI is wider than seed-only",
      (hier["hi"] - hier["lo"]) >= (seed_only["hi"] - seed_only["lo"]),
      f"hier={hier['hi']-hier['lo']:.5f} seed_only={seed_only['hi']-seed_only['lo']:.5f}")
check("a large real flip still survives the hierarchical test", hier["confirmed"],
      f"D={hier['D']:+.4f} CI [{hier['lo']:+.4f},{hier['hi']:+.4f}]")

# False-positive rate: pure noise, no effect at all.
fp = 0
for s in range(40):
    r = np.random.default_rng(100 + s)
    rows = [{"backbone": "reinvent", "cluster": c, "oracle": o, "setting": st,
             "seed": sd, "status": "ok", "auc_top10": 0.4 + r.normal(0, 0.02)}
            for c, os_ in [("A", ["a1", "a2"]), ("B", ["b1", "b2"])]
            for o in os_ for st in ns["SETTING_ORDER"] for sd in range(3)]
    d = pd.DataFrame(rows)
    ww = winning_setting(d, "reinvent", by="cluster")
    if ww["A"] == ww["B"]:
        continue
    if bootstrap_interaction(d, "reinvent", "A", ww["A"], "B", ww["B"], n_boot=1500,
                             rng_seed=s, resample_oracles=True)["confirmed"]:
        fp += 1
check("false positives on pure noise stay low (hierarchical)", fp <= 8,
      f"{fp}/40 noise trials confirmed")

# Same noise, seed-only resampling: the false-positive rate must be HIGHER.
# This is the empirical justification for making the hierarchical test primary.
fp_seed = 0
for s in range(40):
    r = np.random.default_rng(100 + s)
    rows = [{"backbone": "reinvent", "cluster": c, "oracle": o, "setting": st,
             "seed": sd, "status": "ok", "auc_top10": 0.4 + r.normal(0, 0.02)}
            for c, os_ in [("A", ["a1", "a2"]), ("B", ["b1", "b2"])]
            for o in os_ for st in ns["SETTING_ORDER"] for sd in range(3)]
    d = pd.DataFrame(rows)
    ww = winning_setting(d, "reinvent", by="cluster")
    if ww["A"] == ww["B"]:
        continue
    if bootstrap_interaction(d, "reinvent", "A", ww["A"], "B", ww["B"], n_boot=1500,
                             rng_seed=s, resample_oracles=False)["confirmed"]:
        fp_seed += 1
check("seed-only resampling is more permissive than hierarchical", fp_seed >= fp,
      f"seed-only {fp_seed}/40 vs hierarchical {fp}/40 -- why hierarchical is primary")

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("6. build artifacts -- the generated library must satisfy its consumers")
print("=" * 72)

import ast as _ast

LIB = ROOT / "src" / "phase0_lib.py"
check("src/phase0_lib.py exists (run: python src/build_notebook.py)", LIB.exists())

if LIB.exists():
    _lib = _ast.parse(LIB.read_text())
    _defined = set()
    for _n in _lib.body:
        if isinstance(_n, (_ast.FunctionDef, _ast.ClassDef)):
            _defined.add(_n.name)
        elif isinstance(_n, _ast.Assign):
            _defined |= {t_.id for t_ in _n.targets if isinstance(t_, _ast.Name)}
        elif isinstance(_n, (_ast.Import, _ast.ImportFrom)):
            _defined |= {a.asname or a.name.split(".")[0] for a in _n.names}

    _script = ROOT / "scripts" / "run_sweep.py"
    check("scripts/run_sweep.py exists", _script.exists())
    if _script.exists():
        _needed = set()
        for _n in _ast.walk(_ast.parse(_script.read_text())):
            if isinstance(_n, _ast.ImportFrom) and _n.module == "phase0_lib":
                _needed |= {a.name for a in _n.names}
        check("run_sweep.py imports resolve against phase0_lib",
              not (_needed - _defined),
              f"missing: {sorted(_needed - _defined) or 'none'}")

    # The library must not carry demo cells: a bare smoke-test call at module
    # level would fire on import and spend oracle budget.
    _src_txt = LIB.read_text()
    # Match ASSIGNMENTS, not mentions: phase0_verdict legitimately names
    # LAUNCH_SWEEP inside a help string telling the user which cell to edit.
    for _bad in ("Graph GA smoke test", "LM backbone smoke test",
                 "DRY_RUN_FIRST = ", "LAUNCH_SWEEP = ", "ENVIRONMENT_YML = "):
        check(f"library excludes notebook-only code: {_bad.strip(' =')}",
              _bad not in _src_txt)

for _f in ["environment.yml", "scripts/slurm_phase0.sh", ".gitignore",
           "landscape_phase0.ipynb"]:
    check(f"asset present: {_f}", (ROOT / _f).exists())

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("7. Gini")
print("=" * 72)
check("uniform values -> gini 0", abs(_gini(np.ones(100))) < 1e-9)
check("all mass on one value -> gini -> 1",
      _gini(np.array([0.0] * 99 + [1.0])) > 0.98, f"{_gini(np.array([0.0]*99+[1.0])):.4f}")
check("all-zero input is safe", _gini(np.zeros(10)) == 0.0)
g_needle = _gini(np.array([0.0] * 95 + [1.0] * 5))
g_graded = _gini(np.linspace(0, 1, 100))
check("needle landscape has higher gini than graded", g_needle > g_graded,
      f"needle={g_needle:.3f} graded={g_graded:.3f}")

# ---------------------------------------------------------------------------
print()
print("=" * 72)
if FAILED:
    print(f"{len(FAILED)} TEST(S) FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("ALL TESTS PASSED")
sys.exit(0)
