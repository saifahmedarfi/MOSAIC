#!/usr/bin/env python
"""Phase 2 power gate: can this study return anything but BENCHMARK-INSENSITIVE?

    python scripts/phase2_power.py

Phase 1 measured the three backbones within 0.0034 AUC of each other against a median
within-cell seed SD of 0.0063. At 9 oracles a 95% argmax-replication bar may therefore be
unreachable, which would make BENCHMARK-INSENSITIVE close to pre-determined. Phase 0's bar
once required two confirmed cluster pairs when the oracle set admitted at most one, and that
was only found afterwards. This gate quantifies the risk BEFORE the pre-registration freezes.

LEGITIMACY LINE. Choosing the replication bar and the seed count from a simulation driven by
Phase 0/1 data, before any Phase 2 run exists, is pre-registration working as intended.
Changing either after seeing Phase 2 data is forbidden. This module reads no Phase 2 data,
decides the seed count once (there is NO result-triggered seed extension: that would be
optional stopping), and its output is committed alongside the pre-registration so the order
is verifiable from timestamps. If nothing is reachable it halts the build (non-zero exit).

Simulation design.
  * Per-oracle means and the pooled within-cell seed SD come from Phase 1's measured
    `auc_top10` (`Phase 1/phase1_run_manifest.csv`, `explore_mid` rows only, because Phase 2
    runs that setting alone and mixing settings would inflate the SD with setting variance).
    Each oracle contributes its mean over backbones, so the planted null really is a null
    (all backbones identical); the backbone gaps are what `effect_sd` plants. Oracles whose
    pooled SD is below `MIN_SD` (valsartan_smarts scores ~0 for everyone) carry no usable
    noise scale and are excluded. Each replicate draws `n_oracles` of the remaining oracles.
  * Fallback if the manifest is unreadable: 15 synthetic oracles with means spread over
    [0.2, 0.9] and SD 0.0063 (the recorded median).
  * `effect_sd` = that many pooled SDs by which backbone A leads on `auc_top10` and backbone
    B leads on `hypervolume` (the PRIMARY_PAIR). hypervolume gets the same oracle means and
    SDs. Secondary metrics and scalarization_loss are noise, as under the null.
  * The verdict NAME depends only on Criterion 0, A and C (phase2_verdict reads B and G2 only
    for descriptive fields), so the simulation skips B and G2 and passes inert stubs. This
    changes no verdict and removes most of the cost.
  * Replicates are common-random-number across effect sizes: the same `seed` gives identical
    noise draws and only the planted shift differs, so power curves are smooth in effect.
  * Reporting beyond the frozen rule (none of it changes a decision rule): a data-derived
    "Phase-1-sized" effect (observed between-method spread / median pooled seed SD), the
    verdict distribution per cell, the smallest planted effect reaching the target power per
    n (the detectable-effect floor: "we would have detected X and observed none"), and the
    smallest sufficient n at the Phase-1-sized effect and at the floor. The seed count itself
    is still chosen by the frozen rule only.
  * n_boot: the real analysis uses N_BOOT = 10,000. The simulation runs the whole pipeline
    n_rep times, so it uses `n_boot` = 2,000 by default; whether a replication fraction
    clears 0.95 converges well before 10,000 draws, so the power estimate is not materially
    affected, and the speed is what makes an interactive gate and a test suite possible.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import phase2_stats as P  # noqa: E402

MANIFEST = ROOT / "Phase 1" / "phase1_run_manifest.csv"
GATE_SETTING = "explore_mid"
EFFECTS = (0.0, 1.0, 2.0)
BACKBONES = ("A", "B", "C")
MIN_SD = 1e-3
FALLBACK_SD = 0.0063
SIM_N_BOOT = 2000
_cache: Dict[str, Any] = {}


def phase1_oracle_params():
    """(means, sds, source). Per-oracle mean over backbones and pooled within-cell seed SD."""
    if "p" in _cache:
        return _cache["p"]
    try:
        d = pd.read_csv(MANIFEST)
        d = d[(d["setting"] == GATE_SETTING) & (d["status"] == "ok")]
        g = d.groupby(["oracle", "backbone"])["auc_top10"]
        cell = pd.DataFrame({"m": g.mean(), "v": g.var(ddof=1), "n": g.count()}).dropna()
        cell["dof"] = cell["n"] - 1
        per = cell.groupby("oracle").apply(
            lambda c: pd.Series({"mean": c["m"].mean(),
                                 "sd": float(np.sqrt((c["v"] * c["dof"]).sum() / c["dof"].sum()))}),
            include_groups=False)
        per = per[per["sd"] >= MIN_SD]
        if len(per) < 9:
            raise ValueError(f"only {len(per)} usable oracles")
        out = (per["mean"].to_numpy(), per["sd"].to_numpy(), "manifest", list(per.index))
    except Exception as e:  # unreadable or too thin: documented fallback
        print(f"WARNING: manifest unusable ({e}); using recorded summary fallback", file=sys.stderr)
        out = (np.linspace(0.2, 0.9, 15), np.full(15, FALLBACK_SD), "fallback", [])
    _cache["p"] = out
    return out


def _frame(n_seeds: int, effect_sd: float, n_oracles: int, rng) -> pd.DataFrame:
    means, sds, _, _ = phase1_oracle_params()
    pick = rng.choice(len(means), size=n_oracles, replace=False)
    rows = []
    for i, k in enumerate(pick):
        mu, sd = means[k], sds[k]
        for bb in BACKBONES:
            for s in range(n_seeds):
                row = {"oracle": f"o{i}", "backbone": bb, "setting": GATE_SETTING,
                       "seed": s, "status": "ok"}
                row["auc_top10"] = mu + (effect_sd * sd if bb == "A" else 0.0) + sd * rng.standard_normal()
                row["hypervolume"] = mu + (effect_sd * sd if bb == "B" else 0.0) + sd * rng.standard_normal()
                for m in P.SECONDARY_METRICS + ("scalarization_loss",):
                    row[m] = 0.5 + 0.05 * rng.standard_normal()
                rows.append(row)
    return pd.DataFrame(rows)


def simulate(n_seeds: int, effect_sd: float, n_oracles: int = 9, n_rep: int = 200,
             seed: int = 0, n_boot: int = SIM_N_BOOT) -> Dict[str, Any]:
    """Run the full verdict pipeline on `n_rep` synthetic frames; count each verdict."""
    rng = np.random.default_rng(seed)
    counts: Dict[str, int] = {}
    for r in range(n_rep):
        df = _frame(n_seeds, effect_sd, n_oracles, rng)
        bs = int(rng.integers(0, 2**31 - 1))
        c0 = P.criterion_0(df, n_boot=n_boot, seed=bs)
        cA = P.criterion_A(df, n_boot=n_boot, seed=bs)
        cC = P.criterion_C(df, c0, cA)
        # B and G2 never change the verdict name; inert stubs keep phase2_verdict's contract.
        cB = {"per_metric": {}, "degenerate_metrics": []}
        cG = {"mean": 0.0}
        v = P.phase2_verdict(c0, cA, cB, cC, cG)["verdict"]
        counts[v] = counts.get(v, 0) + 1
    power = 1.0 - counts.get("BENCHMARK-INSENSITIVE", 0) / n_rep
    return {"n_seeds": n_seeds, "effect_sd": effect_sd, "n_oracles": n_oracles,
            "n_rep": n_rep, "n_boot": n_boot, "verdict_counts": counts, "power": power}


def _cell(args):
    return simulate(*args)


def run_grid(candidates, effects, n_rep, n_oracles=9, seed=0, n_boot=SIM_N_BOOT,
             workers=1) -> Dict[int, Dict[float, Dict[str, Any]]]:
    """simulate() over candidates x effects. Cells are independent, so they may run in a pool."""
    keys = [(n, e) for n in candidates for e in effects]
    jobs = [(n, e, n_oracles, n_rep, seed, n_boot) for n, e in keys]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_cell, jobs))
    else:
        res = [_cell(j) for j in jobs]
    table: Dict[int, Dict[float, Dict[str, Any]]] = {n: {} for n in candidates}
    for (n, e), r in zip(keys, res):
        table[n][e] = r
    return table


def choose_seeds(candidates: Sequence[int] = (5, 10, 15), target_power: float = 0.80,
                 n_rep: int = 200, n_oracles: int = 9, seed: int = 0,
                 n_boot: int = SIM_N_BOOT, workers: int = 1) -> Dict[str, Any]:
    """Smallest candidate whose power at a 1-SD disagreement reaches `target_power`. Halts
    when no candidate reaches it even at a 2-SD disagreement. Decided once, never extended."""
    table = run_grid(candidates, EFFECTS, n_rep, n_oracles, seed, n_boot, workers)
    chosen: Optional[int] = next((n for n in candidates if table[n][1.0]["power"] >= target_power), None)
    reachable = any(table[n][2.0]["power"] >= target_power for n in candidates)
    return {"table": table, "chosen": chosen, "reachable": reachable, "halt": not reachable,
            "target_power": target_power}


def phase1_sized_effect() -> Dict[str, float]:
    """Phase 1's observed between-method spread (max - min of the backbone grand-mean
    auc_top10 over all ok runs, the recorded 0.0034) over the median pooled seed SD used by
    the simulation, i.e. the planted-effect unit. Derived, not hard-coded."""
    d = pd.read_csv(MANIFEST)
    m = d[d["status"] == "ok"].groupby("backbone")["auc_top10"].mean()
    spread = float(m.max() - m.min())
    _, sds, _, _ = phase1_oracle_params()
    sd = float(np.median(sds))
    return {"spread": spread, "median_sd": sd, "effect_sd": spread / sd}


def smallest_n(table, effect: float, target: float) -> Optional[int]:
    return next((n for n in sorted(table) if table[n][effect]["power"] >= target), None)


def detectable_floor(scan, candidates, target: float, key: str = "power") -> Dict[int, Optional[float]]:
    """Per n, the smallest scanned effect whose `key` fraction reaches `target`.
    key='power' (not BENCHMARK-INSENSITIVE) or 'dependent' (the planted verdict itself)."""
    out: Dict[int, Optional[float]] = {}
    for n in candidates:
        hit = [e for e in sorted(scan[n]) if _frac(scan[n][e], key) >= target]
        out[n] = hit[0] if hit else None
    return out


def _frac(r: Dict[str, Any], key: str) -> float:
    if key == "power":
        return r["power"]
    return r["verdict_counts"].get("METRIC-DEPENDENT", 0) / r["n_rep"]


def _dist(r) -> str:
    return " ".join(f"{k}={v / r['n_rep']:.2f}" for k, v in sorted(r["verdict_counts"].items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--floor-rep", type=int, default=100)
    ap.add_argument("--target-power", type=float, default=0.80)
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    a = ap.parse_args()
    t0 = time.time()
    cands = (5, 10, 15)
    means, sds, src, names = phase1_oracle_params()
    print(f"data source: {src}; {len(means)} oracles; median pooled SD {np.median(sds):.4f}")
    p1 = phase1_sized_effect() if src == "manifest" else None
    if p1:
        print(f"Phase-1-sized effect: spread {p1['spread']:.5f} / median pooled SD "
              f"{p1['median_sd']:.5f} = {p1['effect_sd']:.3f} SD")
    p1e = round(p1["effect_sd"], 2) if p1 else 0.4
    res = choose_seeds(cands, a.target_power, a.n_rep, workers=a.workers)
    extra = run_grid(cands, (p1e,), a.n_rep, workers=a.workers)
    for n in cands:
        res["table"][n][p1e] = extra[n][p1e]
    print(f"\nn_rep={a.n_rep} n_oracles=9 n_boot={SIM_N_BOOT} (real analysis: {P.N_BOOT})")
    print("power = P(verdict != BENCHMARK-INSENSITIVE); fractions are of replicates")
    for n in cands:
        for e in sorted(res["table"][n]):
            r = res["table"][n][e]
            tag = " (Phase-1-sized)" if e == p1e else ""
            print(f"  n_seeds={n:<3} effect={e:<5} power={r['power']:.3f}  {_dist(r)}{tag}")
    scan_effects = tuple(round(0.2 * k, 1) for k in range(1, 11))
    scan = run_grid(cands, scan_effects, a.floor_rep, workers=a.workers)
    print(f"\nfloor scan: effects {scan_effects}, n_rep={a.floor_rep}")
    for n in cands:
        print(f"  n={n:<3} " + " ".join(f"{e}:{scan[n][e]['power']:.2f}/{_frac(scan[n][e], 'dependent'):.2f}"
                                         for e in scan_effects) + "   (power/METRIC-DEPENDENT)")
    fl = detectable_floor(scan, cands, a.target_power)
    fld = detectable_floor(scan, cands, a.target_power, "dependent")
    print(f"\ndetectable-effect floor (smallest scanned SD with power >= {a.target_power}): {fl}")
    print(f"floor for the planted verdict METRIC-DEPENDENT itself (>= {a.target_power}): {fld}")
    print(f"smallest sufficient n at {p1e} SD: {smallest_n(res['table'], p1e, a.target_power)}")
    print(f"smallest sufficient n at the floor of the frozen-rule n: "
          f"{[(n, smallest_n(scan, fl[n], a.target_power)) for n in cands if fl[n] is not None]}")
    print(f"\nwall-clock {time.time() - t0:.0f}s")
    if res["halt"]:
        print(f"HALT: no candidate reaches power {a.target_power} even at 2 SD. Do not freeze.")
        return 1
    if res["chosen"] is None:
        print(f"NO SEED COUNT: reachable at 2 SD but no candidate reaches {a.target_power} at 1 SD.")
        return 2
    print(f"CHOSEN seeds = {res['chosen']} (frozen rule: 1 SD)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
