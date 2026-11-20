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
  * n_boot: the real analysis uses N_BOOT = 10,000. The simulation runs the whole pipeline
    n_rep times, so it uses `n_boot` = 2,000 by default; whether a replication fraction
    clears 0.95 converges well before 10,000 draws, so the power estimate is not materially
    affected, and the speed is what makes an interactive gate and a test suite possible.
"""
import argparse
import os
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
        cB = P.criterion_B(df, n_boot=n_boot, seed=bs)
        cC = P.criterion_C(df, c0, cA)
        cG = P.criterion_G2(df, n_boot=n_boot, seed=bs)
        v = P.phase2_verdict(c0, cA, cB, cC, cG)["verdict"]
        counts[v] = counts.get(v, 0) + 1
    power = 1.0 - counts.get("BENCHMARK-INSENSITIVE", 0) / n_rep
    return {"n_seeds": n_seeds, "effect_sd": effect_sd, "n_oracles": n_oracles,
            "n_rep": n_rep, "n_boot": n_boot, "verdict_counts": counts, "power": power}


def choose_seeds(candidates: Sequence[int] = (5, 10, 15), target_power: float = 0.80,
                 n_rep: int = 200, n_oracles: int = 9, seed: int = 0,
                 n_boot: int = SIM_N_BOOT) -> Dict[str, Any]:
    """Smallest candidate whose power at a 1-SD disagreement reaches `target_power`. Halts
    when no candidate reaches it even at a 2-SD disagreement. Decided once, never extended."""
    table = {n: {e: simulate(n, e, n_oracles, n_rep, seed, n_boot) for e in EFFECTS}
             for n in candidates}
    chosen: Optional[int] = next((n for n in candidates if table[n][1.0]["power"] >= target_power), None)
    reachable = any(table[n][2.0]["power"] >= target_power for n in candidates)
    return {"table": table, "chosen": chosen, "reachable": reachable, "halt": not reachable,
            "target_power": target_power}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--target-power", type=float, default=0.80)
    a = ap.parse_args()
    t0 = time.time()
    means, sds, src, names = phase1_oracle_params()
    print(f"data source: {src}; {len(means)} oracles; median pooled SD {np.median(sds):.4f}")
    res = choose_seeds(target_power=a.target_power, n_rep=a.n_rep)
    print(f"\nn_rep={a.n_rep} n_oracles=9 n_boot={SIM_N_BOOT} (real analysis: {P.N_BOOT})")
    print("power = P(verdict != BENCHMARK-INSENSITIVE)")
    for n, row in res["table"].items():
        for e, r in row.items():
            print(f"  n_seeds={n:<3} effect={e:<4} power={r['power']:.3f}  {r['verdict_counts']}")
    print(f"\nwall-clock {time.time() - t0:.0f}s")
    if res["halt"]:
        print(f"HALT: no candidate reaches power {a.target_power} even at 2 SD. Do not freeze.")
        return 1
    if res["chosen"] is None:
        print(f"NO SEED COUNT: reachable at 2 SD but no candidate reaches {a.target_power} at 1 SD.")
        return 2
    print(f"CHOSEN seeds = {res['chosen']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
