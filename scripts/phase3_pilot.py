#!/usr/bin/env python
"""Phase 3 Stage A: the pilot that sets delta_P, eps_Q and the Stage B seed count.

    python scripts/phase3_pilot.py            # runs the 45-run pilot, writes Phase 3/phase3_pilot.json

THIS IS A THRESHOLD-SETTING DEVICE, NOT AN ESTIMATE OF THE EFFECT. It is exploratory by
construction and reaches no verdict (spec 4.4): every output carries `exploratory: True`.
The pilot has 3 oracles at 5 seeds, and the shared-oracle hierarchical bootstrap resamples
ORACLES as the cluster unit, so with 3 clusters the interval on any effect is very wide. The
number it measures is used only to pick a margin that Stage B (9 oracles, 20-30 seeds) then
tests with fresh data. For that reason `resolvable` is deliberately conservative:

  * the planning effect is the midpoint of the point estimate and its bootstrap bound on the
    unfavourable side (lower for the hypervolume gain, upper for the AUC loss), never the raw
    point estimate;
  * delta_P = 0.5 x that planning gain, and the power scan plants the planning gain, so P's
    power is never computed on the raw point estimate; eps_Q = 1.5 x the point AUC loss
    (planted as that same loss), so a loss the pilot did not see (point <= 0) leaves eps_Q
    under the floor and the phase unresolvable;
  * a margin below the resolution floor RESOLUTION_FLOOR makes the phase unresolvable.

The floor (spec 6.6, ruling 4): the bootstrap SE of an arm difference is about
sqrt(2 / (n_seeds * n_oracles)) = 0.086 at 30 seeds and 9 oracles, so no margin below about
0.14 z (1.645 SE) is testable at 95%. A margin below it is not a threshold, it is noise, and
this module returns `resolvable=False` with `delta_p=None` and `eps_q=None` rather than
invent one. Exit status is 1 on unresolvable, 0 otherwise.

Seed count (spec 4.4): the smallest n in [20, 30] such that ALL of
  (a) Criterion P passes in >= TARGET_POWER of simulated Stage B frames at the planting,
  (b) Criterion Q passes in >= TARGET_POWER of them, and
  (c) Criterion F (anchor == control exactly) passes in >= F_TARGET of them.
The floor is 20 because F cannot reliably pass below it (half-width 0.249 at 15 seeds and
0.208 at 20 against the 0.25 bound). If none of [20, 30] satisfies all three the phase is
unresolvable. The seed count is decided once; there is no result-triggered extension.

Ordering contract (spec 4.4): this runs and its JSON is COMMITTED before the pre-registration
is written; the pre-registration is frozen and committed before any Stage B record exists.

`run_ga_arm` is imported lazily inside `run_pilot`: it transitively needs `phase0_lib`, which
downloads a dataset at import. `pilot_jobs` and `choose_thresholds` need none of that.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import phase3_stats as P3St  # noqa: E402

PILOT_ORACLES = ("median1", "fexofenadine_mpo", "osimertinib_mpo")   # 2, 3, 4 components
PILOT_SEEDS = (0, 1, 2, 3, 4)
SEED_MIN, SEED_MAX = 20, 30
RESOLUTION_FLOOR = 0.14        # ~1.645 x sqrt(2 / (30 * 9)); ruling 4
DELTA_P_FRACTION = 0.5         # delta_P = 0.5 x conservative hypervolume gain
EPS_Q_FACTOR = 1.5             # eps_Q   = 1.5 x conservative AUC loss
TARGET_POWER = 0.80
F_TARGET = 0.90
SIM_N_BOOT = 1000
SIM_N_REP = 40
OUT_PATH = ROOT / "Phase 3" / "phase3_pilot.json"


def pilot_jobs() -> List[Dict[str, Any]]:
    """The 45 pilot runs: 3 oracles x 3 arms x 5 seeds, in a fixed order."""
    return [{"oracle": o, "arm": a, "seed": s}
            for o in PILOT_ORACLES for a in P3St.ARMS for s in PILOT_SEEDS]


# ---------------------------------------------------------------- the pilot's observed effects
def _conservative(point: float, lo: float, hi: float, side: str) -> float:
    """Midpoint of the point estimate and its bound on the unfavourable side."""
    if not np.isfinite(point):
        return float("nan")
    bound = lo if side == "lo" else hi
    return float((point + bound) / 2.0) if np.isfinite(bound) else float("nan")


def _oracle_scatter(df, metric: str, a: str, b: str) -> float:
    """SD across oracles of the per-oracle z-difference a - b (the pilot's measured effect
    heterogeneity; planted in the simulation so power is not computed on a homogeneous effect)."""
    d = df[df["status"] == "ok"] if "status" in df.columns else df
    diffs = []
    for _, g in d.groupby("oracle"):
        cells = g.groupby("arm")[metric]
        sd = np.sqrt(cells.var(ddof=1).mean())
        if cells.ngroups < 2 or not np.isfinite(sd) or sd <= 0 or a not in cells.groups or b not in cells.groups:
            continue
        diffs.append((cells.mean()[a] - cells.mean()[b]) / sd)
    return float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else 0.0


def _sim_frame(n_seeds: int, n_oracles: int, gain: float, loss: float, tau_p: float,
               tau_q: float, rng) -> pd.DataFrame:
    """A synthetic Stage B frame in z units (pooled seed SD = 1): treatment hypervolume is
    `gain` above control, treatment AUC is `loss` below control, anchor == control, with
    per-oracle effect scatter tau. front_size is mean-zero noise."""
    rows = []
    for o in range(n_oracles):
        eff_p = gain + tau_p * rng.standard_normal()
        eff_q = loss + tau_q * rng.standard_normal()
        for arm in P3St.ARMS:
            shift_hv = eff_p if arm == "treatment" else 0.0
            shift_auc = -eff_q if arm == "treatment" else 0.0
            rows.append(pd.DataFrame({
                "oracle": f"o{o}", "arm": arm, "setting": "explore_mid",
                "seed": np.arange(n_seeds), "status": "ok",
                "auc_top10": shift_auc + rng.standard_normal(n_seeds),
                "hypervolume": shift_hv + rng.standard_normal(n_seeds),
                "front_size": rng.standard_normal(n_seeds)}))
    return pd.concat(rows, ignore_index=True)


def _scan(candidates: Sequence[int], gain, loss, tau_p, tau_q, delta_p, eps_q, f_bound,
          n_rep, n_boot, n_oracles, seed) -> List[Dict[str, Any]]:
    out = []
    for n in candidates:
        rng = np.random.default_rng([seed, n])
        p = q = f = 0
        for r in range(n_rep):
            fr = _sim_frame(n, n_oracles, gain, loss, tau_p, tau_q, rng)
            bs = int(rng.integers(0, 2**31 - 1))
            p += P3St.criterion_P(fr, delta_p, n_boot=n_boot, seed=bs)["passes"]
            q += P3St.criterion_Q(fr, eps_q, n_boot=n_boot, seed=bs)["passes"]
            f += P3St.criterion_F(fr, bound=f_bound, n_boot=n_boot, seed=bs)["passes"]
        out.append({"n_seeds": int(n), "power_P": p / n_rep, "power_Q": q / n_rep,
                    "pass_F": f / n_rep, "n_rep": n_rep})
    return out


def choose_thresholds(df, f_bound: float = P3St.F_BOUND, n_rep: int = SIM_N_REP,
                      n_boot: int = SIM_N_BOOT, n_oracles: int = 9, seed: int = 0,
                      pilot_n_boot: int = 2000) -> Dict[str, Any]:
    """Pilot frame -> `{delta_p, eps_q, seeds, seed_ruling, power_scan, resolvable, exploratory}`.

    Exploratory: reaches no verdict and is NOT an estimate of the effect (module docstring).
    """
    gain_r = P3St.arm_difference(df, "hypervolume", "treatment", "control",
                                 n_boot=pilot_n_boot, seed=seed)
    loss_r = P3St.arm_difference(df, "auc_top10", "control", "treatment",
                                 n_boot=pilot_n_boot, seed=seed)
    gain = _conservative(gain_r["diff"], gain_r["lo"], gain_r["hi"], "lo")
    loss = _conservative(loss_r["diff"], loss_r["lo"], loss_r["hi"], "hi")
    res: Dict[str, Any] = {
        "exploratory": True, "resolvable": False, "delta_p": None, "eps_q": None,
        "seeds": None, "seed_ruling": "", "power_scan": [],
        "observed": {"hv_gain": gain_r["diff"], "hv_gain_ci": [gain_r["lo"], gain_r["hi"]],
                     "auc_loss": loss_r["diff"], "auc_loss_ci": [loss_r["lo"], loss_r["hi"]],
                     "hv_gain_conservative": gain, "auc_loss_conservative": loss,
                     "n_oracles": min(gain_r["n_oracles"], loss_r["n_oracles"])},
        "resolution_floor": RESOLUTION_FLOOR, "f_bound": float(f_bound),
        "target_power": TARGET_POWER, "f_target": F_TARGET,
        "note": ("exploratory threshold-setting device; reaches no verdict; with 3 oracles the "
                 "effect is not estimated, only used to set margins")}
    if not (np.isfinite(gain) and np.isfinite(loss)):
        res["seed_ruling"] = "UNRESOLVABLE: the pilot effect could not be measured"
        return res
    # P: margin and planted gain both use the conservative (CI-shrunk) gain, never the raw point.
    # Q: the margin is built from the point loss and the same point loss is planted, so the
    # planting sits 1.5x inside its own margin; planting the CI-inflated loss would put the
    # truth OUTSIDE the margin built from it and make Q's power 0 by construction.
    gain = min(gain, gain_r["diff"])
    loss = loss_r["diff"]
    delta_p = DELTA_P_FRACTION * gain
    eps_q = EPS_Q_FACTOR * loss
    if delta_p < RESOLUTION_FLOOR or eps_q < RESOLUTION_FLOOR:
        res["seed_ruling"] = (f"UNRESOLVABLE: candidate margins delta_P={delta_p:.3f}, eps_Q={eps_q:.3f} "
                              f"fall below the resolution floor {RESOLUTION_FLOOR}; no margin is testable at 95%")
        return res
    scan = _scan(range(SEED_MIN, SEED_MAX + 1), gain, loss,
                 _oracle_scatter(df, "hypervolume", "treatment", "control"),
                 _oracle_scatter(df, "auc_top10", "control", "treatment"),
                 delta_p, eps_q, f_bound, n_rep, n_boot, n_oracles, seed)
    res["power_scan"] = scan
    ok = [s for s in scan if s["power_P"] >= TARGET_POWER and s["power_Q"] >= TARGET_POWER
          and s["pass_F"] >= F_TARGET]
    if not ok:
        res["seed_ruling"] = (f"UNRESOLVABLE: no n in [{SEED_MIN}, {SEED_MAX}] gives power >= {TARGET_POWER} on P and Q "
                              f"and F passing >= {F_TARGET} within +/-{f_bound}")
        return res
    res.update({"resolvable": True, "delta_p": float(delta_p), "eps_q": float(eps_q),
                "seeds": ok[0]["n_seeds"],
                "seed_ruling": (f"smallest n in [{SEED_MIN}, {SEED_MAX}] with power(P) >= {TARGET_POWER}, "
                                f"power(Q) >= {TARGET_POWER} and F pass rate >= {F_TARGET}: {ok[0]['n_seeds']}")})
    return res


# ------------------------------------------------------------------------ the real pilot run
def run_pilot(workers: int = 1) -> pd.DataFrame:
    """Run the 45 jobs. `run_ga_arm` / `phase0_lib` are imported HERE, never at module level,
    so the rest of this module stays importable where phase0_lib cannot load."""
    from phase3_pareto import run_ga_arm            # noqa: PLC0415  (lazy by design)
    import phase0_lib as L                          # noqa: PLC0415
    rows = []
    for j in pilot_jobs():
        row = {"oracle": j["oracle"], "arm": j["arm"], "setting": "explore_mid",
               "seed": j["seed"], "status": "ok"}
        try:
            if j["arm"] == "anchor":
                r = L.run_ga_optimizer(L.make_oracle(j["oracle"]), L.GAConfig(), seed=j["seed"])
            else:
                sel = "scalar" if j["arm"] == "control" else "pareto"
                r = run_ga_arm(L.make_oracle(j["oracle"]), L.GAConfig(), j["oracle"], sel, seed=j["seed"])
            for m in P3St.METRICS:
                row[m] = float(r[m])
        except Exception as e:  # a failed run is recorded, never dropped silently
            row["status"] = f"failed: {e}"
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--csv", default=None, help="analyse an existing pilot frame instead of running")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.csv) if a.csv else run_pilot()
    out = choose_thresholds(df)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    print(json.dumps({k: out[k] for k in ("resolvable", "delta_p", "eps_q", "seeds", "seed_ruling")}, indent=1))
    print(f"wrote {a.out}")
    return 0 if out["resolvable"] else 1


if __name__ == "__main__":
    sys.exit(main())
