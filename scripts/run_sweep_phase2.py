#!/usr/bin/env python
"""Phase 2 sweep runner.

    python scripts/run_sweep_phase2.py --workers 12
    python scripts/run_sweep_phase2.py --backbone reinvent --seed 0 1 2 --workers 6

Resumable: completed runs are skipped, so re-running after an interruption is safe.
Never pass --force to a full sweep; it discards completed runs.

Refuses to start if the pre-registration hash does not match, or if the reproduction gate
(Phase 2/phase2_reproduction_gate.json, written by the notebook's Part 3) has not cleared at
least 5 oracles. The default oracle list is the gate's survivors.

Threads, not processes: all workers share one CUDA context. Set --workers from CORE count.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("LCE_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from phase2_lib import (PREREG2, build_sweep2, load_results2,  # noqa: E402
                        run_sweep2, surviving_oracles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--budget", type=int, default=PREREG2["budget"])
    ap.add_argument("--oracle", nargs="*", default=None)
    ap.add_argument("--backbone", nargs="*", default=None)
    ap.add_argument("--seed", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    jobs = build_sweep2(a.oracle or surviving_oracles(), a.backbone, a.seed)
    print(f"{len(jobs)} jobs, budget {a.budget}")
    run_sweep2(jobs, budget=a.budget, n_workers=a.workers, force=a.force)
    print(f"{len(load_results2(with_metrics=False))} records on disk")


if __name__ == "__main__":
    main()
