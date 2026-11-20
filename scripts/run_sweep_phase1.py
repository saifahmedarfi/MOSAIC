#!/usr/bin/env python
"""Phase 1 sweep runner.

    python scripts/run_sweep_phase1.py --workers 12
    python scripts/run_sweep_phase1.py --backbone reinvent --workers 6

Resumable: completed runs are skipped, so re-running after an interruption is safe.
Never pass --force to a full sweep; it discards completed runs.

VRAM note for a 12 GB card: this uses a THREAD pool, so all workers share one CUDA
context (~500 MB paid once, not per worker). Phase 0 measured ~4.9 GB at 12 workers on a
4070 Super. Set --workers from CORE count, not VRAM -- the oracles are CPU-bound.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("LCE_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from phase1_lib import (CONFIRMATION, PREREG1, build_sweep1,  # noqa: E402
                        load_results1, run_sweep1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--budget", type=int, default=PREREG1["budget"])
    ap.add_argument("--oracle", nargs="*", default=None)
    ap.add_argument("--backbone", nargs="*", default=None)
    ap.add_argument("--setting", nargs="*", default=None)
    ap.add_argument("--seed", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    jobs = build_sweep1(a.oracle or CONFIRMATION, a.backbone, a.setting, a.seed)
    print(f"{len(jobs)} jobs, budget {a.budget}")
    run_sweep1(jobs, budget=a.budget, n_workers=a.workers, force=a.force)
    print(f"{len(load_results1())} records on disk")


if __name__ == "__main__":
    main()
