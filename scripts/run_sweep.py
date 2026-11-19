#!/usr/bin/env python
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
