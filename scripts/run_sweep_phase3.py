#!/usr/bin/env python
"""Phase 3 sweep runner (Stage B).

    python scripts/run_sweep_phase3.py --workers 12
    python scripts/run_sweep_phase3.py --arm treatment --seed 0 1 2 --workers 6

Resumable: completed runs are skipped, so re-running after an interruption is safe.
Never pass --force to a full sweep; it discards completed runs.

Refuses to start unless the pre-registration is FROZEN -- delta_p and seeds are null until
Stage A fills them, and a sweep against an unfrozen rule would be a Stage B record produced
before the rule existed. Also refuses if the reproduction gate has not cleared the floor.

Threads, not processes. Phase 3 runs graph_ga only, which is CPU-bound Python + RDKit and
needs no GPU, so set --workers from CORE count and expect sub-linear scaling under the GIL.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("LCE_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from phase3_lib import (PREREG3, build_sweep3, run_sweep3,  # noqa: E402
                        surviving_oracles3, sweep_failures3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--budget", type=int, default=PREREG3["budget"])
    ap.add_argument("--oracle", nargs="*", default=None)
    ap.add_argument("--arm", nargs="*", default=None)
    ap.add_argument("--seed", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    jobs = build_sweep3(a.oracle or surviving_oracles3(), a.arm, a.seed)
    print(f"{len(jobs)} jobs, budget {a.budget}")
    run_sweep3(jobs, budget=a.budget, n_workers=a.workers, force=a.force)
    failed = sweep_failures3(jobs)
    if failed:
        print(f"FAILED: {len(failed)} of {len(jobs)} job(s) have no complete ok record, "
              f"e.g. {failed[:3]}")
        sys.exit(1)
    print(f"all {len(jobs)} jobs have a complete ok record")


if __name__ == "__main__":
    main()
