#!/usr/bin/env python
"""Build Phase 1 from src/phase1_source.py.

    python src/build_phase1.py

The percent-format .py is the single source of truth. This emits, and syntax-checks:

  Phase 1/landscape_phase1.ipynb      the notebook
  src/phase1_lib.py                   importable library (no @notebook-only cells)
  scripts/run_sweep_phase1.py         CLI sweep runner, resumable

Cells tagged `# %% @notebook-only` are drivers, smoke tests and plot calls. They go in the
notebook and are excluded from the library, so importing phase1_lib defines things without
launching a sweep or downloading an oracle.

Mirrors src/build_notebook.py deliberately -- same cell grammar, same checks -- so whoever
learned one knows the other. phase0_lib.py and phase0_source.py are never touched.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "phase1_source.py"
OUT = ROOT / "Phase 1" / "landscape_phase1.ipynb"
LIB = ROOT / "src" / "phase1_lib.py"
SWEEP = ROOT / "scripts" / "run_sweep_phase1.py"

LIB_HEADER = '''"""Phase 1 library -- GENERATED from src/phase1_source.py. Do not edit.

Regenerate with:  python src/build_phase1.py

Importing this module defines the Phase 1 pre-registration, cluster tags, diagnostics,
runner and analyses. Driver cells are excluded, so an import has no side effects beyond
what phase0_lib does on import (vendoring mol_opt and loading the prior, both cached after
Phase 0's first run).
"""
'''

SWEEP_SCRIPT = '''#!/usr/bin/env python
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
'''


def parse_percent(text):
    """Split a jupytext-percent file into (kind, source, notebook_only) cells."""
    cells, kind, buf, nb_only = [], None, [], False

    def flush():
        if kind is None:
            return
        src = "\n".join(buf).strip("\n")
        if src.strip():
            cells.append((kind, src, nb_only))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %% [markdown]"):
            flush()
            kind, buf, nb_only = "markdown", [], False
        elif stripped.startswith("# %%"):
            flush()
            kind, buf = "code", []
            nb_only = "@notebook-only" in stripped
        else:
            buf.append(line)
    flush()
    return cells


def check_cells(cells):
    """Every code cell must parse on its own before anything is written."""
    bad = 0
    for i, (kind, src, _) in enumerate(cells):
        if kind != "code":
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"  cell {i} does not parse at line {e.lineno}: {e.msg}")
            bad += 1
    return bad


def write_notebook(cells):
    nb = {"cells": [], "metadata": {
        "kernelspec": {"display_name": "lce", "language": "python", "name": "lce"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}
    for kind, src, _ in cells:
        lines = src.splitlines(keepends=True)
        if kind == "markdown":
            # strip the leading "# " comment marker from markdown cells
            lines = [ln[2:] if ln.startswith("# ") else ln[1:] if ln.startswith("#") else ln
                     for ln in lines]
            nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            nb["cells"].append({"cell_type": "code", "execution_count": None,
                                "metadata": {}, "outputs": [], "source": lines})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    return len(nb["cells"])


def write_library(cells):
    body = [src for kind, src, nb_only in cells if kind == "code" and not nb_only]
    text = LIB_HEADER + "\n\n" + "\n\n\n".join(body) + "\n"
    try:
        ast.parse(text)
    except SyntaxError as e:
        print(f"generated library does not parse at line {e.lineno}: {e.msg}")
        return 0
    LIB.write_text(text)
    return len(body)


def main():
    if not SRC.exists():
        print(f"missing source: {SRC}")
        return 1
    cells = parse_percent(SRC.read_text(encoding="utf-8"))
    print(f"parsed {len(cells)} cells from {SRC.name}")

    bad = check_cells(cells)
    if bad:
        print(f"\n{bad} cell(s) failed to parse -- nothing written")
        return 1

    n_cells = write_notebook(cells)
    n_lib = write_library(cells)
    SWEEP.parent.mkdir(parents=True, exist_ok=True)
    SWEEP.write_text(SWEEP_SCRIPT)

    print(f"  {OUT.relative_to(ROOT)}  {n_cells} cells")
    print(f"  {LIB.relative_to(ROOT)}  {n_lib} code cells")
    print(f"  {SWEEP.relative_to(ROOT)}")
    print("\nNote: the .ipynb is a build product. Edit src/phase1_source.py, never the "
          "notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
