#!/usr/bin/env python
"""Build Phase 3 from src/phase3_source.py.

    python src/build_phase3.py

The percent-format .py is the single source of truth. This emits, and syntax-checks:

  Phase 3/landscape_phase3.ipynb      the notebook
  src/phase3_lib.py                   importable library (no @notebook-only cells)
  scripts/run_sweep_phase3.py         CLI sweep runner, resumable

Cells tagged `# %% @notebook-only` are drivers, smoke tests and plot calls. They go in the
notebook and are excluded from the library, so importing phase3_lib defines things without
launching a sweep or downloading an oracle.

Mirrors src/build_phase2.py deliberately -- same cell grammar, same checks. phase0_lib.py,
phase0_source.py, phase1_*.py and phase2_*.py are never touched.

One extra check over Phase 2's builder: every `import ... as <alias>` the generated library
references must be present IN the library. Phase 1 shipped a library that referenced `P0` six
times with no import, because the import sat in an @notebook-only cell (Phase 2 design L9).
"""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "phase3_source.py"
OUT = ROOT / "Phase 3" / "landscape_phase3.ipynb"
LIB = ROOT / "src" / "phase3_lib.py"
SWEEP = ROOT / "scripts" / "run_sweep_phase3.py"

LIB_HEADER = '''"""Phase 3 library -- GENERATED from src/phase3_source.py. Do not edit.

Regenerate with:  python src/build_phase3.py

Importing this module defines the Phase 3 pre-registration, the three arms, the gates, the
runner and the metric computation. Driver cells are excluded, so an import launches nothing
and downloads nothing: phase0_lib is loaded lazily by p0() on first use, so the library
imports on a machine without torch or TDC.
"""
'''

SWEEP_SCRIPT = '''#!/usr/bin/env python
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


def check_library_imports(text):
    """Every alias the library USES must be imported IN the library.

    Phase 1's generated library referenced `P0` six times while its `import phase0_lib as P0`
    sat in an @notebook-only cell, so the library was not importable. Catch that here rather
    than at someone else's import time.
    """
    tree = ast.parse(text)
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                bound.add(al.asname or al.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)          # assigned in the library, so not a missing import
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    # Only attribute access on a bare Name counts as "using an alias".
    used = {n.value.id for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    return sorted(a for a in used - bound if re.fullmatch(r"P\d?[A-Z][A-Za-z]{0,3}|P\d", a))


def write_notebook(cells):
    nb = {"cells": [], "metadata": {
        "kernelspec": {"display_name": "lce", "language": "python", "name": "lce"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}
    for kind, src, _ in cells:
        lines = src.splitlines(keepends=True)
        if kind == "markdown":
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
    missing = check_library_imports(text)
    if missing:
        print(f"generated library uses {missing} with no import in the library -- "
              f"move the import out of its @notebook-only cell")
        return 0
    LIB.write_text(text)
    return len(body)


def _rel(p):
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


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

    n_lib = write_library(cells)
    if not n_lib:
        print("\nthe library was not written -- build FAILED (previous library left in place)")
        return 1
    n_cells = write_notebook(cells)
    SWEEP.parent.mkdir(parents=True, exist_ok=True)
    SWEEP.write_text(SWEEP_SCRIPT)

    print(f"  {_rel(OUT)}  {n_cells} cells")
    print(f"  {_rel(LIB)}  {n_lib} code cells")
    print(f"  {_rel(SWEEP)}")
    print("\nNote: the .ipynb is a build product. Edit src/phase3_source.py, never the "
          "notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
