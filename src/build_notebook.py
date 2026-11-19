#!/usr/bin/env python
"""Build everything from src/phase0_source.py.

    python src/build_notebook.py

The percent-format .py is the single source of truth. This emits, and syntax-checks:

  landscape_phase0.ipynb   the notebook
  src/phase0_lib.py                                importable library (no demo cells)
  environment.yml, scripts/run_sweep.py,
  scripts/slurm_phase0.sh, .gitignore              extracted from Part 11's constants

Cells tagged `# %% @notebook-only` are demonstrations, smoke tests and plot calls.
They go in the notebook and are excluded from the library, so importing phase0_lib
defines things without running a sweep or downloading an oracle.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "phase0_source.py"
OUT = ROOT / "landscape_phase0.ipynb"
LIB = ROOT / "src" / "phase0_lib.py"


def parse_percent(text: str):
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


LIB_HEADER = '''"""Phase 0 library -- GENERATED from src/phase0_source.py. Do not edit.

Regenerate with:  python src/build_notebook.py

Importing this module defines the oracles, backbones, diagnostics and analysis
used by the notebook and by scripts/run_sweep.py. Demonstration and driver cells
are excluded, so an import has no side effects beyond loading the vendored prior
and the ZINC pool (both cached on disk after the first notebook run).
"""
'''


def write_library(cells) -> int:
    """Emit the importable library: every code cell except the notebook-only ones."""
    body = [src for kind, src, nb_only in cells if kind == "code" and not nb_only]
    text = LIB_HEADER + "\n\n" + "\n\n\n".join(body) + "\n"
    try:
        ast.parse(text)
    except SyntaxError as e:
        print(f"generated library does not parse at line {e.lineno}: {e.msg}")
        return 0
    LIB.write_text(text)
    return len(body)


def write_assets(cells) -> list:
    """Extract Part 11's file constants and write them as real files."""
    targets = {
        "ENVIRONMENT_YML": ROOT / "environment.yml",
        "RUN_SWEEP_PY": ROOT / "scripts" / "run_sweep.py",
        "SLURM_SH": ROOT / "scripts" / "slurm_phase0.sh",
        "GITIGNORE": ROOT / ".gitignore",
    }
    found, written = {}, []
    for kind, src, _ in cells:
        if kind != "code":
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            name = getattr(node.targets[0], "id", None)
            if name in targets and isinstance(node.value, ast.Constant):
                found[name] = node.value.value

    for name, path in targets.items():
        if name not in found:
            print(f"  WARNING: {name} not found in source; {path.name} not written")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(found[name])
        if path.suffix == ".py":
            try:
                ast.parse(found[name])
            except SyntaxError as e:
                print(f"  WARNING: {path.name} does not parse: {e.msg}")
        if path.suffix in (".py", ".sh"):
            path.chmod(0o755)
        written.append(path)
    return written


def demark(src: str) -> str:
    """Strip the leading '# ' from markdown cell lines."""
    out = []
    for line in src.split("\n"):
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip("\n")


def main() -> int:
    if not SRC.exists():
        print(f"missing source: {SRC}")
        return 1

    cells = parse_percent(SRC.read_text())

    # --- syntax check every code cell independently -------------------------
    errors = []
    for i, (kind, src, _nb) in enumerate(cells):
        if kind != "code":
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            errors.append((i, e.lineno, e.msg, src.split("\n")[max(0, (e.lineno or 1) - 1)]))

    if errors:
        print(f"SYNTAX ERRORS in {len(errors)} cell(s):\n")
        for i, lineno, msg, line in errors:
            print(f"  cell {i}  line {lineno}: {msg}")
            print(f"    {line!r}\n")
        return 1

    n_code = sum(1 for k, _s, _n in cells if k == "code")
    n_md = len(cells) - n_code
    n_nb = sum(1 for k, _s, nb in cells if k == "code" and nb)
    print(f"parsed {len(cells)} cells ({n_code} code, {n_md} markdown) -- all code parses")
    print(f"  {n_nb} code cells are notebook-only (demos, smoke tests, plot calls)")

    nb = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": ({"tags": ["notebook-only"]} if nb_only else {}),
                "source": (demark(src) if kind == "markdown" else src).split("\n"),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, src, nb_only in cells
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (lce)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "codemirror_mode": {"name": "ipython", "version": 3},
            },
            "lce": {
                "project": "Landscape-Conditional Exploration",
                "phase": "0 (pre-registered falsification)",
                "generated_from": "src/phase0_source.py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    # nbformat stores source as a list of lines WITH trailing newlines except the last
    for cell in nb["cells"]:
        lines = cell["source"]
        cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []

    OUT.write_text(json.dumps(nb, indent=1))
    print(f"\nwrote {OUT.name}  ({OUT.stat().st_size/1024:.0f} KB)")

    n_lib = write_library(cells)
    if n_lib:
        print(f"wrote {LIB.relative_to(ROOT)}  ({n_lib} cells, "
              f"{LIB.stat().st_size/1024:.0f} KB)")
    else:
        print("library NOT written -- see error above")
        return 1

    for path in write_assets(cells):
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
