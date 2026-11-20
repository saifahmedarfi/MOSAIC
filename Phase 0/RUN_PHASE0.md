
Audience: an autonomous coding agent (Claude Code) running Phase 0 of the research plan held
outside this repository.

Authoritative background: [README.md](README.md) and [GUIDE.md](GUIDE.md). This file is the
*execution* contract — what to run, in what order, what to never do, and when to stop and ask.
Where this file and GUIDE.md disagree about intent, GUIDE.md wins; where they disagree about a
command, trust this file and flag the drift.

---

## 0. Invariants — read before touching anything

1. **`src/phase0_source.py` is the only editable source.** `Phase 0/landscape_phase0.ipynb`,
   `src/phase0_lib.py`, `environment.yml`, `scripts/run_sweep.py`, `scripts/slurm_phase0.sh`
   and `.gitignore` are build products of `python src/build_notebook.py`. Never edit them, and
   never edit the `.ipynb` in Jupyter — the next build silently overwrites it.
2. **`preregistration.json` is frozen once written.** It must be committed *before* any sweep
   run exists. After that, do not edit, regenerate, delete or `--force` around it. If
   `PREREG_STATUS == "mismatch"`, Part 10 returns `verdict: REFUSED` — that is the mechanism
   working, not a bug to fix.
3. **Never change a criterion, α, feature list, oracle set, seed set or budget to obtain a
   different verdict.** If a change is genuinely required, stop and ask the user; if they
   approve, it is a documented post-hoc deviation, not a re-registration.
4. **The verdict is binding.** `PIVOT` means stop and write up the null. Do not propose
   "one more analysis" to rescue it, and do not start Phase 1/2/3 work on anything but `GO`.
5. **No fabricated numbers.** Every number reported to the user must come from a command whose
   output is in the transcript. If a stage was skipped or failed, say so with the output.
6. Writes land under `results/`, `figures/`, `vendor/`, `data/`, `logs/` — all gitignored
   except `preregistration.json` and `results/phase0_verdict.json`.

---

## 1. Preflight (facts verified on this machine 2026-11-19 — re-verify, do not assume)

```bash
cd /home/saifarfi/Volumetric/MOSAIC
git status --porcelain && git log --oneline -3
nproc
command -v conda git sbatch jupyter
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "NO GPU"
```

Known state of this box:

| Fact | Value | Consequence |
|---|---|---|
| Cores | 16 | `--workers 8`–`12`; throughput is CPU-bound (oracles + Graph GA) |
| GPU | **NVIDIA driver not responding**, `torch.cuda.is_available() == False` | LM backbone runs on CPU → the 144 LM runs are *much* slower than the README's "~1h at 8 workers". Prefer the cluster. |
| `sbatch` | present (`/usr/bin/sbatch`) | `scripts/slurm_phase0.sh` is the intended path for the full sweep |
| `conda` envs | `base`, `research-base` — **no `lce`** | see §2 |
| `jupyter` on PATH | missing (present inside `research-base`) | always invoke as `<env>/bin/python -m jupyter ...` |

Stop and ask the user if: the working tree is dirty, the GPU is still unavailable *and* they
want the sweep run locally anyway, or `preregistration.json` already exists while
`results/runs/` is non-empty (ordering may already be compromised).

---

## 2. Environment

Two viable paths. Ask the user which, unless they already said.

**A — the documented path (correct for the cluster):**
```bash
conda env create -f environment.yml && conda activate lce
```
`environment.yml` pins `pytorch-cuda=11.8`; its own comment says *match your cluster's driver*.
On a box with no working driver this install is wrong-but-harmless (CPU torch), so confirm the
pin against the target machine before creating the env there.

**B — reuse the existing env (fastest here):** `research-base` already has numpy, scipy, pandas,
scikit-learn, matplotlib, joblib, tqdm, rdkit 2026.09.1, torch 2.14.1 (CPU), nbconvert,
ipykernel. Only `PyTDC` is missing:
```bash
/home/saifarfi/miniconda3/envs/research-base/bin/python -m pip install "PyTDC>=0.4.1"
```

Export the interpreter once and use it everywhere below:
```bash
PY=/home/saifarfi/miniconda3/envs/research-base/bin/python   # or .../envs/lce/bin/python
```

**Network.** Several stages need egress; the sandbox blocks it by default. Declare
`allowed_domains` on those Bash calls: `github.com`, `codeload.github.com`,
`raw.githubusercontent.com`, `objects.githubusercontent.com` (mol_opt clone),
`dataverse.harvard.edu`, `*.harvard.edu` (TDC oracle data), plus `pypi.org`,
`files.pythonhosted.org`, `conda.anaconda.org`, `repo.anaconda.com` for installs. Add a host
only when a `<sandbox_violations>` block names it.

---

## 3. Stage 1 — rebuild, then verify the build is clean

```bash
$PY src/build_notebook.py
git status --porcelain
```

`git status` must come back **empty**. Non-empty means the committed build products drifted from
`src/phase0_source.py`. Do not continue: show the user the diff and ask whether to commit the
rebuild or investigate. Silently committing a rebuild hides an edit someone made to a generated
file.

## 4. Stage 2 — statistics gate

```bash
$PY tests/test_core.py        # ~90s, no torch/rdkit/GPU needed
```

56 assertions covering the AUC formula (verified to 1e-12 against mol_opt's published function),
the hierarchical bootstrap, the exact permutation Spearman, Holm correction, and the build
artifacts. **Any failure halts Phase 0.** Do not run a sweep whose statistics are unverified;
report the failing assertion and stop.

## 5. Stage 3 — notebook pass 1: vendoring, oracles, smoke tests, diagnostics, pre-registration

The notebook must run once end to end before `scripts/run_sweep.py` works — importing
`src/phase0_lib.py` calls `vendor_mol_opt()` at import time (line ~397), which needs `git` and
network, and the TDC oracles need their first download.

Run it headless; do not ask the user to click through Jupyter:

```bash
mkdir -p logs
cd /home/saifarfi/Volumetric/MOSAIC && LCE_ROOT="$PWD" $PY -m jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 \
  "Phase 0/landscape_phase0.ipynb" > logs/nbconvert_pass1.log 2>&1
```

Notes and gotchas:

- `--inplace` rewrites the `.ipynb` with outputs. That is fine (it is a build product) but it
  *will* show as a git diff — do not commit it; `git checkout -- "Phase 0/landscape_phase0.ipynb"`
  after you have read the outputs, or rebuild with `src/build_notebook.py`.
- Always set `LCE_ROOT` to the repo root. Without it, `results/` and `preregistration.json` can
  land in the wrong directory.
- The notebook's own driver toggles are deliberately off: `DO_INSTALL = False`,
  `LAUNCH_SWEEP = False`, `DRY_RUN_FIRST = True`. **Leave them.** The 2-run dry check at reduced
  budget (200 calls) is the pipeline test and it restores/removes its own result files. The full
  sweep is launched from the CLI in Stage 5, never from a notebook kernel.
- Run it as a **background** Bash call (it is long) and read `logs/nbconvert_pass1.log`.

After it finishes, confirm the artifacts exist and report what they say:

```bash
ls -la preregistration.json results/diagnostics.json vendor/checksums.json
$PY -c "import json;d=json.load(open('preregistration.json'));print(d.get('sha256') or list(d))"
grep -iE "dry check|pipeline OK|device|vram|REFUS|Traceback" logs/nbconvert_pass1.log | head -40
```

## 6. Stage 4 — HARD GATE: commit the pre-registration

Its entire value is a timestamp earlier than the results. Before any sweep run exists:

```bash
git add preregistration.json
git commit -m "Phase 0: freeze pre-registration (sha256 <first 12 chars>)"
```

Ask the user before committing (commits are theirs to authorize), but do **not** proceed to
Stage 5 until `preregistration.json` is committed. If they decline, say plainly that the sweep's
pre-registration claim is void and stop.

Append to the commit message:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

## 7. Stage 5 — the sweep (216 runs = 8 oracles × 3 backbones × 3 settings × 3 seeds, 1000 calls each)

Resumable by design: completed runs are skipped, so interruption and re-launch are safe. Never
pass `--force` to a sweep — it discards completed runs.

**On the cluster (preferred):**
```bash
sbatch scripts/slurm_phase0.sh        # 1 GPU, 16 cpus, 32G, 8h; logs/phase0_%j.out
squeue -u "$USER"
```

**Locally:** run in the background, never in the foreground (it outlives the 600s Bash cap):
```bash
cd /home/saifarfi/Volumetric/MOSAIC && LCE_ROOT="$PWD" nohup $PY scripts/run_sweep.py \
  --workers 8 > logs/sweep.log 2>&1
```

Useful narrowings (same flags, all resumable): `--backbone reinvent`, `--oracle <names>`,
`--setting <names>`, `--seed 0 1 2`, `--budget <n>`. Splitting across GPUs:
`CUDA_VISIBLE_DEVICES=0 ... --backbone reinvent &` — see the docstring in
[scripts/run_sweep.py](scripts/run_sweep.py).

Progress, without polling in a tight loop:
```bash
ls results/runs/*.json | wc -l     # target 216
tail -20 logs/sweep.log
```

Before declaring the sweep done, count failures:
```bash
$PY - <<'EOF'
import json,glob
rows=[json.load(open(p)) for p in glob.glob('results/runs/*.json')]
bad=[r for r in rows if r.get('status')!='ok']
print(len(rows),'runs,',len(bad),'failed')
for r in bad[:10]: print(' ',r.get('oracle'),r.get('backbone'),r.get('setting'),r.get('seed'),r.get('status'))
EOF
```

A verdict on an incomplete sweep is labelled provisional by the code itself. Report the
completion percentage with the verdict; do not round it up in prose.

## 8. Stage 6 — the verdict

Re-execute the notebook (same command as Stage 3) so Parts 10–12 run against the finished
results, **or** drive the library directly:

```bash
cd /home/saifarfi/Volumetric/MOSAIC && LCE_ROOT="$PWD" $PY - <<'EOF'
import sys; sys.path.insert(0,'src')
from phase0_lib import (load_results, compute_all_diagnostics, phase0_verdict,
                        plot_setting_by_cluster, plot_per_oracle_heatmap)
df   = load_results()                     # results/runs/*.json
diag = compute_all_diagnostics()          # cached in results/diagnostics.json
out  = phase0_verdict(df, diag)           # writes results/phase0_verdict.json
plot_setting_by_cluster(df); plot_per_oracle_heatmap(df, backbone="reinvent")
print(out["verdict"])
EOF
```

`phase0_verdict` prints Criterion A (hierarchical bootstrap over the cluster rank-flip) and
Criterion B (exact permutation Spearman, Holm-corrected, α=0.10, declared a **screen**), then
`GO` / `PARTIAL` / `PIVOT`, and writes `results/phase0_verdict.json`.

Possible returns that are **not** verdicts: `REFUSED` (prereg hash mismatch — restore the frozen
rule, do not re-register) and `NO_DATA` (no runs on disk).

Then commit the result and report:

```bash
git add results/phase0_verdict.json && git commit -m "Phase 0: <VERDICT>"
```

## 9. Reporting the verdict to the user

State the verdict, both criteria's statistics, the sweep completion %, and the figures written
to `figures/`. Then the mandated action from GUIDE.md §9 — unchanged, including when unwelcome:

| Verdict | Action |
|---|---|
| **GO** | Proceed to Phase 1 as written. |
| **PARTIAL** | Characterisation (C1) only; redesign the feature set before the controller; downgrade venue expectation. |
| **PIVOT** | **Stop.** The premise is dead. Switch to the proxy-gap direction and write up the null — it is informative, because the field has assumed this dependence since PMO without testing it. |

Two caveats that must accompany any number reported:

1. **AUC-top-10 is not a mean reward.** mol_opt's integral starts from `prev = 0`, so a
   constant-0.42 run scores 0.4179. The formula is reproduced exactly on purpose; absolute AUC
   values must never be described as mean rewards.
2. **Criterion B at n=8 is a screen, not a result.** Phase 1 (23 oracles, α=0.05) is the
   confirmatory test.

---

## 10. Out of scope for this runbook

Do not start, and do not let scope creep into Phase 0: the controller, Genetic-GFN, Mol GA, the
full 23-oracle sweep, the docking oracle, transfer experiments (Phases 1–3 — all gated on `GO`).
Three open items are *not* coding tasks and must not be improvised:
the step-0 novelty check (a literature search), adding the online bandit baseline to a later
phase, and settling the submission timeline. Raise them with the user; do not decide them.
