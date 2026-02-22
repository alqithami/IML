# IML: Institutional Monitoring & Ledger (IML) for Sequential Social Dilemmas

This repository contains an experiment pipeline and reference implementation of an **Institutional Monitoring and Ledger (IML)** wrapper for **sequential social dilemma** environments (**Harvest** and **Cleanup**) from *Sequential Social Dilemma Games* (SSD). The core idea is to separate **base game rewards** from an **auditable institutional layer** that (i) monitors norm-relevant events, (ii) logs evidence to a ledger, and (iii) applies **delayed, contestable settlement** (sanctions/remedies).

The codebase is designed to support:

- **Multi-seed MARL training** (parameter-shared PPO) under Baseline vs IML conditions.
- **Auditable institutional traces** (violations, detections, false positives, review overturns, net sanctions).
- **Aggregation + paper-ready plotting** from CSV logs.
- **Evaluation robustness checks** (e.g., evaluation seed sweeps) and **evaluation-only sensitivity sweeps** over institutional parameters.

---

## Repository layout

- `iml_ssd/` — core library code (env wrapper, PPO training, evaluation, analysis).
- `configs/` — YAML configs for Baseline/IML in Harvest/Cleanup.
- `scripts/` — convenience scripts (SSD install without Ray/RLlib, multi-seed sweeps).
- `runs/` — raw run outputs (per-seed logs/checkpoints). **Large**.
- `results/` — aggregated CSV summaries (e.g., `summary.csv`, `learning_curves.csv`, `eval_seed_sweep*.csv`).
- `figures/` — generated figures (created by the plotting scripts; not always tracked).
- `robustness/` — robustness/sensitivity outputs (CSV + figures).

---

## Quickstart

### 0) Create a clean environment

**Option A: Conda (recommended)**

```bash
conda create -n imlssd python=3.9 -y
conda activate imlssd
python -m pip install --upgrade pip setuptools wheel
```

**Option B: Use `environment.yml`**

```bash
conda env create -f environment.yml
conda activate imlssd
```

### 1) Install SSD (without Ray/RLlib)

From the repo root:

```bash
bash scripts/install_ssd_no_ray.sh
```

This clones SSD into `sequential_social_dilemma_games/`, patches optional RLlib imports, and installs `social-dilemmas` in editable mode.

### 2) Install this package

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 3) Sanity check (smoke test)

```bash
python -m iml_ssd.tools.smoke_test --env cleanup --num_agents 5 --steps 50
```

If this fails with `No module named 'cv2'`, make sure `opencv-python` is installed in your active env:

```bash
python -m pip install "numpy<2" "opencv-python<4.13"
```

---

## Reproducing the main experiments

### Run the full multi-seed sweep

This runs **Harvest + Cleanup** × **Baseline + IML** for training seeds `0..4`:

```bash
bash scripts/run_sweep.sh
```

All raw outputs are written under `runs/`.

### Aggregate learning curves and evaluation summaries

```bash
python -m iml_ssd.analysis.aggregate --runs_dir runs --out_dir results
```

This writes (at minimum) `results/summary.csv` and `results/learning_curves.csv`.

### Generate figures

```bash
python -m iml_ssd.analysis.plot --results_dir results --out_dir figures
```

---

## Rebuilding evaluation-seed sweeps

If you have per-run evaluation-seed CSVs at:

- `runs/<run_name>/eval_seed0.csv`
- `runs/<run_name>/eval_seed1.csv`
- …

then you can rebuild the consolidated sweep tables with:

```bash
python rebuild_eval_seed_sweep.py
```

This writes:

- `results/eval_seed_sweep.csv` (one row per run × eval seed)
- `results/eval_seed_sweep_agg.csv` (aggregated mean/std over eval seeds, per run)

---

## Compute backend notes (CPU / CUDA / Apple Silicon)

This project uses PyTorch. You can check which accelerator backend is available:

```bash
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("mps", hasattr(torch.backends,"mps") and torch.backends.mps.is_available())'
```

- **CUDA**: supported if `torch.cuda.is_available()` is `True`.
- **Apple Silicon (MPS)**: supported if `torch.backends.mps.is_available()` is `True`.
- **CPU-only**: works, but training sweeps are compute-intensive.

---

## Troubleshooting

### `No module named 'social_dilemmas'`

You likely skipped SSD installation (or installed in a different environment). Re-run:

```bash
bash scripts/install_ssd_no_ray.sh
```

### `No module named 'cv2'`

Install OpenCV into the *active* environment:

```bash
python -m pip install "numpy<2" "opencv-python<4.13"
```

### Gym / NumPy warnings

SSD depends on the legacy `gym` package, which emits warnings under NumPy 2. To avoid incompatibilities, this repo constrains NumPy to `<2`.

---

## Citation

If you use this code in academic work, please cite the accompanying paper:

```bibtex
@article{alqithamiIML2026,
  title   = {From Behavioral Influence to Accountable Institutions: Institutional Monitoring and Ledger Mechanisms for Cooperative Human--AI Systems},
  author  = {Alqithami, Saad},
  year    = {2026},
  note    = {Manuscript under review. Code: https://github.com/alqithami/IML}
}
```

---

## Acknowledgements

This repository builds on *Sequential Social Dilemma Games* (SSD) by Leibo et al. and related work on sequential social dilemmas.
