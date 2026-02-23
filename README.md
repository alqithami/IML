# IML: Institutional Monitoring & Ledger for Sequential Social Dilemmas

This repository provides an experiment pipeline and reference implementation of an **Institutional Monitoring and Ledger (IML)** wrapper for **sequential social dilemma** environments (**Harvest** and **Cleanup**) from *Sequential Social Dilemma Games* (SSD). The key idea is to keep the **base Markov game** intact while adding an **auditable institutional layer** that (i) monitors norm-relevant events, (ii) logs evidence to a ledger, and (iii) applies **delayed, contestable settlement** (sanctions/remedies).

> **Important:** GitHub hosts the code, but it does not “run” the experiments for you. To execute anything, you must **clone/download the repository to your machine** and run the commands locally.

---

## Repository layout (high level)

- `iml_ssd/` — core library code (IML wrapper, PPO training, evaluation, analysis).
- `configs/` — YAML configs for Baseline/IML in Harvest/Cleanup.
- `scripts/` — helper scripts (SSD install without Ray/RLlib, multi-seed sweep).
- `runs/` — raw run outputs (per-seed logs/checkpoints). **Large; typically not committed.**
- `results/` — aggregated CSV summaries written by the analysis scripts.
- `figures/` — generated figures written by the plotting scripts.

---

## System requirements

- **Python:** 3.9 (SSD is not compatible with newer Python/Gym/NumPy stacks)
- **OS:** macOS or Linux recommended  
  - **Windows:** use **WSL2 (Ubuntu)** or another Linux environment; the scripts are `bash`-based.
- **Tools:** `git`, and either **conda/miniforge** (recommended) or a compatible Python environment.

---

## Quickstart (copy/paste)

### 1) Clone the repository

```bash
git clone https://github.com/alqithami/IML.git
cd IML
```

If you do not have `git`, install it first (or use GitHub’s “Download ZIP”, then unzip and `cd` into the folder).

### 2) Create and activate a clean environment

**Conda / Miniforge (recommended):**
```bash
conda create -n imlssd python=3.9 -y
conda activate imlssd
python -m pip install --upgrade pip setuptools wheel
```

> Tip: if your shell shows `(base)` and `(imlssd)` at the same time, run `conda deactivate` once, then `conda activate imlssd` to avoid mixing environments.

### 3) Install SSD (without Ray/RLlib)

From the repo root:
```bash
bash scripts/install_ssd_no_ray.sh
```

This will:
- clone SSD into `sequential_social_dilemma_games/`,
- patch RLlib imports to be optional, and
- install `social-dilemmas` in editable mode.

### 4) Install this package

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 5) Run a smoke test

```bash
python -m iml_ssd.tools.smoke_test --env cleanup --num_agents 5 --steps 50
```

If that works, your environment is set up correctly.

---

## Reproducing the main experiments

### 1) Run the full multi-seed sweep

This runs **Harvest + Cleanup** × **Baseline + IML** for training seeds `0..4`:

```bash
bash scripts/run_sweep.sh
```

Raw outputs are written under `runs/` (one folder per run).

### 2) Aggregate learning curves and evaluation summaries

```bash
python -m iml_ssd.analysis.aggregate --runs_dir runs --out_dir results
```

This writes (at minimum):
- `results/summary.csv`
- `results/learning_curves.csv`

### 3) Generate figures

```bash
python -m iml_ssd.analysis.plot --results_dir results --out_dir figures
```

---

## Evaluation-seed sweep rebuild (optional)

If you have per-run evaluation-seed files such as:
- `runs/<run_name>/eval_seed0.csv`
- `runs/<run_name>/eval_seed1.csv`
- …

then you can rebuild consolidated tables with:

```bash
python rebuild_eval_seed_sweep.py
```

This writes:
- `results/eval_seed_sweep.csv` (one row per run × eval seed)
- `results/eval_seed_sweep_agg.csv` (mean/std over eval seeds, per run)

---

## Compute backend notes (CPU / CUDA / Apple Silicon)

This project uses PyTorch. You can check available acceleration:

```bash
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("mps", hasattr(torch.backends,"mps") and torch.backends.mps.is_available())'
```

- **CUDA** is used if `torch.cuda.is_available()` is `True`.
- **Apple Silicon (MPS)** is used if `torch.backends.mps.is_available()` is `True`.
- **CPU-only** works, but sweeps are compute-intensive.

---

## Troubleshooting

### `No module named 'social_dilemmas'`
You likely skipped SSD installation (or installed in a different environment). Re-run:
```bash
bash scripts/install_ssd_no_ray.sh
```

### `No module named 'cv2'`
Install OpenCV into the **active** environment:
```bash
python -m pip install "numpy<2" "opencv-python<4.13"
```

### Gym / NumPy warnings
SSD depends on the legacy `gym` package, which emits warnings under NumPy 2. This repo constrains NumPy to `<2` for compatibility.

---

## Citation

If you use this code in academic work, please cite the accompanying manuscript and the SSD benchmark:

```bibtex
@article{alqithamiIML2026,
  title   = {From Behavioral Influence to Accountable Institutions: Institutional Monitoring and Ledger Mechanisms for Cooperative Human--AI Systems},
  author  = {Alqithami, Saad},
  year    = {2026},
  note    = {Manuscript under review. Code: https://github.com/alqithami/IML}
}

@inproceedings{leibo2017ssd,
  title   = {Multi-Agent Reinforcement Learning in Sequential Social Dilemmas},
  author  = {Leibo, Joel Z. and others},
  booktitle = {Proceedings of the 16th Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year    = {2017}
}
```

---

## License

This repository is released under the **MIT License**. See `LICENSE`.

---

## Acknowledgements

This repository builds on *Sequential Social Dilemma Games* (SSD) and the sequential social dilemmas introduced by Leibo et al.
