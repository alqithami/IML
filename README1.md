# IML: Institutional Monitoring & Ledger for Sequential Social Dilemmas

This repository contains the reference implementation and paper artifact for **Institutional Monitoring and Ledgers for Cooperative Human–AI Systems: A Framework with Pilot Evidence**.

The code implements an **Institutional Monitoring and Ledger (IML)** wrapper for Sequential Social Dilemma environments, with experiments in **Harvest** and **Cleanup**. IML keeps the underlying Markov game intact while adding an explicit institutional layer that monitors norm-relevant events, records evidence in an auditable ledger, applies delayed sanctions or remedies, and exposes review/contestation as explicit mechanism parameters.

<p align="center">
  <img src="figures/graphical_abstract.png" width="500" alt="Graphical abstract for the IML framework">
</p>

---

## Repository contents

### Code artifact

- `iml_ssd/` — IML wrapper, PPO training loop, evaluation pipeline, and analysis utilities.
- `configs/` — experiment configurations for all reported conditions across both environments:
  - Baseline PPO
  - Inequity Aversion (IA)
  - Social Influence (SI)
  - IML (Full)
  - IML Monitor Only
  - IML Sanction (No Review)
  - IML High Review
- `scripts/` — setup and sweep helpers, including:
  - `install_ssd_no_ray.sh`
  - `run_sweep.sh` for a small Baseline/IML sweep
  - `run_full_sweep.sh` for the full reported-condition sweep
- `runs/` — committed per-run outputs where available.
- `results/` — aggregated CSV summaries.
- `figures/` — generated figures from the public analysis pipeline.
- `robustness/` — evaluation-seed robustness and institutional sensitivity analyses.

### Paper artifact

- `paper_artifact/main.tex` — manuscript source.
- `paper_artifact/main.pdf` — final manuscript PDF.
- `paper_artifact/references.bib` — bibliography support file where retained.
- `paper_artifact/figures/` — publication figures.
- `paper_artifact/tables/` — manuscript tables.
- `paper_artifact/results/` — paper-side result files.

---

## System requirements

- **Python:** 3.9
- **Operating system:** Linux or macOS recommended. Windows users should use WSL2 or another Linux-compatible environment.
- **Tools:** `git`, `bash`, and either conda/miniforge or a compatible Python 3.9 environment.

The SSD dependency used here relies on an older Gym/NumPy stack. This repository therefore provides a Ray-free SSD installation helper and pins NumPy below 2.

---

## Installation

### Option A: Conda / Miniforge

```bash
git clone https://github.com/alqithami/IML.git
cd IML
conda env create -f environment.yml
conda activate imlssd
python -m pip install --upgrade pip setuptools wheel
```

### Option B: Manual environment setup

```bash
git clone https://github.com/alqithami/IML.git
cd IML
conda create -n imlssd python=3.9 -y
conda activate imlssd
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Install SSD without Ray/RLlib

```bash
bash scripts/install_ssd_no_ray.sh
```

This script clones or reuses the SSD codebase in `sequential_social_dilemma_games/`, patches Ray/RLlib imports out of the dependency, and installs the SSD package in editable mode.

### Install this package

```bash
python -m pip install -e .
```

### Smoke test

```bash
python -m iml_ssd.tools.smoke_test --env cleanup --num_agents 5 --steps 50
```

---

## Reproducing the experiments

### Train one run

```bash
python -m iml_ssd.experiments.train --config configs/cleanup_iml.yaml --seed 0
```

This writes a run directory under `runs/`, for example:

```text
runs/cleanup_iml_agents5_seed0/
```

### Evaluate one trained run

```bash
python -m iml_ssd.experiments.evaluate \
  --run_dir runs/cleanup_iml_agents5_seed0 \
  --episodes 50 \
  --seed 0
```

By default this writes `eval.csv` into the run directory. Use `--out_suffix` to write files such as `eval_seed0.csv`.

### Small example sweep

```bash
bash scripts/run_sweep.sh
```

This runs Cleanup and Harvest under Baseline and IML for training seeds `0..4`.

### Full manuscript sweep

```bash
bash scripts/run_full_sweep.sh
```

This script runs all reported conditions across both environments:

- Baseline
- Inequity Aversion
- Social Influence
- Monitor Only
- Sanction (No Review)
- IML (Full)
- High Review

The sweep covers Cleanup and Harvest with five training seeds, for a total of 70 training runs. It then evaluates each trained policy under five evaluation seeds (`0..4`) with 50 episodes per evaluation seed.

### Aggregate public results

```bash
python -m iml_ssd.analysis.aggregate --runs_dir runs --out_dir results
```

Expected outputs include:

- `results/summary.csv`
- `results/learning_curves.csv`

### Generate figures

```bash
python -m iml_ssd.analysis.plot --results_dir results --out_dir figures
```

### Statistical analysis

```bash
python -m iml_ssd.analysis.statistics --results_dir results --out_dir results/statistics
```

---

## Robustness and sensitivity analyses

The `robustness/` directory contains scripts and outputs for the additional analyses reported in the manuscript, including evaluation-seed robustness, paired deltas across conditions, and Cleanup sensitivity to false-positive rate and review probability.

Typical entry points are:

```bash
python robustness/robust_eval_seed.py
python robustness/sensitivity_cleanup_iml.py
```

To rebuild consolidated evaluation-seed tables from files such as `runs/<run_name>/eval_seed0.csv` through `eval_seed4.csv`, run:

```bash
python rebuild_eval_seed_sweep.py
```

This writes:

- `results/eval_seed_sweep.csv`
- `results/eval_seed_sweep_agg.csv`

---

## Using the committed artifacts

The repository includes committed run outputs, aggregated result tables, robustness files, and the manuscript artifact so readers can inspect the paper, examine raw and aggregated outputs, verify the analysis structure, and reproduce or extend the experiments from the provided configs.

For the manuscript artifact, start with:

```text
paper_artifact/main.pdf
paper_artifact/main.tex
paper_artifact/figures/
paper_artifact/tables/
```

---

## Compute backend notes

The project uses PyTorch. To check available acceleration:

```bash
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("mps", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())'
```

CUDA is used when available. Apple Silicon MPS may work depending on the local PyTorch install. CPU-only execution is supported, but the full sweep is compute-intensive.

---

## Troubleshooting

### `No module named 'social_dilemmas'`

The SSD dependency is not installed in the active environment. Re-run:

```bash
bash scripts/install_ssd_no_ray.sh
```

### `No module named 'cv2'`

Install OpenCV into the active environment:

```bash
python -m pip install "numpy<2" "opencv-python<4.13"
```

### Python version errors during SSD install

Use Python 3.9. The SSD setup script explicitly refuses Python 3.10+.

### Gym / NumPy compatibility warnings

The original SSD dependency uses legacy Gym. This project constrains NumPy to `<2` to remain compatible with that stack.

---

## Citation

If you use this repository, please cite the software artifact and the accompanying article.

```bibtex
@software{alqithami_iml,
  author = {Alqithami, Saad},
  title = {IML: Institutional Monitoring and Ledger for Sequential Social Dilemmas},
  year = {2026},
  url = {https://github.com/alqithami/IML}
}

@article{alqithami2026iml,
  author  = {Alqithami, Saad},
  title   = {Institutional Monitoring and Ledgers for Cooperative Human--AI Systems: A Framework with Pilot Evidence},
  journal = {Mathematical and Computational Applications},
  year    = {2026}
}

@inproceedings{leibo2017multi,
  title     = {Multi-agent Reinforcement Learning in Sequential Social Dilemmas},
  author    = {Leibo, Joel Z. and Zambaldi, Vinicius and Lanctot, Marc and Marecki, Janusz and Graepel, Thore},
  booktitle = {Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems},
  pages     = {464--473},
  year      = {2017}
}
```

---

## License

This repository is released under the **MIT License**. See `LICENSE` for details.

---

## Acknowledgements

This project builds on the open-source Sequential Social Dilemma Games environments and on the broader cooperative AI and multi-agent reinforcement learning literature.
