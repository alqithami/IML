#!/usr/bin/env bash
set -euo pipefail

# Full experimental sweep for the IML paper artifact.
# Conditions:
#   1. baseline — PPO only (no institutional wrapper)
#   2. ia — Inequity Aversion
#   3. si — Social Influence
#   4. monitor_only — detection + ledger, no sanctions
#   5. sanction_no_review — detection + sanctions, no review
#   6. iml — full IML (detection + sanctions + review)
#   7. high_review — IML with high review probability (0.8)
# Environments: cleanup, harvest
# Training seeds: 0, 1, 2, 3, 4
# Evaluation seeds: 0, 1, 2, 3, 4

SEEDS=(0 1 2 3 4)
EVAL_SEEDS=(0 1 2 3 4)

CONFIGS=(
  # Cleanup
  "configs/cleanup_baseline.yaml"
  "configs/cleanup_ia.yaml"
  "configs/cleanup_si.yaml"
  "configs/cleanup_iml_monitor_only.yaml"
  "configs/cleanup_iml_sanction_no_review.yaml"
  "configs/cleanup_iml.yaml"
  "configs/cleanup_iml_high_review.yaml"

  # Harvest
  "configs/harvest_baseline.yaml"
  "configs/harvest_ia.yaml"
  "configs/harvest_si.yaml"
  "configs/harvest_iml_monitor_only.yaml"
  "configs/harvest_iml_sanction_no_review.yaml"
  "configs/harvest_iml.yaml"
  "configs/harvest_iml_high_review.yaml"
)

echo "=============================="
echo " IML Full Experimental Sweep"
echo " ${#CONFIGS[@]} configs × ${#SEEDS[@]} seeds = $(( ${#CONFIGS[@]} * ${#SEEDS[@]} )) training runs"
echo " Evaluation seeds: ${EVAL_SEEDS[*]}"
echo "=============================="

TOTAL=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
COUNT=0
for CFG in "${CONFIGS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] Training: $CFG seed=$SEED"
    python -m iml_ssd.experiments.train --config "$CFG" --seed "$SEED"
  done
done

echo ""
echo "=============================="
echo " Training complete ($TOTAL runs)"
echo "=============================="

echo ""
echo "Running evaluation sweep..."
for RUN_DIR in runs/*/; do
  if [[ ! -f "${RUN_DIR}model.pt" ]]; then
    echo "[skip] No model.pt in $RUN_DIR"
    continue
  fi

  for ES in "${EVAL_SEEDS[@]}"; do
    echo "Evaluating: $RUN_DIR eval_seed=$ES"
    python -m iml_ssd.experiments.evaluate \
      --run_dir "$RUN_DIR" \
      --episodes 50 \
      --seed "$ES" \
      --out_suffix "_seed${ES}"
  done
done

echo ""
echo "=============================="
echo " Evaluation complete"
echo "=============================="

echo ""
echo "Aggregating results..."
python -m iml_ssd.analysis.aggregate --runs_dir runs --out_dir results

echo ""
echo "Generating figures..."
python -m iml_ssd.analysis.plot --results_dir results --out_dir figures

echo ""
echo "Running statistical analysis..."
python -m iml_ssd.analysis.statistics --results_dir results --out_dir results/statistics

echo ""
echo "=============================="
echo " Full sweep pipeline complete"
echo " Results: results/"
echo " Figures: figures/"
echo " Stats: results/statistics/"
echo "=============================="
