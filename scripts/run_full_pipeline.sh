#!/usr/bin/env bash
set -euo pipefail

# End-to-end reproducibility pipeline for the final project artifacts.
# Default setup is the robust protocol using 120 Google trace shards.

START_SHARD="${1:-0}"
END_SHARD="${2:-119}"
RAW_DIR="${3:-data/raw}"
TRACE_OUT="${4:-data/processed/workload_trace_120.csv}"

SEEDS="${SEEDS:-11,22,33,44,55}"
EPISODE_LEN="${EPISODE_LEN:-288}"

echo "[1/8] Download trace shards ${START_SHARD}-${END_SHARD} -> ${RAW_DIR}"
bash scripts/download_google_trace_shards.sh "${START_SHARD}" "${END_SHARD}" "${RAW_DIR}"

echo "[2/8] Preprocess trace -> ${TRACE_OUT}"
python3 scripts/prepare_workload_trace.py \
  --raw-dir "${RAW_DIR}" \
  --pattern 'task_events_part_*.csv.gz' \
  --max-files $((END_SHARD - START_SHARD + 1)) \
  --bucket-seconds 60 \
  --output "${TRACE_OUT}"

echo "[3/8] Run baselines"
python3 scripts/run_phase2_baselines.py \
  --trace-path "${TRACE_OUT}" \
  --episode-length "${EPISODE_LEN}" \
  --stride "${EPISODE_LEN}" \
  --max-train-episodes 20 \
  --max-test-episodes 6 \
  --output-prefix results/phase2_baselines_120

echo "[4/8] Train single-run tabular + approx models for learning curves/policy summary"
python3 scripts/run_tabular_q_learning.py \
  --trace-path "${TRACE_OUT}" \
  --episode-length "${EPISODE_LEN}" \
  --stride "${EPISODE_LEN}" \
  --max-train-episodes 20 \
  --max-test-episodes 6 \
  --num-epochs 300 \
  --eval-every 5 \
  --seed 55 \
  --output-dir results/tabular_q_learning_120

python3 scripts/run_approx_q_learning.py \
  --trace-path "${TRACE_OUT}" \
  --episode-length "${EPISODE_LEN}" \
  --stride "${EPISODE_LEN}" \
  --max-train-episodes 20 \
  --max-test-episodes 6 \
  --num-epochs 300 \
  --alpha 0.03 \
  --gamma 0.98 \
  --epsilon-decay 0.999 \
  --eval-every 5 \
  --seed 55 \
  --output-dir results/approx_q_learning_120

echo "[5/8] Multi-seed method comparison"
python3 scripts/run_phase3_multiseed.py \
  --trace-path "${TRACE_OUT}" \
  --episode-length "${EPISODE_LEN}" \
  --stride "${EPISODE_LEN}" \
  --max-train-episodes 20 \
  --max-test-episodes 6 \
  --seeds "${SEEDS}" \
  --output-prefix results/phase3_multiseed_120

echo "[6/8] Reward and hyperparameter sensitivity"
python3 scripts/run_reward_sensitivity.py \
  --trace-path "${TRACE_OUT}" \
  --episode-length "${EPISODE_LEN}" \
  --stride "${EPISODE_LEN}" \
  --max-train-episodes 20 \
  --max-test-episodes 6 \
  --seeds "${SEEDS}" \
  --methods approx \
  --output-prefix results/reward_sensitivity_120

python3 scripts/run_hyperparam_sensitivity.py \
  --trace-path "${TRACE_OUT}" \
  --episode-length "${EPISODE_LEN}" \
  --stride "${EPISODE_LEN}" \
  --max-train-episodes 20 \
  --max-test-episodes 6 \
  --seeds "${SEEDS}" \
  --method approx \
  --approx-num-epochs 150 \
  --output-prefix results/hyperparam_sensitivity_120

echo "[7/8] Slicing generalization check"
python3 scripts/run_generalization_check.py \
  --trace-path "${TRACE_OUT}" \
  --episode-length "${EPISODE_LEN}" \
  --seeds "${SEEDS}" \
  --method approx \
  --approx-num-epochs 120 \
  --output-prefix results/generalization_check_120

echo "[8/8] Generate final figures/tables"
python3 scripts/generate_final_artifacts.py \
  --trace-path "${TRACE_OUT}" \
  --phase3-json results/phase3_multiseed_120.json \
  --reward-json results/reward_sensitivity_120.json \
  --hyper-json results/hyperparam_sensitivity_120.json \
  --generalization-json results/generalization_check_120.json \
  --tabular-history-csv results/tabular_q_learning_120/history.csv \
  --approx-history-csv results/approx_q_learning_120/history.csv \
  --tabular-q-table results/tabular_q_learning_120/q_table_best.npy \
  --approx-weights results/approx_q_learning_120/weights_best.npy \
  --output-dir results/final_figures

echo "Pipeline completed."
