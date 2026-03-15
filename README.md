# EE473 Final Project

Energy-aware scheduling for edge devices using reinforcement learning.

## Phase 1 quick start

1. Download Google task event shards (example: parts 0-19):
```bash
bash scripts/download_google_trace_shards.sh 0 19 data/raw
```

2. Preprocess downloaded shards:
```bash
python3 scripts/prepare_workload_trace.py \
  --raw-dir data/raw \
  --pattern 'task_events_part_*.csv.gz' \
  --max-files 20 \
  --bucket-seconds 60 \
  --output data/processed/workload_trace.csv
```

3. Run fixed-policy sanity checks:
```bash
python3 scripts/run_phase1_sanity.py \
  --trace-path data/processed/workload_trace.csv \
  --episode-length 288 \
  --stride 288 \
  --output results/phase1_sanity.json
```

4. Run Phase 2 baseline table (with threshold search):
```bash
python3 scripts/run_phase2_baselines.py \
  --trace-path data/processed/workload_trace.csv \
  --episode-length 288 \
  --stride 288 \
  --output-prefix results/phase2_baselines
```

5. Train tabular Q-learning:
```bash
python3 scripts/run_tabular_q_learning.py \
  --trace-path data/processed/workload_trace.csv \
  --episode-length 288 \
  --stride 288 \
  --num-epochs 300 \
  --eval-every 5 \
  --output-dir results/tabular_q_learning
```

6. Train linear approximation Q-learning:
```bash
python3 scripts/run_approx_q_learning.py \
  --trace-path data/processed/workload_trace.csv \
  --episode-length 288 \
  --stride 288 \
  --num-epochs 400 \
  --eval-every 5 \
  --output-dir results/approx_q_learning
```

7. Build Phase 3 comparison table:
```bash
python3 scripts/compare_phase3_results.py \
  --baseline-json results/phase2_baselines.json \
  --tabular-json results/tabular_q_learning/summary.json \
  --approx-json results/approx_q_learning/summary.json \
  --output-md results/phase3_comparison.md \
  --output-json results/phase3_comparison.json
```

8. Run robust multi-seed comparison (mean/std + training wall time):
```bash
python3 scripts/run_phase3_multiseed.py \
  --trace-path data/processed/workload_trace_120.csv \
  --episode-length 288 \
  --stride 288 \
  --max-train-episodes 20 \
  --max-test-episodes 6 \
  --seeds 11,22,33,44,55 \
  --output-prefix results/phase3_multiseed_120
```

9. Run reward sensitivity ablation:
```bash
python3 scripts/run_reward_sensitivity.py \
  --trace-path data/processed/workload_trace_120.csv \
  --seeds 11,22,33,44,55 \
  --methods approx \
  --output-prefix results/reward_sensitivity_120
```

10. Run hyperparameter sensitivity ablation:
```bash
python3 scripts/run_hyperparam_sensitivity.py \
  --trace-path data/processed/workload_trace_120.csv \
  --seeds 11,22,33,44,55 \
  --method approx \
  --approx-num-epochs 150 \
  --output-prefix results/hyperparam_sensitivity_120
```

11. Run train/test slicing generalization check:
```bash
python3 scripts/run_generalization_check.py \
  --trace-path data/processed/workload_trace_120.csv \
  --seeds 11,22,33,44,55 \
  --method approx \
  --approx-num-epochs 120 \
  --output-prefix results/generalization_check_120
```

12. Build reward sensitivity context table (setting-wise gap vs always-low):
```bash
python3 scripts/build_reward_context_table.py \
  --trace-path data/processed/workload_trace_120.csv \
  --reward-json results/reward_sensitivity_120.json \
  --output-prefix results/reward_sensitivity_context_120
```

13. Run deadline-threshold stress test for miss-rate sensitivity:
```bash
python3 scripts/run_deadline_stress_test.py \
  --trace-path data/processed/workload_trace_120.csv \
  --deadline-thresholds 2.5,1.5,1.0,0.75,0.5 \
  --tabular-q-table results/tabular_q_learning_120/q_table_best.npy \
  --approx-weights results/approx_q_learning_120/weights_best.npy \
  --output-prefix results/deadline_stress_test_120
```

14. Reproduce full pipeline end-to-end:
```bash
bash scripts/run_full_pipeline.sh 0 119 data/raw data/processed/workload_trace_120.csv
```
