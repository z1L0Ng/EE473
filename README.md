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
