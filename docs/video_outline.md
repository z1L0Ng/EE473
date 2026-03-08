# 3-5 Minute Video Outline

## 0:00-0:30 Problem and Motivation
- Introduce single-device edge scheduling under time-varying workload.
- Goal: balance energy cost and service quality (latency + deadline misses).

## 0:30-1:10 Dataset and Environment
- Workload source: Google cluster trace task events.
- Show preprocessing pipeline: submit-event aggregation, normalization, train/test split.
- Show custom simulator design: state bins (workload/queue/battery), 3 actions (low/medium/high), reward function.

## 1:10-1:50 Baselines and RL Methods
- Baselines: always low/medium/high + threshold heuristic.
- RL methods: tabular Q-learning and linear approximate Q-learning.
- Explain why these methods align with course scope and comparison goals.

## 1:50-2:50 Evaluation Protocol and Main Results
- Protocol: 120 shards, non-overlap test episodes (`episode_length=288`, `stride=288`), 5 random seeds.
- Report key table from `results/phase3_multiseed_120.md`.
- Highlight best method and margin over baseline.

## 2:50-3:40 Ablation and Generalization
- Reward sensitivity summary from `results/reward_sensitivity_120.md`.
- Hyperparameter sensitivity summary from `results/hyperparam_sensitivity_120.md`.
- Slicing generalization check from `results/generalization_check_120.md`.

## 3:40-4:30 Limitations and Future Work
- Current simplifications: single device, simplified battery model, discrete actions.
- Future extensions: richer constraints, stronger policies, broader workload coverage.

## 4:30-5:00 Closing
- Restate contribution: reproducible custom simulator + robust multi-seed comparison + ablation suite.
- Mention repository artifacts: scripts, figures, notebook.
