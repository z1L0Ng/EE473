# Presentation Script (First Half, Updated for 13-Slide Deck)

Target split: first half ~2:10-2:30 (Slides 1-6), then hand off to teammate for Slides 7-12.

## Slide 1 (Title) - 0:00 to 0:15

Hello everyone, we are Zilong Zeng and Meng Wu.
Our project is **Energy-Aware Edge Scheduling with Reinforcement Learning**.
I will cover motivation, data, environment, and methods first, then my teammate will present results, ablations, and conclusion.

## Slide 2 (Contents) - 0:15 to 0:30

This presentation has six parts:
goal, dataset and environment, methods, main results, robustness analysis, and future work.
I will focus on the setup and method side.

## Slide 3 (Introduction & Goal) - 0:30 to 1:00

Our task is edge scheduling under a practical trade-off.
At each time step, the device chooses a performance mode while workload changes over time.
Low mode saves energy but can increase queueing delay.
High mode reduces delay but consumes more energy.
So we optimize three objectives together: energy, latency, and deadline misses.

## Slide 4 (Dataset & Environment) - 1:00 to 1:35

We use the Google cluster task-event trace with 120 shards.
The preprocessing pipeline is submit-event aggregation, normalization, and reproducible train/test splitting.

The simulator is a single-device environment with three discrete actions: Low, Medium, and High.
The reward is:
`r_t = -(alpha_energy * energy_t + beta_latency * latency_t + gamma_miss * miss_t)`.
Our default weights are `(1.0, 0.6, 2.0)`.

## Slide 5 (Baselines & RL Methods) - 1:35 to 2:05

For baselines, we use always_low, always_medium, always_high, and a threshold policy.
For RL, we use two course-aligned algorithms:
Tabular Q-learning and Linear Approximation Q-learning.

Tabular Q-learning works on discretized states.
Linear Approximation Q-learning uses a 66-dimensional feature vector built from continuous terms, one-hot bins, and interaction terms.
For reproducibility, final reporting uses 5 seeds with mean and standard deviation.

## Slide 6 (Evaluation Protocol & Main Result Preview) - 2:05 to 2:25

Before handing off, here is the evaluation protocol:
we benchmark RL against heuristics on fixed train/test episodes and report comparable metrics.
At a high level, both RL methods outperform the strongest heuristic baseline, and Approx Q is slightly better in return while Tabular Q is faster.

## Handoff Line

Now I’ll hand it over to my teammate, who will walk through the detailed figures, ablation and generalization results, trials and setbacks, and final conclusion.
