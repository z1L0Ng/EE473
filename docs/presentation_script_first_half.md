# Presentation Script (First Half)

Target split: first half ~2:00-2:20 (Slides 1-5), then hand off to teammate.

## Slide 1 (Title) - 0:00 to 0:15

Hello everyone, we are Zilong Zeng and Meng Wu.
Our project is **Energy-Aware Edge Scheduling with Reinforcement Learning**.
Today I will cover the problem setup, dataset, environment, and methods, and then my teammate will present the results and conclusions.

## Slide 2 (Contents) - 0:15 to 0:25

This talk is organized into six parts:
motivation and goal, dataset and environment, methods, evaluation results, robustness analysis, and future work.
I will focus on the first three parts.

## Slide 3 (Introduction and Goal) - 0:25 to 0:55

The core problem is a practical scheduling trade-off in edge systems.
At each time step, a device must choose a performance mode while workload changes over time.
If we stay in low-power mode, we save energy but may increase delay.
If we stay in high-performance mode, we reduce delay but spend much more energy.
So our objective is to learn a policy that balances **energy consumption**, **queueing latency**, and **deadline misses**.

## Slide 4 (Dataset and Environment) - 0:55 to 1:40

For realism, we use the Google cluster task-event trace, with 120 shards.
We preprocess the data by aggregating submit events, normalizing workload, and generating reproducible train-test episode splits.

On top of this trace, we build a custom single-device simulator with three discrete actions:
Low, Medium, and High.

Our reward is:
`r_t = -(alpha_energy * energy_t + beta_latency * latency_t + gamma_miss * miss_t)`.
The default reward weights are `(1.0, 0.6, 2.0)`.
This makes the optimization target explicit and interpretable.

## Slide 5 (Baselines and RL Methods) - 1:40 to 2:15

We compare four heuristic baselines:
always_low, always_medium, always_high, and a threshold policy.

Then we evaluate two course-aligned RL methods on the same environment:
Tabular Q-learning and Linear Approximation Q-learning.

The tabular method uses discretized states.
The linear approximation method uses a 66-dimensional feature vector combining continuous terms, one-hot bins, and interaction terms.

To ensure reproducibility, we run multi-seed evaluation and report mean and standard deviation.

## Handoff Line (End of First Half)

Now I’ll hand it over to my teammate, who will present the evaluation protocol, key results, ablations, and final conclusions.

