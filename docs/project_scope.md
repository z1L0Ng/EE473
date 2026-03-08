# Final Project Scope Statement

We implement a single-device energy-aware scheduling environment driven by real Google cluster workload traces, where the agent selects one of three discrete performance modes (low, medium, high) at each time step to trade off energy use against queueing delay and deadline misses; we keep the scope deliberately narrow (single device, simple queue and battery dynamics, interpretable reward), and evaluate reproducible train/test episodes by comparing simple heuristic baselines and course-aligned RL methods on the same processed trace.

