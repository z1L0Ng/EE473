# Final Artifact Summary

## Multi-seed Comparison

| Method | Avg Return | Energy | Latency | Miss Rate |
| --- | --- | --- | --- | --- |
| linear_approx_q_learning(best-per-seed) | -56.016 +/- 0.117 | 0.190 +/- 0.001 | 0.008 +/- 0.001 | 0.000 +/- 0.000 |
| tabular_q_learning(best-per-seed) | -56.200 +/- 0.084 | 0.191 +/- 0.001 | 0.007 +/- 0.001 | 0.000 +/- 0.000 |
| baseline:always_low | -57.289 +/- 0.000 | 0.180 +/- 0.000 | 0.032 +/- 0.000 | 0.000 +/- 0.000 |

## Best Reward Setting

- method=approx, (alpha,beta,gamma)=(0.80, 0.60, 2.00), return=-45.098 +/- 0.050

## Best Hyperparameter Setting

- method=approx, (alpha,epsilon_decay,gamma)=(0.030, 0.999, 0.980), return=-56.017 +/- 0.144

## Generalization Check

- non_overlap: return=-45.348 +/- 0.406 (train/test episodes=20/6)
- overlap_test: return=-45.709 +/- 0.400 (train/test episodes=20/10)
