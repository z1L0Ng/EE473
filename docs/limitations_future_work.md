# Limitations and Future Work

## Limitations
- The current simulator models a single device only; multi-device interactions and contention are not represented.
- The battery/energy dynamics are intentionally simplified and do not include hardware-level thermal or nonlinear effects.
- The action space is discrete (`low`, `medium`, `high`), which limits fine-grained control decisions.
- RL evaluation is robust across seeds and slicing settings, but still tied to one workload trace family (Google cluster task events).
- The current policy class is value-based and lightweight; richer function classes may capture more complex scheduling behavior.

## Future Work
- Extend from single-device scheduling to multi-device coordination with shared workload pools.
- Add richer constraints (mode-switch overhead calibration, hard service-level targets, battery safety constraints).
- Explore continuous or larger discrete action spaces with actor-critic or constrained RL methods.
- Increase workload diversity by evaluating across additional trace windows and domain-shifted test conditions.
- Add uncertainty-aware or risk-sensitive objectives (e.g., CVaR-style penalties for tail latency/miss events).
- Integrate a deployment-oriented evaluation layer (runtime overhead, on-device inference budget, online adaptation).
