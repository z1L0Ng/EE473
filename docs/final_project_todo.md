# Final Project TODO Plan

## 0. Progress snapshot (updated 2026-03-08)

### Current status summary
- Scope locked to single-device scheduling and documented in `docs/project_scope.md`
- Dataset pipeline implemented (download + preprocess + reproducible train/test split)
- Custom simulator implemented (`state/action/transition/reward/done`)
- Baselines implemented and evaluated (`always_low`, `always_medium`, `always_high`, threshold heuristic)
- RL algorithms implemented and run (Tabular Q-learning + Linear approximation Q-learning)
- Comparison table generated in `results/phase3_comparison.md`
- Multi-seed evaluation pipeline script added: `scripts/run_phase3_multiseed.py` (mean/std reporting + wall time aggregation)
- Ablation scripts added and run: reward sensitivity, hyperparameter sensitivity, slicing generalization check
- Final packaging artifacts generated (`notebooks/final_project.ipynb`, `results/final_figures/*`, `docs/video_outline.md`, `docs/limitations_future_work.md`)

### Current best test results (single-seed setting)
- Baseline best: `threshold(w=0.80)` return `-108.446`
- Tabular Q-learning best checkpoint: return `-103.573`
- Linear approximation Q-learning best checkpoint: return `-101.357`

### Current multi-seed results (5 seeds, non-overlap test episodes)
- Dataset: `data/processed/workload_trace_120.csv` (120 shards)
- Baseline best: `always_low` return mean/std `-57.289 +/- 0.000`
- Tabular Q-learning (best-per-seed): return mean/std `-56.200 +/- 0.084`
- Linear approximation Q-learning (best-per-seed): return mean/std `-56.016 +/- 0.117`

### Important caveat
- Legacy 20-shard file still has low test coverage; the robust protocol should use `data/processed/workload_trace_120.csv` where `episode_length=288, stride=288` yields 6 non-overlapping test episodes.

---

## 1. Lock project scope

### Goal
Implement a **single-device energy-delay aware scheduling problem** driven by real workload traces, and compare a small set of RL methods against simple heuristic baselines.

### Final framing
- Environment type: custom RL scheduling simulator
- Scenario: one edge device with discrete performance modes
- Input: time-varying workload trace
- Objective: balance workload service quality and energy consumption
- Deliverable focus: clear simulator, clear reward, limited scope, reproducible experiments

### Done condition
- A one-paragraph project statement is finalized and used consistently in notebook/report/video
- Team agrees to avoid expanding into multi-device or full cluster scheduling

---

## 2. Confirm dataset choice

### Primary dataset choice
Use **Google cluster workload trace** only as the source of time-varying workload arrivals.

### What the dataset is used for
- Extract one or more workload time series
- Normalize workload intensity to a fixed range
- Segment the trace into episodes for RL training and evaluation

### What the dataset is **not** used for
- Not used as a ready-made RL environment
- Not used as direct energy labels
- Not used to reconstruct a full real scheduler

### Tasks
- [x] Download and inspect the workload trace files
- [x] Choose the workload field to use as the main signal
- [x] Define the time resolution after downsampling
- [x] Normalize the selected workload signal
- [x] Slice the trace into training and testing episodes
- [x] Save a lightweight processed dataset for repeated experiments

### Done condition
- A processed workload time series file exists
- Training/test episode generation is reproducible
- The notebook can load the processed trace in one cell

---

## 3. Define the custom simulator

### Required simulator components
The environment must explicitly define:
- state
- action
- transition logic
- reward
- termination condition
- evaluation metrics

### Minimal environment design
#### State
Start with a compact discrete state such as:
- workload bin
- queue/backlog bin
- battery/energy bin

Optional later addition:
- previous mode

#### Action
Discrete performance mode:
- low
- medium
- high

#### Transition dynamics
At each step:
1. Read workload arrival from trace
2. Apply selected performance mode
3. Update processed workload / queue backlog
4. Update battery or energy budget
5. Compute reward
6. Move to next timestep

### Tasks
- [x] Write a clear state definition
- [x] Write a clear action definition
- [x] Define queue update logic
- [x] Define battery / energy update logic
- [x] Define episode length and reset logic
- [x] Implement `reset()` and `step()` functions
- [x] Verify one full episode can run with a dummy policy

### Done condition
- The simulator can run end-to-end with a random or fixed policy
- Each step returns next state, reward, done, and metrics info

---

## 4. Specify the energy model and scheduling constraints

### Energy model
Each mode must have explicit service rate and energy cost.

Example starting point:
- low: low service rate, low power draw
- medium: medium service rate, medium power draw
- high: high service rate, high power draw

A simple first version is enough as long as it is explicit and consistent.

### Scheduling constraints
Keep only a few constraints to control scope.

Recommended first version:
- bounded queue/backlog
- deadline miss indicator or backlog penalty
- optional battery lower bound
- optional switching cost between modes

### Tasks
- [x] Pick numeric values for service rate in each mode
- [x] Pick numeric values for energy cost in each mode
- [x] Decide whether to include switching cost
- [x] Decide whether to include hard battery constraints
- [x] Decide how deadline misses are defined
- [x] Document all constants in one config section

### Done condition
- The simulator parameters are fully listed in one place
- Another teammate can understand the environment without verbal explanation

---

## 5. Finalize the reward function

### First reward version
Use a simple weighted penalty:

```text
r_t = - alpha * energy_t - beta * latency_t - gamma * miss_t
```

Where:
- `energy_t` = energy consumed this step
- `latency_t` = queue length, backlog, or delay proxy
- `miss_t` = deadline violation indicator or count

### Design principle
Reward should match the actual scheduling goal and remain interpretable.

### Tasks
- [x] Define the exact form of `energy_t`
- [x] Define the exact form of `latency_t`
- [x] Define the exact form of `miss_t`
- [x] Choose initial values for `alpha`, `beta`, `gamma`
- [x] Run a few hand-check examples to verify reward signs make sense
- [x] Confirm that better scheduling behavior produces higher return

### Done condition
- Reward is mathematically written down
- Reward can be explained in 3–4 sentences in the final notebook
- Sanity checks show the reward is directionally correct

---

## 6. Build baselines before RL

### Required baselines
At minimum implement:
- always low mode
- always high mode
- threshold heuristic

### Threshold heuristic idea
- Use high mode when workload is above a threshold
- Use low mode when workload is below a threshold
- Optionally force low mode when battery is low

### Tasks
- [x] Implement always-low baseline
- [x] Implement always-high baseline
- [x] Implement threshold heuristic baseline
- [x] Run all baselines on the same episodes
- [x] Record return, energy, latency, and miss rate

### Done condition
- Baseline metrics table exists
- RL results can later be compared against non-learning methods

---

## 7. Implement RL algorithms

### Recommended algorithm set
Keep the algorithm set small and aligned with the course.

#### Core comparison
- Tabular Q-learning
- Linear function approximation Q-learning

#### Optional fallback if approximation becomes too heavy
- Tabular SARSA
- Tabular Q-learning

### Why this choice
- Matches course expectations
- Keeps coding manageable
- Supports a meaningful comparison between tabular and approximate methods

### Tasks
- [x] Implement tabular Q-learning
- [x] Define state discretization for tabular learning
- [x] Implement linear feature representation for approximate Q-learning
- [x] Implement training loop and epsilon-greedy exploration
- [x] Save training curves and final learned policies

### Done condition
- At least two RL methods run successfully on the custom simulator
- Training output is reproducible and comparable across methods

---

## 8. Define evaluation metrics and experiment protocol

### Core metrics
Use a small, well-defined metric set:
- cumulative return
- average energy consumption
- average latency or backlog
- deadline miss rate
- training time

### Recommended experiment dimensions
- algorithm comparison
- reward weight sensitivity
- epsilon / learning rate sensitivity
- workload episode generalization from train to test

### Tasks
- [x] Create a shared evaluation function for all policies
- [x] Define train/test split for episodes
- [x] Decide number of runs per method
- [x] Decide random seeds for reproducibility
- [x] Define the minimum set of ablations

### Done condition
- Every method is evaluated under the same protocol
- Results are ready to be summarized in tables and figures

---

## 9. Plan figures and notebook outputs

### Must-have outputs
- learning curve for RL methods
- comparison table of all methods
- energy-latency tradeoff figure
- reward sensitivity figure
- policy visualization or action-frequency summary

### Tasks
- [x] Implement result logging during training
- [x] Implement plotting utilities
- [x] Generate a clean summary table
- [x] Generate final publication-style figures for notebook/video

### Done condition
- The final notebook has all necessary figures without manual editing

---

## 10. Estimate code modules and implementation scale

### Expected code scale
A realistic implementation target is a few hundred lines of core code, not a large system.

### Suggested structure
```text
project/
  data/
  src/
    data_loader.py
    env.py
    baselines.py
    q_learning.py
    approx_q.py
    metrics.py
    plots.py
  notebooks/
    final_project.ipynb
```

### Rough code size estimate
- environment: 150–250 lines
- baselines + metrics: 50–100 lines
- tabular Q-learning: 80–120 lines
- approximate Q-learning: 100–180 lines
- notebook analysis: 100–200 lines

### Tasks
- [x] Create repo/module structure
- [x] Separate simulator code from notebook code
- [x] Keep constants in one config block
- [x] Keep all experiments reproducible with fixed seeds

### Done condition
- The project is modular enough to debug quickly
- The notebook mainly presents results instead of containing all raw logic

---

## 11. Immediate execution order

### Phase 1: This week
- [x] Finalize problem scope in one paragraph
- [x] Confirm workload trace source and preprocessing plan
- [x] Implement simulator v1
- [x] Define reward v1
- [x] Run fixed-policy sanity checks

### Phase 2: Next step after simulator works
- [x] Implement baseline policies
- [x] Produce first metrics table
- [x] Implement tabular Q-learning
- [x] Verify learning curve is sensible

### Phase 3: After first RL result exists
- [x] Implement linear approximate Q-learning
- [x] Run comparison experiments
- [x] Tune key parameters lightly
- [x] Produce summary figures

### Phase 4: Final packaging
- [x] Write concise notebook narrative
- [x] Prepare 3–5 minute video outline
- [x] Summarize challenges, limitations, and future improvements

---

## 12. Scope guardrails

### Do not add unless everything else already works
- multi-device scheduling
- full cluster reconstruction
- complex battery physics
- continuous action control
- deep RL as the first implementation
- too many reward terms
- too many baselines

### Rule of thumb
If a new feature does not directly improve the clarity of the scheduling scenario or algorithm comparison, do not add it.

---

## 13. Final project success checklist

- [x] The scheduling scenario is narrow and clearly defined
- [x] The simulator is custom and fully specified
- [x] The reward is explicit and interpretable
- [x] The dataset is real and used appropriately
- [x] At least two course-relevant RL methods are compared
- [x] Simple baselines are included
- [x] Metrics are well defined
- [x] The notebook is concise and reproducible
- [x] The video can explain the problem, method, and result in under 5 minutes

---

## 14. New TODO for next sprint

### A. Strengthen evaluation protocol (high priority)
- [x] Increase test coverage: download more shards and generate at least 5 test episodes under the same episode length
- [x] Run all methods with multiple random seeds (recommended: 5 seeds)
- [x] Report mean/std for return, energy, latency, miss rate
- [x] Add wall-clock training time into `summary.json` for tabular and approximate Q-learning

### B. Add required ablations
- [x] Reward sensitivity: small grid over `alpha`, `beta`, `gamma`
- [x] Hyperparameter sensitivity: `alpha`, `epsilon_decay`, and possibly `gamma`
- [x] Train/test generalization check under at least two train/test slicing settings

### C. Final deliverable packaging (Phase 4)
- [x] Create `notebooks/final_project.ipynb` that loads saved artifacts and regenerates all figures/tables
- [x] Add publication-style learning curves (tabular vs approx)
- [x] Add publication-style baseline vs RL comparison table/plot
- [x] Add publication-style energy-latency tradeoff plot
- [x] Add publication-style reward/hyperparameter sensitivity plot
- [x] Add publication-style policy action-frequency summary
- [x] Prepare a concise 3-5 minute video outline (`docs/video_outline.md`)
- [x] Write a short limitations + future work section for final report/notebook

### D. Documentation cleanup
- [x] Add one reproducibility script to run full pipeline end-to-end
- [x] Remove temporary/smoke artifacts from final deliverable directories
