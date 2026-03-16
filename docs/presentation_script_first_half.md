
## Slide 1 - Title (0:00-0:15)

Hello everyone, we are Zilong Zeng and Meng Wu.  
Our project is **Energy-Aware Edge Scheduling with Reinforcement Learning**.  

## Slide 2 - Contents (0:15-0:28)
 
In my part, I focus the first few parts and my teammate will cover the rest.
## Slide 3 - Introduction & Goal (0:28-0:55)

The task is edge scheduling under a practical trade-off.  
At each time step, the controller chooses one of three performance modes while workload keeps changing.  
Using low mode saves energy but can increase queueing delay, while high mode reduces delay but costs more energy.  
So our objective is to optimize **energy**, **latency**, and **deadline misses** together.

## Slide 4 - Dataset & Environment (0:55-1:28)

We use the Google cluster task-event trace.  

The environment is a single-device simulator with three actions: Low, Medium, and High.  
At each step, an action determines service rate and energy cost, queue and battery are updated, and we compute latency and miss indicators.

Our reward is:
`r_t = -(alpha_energy * energy_t + beta_latency * latency_t + gamma_miss * miss_t)`.  
The default weights are `(alpha, beta, gamma) = (1.0, 0.6, 2.0)`.

## Slide 5 - Baselines & RL Methods (1:28-1:58)

For baselines, we evaluate `always_low`, `always_medium`, `always_high`, and a threshold policy.  
For RL, we use two methods: Tabular Q-learning and Linear Approximation Q-learning.


final claims are based on 5-seed reporting with mean and standard deviation.

Adding a deep RL model such as DQN is treated as future extension, so we didn't cover that in this project

## Slide 6 - Main Results: Learning Curves (1:58-2:25)

This figure shows the learning behavior across seeds.  
Both RL methods converge stably, and both outperform the best heuristic baseline in final return.

Using the same 5-seed protocol, Tabular Q is much faster to train, about **6.3 seconds**, while Approx Q is about **34.4 seconds**, which is roughly **5.4x** slower but gives slightly better return.  
So the key takeaway is a quality-versus-training-cost trade-off.

## Handoff (2:25-2:30)

Now I’ll hand it over to my teammate for detailed result comparisons, ablation and generalization, trials and setbacks, and final conclusion.
