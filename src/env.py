from typing import Dict, Optional, Sequence, Tuple

from config import DEFAULT_ENV_CONFIG, EnvConfig

Observation = Tuple[int, int, int]


def _digitize(value: float, bins: Sequence[float]) -> int:
    for idx in range(len(bins) - 1):
        if value < bins[idx + 1]:
            return idx
    return len(bins) - 2


class EnergySchedulingEnv:
    def __init__(self, workload_episode: Sequence[float], config: EnvConfig = DEFAULT_ENV_CONFIG):
        if len(workload_episode) == 0:
            raise ValueError("workload_episode cannot be empty")
        self.workload_episode = list(workload_episode)
        self.config = config
        self._episode_limit = min(self.config.episode_length, len(self.workload_episode))
        self._t = 0
        self._queue = 0.0
        self._battery = self.config.initial_battery
        self._prev_action: Optional[int] = None
        self._done = False

    def reset(self) -> Tuple[Observation, Dict[str, float]]:
        self._t = 0
        self._queue = 0.0
        self._battery = self.config.initial_battery
        self._prev_action = None
        self._done = False
        current_workload = self.workload_episode[self._t]
        obs = self._make_observation(current_workload)
        info = {
            "timestep": float(self._t),
            "current_workload": float(current_workload),
            "queue": float(self._queue),
            "battery": float(self._battery),
            "battery_ratio": float(self._battery / self.config.battery_capacity),
            "step_energy": 0.0,
            "step_latency": 0.0,
            "step_miss": 0.0,
            "step_reward": 0.0,
        }
        return obs, info

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict[str, float]]:
        if self._done:
            raise RuntimeError("Episode already ended. Call reset().")
        if action < 0 or action >= len(self.config.action_names):
            raise ValueError(f"Invalid action {action}")

        arrival_norm = self.workload_episode[self._t]
        arrival = arrival_norm * self.config.workload_scale
        service = min(self.config.service_rates[action], self._queue + arrival)
        next_queue = max(0.0, self._queue + arrival - service)
        next_queue = min(next_queue, self.config.max_queue)

        switch_penalty = 0.0
        if self._prev_action is not None and self._prev_action != action:
            switch_penalty = self.config.switch_cost
        energy = self.config.energy_costs[action] + switch_penalty
        next_battery = max(0.0, self._battery - energy)

        latency = next_queue
        miss = 1.0 if latency > self.config.deadline_queue_threshold else 0.0
        reward = -(
            self.config.alpha_energy * energy
            + self.config.beta_latency * latency
            + self.config.gamma_miss * miss
        )

        self._queue = next_queue
        self._battery = next_battery
        self._prev_action = action
        self._t += 1

        episode_done = self._t >= self._episode_limit
        battery_done = self.config.hard_battery_cutoff and self._battery <= self.config.min_battery
        self._done = episode_done or battery_done

        next_workload = 0.0 if self._done else self.workload_episode[self._t]
        obs = self._make_observation(next_workload)
        info = {
            "timestep": float(self._t),
            "current_workload": float(next_workload),
            "arrival_norm": float(arrival_norm),
            "arrival": float(arrival),
            "service": float(service),
            "action": float(action),
            "action_name": self.config.action_names[action],
            "queue": float(self._queue),
            "battery": float(self._battery),
            "battery_ratio": float(self._battery / self.config.battery_capacity),
            "step_energy": float(energy),
            "step_latency": float(latency),
            "step_miss": float(miss),
            "step_reward": float(reward),
        }
        return obs, reward, self._done, info

    def _make_observation(self, workload: float) -> Observation:
        workload = min(max(workload, 0.0), 1.0)
        battery_ratio = min(max(self._battery / self.config.battery_capacity, 0.0), 1.0)
        workload_bin = _digitize(workload, self.config.workload_bins)
        queue_bin = _digitize(self._queue, self.config.queue_bins)
        battery_bin = _digitize(battery_ratio, self.config.battery_bins)
        return workload_bin, queue_bin, battery_bin
