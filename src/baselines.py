from typing import Callable, Dict, List, Sequence

from config import DEFAULT_ENV_CONFIG, EnvConfig
from env import EnergySchedulingEnv, Observation

Policy = Callable[[Observation, Dict[str, float]], int]


def always_low_policy(_: Observation, __: Dict[str, float]) -> int:
    return 0


def always_high_policy(_: Observation, __: Dict[str, float]) -> int:
    return 2


def always_medium_policy(_: Observation, __: Dict[str, float]) -> int:
    return 1


def threshold_policy_factory(workload_threshold: float = 0.6, battery_threshold: float = 0.15) -> Policy:
    def _policy(_: Observation, info: Dict[str, float]) -> int:
        if float(info.get("battery_ratio", 1.0)) < battery_threshold:
            return 0
        if float(info.get("current_workload", 0.0)) >= workload_threshold:
            return 2
        return 0

    return _policy


def evaluate_policy(
    workload_episodes: Sequence[Sequence[float]],
    policy: Policy,
    config: EnvConfig = DEFAULT_ENV_CONFIG,
) -> Dict[str, float]:
    if len(workload_episodes) == 0:
        raise ValueError("workload_episodes is empty")

    total_return = 0.0
    total_energy = 0.0
    total_latency = 0.0
    total_miss = 0.0
    total_steps = 0

    episode_returns: List[float] = []
    for episode in workload_episodes:
        env = EnergySchedulingEnv(episode, config=config)
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action = int(policy(obs, info))
            obs, reward, done, info = env.step(action)
            episode_return += reward
            total_energy += info["step_energy"]
            total_latency += info["step_latency"]
            total_miss += info["step_miss"]
            total_steps += 1
        episode_returns.append(episode_return)
        total_return += episode_return

    n_episodes = len(workload_episodes)
    return {
        "episodes": float(n_episodes),
        "total_steps": float(total_steps),
        "avg_episode_return": total_return / n_episodes,
        "avg_step_energy": total_energy / total_steps,
        "avg_step_latency": total_latency / total_steps,
        "miss_rate": total_miss / total_steps,
        "avg_episode_length": total_steps / n_episodes,
        "best_episode_return": max(episode_returns),
        "worst_episode_return": min(episode_returns),
    }
