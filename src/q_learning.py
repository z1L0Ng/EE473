from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from baselines import evaluate_policy
from config import DEFAULT_ENV_CONFIG, EnvConfig
from env import EnergySchedulingEnv, Observation


@dataclass(frozen=True)
class QLearningConfig:
    num_epochs: int = 300
    alpha: float = 0.15
    gamma: float = 0.98
    epsilon_start: float = 0.30
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.995
    eval_every: int = 5
    seed: int = 42


def q_table_shape(env_config: EnvConfig) -> Tuple[int, int, int, int]:
    return (
        len(env_config.workload_bins) - 1,
        len(env_config.queue_bins) - 1,
        len(env_config.battery_bins) - 1,
        len(env_config.action_names),
    )


def init_q_table(env_config: EnvConfig = DEFAULT_ENV_CONFIG) -> np.ndarray:
    return np.zeros(q_table_shape(env_config), dtype=np.float64)


def greedy_action(q_table: np.ndarray, obs: Observation) -> int:
    return int(np.argmax(q_table[obs]))


def evaluate_q_policy(
    q_table: np.ndarray,
    workload_episodes: Sequence[Sequence[float]],
    env_config: EnvConfig = DEFAULT_ENV_CONFIG,
) -> Dict[str, float]:
    def _policy(obs: Observation, _: Dict[str, float]) -> int:
        return greedy_action(q_table, obs)

    return evaluate_policy(workload_episodes, _policy, config=env_config)


def train_tabular_q_learning(
    train_episodes: Sequence[Sequence[float]],
    test_episodes: Sequence[Sequence[float]],
    env_config: EnvConfig,
    algo_config: QLearningConfig,
) -> Dict[str, object]:
    if not train_episodes:
        raise ValueError("train_episodes is empty")
    if not test_episodes:
        raise ValueError("test_episodes is empty")

    rng = np.random.default_rng(algo_config.seed)
    q_table = init_q_table(env_config)
    epsilon = algo_config.epsilon_start

    history: List[Dict[str, float]] = []
    best_q_table = q_table.copy()
    best_test_return = -float("inf")
    best_epoch = 0

    for epoch in range(1, algo_config.num_epochs + 1):
        epoch_return = 0.0
        epoch_steps = 0

        order = rng.permutation(len(train_episodes))
        for idx in order:
            env = EnergySchedulingEnv(train_episodes[int(idx)], config=env_config)
            obs, _ = env.reset()
            done = False

            while not done:
                if rng.random() < epsilon:
                    action = int(rng.integers(0, len(env_config.action_names)))
                else:
                    action = greedy_action(q_table, obs)

                next_obs, reward, done, _ = env.step(action)

                q_old = q_table[obs + (action,)]
                next_max = 0.0 if done else float(np.max(q_table[next_obs]))
                target = reward + algo_config.gamma * next_max
                q_table[obs + (action,)] = q_old + algo_config.alpha * (target - q_old)

                obs = next_obs
                epoch_return += reward
                epoch_steps += 1

        avg_train_return = epoch_return / len(train_episodes)
        epsilon = max(algo_config.epsilon_end, epsilon * algo_config.epsilon_decay)

        should_eval = (
            epoch == 1
            or epoch == algo_config.num_epochs
            or epoch % algo_config.eval_every == 0
        )
        if not should_eval:
            continue

        eval_train = evaluate_q_policy(q_table, train_episodes, env_config=env_config)
        eval_test = evaluate_q_policy(q_table, test_episodes, env_config=env_config)

        row = {
            "epoch": float(epoch),
            "epsilon": float(epsilon),
            "train_online_avg_return": float(avg_train_return),
            "train_online_total_steps": float(epoch_steps),
            "eval_train_avg_return": float(eval_train["avg_episode_return"]),
            "eval_train_avg_energy": float(eval_train["avg_step_energy"]),
            "eval_train_avg_latency": float(eval_train["avg_step_latency"]),
            "eval_train_miss_rate": float(eval_train["miss_rate"]),
            "eval_test_avg_return": float(eval_test["avg_episode_return"]),
            "eval_test_avg_energy": float(eval_test["avg_step_energy"]),
            "eval_test_avg_latency": float(eval_test["avg_step_latency"]),
            "eval_test_miss_rate": float(eval_test["miss_rate"]),
        }
        history.append(row)

        if eval_test["avg_episode_return"] > best_test_return:
            best_test_return = eval_test["avg_episode_return"]
            best_q_table = q_table.copy()
            best_epoch = epoch

    final_train_metrics = evaluate_q_policy(q_table, train_episodes, env_config=env_config)
    final_test_metrics = evaluate_q_policy(q_table, test_episodes, env_config=env_config)
    best_train_metrics = evaluate_q_policy(best_q_table, train_episodes, env_config=env_config)
    best_test_metrics = evaluate_q_policy(best_q_table, test_episodes, env_config=env_config)

    return {
        "q_table": q_table,
        "best_q_table": best_q_table,
        "history": history,
        "best_epoch": float(best_epoch),
        "final_train_metrics": final_train_metrics,
        "final_test_metrics": final_test_metrics,
        "best_train_metrics": best_train_metrics,
        "best_test_metrics": best_test_metrics,
    }

