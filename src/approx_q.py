from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from baselines import evaluate_policy
from config import DEFAULT_ENV_CONFIG, EnvConfig
from env import EnergySchedulingEnv, Observation


@dataclass(frozen=True)
class ApproxQLearningConfig:
    num_epochs: int = 400
    alpha: float = 0.02
    gamma: float = 0.98
    epsilon_start: float = 0.30
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.997
    eval_every: int = 5
    seed: int = 42


class LinearQApproximator:
    def __init__(self, env_config: EnvConfig = DEFAULT_ENV_CONFIG):
        self.env_config = env_config
        self.num_actions = len(env_config.action_names)
        self.w_bins = len(env_config.workload_bins) - 1
        self.q_bins = len(env_config.queue_bins) - 1
        self.b_bins = len(env_config.battery_bins) - 1

        # Base features are shared across actions, action selection is handled by block encoding.
        self.base_dim = 4 + self.w_bins + self.q_bins + self.b_bins + 3
        self.feature_dim = self.base_dim * self.num_actions

    def _base_features(self, obs: Observation, info: Dict[str, float]) -> np.ndarray:
        workload_cont = float(info.get("current_workload", 0.0))
        queue_cont = float(info.get("queue", 0.0))
        battery_ratio = float(info.get("battery_ratio", 1.0))
        queue_norm = min(max(queue_cont / self.env_config.max_queue, 0.0), 1.0)
        battery_ratio = min(max(battery_ratio, 0.0), 1.0)

        features = np.zeros(self.base_dim, dtype=np.float64)
        idx = 0

        # Low-order continuous summary terms.
        features[idx] = 1.0
        idx += 1
        features[idx] = workload_cont
        idx += 1
        features[idx] = queue_norm
        idx += 1
        features[idx] = battery_ratio
        idx += 1

        # One-hot discretized bins preserve tabular-like structure where helpful.
        w_bin, q_bin, b_bin = obs
        features[idx + int(w_bin)] = 1.0
        idx += self.w_bins
        features[idx + int(q_bin)] = 1.0
        idx += self.q_bins
        features[idx + int(b_bin)] = 1.0
        idx += self.b_bins

        # Interaction terms improve representation capacity while staying linear.
        features[idx] = workload_cont * queue_norm
        idx += 1
        features[idx] = queue_norm * (1.0 - battery_ratio)
        idx += 1
        features[idx] = workload_cont * (1.0 - battery_ratio)
        return features

    def features(self, obs: Observation, info: Dict[str, float], action: int) -> np.ndarray:
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action {action}")
        vec = np.zeros(self.feature_dim, dtype=np.float64)
        block_start = action * self.base_dim
        block_end = block_start + self.base_dim
        vec[block_start:block_end] = self._base_features(obs, info)
        return vec

    def q_value(self, weights: np.ndarray, obs: Observation, info: Dict[str, float], action: int) -> float:
        phi = self.features(obs, info, action)
        return float(np.dot(weights, phi))

    def q_values(self, weights: np.ndarray, obs: Observation, info: Dict[str, float]) -> np.ndarray:
        values = np.zeros(self.num_actions, dtype=np.float64)
        for action in range(self.num_actions):
            values[action] = self.q_value(weights, obs, info, action)
        return values

    def greedy_action(self, weights: np.ndarray, obs: Observation, info: Dict[str, float]) -> int:
        return int(np.argmax(self.q_values(weights, obs, info)))


def init_weights(approximator: LinearQApproximator) -> np.ndarray:
    return np.zeros(approximator.feature_dim, dtype=np.float64)


def evaluate_linear_q_policy(
    weights: np.ndarray,
    approximator: LinearQApproximator,
    workload_episodes: Sequence[Sequence[float]],
    env_config: EnvConfig = DEFAULT_ENV_CONFIG,
) -> Dict[str, float]:
    def _policy(obs: Observation, info: Dict[str, float]) -> int:
        return approximator.greedy_action(weights, obs, info)

    return evaluate_policy(workload_episodes, _policy, config=env_config)


def train_linear_approx_q_learning(
    train_episodes: Sequence[Sequence[float]],
    test_episodes: Sequence[Sequence[float]],
    env_config: EnvConfig,
    algo_config: ApproxQLearningConfig,
) -> Dict[str, object]:
    if not train_episodes:
        raise ValueError("train_episodes is empty")
    if not test_episodes:
        raise ValueError("test_episodes is empty")

    approximator = LinearQApproximator(env_config)
    weights = init_weights(approximator)
    best_weights = weights.copy()
    epsilon = algo_config.epsilon_start
    rng = np.random.default_rng(algo_config.seed)

    history: List[Dict[str, float]] = []
    best_test_return = -float("inf")
    best_epoch = 0

    for epoch in range(1, algo_config.num_epochs + 1):
        epoch_return = 0.0
        epoch_steps = 0
        order = rng.permutation(len(train_episodes))

        for idx in order:
            env = EnergySchedulingEnv(train_episodes[int(idx)], config=env_config)
            obs, info = env.reset()
            done = False

            while not done:
                if rng.random() < epsilon:
                    action = int(rng.integers(0, approximator.num_actions))
                else:
                    action = approximator.greedy_action(weights, obs, info)

                next_obs, reward, done, next_info = env.step(action)
                q_sa = approximator.q_value(weights, obs, info, action)

                if done:
                    target = reward
                else:
                    next_q_max = float(np.max(approximator.q_values(weights, next_obs, next_info)))
                    target = reward + algo_config.gamma * next_q_max

                td_error = target - q_sa
                phi = approximator.features(obs, info, action)
                weights = weights + algo_config.alpha * td_error * phi

                obs, info = next_obs, next_info
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

        eval_train = evaluate_linear_q_policy(weights, approximator, train_episodes, env_config=env_config)
        eval_test = evaluate_linear_q_policy(weights, approximator, test_episodes, env_config=env_config)

        history.append(
            {
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
        )

        if eval_test["avg_episode_return"] > best_test_return:
            best_test_return = eval_test["avg_episode_return"]
            best_weights = weights.copy()
            best_epoch = epoch

    final_train_metrics = evaluate_linear_q_policy(weights, approximator, train_episodes, env_config=env_config)
    final_test_metrics = evaluate_linear_q_policy(weights, approximator, test_episodes, env_config=env_config)
    best_train_metrics = evaluate_linear_q_policy(
        best_weights, approximator, train_episodes, env_config=env_config
    )
    best_test_metrics = evaluate_linear_q_policy(best_weights, approximator, test_episodes, env_config=env_config)

    return {
        "weights": weights,
        "best_weights": best_weights,
        "history": history,
        "best_epoch": float(best_epoch),
        "feature_dim": float(approximator.feature_dim),
        "final_train_metrics": final_train_metrics,
        "final_test_metrics": final_test_metrics,
        "best_train_metrics": best_train_metrics,
        "best_test_metrics": best_test_metrics,
    }
