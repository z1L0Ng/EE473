from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DataConfig:
    bucket_seconds: int = 60
    episode_length: int = 288
    train_ratio: float = 0.8


@dataclass(frozen=True)
class EnvConfig:
    action_names: Tuple[str, ...] = ("low", "medium", "high")
    service_rates: Tuple[float, ...] = (0.35, 0.75, 1.20)
    energy_costs: Tuple[float, ...] = (0.18, 0.42, 0.85)
    switch_cost: float = 0.05
    workload_scale: float = 1.0

    max_queue: float = 8.0
    deadline_queue_threshold: float = 2.5

    battery_capacity: float = 120.0
    initial_battery: float = 120.0
    min_battery: float = 0.0
    hard_battery_cutoff: bool = False

    alpha_energy: float = 1.0
    beta_latency: float = 0.6
    gamma_miss: float = 2.0

    episode_length: int = 288

    workload_bins: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.01)
    queue_bins: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.01)
    battery_bins: Tuple[float, ...] = (0.0, 0.15, 0.35, 0.6, 0.8, 1.01)


DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_ENV_CONFIG = EnvConfig()
