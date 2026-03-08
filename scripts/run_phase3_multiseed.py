#!/usr/bin/env python3
import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from approx_q import ApproxQLearningConfig, train_linear_approx_q_learning  # noqa: E402
from baselines import (  # noqa: E402
    always_high_policy,
    always_low_policy,
    always_medium_policy,
    evaluate_policy,
    threshold_policy_factory,
)
from config import DEFAULT_ENV_CONFIG  # noqa: E402
from data_loader import build_episodes, load_workload_trace  # noqa: E402
from q_learning import QLearningConfig, train_tabular_q_learning  # noqa: E402


METRIC_KEYS = [
    "avg_episode_return",
    "avg_step_energy",
    "avg_step_latency",
    "miss_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed Phase 3 comparison (baseline vs RL).")
    parser.add_argument("--trace-path", type=Path, default=Path("data/processed/workload_trace.csv"))
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--stride", type=int, default=288)
    parser.add_argument("--train-stride", type=int, default=None)
    parser.add_argument("--test-stride", type=int, default=None)
    parser.add_argument("--max-train-episodes", type=int, default=20)
    parser.add_argument("--max-test-episodes", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="11,22,33,44,55")
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--battery-threshold", type=float, default=0.10)
    parser.add_argument("--tabular-num-epochs", type=int, default=300)
    parser.add_argument("--tabular-alpha", type=float, default=0.15)
    parser.add_argument("--tabular-gamma", type=float, default=0.98)
    parser.add_argument("--tabular-epsilon-start", type=float, default=0.30)
    parser.add_argument("--tabular-epsilon-end", type=float, default=0.02)
    parser.add_argument("--tabular-epsilon-decay", type=float, default=0.995)
    parser.add_argument("--tabular-eval-every", type=int, default=5)
    parser.add_argument("--approx-num-epochs", type=int, default=400)
    parser.add_argument("--approx-alpha", type=float, default=0.02)
    parser.add_argument("--approx-gamma", type=float, default=0.98)
    parser.add_argument("--approx-epsilon-start", type=float, default=0.30)
    parser.add_argument("--approx-epsilon-end", type=float, default=0.02)
    parser.add_argument("--approx-epsilon-decay", type=float, default=0.997)
    parser.add_argument("--approx-eval-every", type=int, default=5)
    parser.add_argument("--output-prefix", type=Path, default=Path("results/phase3_multiseed"))
    return parser.parse_args()


def parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def threshold_grid(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("threshold-step must be positive")
    values = []
    cur = start
    while cur <= end + 1e-9:
        values.append(round(cur, 6))
        cur += step
    return values


def aggregate_runs(runs: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    if not runs:
        raise ValueError("aggregate_runs received empty runs")

    out: Dict[str, Dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = [float(run["best_test_metrics"][key]) for run in runs]  # type: ignore[index]
        out[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }

    wall_times = [float(run["training_wall_time_sec"]) for run in runs]
    out["training_wall_time_sec"] = {
        "mean": float(np.mean(wall_times)),
        "std": float(np.std(wall_times, ddof=0)),
    }
    return out


def format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def write_markdown_table(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    headers = [
        "Method",
        "Avg Return (mean+/-std)",
        "Energy (mean+/-std)",
        "Latency (mean+/-std)",
        "Miss Rate (mean+/-std)",
        "Train Time sec (mean+/-std)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    format_mean_std(
                        float(row["avg_episode_return_mean"]),
                        float(row["avg_episode_return_std"]),
                    ),
                    format_mean_std(
                        float(row["avg_step_energy_mean"]),
                        float(row["avg_step_energy_std"]),
                    ),
                    format_mean_std(
                        float(row["avg_step_latency_mean"]),
                        float(row["avg_step_latency_std"]),
                    ),
                    format_mean_std(
                        float(row["miss_rate_mean"]),
                        float(row["miss_rate_std"]),
                    ),
                    format_mean_std(
                        float(row["training_wall_time_sec_mean"]),
                        float(row["training_wall_time_sec_std"]),
                    ),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row_from_aggregate(method: str, stats: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    return {
        "method": method,
        "avg_episode_return_mean": stats["avg_episode_return"]["mean"],
        "avg_episode_return_std": stats["avg_episode_return"]["std"],
        "avg_step_energy_mean": stats["avg_step_energy"]["mean"],
        "avg_step_energy_std": stats["avg_step_energy"]["std"],
        "avg_step_latency_mean": stats["avg_step_latency"]["mean"],
        "avg_step_latency_std": stats["avg_step_latency"]["std"],
        "miss_rate_mean": stats["miss_rate"]["mean"],
        "miss_rate_std": stats["miss_rate"]["std"],
        "training_wall_time_sec_mean": stats["training_wall_time_sec"]["mean"],
        "training_wall_time_sec_std": stats["training_wall_time_sec"]["std"],
    }


def evaluate_baselines(
    train_episodes: Sequence[Sequence[float]],
    test_episodes: Sequence[Sequence[float]],
    env_config,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    battery_threshold: float,
) -> Tuple[str, Dict[str, float], float, Dict[str, Dict[str, float]]]:
    static_policies: Dict[str, Callable] = {
        "always_low": always_low_policy,
        "always_medium": always_medium_policy,
        "always_high": always_high_policy,
    }

    test_results: Dict[str, Dict[str, float]] = {}
    for name, policy in static_policies.items():
        test_results[name] = evaluate_policy(test_episodes, policy, config=env_config)

    best_threshold: Tuple[float, Dict[str, float]] = (-1.0, {})
    for wt in threshold_grid(threshold_min, threshold_max, threshold_step):
        policy = threshold_policy_factory(workload_threshold=wt, battery_threshold=battery_threshold)
        metrics = evaluate_policy(train_episodes, policy, config=env_config)
        if best_threshold[0] < 0 or metrics["avg_episode_return"] > best_threshold[1]["avg_episode_return"]:
            best_threshold = (wt, metrics)

    selected_wt = best_threshold[0]
    selected_policy_name = f"threshold(w={selected_wt:.2f})"
    selected_policy = threshold_policy_factory(
        workload_threshold=selected_wt,
        battery_threshold=battery_threshold,
    )
    test_results[selected_policy_name] = evaluate_policy(test_episodes, selected_policy, config=env_config)

    best_policy_name = max(test_results.keys(), key=lambda k: test_results[k]["avg_episode_return"])
    best_metrics = test_results[best_policy_name]
    return best_policy_name, best_metrics, selected_wt, test_results


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    train_stride = args.train_stride if args.train_stride is not None else args.stride
    test_stride = args.test_stride if args.test_stride is not None else args.stride
    env_config = replace(DEFAULT_ENV_CONFIG, episode_length=args.episode_length)

    train_series = load_workload_trace(args.trace_path, split="train")
    test_series = load_workload_trace(args.trace_path, split="test")
    train_episodes = build_episodes(train_series, args.episode_length, stride=train_stride, drop_last=True)[
        : args.max_train_episodes
    ]
    test_episodes = build_episodes(test_series, args.episode_length, stride=test_stride, drop_last=True)[
        : args.max_test_episodes
    ]
    if not train_episodes or not test_episodes:
        raise RuntimeError("Not enough train/test episodes. Regenerate data or reduce episode length.")

    baseline_best_name, baseline_best_metrics, selected_wt, all_baseline_test = evaluate_baselines(
        train_episodes=train_episodes,
        test_episodes=test_episodes,
        env_config=env_config,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        battery_threshold=args.battery_threshold,
    )

    tabular_runs: List[Dict[str, object]] = []
    approx_runs: List[Dict[str, object]] = []

    for seed in seeds:
        tabular_cfg = QLearningConfig(
            num_epochs=args.tabular_num_epochs,
            alpha=args.tabular_alpha,
            gamma=args.tabular_gamma,
            epsilon_start=args.tabular_epsilon_start,
            epsilon_end=args.tabular_epsilon_end,
            epsilon_decay=args.tabular_epsilon_decay,
            eval_every=args.tabular_eval_every,
            seed=seed,
        )
        tabular_start = time.perf_counter()
        tabular_result = train_tabular_q_learning(
            train_episodes=train_episodes,
            test_episodes=test_episodes,
            env_config=env_config,
            algo_config=tabular_cfg,
        )
        tabular_wall = time.perf_counter() - tabular_start
        tabular_runs.append(
            {
                "seed": seed,
                "best_epoch": int(tabular_result["best_epoch"]),
                "best_test_metrics": tabular_result["best_test_metrics"],
                "final_test_metrics": tabular_result["final_test_metrics"],
                "training_wall_time_sec": tabular_wall,
            }
        )

        approx_cfg = ApproxQLearningConfig(
            num_epochs=args.approx_num_epochs,
            alpha=args.approx_alpha,
            gamma=args.approx_gamma,
            epsilon_start=args.approx_epsilon_start,
            epsilon_end=args.approx_epsilon_end,
            epsilon_decay=args.approx_epsilon_decay,
            eval_every=args.approx_eval_every,
            seed=seed,
        )
        approx_start = time.perf_counter()
        approx_result = train_linear_approx_q_learning(
            train_episodes=train_episodes,
            test_episodes=test_episodes,
            env_config=env_config,
            algo_config=approx_cfg,
        )
        approx_wall = time.perf_counter() - approx_start
        approx_runs.append(
            {
                "seed": seed,
                "best_epoch": int(approx_result["best_epoch"]),
                "feature_dim": int(approx_result["feature_dim"]),
                "best_test_metrics": approx_result["best_test_metrics"],
                "final_test_metrics": approx_result["final_test_metrics"],
                "training_wall_time_sec": approx_wall,
            }
        )
        print(f"Completed seed={seed}")

    tabular_stats = aggregate_runs(tabular_runs)
    approx_stats = aggregate_runs(approx_runs)
    baseline_stats = {
        key: {"mean": float(baseline_best_metrics[key]), "std": 0.0}
        for key in METRIC_KEYS
    }
    baseline_stats["training_wall_time_sec"] = {"mean": 0.0, "std": 0.0}

    rows = [
        row_from_aggregate(f"baseline:{baseline_best_name}", baseline_stats),
        row_from_aggregate("tabular_q_learning(best-per-seed)", tabular_stats),
        row_from_aggregate("linear_approx_q_learning(best-per-seed)", approx_stats),
    ]
    rows.sort(key=lambda r: float(r["avg_episode_return_mean"]), reverse=True)

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{output_prefix}.json")
    md_path = Path(f"{output_prefix}.md")

    payload = {
        "config": {
            "trace_path": str(args.trace_path),
            "episode_length": args.episode_length,
            "stride": args.stride,
            "train_stride": train_stride,
            "test_stride": test_stride,
            "num_train_episodes": len(train_episodes),
            "num_test_episodes": len(test_episodes),
            "seeds": seeds,
            "tabular_algorithm": asdict(
                QLearningConfig(
                    num_epochs=args.tabular_num_epochs,
                    alpha=args.tabular_alpha,
                    gamma=args.tabular_gamma,
                    epsilon_start=args.tabular_epsilon_start,
                    epsilon_end=args.tabular_epsilon_end,
                    epsilon_decay=args.tabular_epsilon_decay,
                    eval_every=args.tabular_eval_every,
                    seed=seeds[0],
                )
            ),
            "approx_algorithm": asdict(
                ApproxQLearningConfig(
                    num_epochs=args.approx_num_epochs,
                    alpha=args.approx_alpha,
                    gamma=args.approx_gamma,
                    epsilon_start=args.approx_epsilon_start,
                    epsilon_end=args.approx_epsilon_end,
                    epsilon_decay=args.approx_epsilon_decay,
                    eval_every=args.approx_eval_every,
                    seed=seeds[0],
                )
            ),
            "threshold_search": {
                "min": args.threshold_min,
                "max": args.threshold_max,
                "step": args.threshold_step,
                "battery_threshold": args.battery_threshold,
                "selected_workload_threshold": selected_wt,
            },
        },
        "baseline": {
            "best_policy_name": baseline_best_name,
            "best_test_metrics": baseline_best_metrics,
            "all_test_metrics": all_baseline_test,
        },
        "tabular": {
            "runs": tabular_runs,
            "aggregate_best_test_metrics": tabular_stats,
        },
        "approx": {
            "runs": approx_runs,
            "aggregate_best_test_metrics": approx_stats,
        },
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_markdown_table(md_path, rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("\n[Ranking by avg return mean]")
    for row in rows:
        print(
            f"{str(row['method']):40s} "
            f"return={format_mean_std(float(row['avg_episode_return_mean']), float(row['avg_episode_return_std']))} "
            f"energy={format_mean_std(float(row['avg_step_energy_mean']), float(row['avg_step_energy_std']))} "
            f"latency={format_mean_std(float(row['avg_step_latency_mean']), float(row['avg_step_latency_std']))} "
            f"miss={format_mean_std(float(row['miss_rate_mean']), float(row['miss_rate_std']))}"
        )


if __name__ == "__main__":
    main()
