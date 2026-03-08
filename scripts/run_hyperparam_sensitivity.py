#!/usr/bin/env python3
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from approx_q import ApproxQLearningConfig, train_linear_approx_q_learning  # noqa: E402
from config import DEFAULT_ENV_CONFIG  # noqa: E402
from data_loader import build_episodes, load_workload_trace  # noqa: E402
from q_learning import QLearningConfig, train_tabular_q_learning  # noqa: E402


METRIC_KEYS = ["avg_episode_return", "avg_step_energy", "avg_step_latency", "miss_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hyperparameter sensitivity experiments.")
    parser.add_argument("--trace-path", type=Path, default=Path("data/processed/workload_trace_120.csv"))
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--stride", type=int, default=288)
    parser.add_argument("--max-train-episodes", type=int, default=20)
    parser.add_argument("--max-test-episodes", type=int, default=6)
    parser.add_argument("--seeds", type=str, default="11,22,33,44,55")
    parser.add_argument("--method", type=str, default="approx", choices=["tabular", "approx"])
    parser.add_argument(
        "--hyper-grid",
        type=str,
        default="0.015:0.995:0.98,0.020:0.997:0.98,0.030:0.999:0.98,0.020:0.997:0.95,0.020:0.997:0.99",
        help="Comma-separated alpha:epsilon_decay:gamma triples.",
    )
    parser.add_argument("--tabular-num-epochs", type=int, default=200)
    parser.add_argument("--tabular-epsilon-start", type=float, default=0.30)
    parser.add_argument("--tabular-epsilon-end", type=float, default=0.02)
    parser.add_argument("--tabular-eval-every", type=int, default=5)
    parser.add_argument("--approx-num-epochs", type=int, default=250)
    parser.add_argument("--approx-epsilon-start", type=float, default=0.30)
    parser.add_argument("--approx-epsilon-end", type=float, default=0.02)
    parser.add_argument("--approx-eval-every", type=int, default=5)
    parser.add_argument("--output-prefix", type=Path, default=Path("results/hyperparam_sensitivity"))
    return parser.parse_args()


def parse_seeds(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def parse_hyper_grid(raw: str) -> List[Tuple[float, float, float]]:
    grid: List[Tuple[float, float, float]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid hyper tuple '{token}', expected alpha:epsilon_decay:gamma")
        grid.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not grid:
        raise ValueError("Hyper grid is empty.")
    return grid


def aggregate(runs: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    if not runs:
        raise ValueError("aggregate received empty runs")
    out: Dict[str, Dict[str, float]] = {}
    for metric in METRIC_KEYS:
        vals = [float(run["best_test_metrics"][metric]) for run in runs]  # type: ignore[index]
        out[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}
    times = [float(run["training_wall_time_sec"]) for run in runs]
    out["training_wall_time_sec"] = {"mean": float(np.mean(times)), "std": float(np.std(times, ddof=0))}
    return out


def run_one(
    method: str,
    alpha: float,
    epsilon_decay: float,
    gamma: float,
    seed: int,
    train_episodes: Sequence[Sequence[float]],
    test_episodes: Sequence[Sequence[float]],
    env_config,
    args: argparse.Namespace,
) -> Dict[str, object]:
    if method == "tabular":
        cfg = QLearningConfig(
            num_epochs=args.tabular_num_epochs,
            alpha=alpha,
            gamma=gamma,
            epsilon_start=args.tabular_epsilon_start,
            epsilon_end=args.tabular_epsilon_end,
            epsilon_decay=epsilon_decay,
            eval_every=args.tabular_eval_every,
            seed=seed,
        )
        start = time.perf_counter()
        result = train_tabular_q_learning(
            train_episodes=train_episodes,
            test_episodes=test_episodes,
            env_config=env_config,
            algo_config=cfg,
        )
        return {
            "seed": seed,
            "best_epoch": int(result["best_epoch"]),
            "best_test_metrics": result["best_test_metrics"],
            "training_wall_time_sec": time.perf_counter() - start,
        }

    cfg = ApproxQLearningConfig(
        num_epochs=args.approx_num_epochs,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=args.approx_epsilon_start,
        epsilon_end=args.approx_epsilon_end,
        epsilon_decay=epsilon_decay,
        eval_every=args.approx_eval_every,
        seed=seed,
    )
    start = time.perf_counter()
    result = train_linear_approx_q_learning(
        train_episodes=train_episodes,
        test_episodes=test_episodes,
        env_config=env_config,
        algo_config=cfg,
    )
    return {
        "seed": seed,
        "best_epoch": int(result["best_epoch"]),
        "best_test_metrics": result["best_test_metrics"],
        "training_wall_time_sec": time.perf_counter() - start,
    }


def format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def write_markdown(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    headers = [
        "Method",
        "(alpha,epsilon_decay,gamma)",
        "Avg Return",
        "Energy",
        "Latency",
        "Miss Rate",
        "Train Time sec",
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
                    f"({float(row['alpha']):.3f}, {float(row['epsilon_decay']):.3f}, {float(row['gamma']):.3f})",
                    format_mean_std(float(row["avg_episode_return_mean"]), float(row["avg_episode_return_std"])),
                    format_mean_std(float(row["avg_step_energy_mean"]), float(row["avg_step_energy_std"])),
                    format_mean_std(float(row["avg_step_latency_mean"]), float(row["avg_step_latency_std"])),
                    format_mean_std(float(row["miss_rate_mean"]), float(row["miss_rate_std"])),
                    format_mean_std(
                        float(row["training_wall_time_sec_mean"]),
                        float(row["training_wall_time_sec_std"]),
                    ),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    hyper_grid = parse_hyper_grid(args.hyper_grid)
    env_config = replace(DEFAULT_ENV_CONFIG, episode_length=args.episode_length)

    train_series = load_workload_trace(args.trace_path, split="train")
    test_series = load_workload_trace(args.trace_path, split="test")
    train_episodes = build_episodes(train_series, args.episode_length, stride=args.stride, drop_last=True)[
        : args.max_train_episodes
    ]
    test_episodes = build_episodes(test_series, args.episode_length, stride=args.stride, drop_last=True)[
        : args.max_test_episodes
    ]
    if not train_episodes or not test_episodes:
        raise RuntimeError("Not enough train/test episodes for hyperparameter sensitivity run.")

    records: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []

    for alpha, epsilon_decay, gamma in hyper_grid:
        runs: List[Dict[str, object]] = []
        for seed in seeds:
            runs.append(
                run_one(
                    method=args.method,
                    alpha=alpha,
                    epsilon_decay=epsilon_decay,
                    gamma=gamma,
                    seed=seed,
                    train_episodes=train_episodes,
                    test_episodes=test_episodes,
                    env_config=env_config,
                    args=args,
                )
            )
        stats = aggregate(runs)
        records.append(
            {
                "method": args.method,
                "alpha": alpha,
                "epsilon_decay": epsilon_decay,
                "gamma": gamma,
                "runs": runs,
                "aggregate": stats,
            }
        )
        rows.append(
            {
                "method": args.method,
                "alpha": alpha,
                "epsilon_decay": epsilon_decay,
                "gamma": gamma,
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
        )
        print(
            f"Completed method={args.method} "
            f"hyper=({alpha:.3f},{epsilon_decay:.3f},{gamma:.3f}) "
            f"return={stats['avg_episode_return']['mean']:.3f}+/-{stats['avg_episode_return']['std']:.3f}"
        )

    rows.sort(key=lambda r: -float(r["avg_episode_return_mean"]))

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{output_prefix}.json")
    md_path = Path(f"{output_prefix}.md")
    payload = {
        "config": {
            "trace_path": str(args.trace_path),
            "episode_length": args.episode_length,
            "stride": args.stride,
            "num_train_episodes": len(train_episodes),
            "num_test_episodes": len(test_episodes),
            "seeds": seeds,
            "method": args.method,
            "hyper_grid": [
                {"alpha": a, "epsilon_decay": e, "gamma": g}
                for a, e, g in hyper_grid
            ],
        },
        "records": records,
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_markdown(md_path, rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
