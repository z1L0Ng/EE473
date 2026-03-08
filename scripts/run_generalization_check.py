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
    parser = argparse.ArgumentParser(description="Run train/test slicing generalization checks.")
    parser.add_argument("--trace-path", type=Path, default=Path("data/processed/workload_trace_120.csv"))
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--seeds", type=str, default="11,22,33,44,55")
    parser.add_argument("--method", type=str, default="approx", choices=["tabular", "approx"])
    parser.add_argument("--alpha-energy", type=float, default=0.8)
    parser.add_argument("--beta-latency", type=float, default=0.6)
    parser.add_argument("--gamma-miss", type=float, default=2.0)
    parser.add_argument(
        "--settings",
        type=str,
        default="non_overlap:288:288:20:6,overlap_test:288:144:20:10",
        help="Comma-separated name:train_stride:test_stride:max_train:max_test entries.",
    )
    parser.add_argument("--tabular-num-epochs", type=int, default=150)
    parser.add_argument("--tabular-alpha", type=float, default=0.15)
    parser.add_argument("--tabular-gamma", type=float, default=0.98)
    parser.add_argument("--tabular-epsilon-start", type=float, default=0.30)
    parser.add_argument("--tabular-epsilon-end", type=float, default=0.02)
    parser.add_argument("--tabular-epsilon-decay", type=float, default=0.995)
    parser.add_argument("--tabular-eval-every", type=int, default=5)
    parser.add_argument("--approx-num-epochs", type=int, default=150)
    parser.add_argument("--approx-alpha", type=float, default=0.03)
    parser.add_argument("--approx-gamma", type=float, default=0.98)
    parser.add_argument("--approx-epsilon-start", type=float, default=0.30)
    parser.add_argument("--approx-epsilon-end", type=float, default=0.02)
    parser.add_argument("--approx-epsilon-decay", type=float, default=0.999)
    parser.add_argument("--approx-eval-every", type=int, default=5)
    parser.add_argument("--output-prefix", type=Path, default=Path("results/generalization_check"))
    return parser.parse_args()


def parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_settings(raw: str) -> List[Dict[str, object]]:
    settings: List[Dict[str, object]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 5:
            raise ValueError(
                f"Invalid setting '{token}', expected name:train_stride:test_stride:max_train:max_test"
            )
        settings.append(
            {
                "name": parts[0],
                "train_stride": int(parts[1]),
                "test_stride": int(parts[2]),
                "max_train_episodes": int(parts[3]),
                "max_test_episodes": int(parts[4]),
            }
        )
    if len(settings) < 2:
        raise ValueError("Need at least two slicing settings for generalization check.")
    return settings


def aggregate(runs: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for metric in METRIC_KEYS:
        vals = [float(run["best_test_metrics"][metric]) for run in runs]  # type: ignore[index]
        out[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}
    wall = [float(run["training_wall_time_sec"]) for run in runs]
    out["training_wall_time_sec"] = {"mean": float(np.mean(wall)), "std": float(np.std(wall, ddof=0))}
    return out


def run_one(
    method: str,
    seed: int,
    train_episodes: Sequence[Sequence[float]],
    test_episodes: Sequence[Sequence[float]],
    env_config,
    args: argparse.Namespace,
) -> Dict[str, object]:
    if method == "tabular":
        cfg = QLearningConfig(
            num_epochs=args.tabular_num_epochs,
            alpha=args.tabular_alpha,
            gamma=args.tabular_gamma,
            epsilon_start=args.tabular_epsilon_start,
            epsilon_end=args.tabular_epsilon_end,
            epsilon_decay=args.tabular_epsilon_decay,
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
        wall = time.perf_counter() - start
    else:
        cfg = ApproxQLearningConfig(
            num_epochs=args.approx_num_epochs,
            alpha=args.approx_alpha,
            gamma=args.approx_gamma,
            epsilon_start=args.approx_epsilon_start,
            epsilon_end=args.approx_epsilon_end,
            epsilon_decay=args.approx_epsilon_decay,
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
        wall = time.perf_counter() - start

    return {
        "seed": seed,
        "best_epoch": int(result["best_epoch"]),
        "best_test_metrics": result["best_test_metrics"],
        "training_wall_time_sec": wall,
    }


def format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def write_markdown(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    headers = [
        "Method",
        "Slicing Setting",
        "Train/Test Episodes",
        "Avg Return",
        "Energy",
        "Latency",
        "Miss Rate",
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
                    str(row["setting_name"]),
                    f"{int(row['num_train_episodes'])}/{int(row['num_test_episodes'])}",
                    format_mean_std(float(row["avg_episode_return_mean"]), float(row["avg_episode_return_std"])),
                    format_mean_std(float(row["avg_step_energy_mean"]), float(row["avg_step_energy_std"])),
                    format_mean_std(float(row["avg_step_latency_mean"]), float(row["avg_step_latency_std"])),
                    format_mean_std(float(row["miss_rate_mean"]), float(row["miss_rate_std"])),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    settings = parse_settings(args.settings)
    env_config = replace(
        DEFAULT_ENV_CONFIG,
        episode_length=args.episode_length,
        alpha_energy=args.alpha_energy,
        beta_latency=args.beta_latency,
        gamma_miss=args.gamma_miss,
    )

    train_series = load_workload_trace(args.trace_path, split="train")
    test_series = load_workload_trace(args.trace_path, split="test")

    records: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []

    for setting in settings:
        train_episodes = build_episodes(
            train_series,
            args.episode_length,
            stride=int(setting["train_stride"]),
            drop_last=True,
        )[: int(setting["max_train_episodes"])]
        test_episodes = build_episodes(
            test_series,
            args.episode_length,
            stride=int(setting["test_stride"]),
            drop_last=True,
        )[: int(setting["max_test_episodes"])]
        if not train_episodes or not test_episodes:
            raise RuntimeError(f"Insufficient episodes for setting {setting['name']}")

        runs: List[Dict[str, object]] = []
        for seed in seeds:
            runs.append(
                run_one(
                    method=args.method,
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
                "setting": setting,
                "num_train_episodes": len(train_episodes),
                "num_test_episodes": len(test_episodes),
                "runs": runs,
                "aggregate": stats,
            }
        )
        rows.append(
            {
                "method": args.method,
                "setting_name": setting["name"],
                "num_train_episodes": len(train_episodes),
                "num_test_episodes": len(test_episodes),
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
            f"Completed setting={setting['name']} "
            f"train_eps={len(train_episodes)} test_eps={len(test_episodes)} "
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
            "seeds": seeds,
            "method": args.method,
            "reward_weights": {
                "alpha_energy": args.alpha_energy,
                "beta_latency": args.beta_latency,
                "gamma_miss": args.gamma_miss,
            },
            "settings": settings,
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
