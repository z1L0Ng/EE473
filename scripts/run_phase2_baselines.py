#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from baselines import (  # noqa: E402
    always_high_policy,
    always_low_policy,
    always_medium_policy,
    evaluate_policy,
    threshold_policy_factory,
)
from config import DEFAULT_ENV_CONFIG  # noqa: E402
from data_loader import build_episodes, load_workload_trace  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 baseline experiments.")
    parser.add_argument("--trace-path", type=Path, default=Path("data/processed/workload_trace.csv"))
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--stride", type=int, default=288)
    parser.add_argument("--train-stride", type=int, default=None)
    parser.add_argument("--test-stride", type=int, default=None)
    parser.add_argument("--max-train-episodes", type=int, default=20)
    parser.add_argument("--max-test-episodes", type=int, default=10)
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--battery-threshold", type=float, default=0.10)
    parser.add_argument("--output-prefix", type=Path, default=Path("results/phase2_baselines"))
    return parser.parse_args()


def threshold_grid(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("threshold-step must be positive")
    values = []
    cur = start
    while cur <= end + 1e-9:
        values.append(round(cur, 6))
        cur += step
    return values


def flatten_metrics(split: str, policy_name: str, metrics: Dict[str, float]) -> Dict[str, float]:
    row: Dict[str, float] = {"split": split, "policy": policy_name}
    row.update(metrics)
    return row


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, test_metrics: Dict[str, Dict[str, float]]) -> None:
    headers = [
        "Policy",
        "Avg Return",
        "Avg Energy",
        "Avg Latency",
        "Miss Rate",
        "Avg Ep Length",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for policy_name, metrics in sorted(
        test_metrics.items(), key=lambda item: item[1]["avg_episode_return"], reverse=True
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    policy_name,
                    f"{metrics['avg_episode_return']:.3f}",
                    f"{metrics['avg_step_energy']:.3f}",
                    f"{metrics['avg_step_latency']:.3f}",
                    f"{metrics['miss_rate']:.3f}",
                    f"{metrics['avg_episode_length']:.1f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = replace(DEFAULT_ENV_CONFIG, episode_length=args.episode_length)
    train_stride = args.train_stride if args.train_stride is not None else args.stride
    test_stride = args.test_stride if args.test_stride is not None else args.stride

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

    static_policies = {
        "always_low": always_low_policy,
        "always_medium": always_medium_policy,
        "always_high": always_high_policy,
    }

    train_results: Dict[str, Dict[str, float]] = {}
    test_results: Dict[str, Dict[str, float]] = {}

    for name, policy in static_policies.items():
        train_results[name] = evaluate_policy(train_episodes, policy, config=config)
        test_results[name] = evaluate_policy(test_episodes, policy, config=config)

    best_threshold: Tuple[float, Dict[str, float]] = (-1.0, {})
    all_threshold_train: List[Dict[str, float]] = []
    for wt in threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step):
        policy = threshold_policy_factory(workload_threshold=wt, battery_threshold=args.battery_threshold)
        metrics = evaluate_policy(train_episodes, policy, config=config)
        all_threshold_train.append({"workload_threshold": wt, **metrics})
        if best_threshold[0] < 0 or metrics["avg_episode_return"] > best_threshold[1]["avg_episode_return"]:
            best_threshold = (wt, metrics)

    selected_wt = best_threshold[0]
    selected_threshold_policy = threshold_policy_factory(
        workload_threshold=selected_wt,
        battery_threshold=args.battery_threshold,
    )
    train_results[f"threshold(w={selected_wt:.2f})"] = evaluate_policy(
        train_episodes, selected_threshold_policy, config=config
    )
    test_results[f"threshold(w={selected_wt:.2f})"] = evaluate_policy(
        test_episodes, selected_threshold_policy, config=config
    )

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    result_obj = {
        "config": {
            "trace_path": str(args.trace_path),
            "episode_length": args.episode_length,
            "stride": args.stride,
            "train_stride": train_stride,
            "test_stride": test_stride,
            "num_train_episodes": len(train_episodes),
            "num_test_episodes": len(test_episodes),
            "threshold_search": {
                "min": args.threshold_min,
                "max": args.threshold_max,
                "step": args.threshold_step,
                "battery_threshold": args.battery_threshold,
                "selected_workload_threshold": selected_wt,
            },
        },
        "train": train_results,
        "test": test_results,
        "threshold_train_grid": all_threshold_train,
    }

    json_path = Path(f"{output_prefix}.json")
    csv_path = Path(f"{output_prefix}.csv")
    md_path = Path(f"{output_prefix}.md")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result_obj, f, indent=2)

    rows: List[Dict[str, float]] = []
    for split_name, split_metrics in (("train", train_results), ("test", test_results)):
        for policy_name, metrics in split_metrics.items():
            rows.append(flatten_metrics(split_name, policy_name, metrics))
    write_csv(csv_path, rows)
    write_markdown_table(md_path, test_results)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Selected threshold (train best avg return): {selected_wt:.2f}")
    print("\n[Test ranking by avg return]")
    for policy_name, metrics in sorted(
        test_results.items(), key=lambda item: item[1]["avg_episode_return"], reverse=True
    ):
        print(
            f"{policy_name:20s} return={metrics['avg_episode_return']:.3f} "
            f"energy={metrics['avg_step_energy']:.3f} "
            f"latency={metrics['avg_step_latency']:.3f} "
            f"miss={metrics['miss_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
