#!/usr/bin/env python3
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from approx_q import LinearQApproximator  # noqa: E402
from baselines import (  # noqa: E402
    always_high_policy,
    always_low_policy,
    always_medium_policy,
    evaluate_policy,
    threshold_policy_factory,
)
from config import DEFAULT_ENV_CONFIG  # noqa: E402
from data_loader import build_episodes, load_workload_trace  # noqa: E402
from env import Observation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deadline-threshold stress test for policy miss-rate sensitivity.")
    parser.add_argument("--trace-path", type=Path, default=Path("data/processed/workload_trace_120.csv"))
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--stride", type=int, default=288)
    parser.add_argument("--max-test-episodes", type=int, default=6)
    parser.add_argument("--deadline-thresholds", type=str, default="2.5,1.5,1.0,0.75,0.5")
    parser.add_argument("--threshold-workload", type=float, default=0.65)
    parser.add_argument("--threshold-battery", type=float, default=0.10)
    parser.add_argument("--tabular-q-table", type=Path, default=Path("results/tabular_q_learning_120/q_table_best.npy"))
    parser.add_argument("--approx-weights", type=Path, default=Path("results/approx_q_learning_120/weights_best.npy"))
    parser.add_argument("--output-prefix", type=Path, default=Path("results/deadline_stress_test_120"))
    return parser.parse_args()


def parse_thresholds(raw: str) -> List[float]:
    out: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("No deadline thresholds provided.")
    return out


def format_float(value: float, precision: int = 3) -> str:
    return f"{value:.{precision}f}"


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    headers = [
        "Deadline Threshold",
        "Policy",
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
                    format_float(float(row["deadline_queue_threshold"]), 2),
                    str(row["policy"]),
                    format_float(float(row["avg_episode_return"])),
                    format_float(float(row["avg_step_energy"])),
                    format_float(float(row["avg_step_latency"])),
                    format_float(float(row["miss_rate"])),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    deadlines = parse_thresholds(args.deadline_thresholds)

    q_table = np.load(args.tabular_q_table)
    approx_weights = np.load(args.approx_weights)
    approximator = LinearQApproximator(DEFAULT_ENV_CONFIG)

    test_series = load_workload_trace(args.trace_path, split="test")
    test_episodes = build_episodes(
        test_series,
        episode_length=args.episode_length,
        stride=args.stride,
        drop_last=True,
    )[: args.max_test_episodes]
    if not test_episodes:
        raise RuntimeError("No test episodes available.")

    def tabular_policy(obs: Observation, _: Dict[str, float]) -> int:
        return int(np.argmax(q_table[obs]))

    def approx_policy(obs: Observation, info: Dict[str, float]) -> int:
        return approximator.greedy_action(approx_weights, obs, info)

    policies: Dict[str, Callable[[Observation, Dict[str, float]], int]] = {
        "always_low": always_low_policy,
        "always_medium": always_medium_policy,
        "always_high": always_high_policy,
        f"threshold(w={args.threshold_workload:.2f})": threshold_policy_factory(
            workload_threshold=args.threshold_workload,
            battery_threshold=args.threshold_battery,
        ),
        "tabular_q_best": tabular_policy,
        "approx_q_best": approx_policy,
    }

    records: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    for deadline in deadlines:
        env_config = replace(
            DEFAULT_ENV_CONFIG,
            episode_length=args.episode_length,
            deadline_queue_threshold=deadline,
        )
        policy_metrics: Dict[str, Dict[str, float]] = {}
        for name, policy in policies.items():
            metrics = evaluate_policy(test_episodes, policy, config=env_config)
            policy_metrics[name] = metrics
            rows.append(
                {
                    "deadline_queue_threshold": deadline,
                    "policy": name,
                    "avg_episode_return": metrics["avg_episode_return"],
                    "avg_step_energy": metrics["avg_step_energy"],
                    "avg_step_latency": metrics["avg_step_latency"],
                    "miss_rate": metrics["miss_rate"],
                }
            )
        records.append(
            {
                "deadline_queue_threshold": deadline,
                "policy_metrics": policy_metrics,
            }
        )
        print(f"Completed deadline={deadline:.2f}")

    rows.sort(
        key=lambda r: (float(r["deadline_queue_threshold"]), -float(r["avg_episode_return"])),
    )

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{output_prefix}.json")
    md_path = Path(f"{output_prefix}.md")
    payload = {
        "config": {
            "trace_path": str(args.trace_path),
            "episode_length": args.episode_length,
            "stride": args.stride,
            "max_test_episodes": args.max_test_episodes,
            "deadline_thresholds": deadlines,
            "tabular_q_table": str(args.tabular_q_table),
            "approx_weights": str(args.approx_weights),
        },
        "records": records,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
