#!/usr/bin/env python3
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from baselines import (  # noqa: E402
    always_high_policy,
    always_low_policy,
    evaluate_policy,
    threshold_policy_factory,
)
from config import DEFAULT_ENV_CONFIG  # noqa: E402
from data_loader import build_episodes, load_workload_trace  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-1 sanity checks on fixed policies.")
    parser.add_argument("--trace-path", type=Path, default=Path("data/processed/workload_trace.csv"))
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--stride", type=int, default=288)
    parser.add_argument("--max-train-episodes", type=int, default=20)
    parser.add_argument("--max-test-episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/phase1_sanity.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(DEFAULT_ENV_CONFIG, episode_length=args.episode_length)

    train_series = load_workload_trace(args.trace_path, split="train")
    test_series = load_workload_trace(args.trace_path, split="test")

    train_episodes = build_episodes(train_series, args.episode_length, stride=args.stride, drop_last=True)
    test_episodes = build_episodes(test_series, args.episode_length, stride=args.stride, drop_last=True)

    if not train_episodes:
        raise RuntimeError("No train episodes available. Check episode-length/stride or preprocessing output.")
    if not test_episodes:
        raise RuntimeError("No test episodes available. Check episode-length/stride or preprocessing output.")

    train_episodes = train_episodes[: args.max_train_episodes]
    test_episodes = test_episodes[: args.max_test_episodes]

    policies = {
        "always_low": always_low_policy,
        "always_high": always_high_policy,
        "threshold": threshold_policy_factory(workload_threshold=0.6, battery_threshold=0.1),
    }

    results = {
        "config": {
            "trace_path": str(args.trace_path),
            "episode_length": args.episode_length,
            "stride": args.stride,
            "num_train_episodes": len(train_episodes),
            "num_test_episodes": len(test_episodes),
        },
        "train": {},
        "test": {},
    }

    for name, policy in policies.items():
        results["train"][name] = evaluate_policy(train_episodes, policy, config=config)
        results["test"][name] = evaluate_policy(test_episodes, policy, config=config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote sanity results to {args.output}")
    for split in ("train", "test"):
        print(f"\n[{split}]")
        for name, metrics in results[split].items():
            print(
                f"{name:12s} return={metrics['avg_episode_return']:.3f} "
                f"energy={metrics['avg_step_energy']:.3f} "
                f"latency={metrics['avg_step_latency']:.3f} "
                f"miss={metrics['miss_rate']:.3f}"
            )


if __name__ == "__main__":
    main()

