#!/usr/bin/env python3
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from baselines import always_low_policy, evaluate_policy  # noqa: E402
from config import DEFAULT_ENV_CONFIG  # noqa: E402
from data_loader import build_episodes, load_workload_trace  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reward-sensitivity context table with an always-low baseline for each reward setting."
    )
    parser.add_argument("--trace-path", type=Path, default=Path("data/processed/workload_trace_120.csv"))
    parser.add_argument("--reward-json", type=Path, default=Path("results/reward_sensitivity_120.json"))
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--stride", type=int, default=288)
    parser.add_argument("--max-test-episodes", type=int, default=6)
    parser.add_argument("--output-prefix", type=Path, default=Path("results/reward_sensitivity_context_120"))
    return parser.parse_args()


def reward_key(alpha_energy: float, beta_latency: float, gamma_miss: float) -> Tuple[float, float, float]:
    return (round(alpha_energy, 10), round(beta_latency, 10), round(gamma_miss, 10))


def format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    headers = [
        "Method",
        "(alpha,beta,gamma)",
        "Method Return",
        "AlwaysLow Return",
        "Gap vs AlwaysLow",
        "Relative Gain",
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
                    f"({float(row['alpha_energy']):.2f}, {float(row['beta_latency']):.2f}, {float(row['gamma_miss']):.2f})",
                    format_mean_std(float(row["avg_episode_return_mean"]), float(row["avg_episode_return_std"])),
                    f"{float(row['always_low_return']):.3f}",
                    f"{float(row['gap_vs_always_low']):+.3f}",
                    f"{float(row['relative_gain_pct']):+.2f}%",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reward_payload = json.loads(args.reward_json.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = reward_payload["rows"]

    test_series = load_workload_trace(args.trace_path, split="test")
    test_episodes = build_episodes(
        test_series,
        episode_length=args.episode_length,
        stride=args.stride,
        drop_last=True,
    )[: args.max_test_episodes]
    if not test_episodes:
        raise RuntimeError("No test episodes available.")

    baseline_map: Dict[Tuple[float, float, float], Dict[str, float]] = {}
    for row in rows:
        key = reward_key(
            float(row["alpha_energy"]),
            float(row["beta_latency"]),
            float(row["gamma_miss"]),
        )
        if key in baseline_map:
            continue
        env_config = replace(
            DEFAULT_ENV_CONFIG,
            episode_length=args.episode_length,
            alpha_energy=key[0],
            beta_latency=key[1],
            gamma_miss=key[2],
        )
        baseline_map[key] = evaluate_policy(test_episodes, always_low_policy, config=env_config)

    enriched_rows: List[Dict[str, object]] = []
    for row in rows:
        key = reward_key(
            float(row["alpha_energy"]),
            float(row["beta_latency"]),
            float(row["gamma_miss"]),
        )
        baseline_return = float(baseline_map[key]["avg_episode_return"])
        method_return = float(row["avg_episode_return_mean"])
        gap = method_return - baseline_return
        denom = max(abs(baseline_return), 1e-9)
        enriched = dict(row)
        enriched["always_low_return"] = baseline_return
        enriched["gap_vs_always_low"] = gap
        enriched["relative_gain_pct"] = 100.0 * gap / denom
        enriched_rows.append(enriched)

    enriched_rows.sort(
        key=lambda r: (str(r["method"]), -float(r["gap_vs_always_low"])),
    )

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{output_prefix}.json")
    md_path = Path(f"{output_prefix}.md")

    payload = {
        "config": {
            "trace_path": str(args.trace_path),
            "reward_json": str(args.reward_json),
            "episode_length": args.episode_length,
            "stride": args.stride,
            "max_test_episodes": args.max_test_episodes,
            "baseline": "always_low",
        },
        "rows": enriched_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, enriched_rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
