#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline/tabular/approx RL results.")
    parser.add_argument("--baseline-json", type=Path, default=Path("results/phase2_baselines.json"))
    parser.add_argument("--tabular-json", type=Path, default=Path("results/tabular_q_learning/summary.json"))
    parser.add_argument("--approx-json", type=Path, default=Path("results/approx_q_learning/summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/phase3_comparison.md"))
    parser.add_argument("--output-json", type=Path, default=Path("results/phase3_comparison.json"))
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline_json)
    tabular = load_json(args.tabular_json)
    approx = load_json(args.approx_json)

    baseline_test = baseline["test"]  # type: ignore[index]
    baseline_best_name = max(
        baseline_test.keys(), key=lambda k: baseline_test[k]["avg_episode_return"]  # type: ignore[index]
    )
    baseline_best = baseline_test[baseline_best_name]  # type: ignore[index]
    tabular_best = tabular["best_test_metrics"]  # type: ignore[index]
    approx_best = approx["best_test_metrics"]  # type: ignore[index]

    rows: List[Dict[str, object]] = [
        {
            "method": f"baseline:{baseline_best_name}",
            "avg_episode_return": baseline_best["avg_episode_return"],
            "avg_step_energy": baseline_best["avg_step_energy"],
            "avg_step_latency": baseline_best["avg_step_latency"],
            "miss_rate": baseline_best["miss_rate"],
        },
        {
            "method": "tabular_q_learning(best)",
            "avg_episode_return": tabular_best["avg_episode_return"],
            "avg_step_energy": tabular_best["avg_step_energy"],
            "avg_step_latency": tabular_best["avg_step_latency"],
            "miss_rate": tabular_best["miss_rate"],
        },
        {
            "method": "linear_approx_q_learning(best)",
            "avg_episode_return": approx_best["avg_episode_return"],
            "avg_step_energy": approx_best["avg_step_energy"],
            "avg_step_latency": approx_best["avg_step_latency"],
            "miss_rate": approx_best["miss_rate"],
        },
    ]

    rows.sort(key=lambda r: float(r["avg_episode_return"]), reverse=True)
    best_return = float(rows[0]["avg_episode_return"])
    for row in rows:
        row["return_gap_vs_best"] = float(row["avg_episode_return"]) - best_return

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)

    headers = [
        "Method",
        "Avg Return",
        "Energy",
        "Latency",
        "Miss Rate",
        "Return Gap vs Best",
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
                    f"{float(row['avg_episode_return']):.3f}",
                    f"{float(row['avg_step_energy']):.3f}",
                    f"{float(row['avg_step_latency']):.3f}",
                    f"{float(row['miss_rate']):.3f}",
                    f"{float(row['return_gap_vs_best']):.3f}",
                ]
            )
            + " |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print("\n[Ranking]")
    for row in rows:
        print(
            f"{str(row['method']):34s} return={float(row['avg_episode_return']):.3f} "
            f"energy={float(row['avg_step_energy']):.3f} "
            f"latency={float(row['avg_step_latency']):.3f} "
            f"miss={float(row['miss_rate']):.3f}"
        )


if __name__ == "__main__":
    main()

