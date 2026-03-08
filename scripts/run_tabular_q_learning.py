#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp/matplotlib")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache_dir = Path("/tmp/xdg-cache")
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)

from config import DEFAULT_ENV_CONFIG  # noqa: E402
from data_loader import build_episodes, load_workload_trace  # noqa: E402
from q_learning import QLearningConfig, train_tabular_q_learning  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate tabular Q-learning.")
    parser.add_argument("--trace-path", type=Path, default=Path("data/processed/workload_trace.csv"))
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--stride", type=int, default=288)
    parser.add_argument("--max-train-episodes", type=int, default=20)
    parser.add_argument("--max-test-episodes", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tabular_q_learning"))
    return parser.parse_args()


def write_history_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot_learning_curve(history: List[Dict[str, float]], out_path: Path) -> str:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return "skipped (matplotlib not available)"

    epochs = [int(r["epoch"]) for r in history]
    train_returns = [float(r["eval_train_avg_return"]) for r in history]
    test_returns = [float(r["eval_test_avg_return"]) for r in history]
    epsilons = [float(r["epsilon"]) for r in history]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs, train_returns, label="Eval Train Return", color="#0d6efd", linewidth=2.0)
    ax1.plot(epochs, test_returns, label="Eval Test Return", color="#198754", linewidth=2.0)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg Episode Return")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, epsilons, label="Epsilon", color="#dc3545", linestyle="--", linewidth=1.8)
    ax2.set_ylabel("Epsilon")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return "saved"


def main() -> None:
    args = parse_args()
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
        raise RuntimeError("Not enough train/test episodes. Regenerate data or reduce episode length.")

    algo_config = QLearningConfig(
        num_epochs=args.num_epochs,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        eval_every=args.eval_every,
        seed=args.seed,
    )

    result = train_tabular_q_learning(
        train_episodes=train_episodes,
        test_episodes=test_episodes,
        env_config=env_config,
        algo_config=algo_config,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = result["history"]  # type: ignore[assignment]
    final_train: Dict[str, float] = result["final_train_metrics"]  # type: ignore[assignment]
    final_test: Dict[str, float] = result["final_test_metrics"]  # type: ignore[assignment]
    best_train: Dict[str, float] = result["best_train_metrics"]  # type: ignore[assignment]
    best_test: Dict[str, float] = result["best_test_metrics"]  # type: ignore[assignment]
    best_epoch = int(result["best_epoch"])  # type: ignore[arg-type]

    q_table: np.ndarray = result["q_table"]  # type: ignore[assignment]
    best_q_table: np.ndarray = result["best_q_table"]  # type: ignore[assignment]

    history_csv = output_dir / "history.csv"
    summary_json = output_dir / "summary.json"
    q_table_npy = output_dir / "q_table_final.npy"
    best_q_table_npy = output_dir / "q_table_best.npy"
    curve_png = output_dir / "learning_curve.png"

    write_history_csv(history_csv, history)
    np.save(q_table_npy, q_table)
    np.save(best_q_table_npy, best_q_table)
    plot_status = maybe_plot_learning_curve(history, curve_png)

    summary = {
        "config": {
            "trace_path": str(args.trace_path),
            "episode_length": args.episode_length,
            "stride": args.stride,
            "num_train_episodes": len(train_episodes),
            "num_test_episodes": len(test_episodes),
            "algorithm": asdict(algo_config),
        },
        "best_epoch": best_epoch,
        "final_train_metrics": final_train,
        "final_test_metrics": final_test,
        "best_train_metrics": best_train,
        "best_test_metrics": best_test,
        "artifacts": {
            "history_csv": str(history_csv),
            "q_table_final_npy": str(q_table_npy),
            "q_table_best_npy": str(best_q_table_npy),
            "learning_curve_png": str(curve_png),
            "learning_curve_status": plot_status,
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {history_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {q_table_npy}")
    print(f"Wrote {best_q_table_npy}")
    print(f"Learning curve: {plot_status}")
    print("\n[Best policy metrics]")
    print(
        f"best_epoch={best_epoch} "
        f"test_return={best_test['avg_episode_return']:.3f} "
        f"test_energy={best_test['avg_step_energy']:.3f} "
        f"test_latency={best_test['avg_step_latency']:.3f} "
        f"test_miss={best_test['miss_rate']:.3f}"
    )
    print("\n[Final policy metrics]")
    print(
        f"final_test_return={final_test['avg_episode_return']:.3f} "
        f"final_test_energy={final_test['avg_step_energy']:.3f} "
        f"final_test_latency={final_test['avg_step_latency']:.3f} "
        f"final_test_miss={final_test['miss_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
