#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

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

from approx_q import LinearQApproximator  # noqa: E402
from config import DEFAULT_ENV_CONFIG  # noqa: E402
from data_loader import build_episodes, load_workload_trace  # noqa: E402
from env import EnergySchedulingEnv, Observation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final publication-style artifacts.")
    parser.add_argument("--trace-path", type=Path, default=Path("data/processed/workload_trace_120.csv"))
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--stride", type=int, default=288)
    parser.add_argument("--max-test-episodes", type=int, default=6)
    parser.add_argument("--phase3-json", type=Path, default=Path("results/phase3_multiseed_120.json"))
    parser.add_argument("--reward-json", type=Path, default=Path("results/reward_sensitivity_120.json"))
    parser.add_argument("--hyper-json", type=Path, default=Path("results/hyperparam_sensitivity_120.json"))
    parser.add_argument("--generalization-json", type=Path, default=Path("results/generalization_check_120.json"))
    parser.add_argument("--tabular-history-csv", type=Path, default=Path("results/tabular_q_learning_120/history.csv"))
    parser.add_argument("--approx-history-csv", type=Path, default=Path("results/approx_q_learning_120/history.csv"))
    parser.add_argument("--tabular-q-table", type=Path, default=Path("results/tabular_q_learning_120/q_table_best.npy"))
    parser.add_argument("--approx-weights", type=Path, default=Path("results/approx_q_learning_120/weights_best.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_figures"))
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_history(path: Path) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {
        "epoch": [],
        "eval_train_avg_return": [],
        "eval_test_avg_return": [],
    }
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out["epoch"].append(float(row["epoch"]))
            out["eval_train_avg_return"].append(float(row["eval_train_avg_return"]))
            out["eval_test_avg_return"].append(float(row["eval_test_avg_return"]))
    return out


def _always_low(_: Observation, __: Dict[str, float]) -> int:
    return 0


def evaluate_action_frequency(
    workload_episodes: Sequence[Sequence[float]],
    policy: Callable[[Observation, Dict[str, float]], int],
) -> Dict[str, float]:
    counts = np.zeros(len(DEFAULT_ENV_CONFIG.action_names), dtype=np.float64)
    total_steps = 0.0
    for ep in workload_episodes:
        env = EnergySchedulingEnv(ep, config=DEFAULT_ENV_CONFIG)
        obs, info = env.reset()
        done = False
        while not done:
            action = int(policy(obs, info))
            counts[action] += 1.0
            obs, _, done, info = env.step(action)
            total_steps += 1.0

    if total_steps <= 0:
        raise RuntimeError("No steps collected for action frequency.")
    return {
        DEFAULT_ENV_CONFIG.action_names[i]: float(counts[i] / total_steps)
        for i in range(len(DEFAULT_ENV_CONFIG.action_names))
    }


def write_summary_markdown(
    output_path: Path,
    phase3_rows: Sequence[Dict[str, object]],
    reward_rows: Sequence[Dict[str, object]],
    hyper_rows: Sequence[Dict[str, object]],
    generalization_rows: Sequence[Dict[str, object]],
) -> None:
    lines: List[str] = []
    lines.append("# Final Artifact Summary")
    lines.append("")
    lines.append("## Multi-seed Comparison")
    lines.append("")
    lines.append("| Method | Avg Return | Energy | Latency | Miss Rate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in phase3_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    f"{float(row['avg_episode_return_mean']):.3f} +/- {float(row['avg_episode_return_std']):.3f}",
                    f"{float(row['avg_step_energy_mean']):.3f} +/- {float(row['avg_step_energy_std']):.3f}",
                    f"{float(row['avg_step_latency_mean']):.3f} +/- {float(row['avg_step_latency_std']):.3f}",
                    f"{float(row['miss_rate_mean']):.3f} +/- {float(row['miss_rate_std']):.3f}",
                ]
            )
            + " |"
        )

    if reward_rows:
        best_reward = reward_rows[0]
        lines.append("")
        lines.append("## Best Reward Setting")
        lines.append("")
        lines.append(
            f"- method={best_reward['method']}, "
            f"(alpha,beta,gamma)=({float(best_reward['alpha_energy']):.2f}, {float(best_reward['beta_latency']):.2f}, {float(best_reward['gamma_miss']):.2f}), "
            f"return={float(best_reward['avg_episode_return_mean']):.3f} +/- {float(best_reward['avg_episode_return_std']):.3f}"
        )

    if hyper_rows:
        best_h = hyper_rows[0]
        lines.append("")
        lines.append("## Best Hyperparameter Setting")
        lines.append("")
        lines.append(
            f"- method={best_h['method']}, "
            f"(alpha,epsilon_decay,gamma)=({float(best_h['alpha']):.3f}, {float(best_h['epsilon_decay']):.3f}, {float(best_h['gamma']):.3f}), "
            f"return={float(best_h['avg_episode_return_mean']):.3f} +/- {float(best_h['avg_episode_return_std']):.3f}"
        )

    if generalization_rows:
        lines.append("")
        lines.append("## Generalization Check")
        lines.append("")
        for row in generalization_rows:
            lines.append(
                f"- {row['setting_name']}: return={float(row['avg_episode_return_mean']):.3f} +/- {float(row['avg_episode_return_std']):.3f} "
                f"(train/test episodes={int(row['num_train_episodes'])}/{int(row['num_test_episodes'])})"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to generate final artifacts: {exc}") from exc

    phase3 = load_json(args.phase3_json)
    reward = load_json(args.reward_json)
    hyper = load_json(args.hyper_json)
    generalization = load_json(args.generalization_json)

    phase3_rows: List[Dict[str, object]] = phase3["rows"]  # type: ignore[index]
    reward_rows: List[Dict[str, object]] = reward["rows"]  # type: ignore[index]
    hyper_rows: List[Dict[str, object]] = hyper["rows"]  # type: ignore[index]
    generalization_rows: List[Dict[str, object]] = generalization["rows"]  # type: ignore[index]

    tab_hist = load_history(args.tabular_history_csv)
    app_hist = load_history(args.approx_history_csv)

    # Figure 1: learning curves
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tab_hist["epoch"], tab_hist["eval_test_avg_return"], label="Tabular Q (test)", linewidth=2.2, color="#1f77b4")
    ax.plot(app_hist["epoch"], app_hist["eval_test_avg_return"], label="Approx Q (test)", linewidth=2.2, color="#ff7f0e")
    ax.plot(tab_hist["epoch"], tab_hist["eval_train_avg_return"], label="Tabular Q (train)", linewidth=1.3, linestyle="--", color="#1f77b4", alpha=0.6)
    ax.plot(app_hist["epoch"], app_hist["eval_train_avg_return"], label="Approx Q (train)", linewidth=1.3, linestyle="--", color="#ff7f0e", alpha=0.6)
    ax.set_title("Learning Curves (120-shard Dataset)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Episode Return")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "learning_curves_tabular_vs_approx.png", dpi=180)
    plt.close(fig)

    # Figure 2: baseline vs RL comparison (return with std error bars)
    methods = [str(r["method"]) for r in phase3_rows]
    returns = [float(r["avg_episode_return_mean"]) for r in phase3_rows]
    return_err = [float(r["avg_episode_return_std"]) for r in phase3_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4e79a7", "#f28e2b", "#59a14f"][: len(methods)]
    ax.bar(methods, returns, yerr=return_err, capsize=4, color=colors, alpha=0.9)
    ax.set_title("Baseline vs RL (Multi-seed Mean/Std Return)")
    ax.set_ylabel("Average Episode Return")
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "baseline_vs_rl_return.png", dpi=180)
    plt.close(fig)

    # Figure 3: energy-latency tradeoff
    xs = [float(r["avg_step_energy_mean"]) for r in phase3_rows]
    ys = [float(r["avg_step_latency_mean"]) for r in phase3_rows]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(xs, ys, s=120, color="#d62728", alpha=0.9)
    for i, label in enumerate(methods):
        ax.annotate(label, (xs[i], ys[i]), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_title("Energy-Latency Tradeoff")
    ax.set_xlabel("Avg Step Energy")
    ax.set_ylabel("Avg Step Latency")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "energy_latency_tradeoff.png", dpi=180)
    plt.close(fig)

    # Figure 4: reward sensitivity (sorted by return)
    fig, ax = plt.subplots(figsize=(10, 5))
    reward_labels = [
        f"({float(r['alpha_energy']):.1f},{float(r['beta_latency']):.1f},{float(r['gamma_miss']):.1f})"
        for r in reward_rows
    ]
    reward_ret = [float(r["avg_episode_return_mean"]) for r in reward_rows]
    reward_err = [float(r["avg_episode_return_std"]) for r in reward_rows]
    ax.errorbar(range(len(reward_labels)), reward_ret, yerr=reward_err, fmt="o-", capsize=4, linewidth=2, color="#9467bd")
    ax.set_xticks(range(len(reward_labels)))
    ax.set_xticklabels(reward_labels, rotation=20, ha="right")
    ax.set_title("Reward Sensitivity (approx)")
    ax.set_xlabel("(alpha_energy, beta_latency, gamma_miss)")
    ax.set_ylabel("Average Episode Return")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "reward_sensitivity_return.png", dpi=180)
    plt.close(fig)

    # Figure 5: hyperparameter sensitivity
    fig, ax = plt.subplots(figsize=(10, 5))
    hyper_labels = [
        f"({float(r['alpha']):.3f},{float(r['epsilon_decay']):.3f},{float(r['gamma']):.3f})"
        for r in hyper_rows
    ]
    hyper_ret = [float(r["avg_episode_return_mean"]) for r in hyper_rows]
    hyper_err = [float(r["avg_episode_return_std"]) for r in hyper_rows]
    ax.errorbar(range(len(hyper_labels)), hyper_ret, yerr=hyper_err, fmt="o-", capsize=4, linewidth=2, color="#17becf")
    ax.set_xticks(range(len(hyper_labels)))
    ax.set_xticklabels(hyper_labels, rotation=20, ha="right")
    ax.set_title("Hyperparameter Sensitivity (approx)")
    ax.set_xlabel("(alpha, epsilon_decay, gamma)")
    ax.set_ylabel("Average Episode Return")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "hyperparam_sensitivity_return.png", dpi=180)
    plt.close(fig)

    # Figure 6: policy action frequency summary
    test_series = load_workload_trace(args.trace_path, split="test")
    test_episodes = build_episodes(test_series, args.episode_length, stride=args.stride, drop_last=True)[: args.max_test_episodes]
    if not test_episodes:
        raise RuntimeError("No test episodes available for action-frequency summary.")

    q_table = np.load(args.tabular_q_table)
    weights = np.load(args.approx_weights)
    approximator = LinearQApproximator(DEFAULT_ENV_CONFIG)

    def tabular_policy(obs: Observation, _: Dict[str, float]) -> int:
        return int(np.argmax(q_table[obs]))

    def approx_policy(obs: Observation, info: Dict[str, float]) -> int:
        return approximator.greedy_action(weights, obs, info)

    action_freq = {
        "baseline_always_low": evaluate_action_frequency(test_episodes, _always_low),
        "tabular_q_best": evaluate_action_frequency(test_episodes, tabular_policy),
        "approx_q_best": evaluate_action_frequency(test_episodes, approx_policy),
    }

    methods_action = list(action_freq.keys())
    action_names = list(DEFAULT_ENV_CONFIG.action_names)
    x = np.arange(len(methods_action))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, aname in enumerate(action_names):
        vals = [action_freq[m][aname] for m in methods_action]
        ax.bar(x + (i - 1) * width, vals, width=width, label=aname)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_action, rotation=12, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Policy Action Frequency on Test Episodes")
    ax.set_ylabel("Action Ratio")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "policy_action_frequency.png", dpi=180)
    plt.close(fig)

    with (args.output_dir / "policy_action_frequency.json").open("w", encoding="utf-8") as f:
        json.dump(action_freq, f, indent=2)

    with (args.output_dir / "generalization_check_table.md").open("w", encoding="utf-8") as f:
        f.write("| Setting | Train/Test Episodes | Avg Return |\n")
        f.write("| --- | --- | --- |\n")
        for row in generalization_rows:
            f.write(
                f"| {row['setting_name']} | {int(row['num_train_episodes'])}/{int(row['num_test_episodes'])} | "
                f"{float(row['avg_episode_return_mean']):.3f} +/- {float(row['avg_episode_return_std']):.3f} |\n"
            )

    write_summary_markdown(
        output_path=args.output_dir / "final_summary_table.md",
        phase3_rows=phase3_rows,
        reward_rows=reward_rows,
        hyper_rows=hyper_rows,
        generalization_rows=generalization_rows,
    )

    print(f"Wrote final figures/tables to {args.output_dir}")


if __name__ == "__main__":
    main()
