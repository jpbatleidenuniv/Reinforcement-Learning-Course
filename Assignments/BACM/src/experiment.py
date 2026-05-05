"""
experiment.py

Runs REINFORCE, AC, and A2C for N repetitions and plots their
evaluation curves (mean ± std) over environment steps in a single figure.

Usage:
    python experiment.py --config base_ac --n_reps 5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from agent import ReinforceAgent, ACAgent, A2CAgent
from reinforce import reinforce
from actor_critic import actor_critic
from utils import load_config, smooth
from config import Config


COLORS = {
    "REINFORCE": "blue",
    "AC": "purple",
    "A2C": "red",
}


# ---------------------------------------------------------------------------
# Multi-repetition runner
# ---------------------------------------------------------------------------


def run_experiment(cfg: Config, n_reps: int) -> dict[str, list[dict]]:
    """Run all three algorithms for n_reps repetitions, return collected histories."""
    results = {"REINFORCE": [], "AC": [], "A2C": []}

    for rep in range(n_reps):
        print(f"\n=== Repetition {rep + 1}/{n_reps} ===")

        print("  REINFORCE...")
        agent = ReinforceAgent(cfg.agent, cfg.pi_nn)
        training_info, eval_info = reinforce(cfg, agent)
        results["REINFORCE"].append({**training_info, **eval_info})

        print("  AC...")
        agent = ACAgent(cfg.agent, cfg.pi_nn, value_network_cfg=cfg.v_nn)
        training_info, eval_info = actor_critic(cfg, agent)
        results["AC"].append({**training_info, **eval_info})

        print("  A2C...")
        agent = A2CAgent(cfg.agent, cfg.pi_nn, value_network_cfg=cfg.v_nn)
        training_info, eval_info = actor_critic(cfg, agent)
        results["A2C"].append({**training_info, **eval_info})

    return results


def plot_comparison(
    results: dict[str, list[dict]],
    window: int = 21,
    poly: int = 2,
    save_path: Path | None = None,
):
    """
    Single figure with two panels:
      - Left:  Training return over environment steps (mean ± std)
      - Right: Evaluation return over environment steps (mean ± std)
    """
    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("REINFORCE vs AC vs A2C — CartPole-v1", fontsize=13, fontweight="bold")

    for name, reps in results.items():
        color = COLORS[name]

        # Each rep stores list of (total_steps, return) tuples
        all_steps = [np.array([s for s, _ in rep["Returns"]]) for rep in reps]
        all_returns = [np.array([r for _, r in rep["Returns"]]) for rep in reps]

        # Common x-axis: use the shortest run to be safe
        min_len = min(len(s) for s in all_steps)
        common_x = all_steps[0][:min_len]
        returns_arr = np.array([r[:min_len] for r in all_returns])  # (n_reps, T)

        mean = smooth(returns_arr.mean(axis=0), window, poly)
        std = smooth(returns_arr.std(axis=0), window, poly)

        ax_train.plot(common_x, mean, label=name, color=color, linewidth=2)
        ax_train.fill_between(common_x, mean - std, mean + std, alpha=0.15, color=color)

        # Each rep stores eval_steps and eval_history as separate lists
        eval_steps_list = [np.array(rep["Timesteps"]) for rep in reps]
        eval_returns_list = [np.array(rep["Evaluation Returns"]) for rep in reps]

        min_eval_len = min(len(s) for s in eval_steps_list)
        common_eval_x = eval_steps_list[0][:min_eval_len]
        eval_arr = np.array([r[:min_eval_len] for r in eval_returns_list])

        eval_mean = smooth(eval_arr.mean(axis=0), window, poly)
        eval_std = smooth(eval_arr.std(axis=0), window, poly)

        ax_eval.plot(common_eval_x, eval_mean, label=name, color=color, linewidth=2)
        ax_eval.fill_between(
            common_eval_x,
            eval_mean - eval_std,
            eval_mean + eval_std,
            alpha=0.15,
            color=color,
        )

    for ax, title in [(ax_train, "Training Return"), (ax_eval, "Evaluation Return")]:
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Return")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(framealpha=0.7)

    plt.tight_layout()

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fpath = save_path / "comparison.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {fpath}")

    plt.show()
    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="base_ac", help="Config name (without .yaml)"
    )
    parser.add_argument(
        "--n_reps", type=int, default=5, help="Number of repetitions per algorithm"
    )
    parser.add_argument("--save", default="results", help="Directory to save the plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    print(
        f"Config: {args.config} | Reps: {args.n_reps} | Episodes: {cfg.run.n_episodes}"
    )

    results = run_experiment(cfg, n_reps=args.n_reps)
    plot_comparison(results, save_path=Path(args.save))
