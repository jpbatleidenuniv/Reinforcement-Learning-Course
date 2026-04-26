from typing import Any

from matplotlib.figure import Figure
from utils import smooth
from collections.abc import Mapping
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

def plot_curve(ax: Axes, y, window, poly, label="Return", color="blue") -> Axes:
    """Plot a specific curve for a run, allows for inclusion of non-smoothed curve"""
    ax.plot(y, alpha=0.3, color="red", label="raw")
    try:
        smoothed = smooth(y, window, poly)
    except Exception as e:
        print("Savgol failed:", e)
        smoothed = y
    ax.plot(smoothed, color=color, label=f"smoothed (w={window},p={poly})")
    ax.set_ylabel(label)
    ax.set_xlabel("Episode")
    return ax


def plot_training(training_info: Mapping[str, Any], window, poly, plot=True) -> Figure:
    plot_names = training_info.keys()
    n_plots = len(plot_names)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1)

    axes = axes.flatten()
    for ax, (name, data) in zip(axes, training_info.items()):
        plot_curve(ax, data, window=window, poly=poly, label=name)
    plt.legend()
    if plot:
        plt.show()
    return fig


def plot_results(results: dict, window: int, poly: int, baseline_df: pd.DataFrame | None) -> Figure:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    colors = {"A2C": "blue", "AC": "red", "REINFORCE": "green", "Baseline":"black"}

    for method, repetitions in results.items():
        # repetitions is a list of 5 eval_history lists
        # each eval_history is a list of dicts with "step" and "mean"

        # Use the steps from the first repetition as x-axis
        # (all repetitions share the same eval_interval so steps align)
        steps = [ckpt["step"] for ckpt in repetitions[0]]

        # Stack means across repetitions: shape (n_repetitions, n_checkpoints)
        all_means = np.array([
            [ckpt["mean"] for ckpt in rep]
            for rep in repetitions
        ])

        # Average and std across repetitions
        mean_across_runs = all_means.mean(axis=0)
        std_across_runs = all_means.std(axis=0)
        
        mean_across_runs = smooth(mean_across_runs, window=window, poly=poly)
        std_across_runs = smooth(std_across_runs, window=window, poly=poly)

        color = colors.get(method, None)
        ax.plot(steps, mean_across_runs, label=method, color=color, linewidth=1.5)
        ax.fill_between(
            steps,
            mean_across_runs - std_across_runs,
            mean_across_runs + std_across_runs,
            alpha=0.1,
            color=color,
        )
        ax.grid(alpha=0.3) 
    
    if baseline_df is not None:
        x = baseline_df["env_step"]
        y = smooth(baseline_df["Episode_Return_smooth"], window=window+10, poly=poly)
        ax.plot(x, y, label='DQN', color='black', linewidth=1.5, linestyle='--')

    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel("Steps", fontsize=25)
    ax.set_ylabel("Mean Return (greedy)", fontsize=25)
    ax.legend(fontsize=22)
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    with open("results/results_20260426_090218.pkl", "rb") as f:
        results = pickle.load(f)

    try:
        df = pd.read_csv("results/BaselineDataCartPole.csv")
        baseline_df = df[df['env_step'] <= 500000].sort_values('env_step')
    except:
        baseline_df = None

    fig = plot_results(results, window=20, poly=2, baseline_df=baseline_df)
    fig.savefig("results/comparison.png", dpi=150)