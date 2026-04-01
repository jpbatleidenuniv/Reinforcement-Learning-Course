from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from Helper import LearningCurvePlot, smooth
import pandas as pd

# Specify names of result files
base_name_returns = "*eval_returns.npy"
base_name_timesteps = "*eval_timesteps.npy"

results_dir = Path("results/")

sims = [
    f.name.replace("_" + base_name_returns[1:], "")
    for f in results_dir.rglob(base_name_returns)
]


def get_sims_from_results(
    results_dir: Path,
    base_names: tuple[str, str] = (
        "*eval_returns.npy",
        "*eval_timesteps.npy",
    ),
) -> set[str]:
    sim_sets = []
    for base_name in base_names:
        sims = {
            f.name.replace("_" + base_name[1:], "")
            for f in results_dir.rglob(base_name)
        }
        sim_sets.append(sims)

    common_sims = set.intersection(*sim_sets)
    print(f"Found complete simulations: {common_sims}")

    return common_sims


def results_to_dataframe(results_dir: Path) -> pd.DataFrame:
    rows = []

    # find all return files
    for returns_file in results_dir.rglob(
        "*eval_returns.npy"
    ):
        sim_name = returns_file.stem.replace(
            "_eval_returns", ""
        )
        timesteps_file = (
            results_dir / f"{sim_name}_eval_timesteps.npy"
        )

        if not timesteps_file.exists():
            continue  # skip incomplete sims

        returns = np.load(returns_file)
        timesteps = np.load(timesteps_file)

        for t, r in zip(timesteps, returns):
            rows.append(
                {
                    "simulation": sim_name,
                    "timestep": t,
                    "return": r,
                }
            )

    df = pd.DataFrame(rows)

    return df


def add_experiment_info(df: pd.DataFrame) -> pd.DataFrame:
    def parse_sim(sim_name: str):
        if "_" in sim_name:
            exp, value = sim_name.split("_", 1)
        else:
            exp, value = sim_name, "default"
        return exp, value

    parsed = df["simulation"].apply(parse_sim)

    df["experiment"] = parsed.apply(lambda x: x[0])
    df["value"] = parsed.apply(lambda x: x[1])

    return df


def moving_average(x, window=10):
    if len(x) < window:
        return x
    return np.convolve(
        x, np.ones(window) / window, mode="valid"
    )


def plot_grouped_learning_curves(
    df, smoothing_window=9, poly=2
):
    experiments = sorted(df["experiment"].unique())
    n_exp = len(experiments)

    fig, axes = plt.subplots(
        1,
        n_exp,
        figsize=(5 * n_exp, 4),
        squeeze=False,
        sharey=True,
    )

    for i, exp in enumerate(experiments):
        ax = axes[0, i]
        exp_df = df[df["experiment"] == exp]

        for sim, sim_df in exp_df.groupby("simulation"):
            sim_df = sim_df.sort_values("timestep")

            timesteps = sim_df["timestep"].values
            returns = sim_df["return"].values

            # Apply smoothing
            smooth_returns = smooth(
                returns, smoothing_window, poly
            )

            ax.plot(timesteps, smooth_returns, label=sim)

        ax.set_title(exp)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Returns")
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_dir = Path("results/")
    sims = get_sims_from_results(results_dir)
    df = results_to_dataframe(results_dir)
    df = add_experiment_info(df)
    print(df)
    df.groupby("experiment")

    plot_grouped_learning_curves(df)
    # plot_results(results_dir)
