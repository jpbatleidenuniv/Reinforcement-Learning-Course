from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter


def get_files(dir: Path):
    r_files = sorted(dir.rglob("*all_returns.npy"))
    t_files = sorted(dir.rglob("*all_timesteps.npy"))
    return list(zip(r_files, t_files))


def load_to_dataframe(file_pairs: list) -> pd.DataFrame:
    records = []

    for r_path, t_path in file_pairs:
        returns = np.load(r_path)  # (20, 200)
        timesteps = np.load(t_path)  # (20, 200)

        label = r_path.stem.replace(
            "_all_returns", ""
        )  # extracts {ALGO}
        n_reps, n_steps = returns.shape

        for rep in range(n_reps):
            for step in range(n_steps):
                records.append(
                    {
                        "label": label,
                        "repetition": rep,
                        "timestep": timesteps[rep, step],
                        "return": returns[rep, step],
                    }
                )

    return pd.DataFrame(records)


def plot_curves(
    df: pd.DataFrame,
    smooth: bool = False,
    window=21,
    polyorder=3,
):
    """
    Plot learning curves with 95% CI using seaborn lineplot.

    Args:
        df: tidy DataFrame with columns [label, repetition, timestep, return]
        smooth: apply Savitzky-Golay smoothing before plotting
        window: smoothing window length (must be odd)
        polyorder: polynomial order for smoothing
    """
    if smooth:

        def smooth_group(group):
            group = group.sort_values("timestep")
            group["return"] = savgol_filter(
                group["return"], window, polyorder
            )
            return group

        df = df.groupby(
            ["label", "repetition"], group_keys=False
        ).apply(smooth_group)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="timestep",
        y="return",
        hue="label",
        # seaborn computes mean + 95% CI across repetitions automatically
        errorbar=("ci", 95),
        ax=ax,
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Return")
    ax.set_title("Learning Curves with 95% CI")
    ax.legend(title="Simulation")
    plt.tight_layout()
    plt.show()


# --- Usage ---
if __name__ == "__main__":
    data_dir = Path("your/data/dir")
    file_pairs = get_files(data_dir)

    df = load_to_dataframe(
        file_pairs,
    )
    print(df.head())
    # label  repetition  timestep  return
    # expA            0      1000    12.3
    # expA            0      2000    14.1
    # ...

    plot_curves(df, smooth=True)
