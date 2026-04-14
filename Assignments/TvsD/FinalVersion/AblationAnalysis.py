import matplotlib.pyplot as plt
import numpy as np
import os

from Helper import smooth


def mean_and_std(x: np.ndarray):
    """
    Compute mean and standard deviation across repetitions
    """
    mean = np.mean(x, axis=0)
    std  = np.std(x, axis=0)
    return mean, std


def sort_files(directory):
    """
    Scan a results directory and sort files into the four ablation categories.

    Each category holds a tuple of two lists: ([timestep_files], [return_files]).
    Files are matched by filename keywords; idx=0 for timestep files, idx=1 for return files.

    """
    data_files = {
        "Naive":             ([], []),
        "Buffer":            ([], []),
        "Target Network":    ([], []),
        "Target and Buffer": ([], []),
    }

    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found.")
        return data_files

    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        idx = 0 if "timesteps" in file else 1  # 0 = timestep file, 1 = returns file

        if "Naive_all" in file:
            data_files["Naive"][idx].append(path)
        elif "TargetNetwork_ExperienceReplay_all" in file:
            data_files["Target and Buffer"][idx].append(path)
        elif "TargetNetwork_all" in file:
            data_files["Target Network"][idx].append(path)
        elif ("ExperienceReplay_all" in file) and ("TargetNetwork" not in file):
            data_files["Buffer"][idx].append(path)

    return data_files


def plot_data(data_dict, show=True):
    """
    Plot smoothed learning curves for each ablation condition.

    Each condition is plotted as a mean line with a shaded ±1 std band.
    The mean and std are computed across all repetitions stored in the
    loaded array (shape: (n_repetitions, T)).

    """
    smooth_window = 20
    poly          = 2
    
    linewidth = 1.5
    fontsize = 25

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set_title("Network Ablation Study", fontsize=fontsize)

    for cat, (timesteps, returns) in data_dict.items():
        ax.grid(True, alpha=0.3)

        for t_file, r_file in zip(timesteps, returns):
            # np.load(t_file) has shape (n_reps, T); [0] takes the first rep's timesteps
            # all reps share the same eval timesteps, so any row works
            t, r = np.load(t_file)[0], np.load(r_file)

            mean, std = mean_and_std(r)  # Average over repetitions

            # Savitzky-Golay smoothing to reduce noise in the plotted curves
            s_mean = smooth(mean, smooth_window, poly)
            s_std  = smooth(std,  smooth_window, poly)

            line, = ax.plot(t, s_mean, label=cat, linewidth=linewidth)
            ax.fill_between(t, s_mean - s_std, s_mean + s_std,
                            alpha=0.1, color=line.get_color())
        ax.legend(loc='lower right', fontsize=22)
    ax.set_xlabel(r"Steps ($10^{6}$)", fontsize=fontsize)
    ax.set_ylabel("Return", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=22)

    plt.savefig("AblationResults")
    if show:
        plt.show()


def main_ablation(show=True):
    """Load results from disk and produce the ablation study plot."""
    directory = os.path.join('results', 'combined')
    data_dict = sort_files(directory=directory)
    plot_data(data_dict=data_dict, show=show)


if __name__ == '__main__':
    main_ablation(show=True)
