import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from Helper import smooth


def calculate_auc(y, x):
    """Compute the area under the curve using the trapezoidal rule."""
    return np.trapezoid(y, x)


def mean_and_std(x: np.ndarray):
    """
    Compute mean and standard deviation across repetitions (axis 0).
    """
    mean = np.mean(x, axis=0)
    std  = np.std(x, axis=0)
    return mean, std


def create_baseline(data_files, baseline_keywords):
    """
    Identify the set of return files that correspond to the shared baseline condition.

    A file is considered a baseline if any of the baseline_keywords appear in its path.
    These files are pooled across all categories to form a master baseline.
    """
    baseline_set = set()
    for category in data_files:
        _, rewards = data_files[category]
        for reward in rewards:
            if any(base in reward for base in baseline_keywords):
                baseline_set.add(reward)
    return baseline_set


def sort_files(directory):
    """
    Scan a results directory and sort files into hyperparameter study categories.

    Each category holds a tuple: ([timestep_files], [return_files]).
    Files are matched by their name prefix to determine which sweep they belong to.
    """
    data_files = {
        "Softmax":        ([], []),
        "Epsilon-greedy": ([], []),
        "Width":          ([], []),
        "Depth":          ([], []),
        "Lr":             ([], []),
        "Data-to-update": ([], []),
    }

    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found.")
        return data_files

    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        idx = 0 if "timesteps" in file else 1  # 0 = timestep file, 1 = returns file

        if file.startswith("Softmax"):
            data_files["Softmax"][idx].append(path)
        elif file.startswith("EpsGreedy"):
            data_files["Epsilon-greedy"][idx].append(path)
        elif file.startswith("Width"):
            data_files["Width"][idx].append(path)
        elif file.startswith("Layers"):
            data_files["Depth"][idx].append(path)
        elif file.startswith("LR"):
            data_files["Lr"][idx].append(path)
        elif file.startswith("Batch"):
            data_files["Data-to-update"][idx].append(path)

    return data_files


def calculate_and_rank_performance(data_dict, baseline_set, final_performance_frac=0.2):
    """
    Compute final performance and AUC for every hyperparameter setting and rank them.

    Also computes a master baseline entry by pooling all baseline runs across categories.
    """
    rankings = []
    baseline_steps   = None
    baseline_results = None

    # --- Aggregate master baseline across all categories ---
    for category in data_dict:
        for t, r in zip(data_dict[category][0], data_dict[category][1]):
            if r in baseline_set:
                t_data, r_data = np.load(t)[0], np.load(r)
                if baseline_results is None:
                    baseline_results, baseline_steps = r_data, t_data
                else:
                    # Stack repetitions from multiple baseline files
                    baseline_results = np.concatenate([baseline_results, r_data], axis=0)

    if baseline_results is not None:
        mean_b   = np.mean(baseline_results, axis=0)
        last_10  = max(1, int(len(mean_b) * final_performance_frac))
        rankings.append({
            'Category':          'ALL',
            'HP_Setting':        'MASTER BASELINE',
            'Final_Performance': np.mean(mean_b[-last_10:]),
            'AUC':               calculate_auc(mean_b[-last_10:], baseline_steps[-last_10:])
        })

    # --- Rank all non-baseline settings ---
    for category in data_dict:
        for t_file, r_file in zip(data_dict[category][0], data_dict[category][1]): 
            if r_file in baseline_set:
                continue  # Skip baseline files; already handled above

            t_data, r_data = np.load(t_file)[0], np.load(r_file)
            mean_r = np.mean(r_data, axis=0)
            last_10 = max(1, int(len(mean_r) * final_performance_frac))

            # Extract the HP value label from the filename 
            label = os.path.basename(t_file).split("_")[1]

            # Group Softmax and Epsilon-greedy under a single "Exploration" logic category
            logic_cat = "Exploration" if category in ["Softmax", "Epsilon-greedy"] else category

            rankings.append({
                'Category':          logic_cat,
                'Plot_Category':     category,
                'HP_Setting':        label,
                'Final_Performance': np.mean(mean_r[-last_10:]),
                'AUC':               calculate_auc(mean_r[-last_10:], t_data[-last_10:])
            })

    return pd.DataFrame(rankings)


def plot_data(data_dict, baseline_set, rankings_df, show_only_top_n=2, show=True):
    """
    Plot hyperparameter sweep results in a grid of subplots (one per category).

    Only the top `show_only_top_n` settings per category are plotted to avoid
    clutter. The master baseline is overlaid on every subplot in black.
    """
    smooth_window = 20
    poly          = 2
    ncols, nrows  = 3, 2
    linewidth     = 1.5

    fontsize=24

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 10), sharey=True)
    axes = axes.flatten()

    # --- Pre-compute master baseline curve ---
    baseline_runs  = []
    baseline_steps = None
    for cat in data_dict:
        for t, r in zip(data_dict[cat][0], data_dict[cat][1]):
            if r in baseline_set:
                baseline_runs.append(np.load(r))
                baseline_steps = np.load(t)[0]

    master_data = np.concatenate(baseline_runs, axis=0)
    b_mean, b_std = mean_and_std(master_data)
    sb_mean = smooth(b_mean, window=smooth_window, poly=poly)
    sb_std  = smooth(b_std,  window=smooth_window, poly=poly)

    for i, (category, (timesteps, returns)) in enumerate(data_dict.items()):
        ax = axes[i]
        ax.set_title(category, fontsize=fontsize)
        ax.grid(True, alpha=0.3)

        # Identify the top-N settings for this subplot based on final performance
        cat_rankings = rankings_df[rankings_df['Plot_Category'] == category]
        top_labels   = (cat_rankings
                        .sort_values('Final_Performance', ascending=False)
                        .head(show_only_top_n)['HP_Setting']
                        .tolist())

        for t_file, r_file in zip(timesteps, returns):
            if r_file in baseline_set:
                continue  # Baseline is plotted separately below

            label = os.path.basename(t_file).split("_")[1]
            if show_only_top_n and label not in top_labels:
                continue  # Skip non-top settings

            t, r = np.load(t_file)[0], np.load(r_file)
            mean, std = mean_and_std(r)
            s_mean = smooth(mean, smooth_window, poly)
            s_std  = smooth(std,  smooth_window, poly)

            line, = ax.plot(t, s_mean, label=label, linewidth=linewidth)
            ax.fill_between(t, s_mean - s_std, s_mean + s_std,
                            alpha=0.1, color=line.get_color())

        # Overlay the master baseline on every subplot for reference
        ax.plot(baseline_steps, sb_mean, color='black', linewidth=linewidth,
                label='Baseline', zorder=5)
        ax.fill_between(baseline_steps, sb_mean - sb_std, sb_mean + sb_std,
                        color='black', alpha=0.07)
        ax.legend(loc='lower right', fontsize='x-small')
    
        if i == 4:
            ax.legend(loc='upper left', fontsize=16)
        else:
            ax.legend(loc='lower right', fontsize=16)
        if i == 0 or i == 3:
            ax.set_ylabel("Return", fontsize=fontsize)
        if i >= 3:
            ax.set_xlabel(r"Steps ($10^{6}$)", fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=18)

    plt.tight_layout()
    plt.savefig("HyperParameterResults")
    if show:
        plt.show()


def main_hyperparameters(show=True):
    """
    Full hyperparameter analysis pipeline:
      1. Load and sort result files from disk.
      2. Identify baseline files.
      3. Rank all settings by final performance and AUC.
      4. Print a ranked summary table.
      5. Determine the best setting per category (if it beats the baseline).
      6. Plot the top settings per category.
    """
    directory = os.path.join('results', 'combined')

    # Identifiers that mark the shared baseline condition across all sweep categories
    baseline_identifiers = ["t0.5", "Width_64", "1e-3", "MSE", "Layers_1", "Batch_10"]

    criterion = 'AUC'

    data_files   = sort_files(directory)
    baseline_set = create_baseline(data_files, baseline_identifiers)
    df_rankings  = calculate_and_rank_performance(data_files, baseline_set,
                                                  final_performance_frac=0.1)

    # Start with sensible defaults; overwrite only if a setting beats the baseline
    best_hp_settings = {
        "Data-to-update": '10',
        "Lr":             '1e-3',
        "Depth":          '1',
        "Width":          '64',
        "Exploration":    "t0.5",
    }

    # Print ranked results grouped by logical category
    print("\n" + "=" * 75)
    print(f"{'LOGIC GROUP':<15} | {'SETTING':<15} | {'FINAL REWARD':<12} | {'AUC'}")
    print("-" * 75)

    baseline_final_performance = df_rankings[df_rankings['Category'] == "ALL"][criterion].iloc[0]

    for cat in df_rankings['Category'].unique():
        cat_data = df_rankings[df_rankings['Category'] == cat].sort_values(criterion, ascending=False)

        # Update best_hp_settings if the top setting in this category beats the baseline
        if cat_data['Category'].iloc[0] != "ALL":
            row0 = cat_data.iloc[0]
            if row0[criterion] > baseline_final_performance:
                best_hp_settings[row0["Category"]] = row0["HP_Setting"]

        for _, row in cat_data.iterrows():
            print(f"{row['Category']:<15} | {row['HP_Setting']:<15} | "
                  f"{row['Final_Performance']:<12.2f} | {row['AUC']:.2e}")
        print("-" * 75, '\n')

    # Print the final recommended hyperparameter configuration
    print("=" * 75)
    print(f"{'HP':<15} | {'Value':<15}")
    print("-" * 75)

    for key, value in best_hp_settings.items():
        if key == "Exploration":
            key = "Softmax" if value.startswith('t') else 'Epsilon-Greedy'
        print(f"{key:<15} | {value:<15}")
    print("-" * 75)

    plot_data(data_files, baseline_set, df_rankings, show_only_top_n=6, show=show)


if __name__ == "__main__":
    main_hyperparameters(show=True)
