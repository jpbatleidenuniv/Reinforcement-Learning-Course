import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils import smooth


def load_results(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def extract_mean_rewards(results: dict) -> np.ndarray:
    mean_rewards = []

    for iteration in range(5):
        mean_rewards_iteration = []

        for reward in results[iteration]:
            m = reward['mean']
            mean_rewards_iteration.append(m)

        arr_mean_rewards_iteration = np.pad(
            np.array(mean_rewards_iteration),
            (0, 50 - len(mean_rewards_iteration)),
            constant_values=mean_rewards_iteration[-1]
        )

        mean_rewards.append(arr_mean_rewards_iteration)

    mean_rewards = np.array(mean_rewards)

    return mean_rewards  # shape = (5 runs, 50 episodes)


def plot_all_ablations(ablation_data: dict):
    """
    ablation_data format:
    {
        "parameter_name": {
            hp_value: np.ndarray(shape=(runs, episodes))
        }
    }
    """
    labels = {
        "n_trajectories": "N Trajectories",
        "lambda": r"$\lambda$",
        "epsilon": r"$\epsilon$"
    }
    n_plots = len(ablation_data)

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))


    for ax, (param_name, results_dict) in zip(axes, ablation_data.items()):

        for hp_value, rewards in results_dict.items():

            mean_rewards = np.mean(rewards, axis=0)
            smooth_mean_rewards = smooth(mean_rewards, window=5, poly=2)
            std_rewards = np.std(rewards, axis=0)
            smooth_std_rewards = smooth(std_rewards, window=5, poly=2)

            episodes = np.linspace(0, 2.5, len(mean_rewards))

            ax.plot(
                episodes,
                smooth_mean_rewards,
                label=f"{labels.get(param_name, param_name)}={hp_value}"
            )

            ax.fill_between(
                episodes,
                smooth_mean_rewards - smooth_std_rewards,
                smooth_mean_rewards + smooth_std_rewards,
                alpha=0.2
            )

        # ax.set_title(f"Ablation: {param_name}", fontsize=20)
        ax.set_xlabel(r"Steps ($10^5$)", fontsize=23)
        ax.set_xticklabels([0, 0.5, 1.0, 1.5, 2.0, 2.5])
        ax.tick_params(axis='both', labelsize=18)
        # ax.tick_params(axis='x', Labelsize=18)
        if param_name == "n_trajectories":
            ax.set_ylabel("Average Return", fontsize=23)
        ax.grid(alpha=0.3)
        if param_name != "n_trajectories":
            ax.tick_params(
                        axis='y',          # Target the y-axis
                        which='both',      # Apply to both major and minor ticks
                        left=False,        # Turn off the little tick marks on the left
                        labelleft=False    # Turn off the text labels
                        ) #
        # Legend per subplot
        if param_name == "lambda":
            ax.legend(fontsize=14, loc="lower right")
        else:
            ax.legend(fontsize=14, loc="lower left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Load results
    results = load_results("results/ablation_independent_20260517_114334.pkl")

    # Organize raw data
    n_traj_results = {
        1: results["n_traj: 1"],
        5: results["n_traj: 5"],
        15: results["n_traj: 15"]
    }

    lambda_results = {
        0.7: results["lam: 0.7"],
        0.9: results["lam: 0.9"],
        0.99: results["lam: 0.99"]
    }

    epsilon_results = {
        0.05: results["eps: 0.05"],
        0.1: results["eps: 0.1"],
        0.2: results["eps: 0.2"]
    }

    # Convert to arrays of shape (runs, episodes)
    n_traj_results = {
        k: extract_mean_rewards(v)
        for k, v in n_traj_results.items()
    }

    lambda_results = {
        k: extract_mean_rewards(v)
        for k, v in lambda_results.items()
    }

    epsilon_results = {
        k: extract_mean_rewards(v)
        for k, v in epsilon_results.items()
    }

    # Combine all ablations
    ablation_data = {
        "n_trajectories": n_traj_results,
        "lambda": lambda_results,
        "epsilon": epsilon_results
    }

    # Plot
    plot_all_ablations(ablation_data)