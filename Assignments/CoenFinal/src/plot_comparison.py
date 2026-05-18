import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils import smooth

def load_results(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        return pickle.load(f)

def extract_mean_rewards(results: dict) -> np.ndarray:
    mean_rewards = {}

    for method in results:
        mean_rewards[method] = []
        for iteration in range(5):
            mean_rewards_iteration = []

            for reward in results[method][iteration]:
                m = reward['mean']
                mean_rewards_iteration.append(m)

            if len(mean_rewards_iteration) < 50:
                arr_mean_rewards_iteration = np.pad(
                    np.array(mean_rewards_iteration),
                    (0, 50 - len(mean_rewards_iteration)),
                    constant_values=mean_rewards_iteration[-1]
                )
            elif len(mean_rewards_iteration) > 50:
                arr_mean_rewards_iteration = np.array(mean_rewards_iteration[:50])
            else: 
                arr_mean_rewards_iteration = np.array(mean_rewards_iteration)

            mean_rewards[method].append(arr_mean_rewards_iteration)

    mean_rewards = {method: np.array(mean_rewards[method]) for method in mean_rewards}

    return mean_rewards  # shape = (5 runs, 50 episodes)


def plot_comparison(mean_ppo: np.ndarray):
    for method, rewards in mean_ppo.items():
        episodes = np.arange(rewards.shape[1])/20
        mean = np.mean(rewards, axis=0)
        smooth_mean = smooth(mean, window=5, poly=2)
        std = np.std(rewards, axis=0)
        smooth_std = smooth(std, window=5, poly=2)

        plt.plot(episodes, smooth_mean, label=method)
        plt.fill_between(episodes, smooth_mean - smooth_std, smooth_mean + smooth_std, alpha=0.2)

    plt.tick_params(axis='both', labelsize=18)

    plt.xlabel(r"Steps ($10^5$)", fontsize=23)
    plt.ylabel("Average Return", fontsize=23)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)

def plot_main(data_file: str, figure_file: str="comparison.png"):

    ppo_results = load_results(data_file)['PPO_optimal']
    ass3_results = load_results("results_20260427_204003.pkl")
    a2c_results = ass3_results['A2C']
    reinforce_results = ass3_results['REINFORCE']

    ass2_results = np.load("Naive_all_returns.npy", allow_pickle=True)
    ass2_steps = np.load("Naive_all_timesteps.npy", allow_pickle=True)
    mask = ass2_steps <= 250000

    means = extract_mean_rewards({
                                 "PPO Optimal": ppo_results, 
                                 "A2C": a2c_results,
                                 "REINFORCE": reinforce_results
                                 })
    means["DQN"] = ass2_results[mask].reshape(mask.shape[0], -1)

    plot_comparison(means)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_main(data_file="results/results_20260517_160518.pkl", figure_file="comparison.png")