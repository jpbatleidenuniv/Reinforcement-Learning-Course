import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Helper import smooth

RESULTS = [
    "results/q_learning_rate_0_03_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv",
    "results/q_learning_rate_0_1_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv",
    "results/q_learning_rate_0_3_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv",
    "results/sarsa_learning_rate_0_03_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv",
    "results/sarsa_learning_rate_0_1_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv",
    "results/sarsa_learning_rate_0_3_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv"
]

LABELS = [
          r"Q-learning , $\alpha=0.03$",
          r"Q-learning , $\alpha=0.1$",
          r"Q-learning , $\alpha=0.3$",
          r"SARSA , $\alpha=0.03$",
          r"SARSA , $\alpha=0.1$",
          r"SARSA , $\alpha=0.3$"
         ]

TITLE = "on_off_policy_results.pdf"
SMOOTHING_WINDOW = 9
OPTIMAL_DP = None
FONT_SIZE = 20

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlabel("Timestep", fontsize=15)
ax.set_ylabel("Episode Average Return", fontsize=FONT_SIZE)
ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
ax.tick_params(axis="both", which="minor", labelsize=FONT_SIZE)

for file, label in zip(RESULTS, LABELS):
    df = pd.read_csv(file)
    mean = smooth(df.mean(axis=0), window=SMOOTHING_WINDOW)
    ax.plot(np.arange(len(mean))*1000, mean, label=label)


if OPTIMAL_DP is not None:
    ax.hlines(OPTIMAL_DP, 0, 50001)

ax.grid(True, alpha=0.6)
plt.legend(fontsize=15, loc="lower right")
fig.savefig(TITLE)
plt.show()


