import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Helper import smooth

RESULTS1 = [
    "Assignments/TRL/src/results/q_learning_rate_0_03_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv",
    "Assignments/TRL/src/results/q_learning_rate_0_1_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv",
    "Assignments/TRL/src/results/q_learning_rate_0_3_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv",
    "Assignments/TRL/src/results/sarsa_learning_rate_0_03_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv",
    "Assignments/TRL/src/results/sarsa_learning_rate_0_1_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv",
    "Assignments/TRL/src/results/sarsa_learning_rate_0_3_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv"
    ]

LABELS1 = [
          r"Q-learning, $\alpha$ = 0.03",
          r"Q-learning, $\alpha$ = 0.1",
          r"Q-learning, $\alpha$ = 0.3",
          r"SARSA, $\alpha$ = 0.03",
          r"SARSA, $\alpha$ = 0.1",
          r"SARSA, $\alpha$ = 0.3"
         ]

RESULTS2 = [
    "Assignments/TRL/src/results/q_learning_rate_0_1_gamma_1_policy_egreedy_epsilon_0_03_temp_1_n_5_eval_interval_1000.csv",
    "Assignments/TRL/src/results/q_learning_rate_0_1_gamma_1_policy_egreedy_epsilon_0_1_temp_1_n_5_eval_interval_1000.csv",
    "Assignments/TRL/src/results/q_learning_rate_0_1_gamma_1_policy_egreedy_epsilon_0_3_temp_1_n_5_eval_interval_1000.csv",
    "Assignments/TRL/src/results/q_learning_rate_0_1_gamma_1_policy_softmax_epsilon_0_3_temp_0_01_n_5_eval_interval_1000.csv",
    "Assignments/TRL/src/results/q_learning_rate_0_1_gamma_1_policy_softmax_epsilon_0_3_temp_0_1_n_5_eval_interval_1000.csv",
    "Assignments/TRL/src/results/q_learning_rate_0_1_gamma_1_policy_softmax_epsilon_0_3_temp_1_n_5_eval_interval_1000.csv"
    ]

LABELS2 = [
          r"$\epsilon$-greedy, $\epsilon$ = 0.03",
          r"$\epsilon$-greedy, $\epsilon$ = 0.1",
          r"$\epsilon$-greedy, $\epsilon$ = 0.3",
          r"Softmax, $\tau$ = 0.01",
          r"Softmax, $\tau$ = 0.1",
          r"Softmax, $\tau$ = 1"
         ]

RESULTS3 = [
    "Assignments/TRL/src/results/nstep_learning_rate_0_03_gamma_1_policy_egreedy_epsilon_0_2_temp_1_n_1_eval_interval_1000.csv",
    "Assignments/TRL/src/results/nstep_learning_rate_0_03_gamma_1_policy_egreedy_epsilon_0_2_temp_1_n_3_eval_interval_1000.csv",
    "Assignments/TRL/src/results/nstep_learning_rate_0_03_gamma_1_policy_egreedy_epsilon_0_2_temp_1_n_10_eval_interval_1000.csv",
    "Assignments/TRL/src/results/mc_learning_rate_0_03_gamma_1_policy_egreedy_epsilon_0_2_temp_1_n_10_eval_interval_1000.csv"
]

LABELS3 = [
          r"1-step Q-learning",
          r"3-step Q-learning",
          r"10step Q-learning",
          r"Monte Carlo"
         ]

TITLES = ["on_off_policy.png", "exploration_results.png", "Nstep.png"]



SMOOTHING_WINDOW = 9
OPTIMAL_DP = 83.7
FONT_SIZE = 20

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlabel("Timestep", fontsize=FONT_SIZE)
ax.set_ylabel("Episode Average Return", fontsize=FONT_SIZE)
ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
ax.tick_params(axis="both", which="minor", labelsize=FONT_SIZE)

for file, label in zip(RESULTS2, LABELS2):
    df = pd.read_csv(file)
    mean = smooth(df.mean(axis=0), window=SMOOTHING_WINDOW)
    ax.plot(np.arange(len(mean))*1000, mean, label=label)


if OPTIMAL_DP is not None:
    ax.hlines(OPTIMAL_DP, 0, 50001, label="DP optimum", linestyles='--', colors='black')

ax.grid(True, alpha=0.6)
plt.legend(fontsize=15, loc="center right")
fig.savefig(TITLES[1])
plt.show()


