import gymnasium as gym
import yaml
import pickle
import torch
import numpy as np

from datetime import datetime
from PPO import PPO
from pathlib import Path
from copy import deepcopy
from config import RunConfig, AgentConfig, NNConfig, Config
from plot_hp_tuning import plot_main


def load_base_config(name: str) -> dict:
    with open(f"{name}.yaml") as f:
        return yaml.safe_load(f)


def make_config(base: dict, k: int, lamb: float, epsilon: float) -> Config:
    # Copy dictionaries so we can safely overwrite values
    nn_dict = dict(base["nn"])
    agent_dict = dict(base["agent"])

    nn_dict["lamb"] = lamb
    agent_dict["k"] = k
    agent_dict["epsilon"] = epsilon

    # Create config objects
    nn_cfg = NNConfig(**nn_dict)
    agent_cfg = AgentConfig(
        nn_cfg=nn_cfg,
        **agent_dict
    )
    run_cfg = RunConfig(**base["run"])

    return Config(
                  run=run_cfg,
                  nn=nn_cfg,
                  agent=agent_cfg
                  )


# Varying settings
N_TRAJ_VALUES  = [1, 5, 15]   # n_trajectories: number of episodes collected per update
LAMBDA_VALUES  = [0.7, 0.9, 0.99]
EPSILON_VALUES = [0.05, 0.1, 0.2]

N_REPETITIONS = 5
BASE_CONFIG   = "PPO"

save_path = Path("results/")
save_path.mkdir(exist_ok=True)

base_cfg_dict = load_base_config(BASE_CONFIG)

# Baseline values (taken from PPO.yaml)
BASE_N_TRAJ  = 1                                   # n_trajectories baseline
BASE_K       = base_cfg_dict["agent"]["k"]         # PPO update epochs, kept fixed
BASE_LAMBDA  = base_cfg_dict["nn"]["lamb"]
BASE_EPSILON = base_cfg_dict["agent"]["epsilon"]

# Results dictionary
results: dict[str, list] = {}

experiments = []

# vary n_trajectories 
for n_traj in N_TRAJ_VALUES:
    experiments.append({
                        "name":   f"n_traj: {n_traj}",
                        "n_traj": n_traj,
                        "k":      BASE_K,
                        "lamb":   BASE_LAMBDA,
                        "epsilon": BASE_EPSILON,
                        })

# vary lambda
for lamb in LAMBDA_VALUES:
    experiments.append({
                        "name":   f"lam: {lamb}",
                        "n_traj": BASE_N_TRAJ,
                        "k":      BASE_K,
                        "lamb":   lamb,
                        "epsilon": BASE_EPSILON,
                        })

# vary epsilon
for epsilon in EPSILON_VALUES:
    experiments.append({
                        "name":   f"eps: {epsilon}",
                        "n_traj": BASE_N_TRAJ,
                        "k":      BASE_K,
                        "lamb":   BASE_LAMBDA,
                        "epsilon": epsilon,
                        })

total = len(experiments) * N_REPETITIONS
done  = 0

# Run experiments
for exp in experiments:
    run_key  = exp["name"]
    n_traj   = exp["n_traj"]
    k        = exp["k"]
    lamb     = exp["lamb"]
    epsilon  = exp["epsilon"]

    results[run_key] = []

    print(f"\n{'=' * 60}")
    print(f"  Ablation: {run_key}")
    print(f"    n_trajectories = {n_traj}")
    print(f"    k (epochs)     = {k}")
    print(f"    lambda         = {lamb}")
    print(f"    epsilon        = {epsilon}")
    print(f"{'=' * 60}")

    for rep in range(N_REPETITIONS):
        torch.manual_seed(rep)
        np.random.seed(rep)

        cfg = make_config(
                          deepcopy(base_cfg_dict),
                          k=k,
                          lamb=lamb,
                          epsilon=epsilon,
                          )

        env = gym.make("CartPole-v1")

        eval_history = PPO(
                           config=cfg,
                           env=env,
                           save_plot=None,
                           plot=False,
                           iteration=rep,
                           n_trajectories=n_traj,   
                           )

        results[run_key].append(eval_history)

        env.close()

        done += 1
        print(f"  [{done}/{total}] rep {rep + 1}/{N_REPETITIONS} done")


#  Save results 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path  = save_path / f"ablation_independent_{timestamp}.pkl"

with open(out_path, "wb") as f:
    pickle.dump(results, f)

print(f"\nSaved ablation results to {out_path}")

# Summary table 
print(f"\n{'Run key':<25} {'Mean final return':>20} {'Std':>10}")
print("-" * 60)

for run_key, reps in results.items():
    # Each rep is a list of eval dicts; take the last checkpoint's mean return
    final_means = [
                   rep[-1]["mean"]
                   for rep in reps
                   if rep 
                   ]

    if final_means:
        print(
            f"{run_key:<25}"
            f"{np.mean(final_means):>20.1f}"
            f"{np.std(final_means):>10.1f}"
        )

plot_main(data_file=out_path, figure_file="hp_tuning.png")