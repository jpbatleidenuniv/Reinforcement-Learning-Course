import os
import sys
import numpy as np
import gymnasium as gym
import torch
import time
from dataclasses import dataclass
from run_dqn import train_dqn_naive

from concurrent.futures import ProcessPoolExecutor
from OptimalHyperparameter import main_hyperparameters
from AblationAnalysis import main_ablation


# ------------------------------------------------------------------ #
# Experiment configuration                                            #
# ------------------------------------------------------------------ #

@dataclass
class RunConfig:
    """
    Dataclass that fully specifies one experimental condition.

    Each field maps directly to a hyperparameter passed to train_dqn_naive.
    Defaults here represent the shared baseline used across all studies.
    """
    name: str

    n_envs: int = 10                  # Parallel environments (= batch width during rollout)

    # Experience replay
    buffer: bool = False
    buffer_size: int = 100000         # Maximum transitions stored
    min_buffer_size: int = 100        # Minimum fill before training starts

    # Target network
    target_network: bool = False
    update_target: int = 100          # Steps between hard target network syncs

    # Exploration
    policy: str = "softmax"           # "epsilon_greedy" or "softmax"
    epsilon: float = 0.01
    temperature: float = 0.5

    # Network architecture
    layers: int = 1
    width: int = 64
    lr: float = 1e-3
    batch_size: int = 1               
    loss: str = "MSE"                 # Loss function label (MSE used in DQNAgent)

    reduce_factor: float = 0.5
    patience: int = 1000

    # Training budget and evaluation
    maximum_steps: int = 10**6
    n_eval_timesteps: int = 5000     # Steps between greedy evaluation runs
    n_eval_episodes: int = 10         # Episodes per evaluation


# Experiment definitions:
#   -   Study 1: Hyperparameter sweeps (exploration, architecture, LR, batch size)
#   -   Study 2: Ablation over stabilization techniques

CONFIGS = [
    # STUDY 1: Exploration policies
    RunConfig("EpsGreedy_e0.01", policy="epsilon_greedy", epsilon=0.01),
    RunConfig("EpsGreedy_e0.05", policy="epsilon_greedy", epsilon=0.05),
    RunConfig("EpsGreedy_e0.2",  policy="epsilon_greedy", epsilon=0.2),

    RunConfig("Softmax_t0.5", policy="softmax", temperature=0.5),
    RunConfig("Softmax_t1.0", policy="softmax", temperature=1.0),
    RunConfig("Softmax_t2.0", policy="softmax", temperature=2.0),

    # STUDY 1: Network architectures (wider networks)
    RunConfig("Width_128", layers=1, width=128),
    RunConfig("Width_256", layers=1, width=256),

    RunConfig("Layers_3", layers=3),
    RunConfig("Layers_6", layers=6),

    # STUDY 1: Learning rates
    RunConfig("LR_1e-3", lr=1e-3),
    RunConfig("LR_5e-4", lr=5e-4),
    RunConfig("LR_1e-4", lr=1e-4),
    RunConfig("LR_1e-5", lr=1e-5),

    # STUDY 1: Batch sizes (data-to-update ratio)
    RunConfig("Batch_1",  n_envs=1),
    RunConfig("Batch_32", batch_size=32),
    RunConfig("Batch_64", batch_size=64),

    # STUDY 2: Ablation — stabilization techniques
    # Uses the best hyperparameters identified in Study 1
    RunConfig("Naive",
        policy="softmax", temperature=1.0,
        layers=1, width=64, lr=1e-3, n_envs=10,
    ),
    RunConfig("TargetNetwork",
        policy="softmax", temperature=1.0,
        layers=1, width=64, lr=1e-3, n_envs=10,
        target_network=True, update_target=100,
    ),
    RunConfig("ExperienceReplay",
        policy="softmax", temperature=1.0,
        layers=1, width=64, lr=1e-3, n_envs=10,
        buffer=True, buffer_size=100000, min_buffer_size=1000,
    ),
    RunConfig("TargetNetwork_ExperienceReplay",
        policy="softmax", temperature=1.0,
        layers=1, width=64, lr=1e-3, n_envs=10,
        target_network=True, update_target=100,
        buffer=True, buffer_size=100000, min_buffer_size=1000,
    )
]

# Number of independent repetitions run in parallel (different seeds)
N_REPETITIONS = 5


def single_run(cfg: RunConfig, seed: int = 42):
    """
    Execute one training run for a given config and seed.

    Returns eval_returns and eval_steps (the periodic greedy evaluation
    results), which are used as the primary performance signal for analysis.
    """
    device = "cpu"
    print(f"Using device: {device}")

    # Seed numpy and torch for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    _, _, _, returns, timesteps = train_dqn_naive(
        num_envs=cfg.n_envs,
        total_env_steps=cfg.maximum_steps,
        hidden_layers=[cfg.width] * cfg.layers,
        learning_rate=cfg.lr,
        exploration_strategy=cfg.policy,
        epsilon=cfg.epsilon,
        temperature=cfg.temperature,
        gamma=0.99,
        update_every=cfg.batch_size,
        seed=seed,
        device=device,
        use_target_network=cfg.target_network,
        target_update_every=cfg.update_target,
        eval_every=cfg.n_eval_timesteps,
        n_eval_episodes=10,
        use_replay_buffer=cfg.buffer,
        replay_capacity=cfg.buffer_size,
        batch_size=cfg.n_envs,        # Replay batch size = number of envs
        min_buffer_size=cfg.min_buffer_size,
    )

    return returns, timesteps


def run_repetitions_parallel(cfg: RunConfig):
    """
    Run N_REPETITIONS independent seeds for one config in parallel using
    separate processes
    """
    seeds = [42 + i for i in range(N_REPETITIONS)]
    max_worker = min(N_REPETITIONS, os.cpu_count())

    with ProcessPoolExecutor(max_workers=max_worker) as executor:
        results = list(executor.map(
            single_run,
            [cfg] * N_REPETITIONS,
            seeds
        ))

    returns   = [r[0] for r in results]
    timesteps = [r[1] for r in results]

    return returns, timesteps


def run_all(cfg_index: int | None = None):
    """
    Run experiments and save results.

    If cfg_index is provided, only that config
    is run. Otherwise all configs in CONFIGS are run sequentially.
    """
    configs_to_run = (
        [CONFIGS[cfg_index]] if cfg_index is not None else CONFIGS
    )

    os.makedirs("results/", exist_ok=True)
    os.makedirs("results/combined/", exist_ok=True)

    time0 = time.time()

    all_returns, all_timesteps = [], []
    for cfg in configs_to_run:
        print(f"\n{'=' * 60}")
        print(f"Config: {cfg.name}  ({N_REPETITIONS} parallel repetitions)")
        print(f"{'=' * 60}")

        
        for i in range(N_REPETITIONS):
            returns, timesteps = single_run(cfg, seed=42+i)
            all_returns.append(returns)
            all_timesteps.append(timesteps)     
   


        time1 = time.time()
        print(f"{N_REPETITIONS} took {((time1 - time0) / 60):.2f} minutes to run")

        # Align to the shortest repetition in case of edge-case length mismatch
        min_len = min(len(r) for r in all_returns)
        returns_arr   = np.array([r[:min_len] for r in all_returns])    # (N_REPS, T)
        timesteps_arr = np.array([t[:min_len] for t in all_timesteps])  # (N_REPS, T)

        print(returns_arr.shape)
        print(timesteps_arr.shape)

        # Save per-repetition files (useful for debugging individual seeds)
        for rep in range(N_REPETITIONS):
            np.save(f"results/{cfg.name}_returns_rep{rep}.npy",   returns_arr[rep])
            np.save(f"results/{cfg.name}_timesteps_rep{rep}.npy", timesteps_arr[rep])

        # Save aggregated arrays used by the analysis scripts
        np.save(f"results/combined/{cfg.name}_all_returns.npy",   returns_arr)
        np.save(f"results/combined/{cfg.name}_all_timesteps.npy", timesteps_arr)

        mean = np.mean(returns_arr, axis=0)
        std  = np.std(returns_arr,  axis=0)
        print(f"  Final eval return: {mean[-1]:.1f} ± {std[-1]:.1f}")

        all_returns = []
        all_timesteps = []

    print("\nAll done. Results saved to results/")


if __name__ == "__main__":
    # Optionally accept a config index as a CLI argument
    cfg_index = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_all(cfg_index)

    # Generate analysis plots after all experiments complete
    main_hyperparameters(show=False)
    main_ablation(show=False)
