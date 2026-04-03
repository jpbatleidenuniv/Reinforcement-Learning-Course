import os
import sys
import numpy as np
import gymnasium as gym
import torch
from torch import optim
from gymnasium.wrappers import RecordEpisodeStatistics

from DQN import DQNAgent, ExperienceReplay
from Cartpole import cartpole
from dataclasses import dataclass


# ------------------------------------------------------------------ #
# Configurations                                                       #
# ------------------------------------------------------------------ #


@dataclass
class RunConfig:
    name: str

    # Only for buffer
    buffer: bool = False
    buffer_size: int = 100000
    min_buffer_size: int = 1000

    # Only for target
    target_network: bool = False
    update_target: int = 100

    # Naive
    policy: str = "softmax" # ['epsilon-greedy', 'softmax']
    epsilon: float = 0.05
    temperature: float = 1.0

    # NN
    layers: int = 2
    width: int = 128
    lr: float = 1e-4
    batch_size: int = 32
    loss: str = "MSE" #['MSE', 'MAE']

    # Lr scheduler
    reduce_factor: float = 0.5
    patience: int = 1000

    # Environment
    maximum_steps: int = 10**6
    n_eval_timesteps: int = 5000
    n_eval_episodes: int = 100


CONFIGS = [
    # Policies
    RunConfig("EpsGreedy_e0.01", policy="epsilon-greedy", epsilon=0.01),
    RunConfig("EpsGreedy_e0.05", policy="epsilon-greedy", epsilon=0.05),
    RunConfig("EpsGreedy_e0.2",  policy="epsilon-greedy", epsilon=0.2),

    RunConfig("Softmax_t0.5", policy="softmax", temperature=0.5),
    RunConfig("Softmax_t1.0", policy="softmax", temperature=1.0),
    RunConfig("Softmax_t2.0", policy="softmax", temperature=2.0),

    # Architectures
    RunConfig("Width_64",  layers=3, width=64),
    RunConfig("Width_128",  layers=3, width=128),
    RunConfig("Width_256", layers=3, width=256),

    RunConfig("Layers_1",  layers=1, width=128),
    RunConfig("Layers_2",  layers=2, width=128),
    RunConfig("Layers_3", layers=3, width=128),
    RunConfig("Layers_6", layers=6, width=128),

    # Learning rates
    RunConfig("LR_1e-3",  lr=1e-3),
    RunConfig("LR_5e-4",  lr=5e-4),
    RunConfig("LR_1e-4",  lr=1e-4),
    RunConfig("LR_1e-5",  lr=1e-5),

    # Batch size
    RunConfig("Batch_1",  batch_size=1),
    RunConfig("Batch_10", batch_size=10),
    RunConfig("Batch_20", batch_size=32),
    RunConfig("Batch_64", batch_size=64),

    # Loss functions
    RunConfig("loss_MSE", loss="MSE"),
    RunConfig("loss_MAE", loss="MAE"),


    # RunConfig("Best_Combined",
    #     policy="softmax",       # <-- replace with your findings
    #     temperature=1.0,
    #     layers=2,
    #     width=64,
    #     lr=1e-3,
    #     batch_size=32,
    # ),

    # ================================================================
#     # STUDY 2: Stabilization techniques (use best hyperparams above)
#     # ================================================================
#     RunConfig("Naive",
#         policy="softmax", temperature=1.0,  # <-- replace with best
#         layers=2, width=64, lr=1e-3, batch_size=64,
#     ),
#     RunConfig("TargetNetwork",
#         policy="softmax", temperature=1.0,
#         layers=2, width=64, lr=1e-3, batch_size=64,
#         target_network=True, update_target=100,
#     ),
#     RunConfig("ExperienceReplay",
#         policy="softmax", temperature=1.0,
#         layers=2, width=64, lr=1e-3, batch_size=64,
#         buffer=True, buffer_size=100000, min_buffer_size=1000,
#     ),
#     RunConfig("TargetNetwork_ExperienceReplay",
#         policy="softmax", temperature=1.0,
#         layers=2, width=64, lr=1e-3, batch_size=64,
#         target_network=True, update_target=100,
#         buffer=True, buffer_size=100000, min_buffer_size=1000,
#     ),
]

N_REPETITIONS = 5


def single_run(
    cfg: RunConfig, seed: int
) -> tuple[list[float], list[int]]:
    """This is exactly the same as the code in Cartpole.py"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If using GPU:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # For full determinism (can slow things down):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env = RecordEpisodeStatistics(
        env, buffer_length=cfg.n_eval_episodes
    )
    eval_env = gym.make("CartPole-v1")
    eval_env.reset(seed=seed+1)

    experience_replay = ExperienceReplay(
        buffer=cfg.buffer,
        buffer_size=cfg.buffer_size,
        min_buffer_size=cfg.min_buffer_size,
        batch_size=cfg.batch_size,
    )

    agent = DQNAgent(
        hidden_layers=cfg.layers,
        width=cfg.width,
        learning_rate=cfg.lr,
        policy=cfg.policy,
        epsilon=cfg.epsilon,
        temp=cfg.temperature,
        target=cfg.target_network,
        update_frequence=cfg.update_target,
        loss_function=cfg.loss
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer,
        mode="min",
        factor=cfg.reduce_factor,
        patience=cfg.patience,
    )

    eval_returns, eval_timesteps = cartpole(
        agent=agent,
        buffer=experience_replay,
        env=env,
        scheduler=scheduler,
        eval_env=eval_env,
        maximum_steps=cfg.maximum_steps,
        batch_size=cfg.batch_size,
        n_eval_timesteps=cfg.n_eval_timesteps,
        n_eval_episodes=cfg.n_eval_episodes,
    )

    env.close()
    eval_env.close()
    return eval_returns, eval_timesteps


def run_all(cfg_index: int | None = None):
    """
    If cfg_index is given, only run that config (used for Slurm array jobs).
    Otherwise run all configs sequentially.
    """
    configs_to_run = (
        [CONFIGS[cfg_index]]
        if cfg_index is not None
        else CONFIGS
    )

    os.makedirs("results", exist_ok=True)

    for cfg in configs_to_run:
        print(f"\n{'=' * 60}")
        print(
            f"Config: {cfg.name}  ({N_REPETITIONS} repetitions)"
        )
        print(f"{'=' * 60}")

        all_returns = []  # Saving all results over here
        all_timesteps = []

        for rep in range(N_REPETITIONS):
            seed = rep * 42  # Just change it a little bit
            print(
                f"  Repetition {rep + 1}/{N_REPETITIONS}  (seed={seed})"
            )

            eval_returns, eval_timesteps = single_run(
                cfg, seed=seed
            )
            all_returns.append(eval_returns)
            all_timesteps.append(eval_timesteps)

            # Save after every rep so partial results survive a crash
            np.save(
                f"results/{cfg.name}_returns_rep{rep}.npy",
                np.array(eval_returns),
            )
            np.save(
                f"results/{cfg.name}_timesteps_rep{rep}.npy",
                np.array(eval_timesteps),
            )

        # Stack and save aggregated results
        # Use the shortest run in case of edge-case length mismatch
        min_len = min(len(r) for r in all_returns)
        returns_arr = np.array(
            [r[:min_len] for r in all_returns]
        )  # (N_REPS, T)
        timesteps_arr = np.array(
            [t[:min_len] for t in all_timesteps]
        )  # (N_REPS, T)

        np.save(
            f"results/{cfg.name}_all_returns.npy",
            returns_arr,
        )
        np.save(
            f"results/{cfg.name}_all_timesteps.npy",
            timesteps_arr,
        )

        mean = np.mean(returns_arr, axis=0)
        std = np.std(returns_arr, axis=0)
        print(
            f"  Final eval return: {mean[-1]:.1f} ± {std[-1]:.1f}"
        )

    print("\nAll done. Results saved to results/")


if __name__ == "__main__":
    # Optional: pass a config index as argv[1] for Slurm array jobs
    cfg_index = (
        int(sys.argv[1]) if len(sys.argv) > 1 else None
    )
    run_all(cfg_index)
