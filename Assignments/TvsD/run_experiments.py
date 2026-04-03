import os
import sys
import numpy as np
import gymnasium as gym
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
    buffer: bool = False
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    target_network: bool = False
    update_target: int = 100
    policy: str = "softmax"
    epsilon: float = 0.1
    temperature: float = 2.0
    layers: int = 3
    width: int = 128
    lr: float = 5e-4
    batch_size: int = 24
    reduce_factor: float = 0.5
    patience: int = 200
    maximum_steps: int = 10**6
    n_eval_timesteps: int = 5000
    n_eval_episodes: int = 100


CONFIGS = [
    RunConfig("Baseline"),
    RunConfig(
        "TargetNetwork",
        target_network=True,
        update_target=100,
    ),
    RunConfig(
        "Buffer",
        buffer=True,
        buffer_size=100000,
        min_buffer_size=1000,
        batch_size=64,
    ),
    RunConfig(
        "Target+Buffer",
        target_network=True,
        update_target=100,
        buffer=True,
        buffer_size=100000,
        min_buffer_size=1000,
        batch_size=64,
    ),
]

N_REPETITIONS = 20


def single_run(
    cfg: RunConfig, seed: int
) -> tuple[list[float], list[int]]:
    """This is exactly the same as the code in Cartpole.py"""

    # Seed for reproducibility
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    env = RecordEpisodeStatistics(
        env, buffer_length=cfg.n_eval_episodes
    )
    eval_env = gym.make("CartPole-v1")

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
