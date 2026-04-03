import os
import gymnasium as gym
import numpy as np
import torch
from time import perf_counter
import sys

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from DQN import DQNAgent, ExperienceReplay
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    RecordVideo,
)
from Helper import LearningCurvePlot, smooth


def cartpole(
    agent: DQNAgent,
    buffer: ExperienceReplay,
    env: gym.Env,
    scheduler: ReduceLROnPlateau,
    eval_env: gym.Env,
    maximum_steps: int = 10**6,
    batch_size: int = 5,
    n_eval_timesteps: int = 5000,
    n_eval_episodes: int = 100,
):
    """Main function for running cartpole

    Parameters:
    exp_id (int): Index of the experiment chosen from experiments.py

    Returns:
    Evaluation metrics for each of the runs

    """

    # ------------------ Main simulation loop ------------------

    eval_returns: list[float] = []
    eval_timesteps: list[int] = []
    next_eval_step: int = n_eval_timesteps

    pbar = tqdm(
        total=maximum_steps,
        desc="Training Steps",
        unit="step",
    )
    total_steps_taken = 0
    episode_num = 0

    while total_steps_taken <= maximum_steps:
        episode_num += 1
        obs, _ = env.reset()
        Q_s = agent.eval_Q(state=obs)
        time_0 = perf_counter()

        episode_reward = 0
        step_count = 0
        episode_over = False
        obs_prev = obs
        batch_loss = []
        batch_counter = 0
        episode_loss = 0

        while not episode_over:
            # Replace this with your trained agent's policy
            action, _ = agent.action(Q_s)
            obs_prev = obs
            obs, reward, terminated, truncated, _ = (
                env.step(action)
            )
            Q_s_next = agent.eval_Q(obs)

            done = terminated or truncated

            sequences = buffer.get_sequence(state=obs_prev, 
                                            next_state=obs,
                                            action=action,
                                            reward=float(reward),
                                            done=done)
            

            l = agent.loss(
                sequences=sequences,
                count=total_steps_taken
            )

            episode_over = done

            if l is not None:
                if buffer.use_buffer:
                    # Replay buffer already returns a full batch mean loss, backprop immediately
                    agent.optimizer.zero_grad()
                    l.backward()
                    agent.optimizer.step()
                else:
                    # Accumulate loss for mini-batch updates
                    batch_loss.append(l)
                    batch_counter += 1
                    

                    if batch_counter >= batch_size or episode_over:
                        agent.optimizer.zero_grad()
                        sum(batch_loss).backward()
                        agent.optimizer.step()
                        batch_loss = []
                        batch_counter = 0
                
                step_count += 1
                episode_loss += l.item()

            Q_s = Q_s_next.detach()
            episode_reward += float(reward)
            total_steps_taken += 1
            pbar.update(1)

            # --- Evaluation at fixed timestep intervals ---
            if total_steps_taken >= next_eval_step:
                mean_return = agent.evaluate(
                    eval_env, n_eval_episodes
                )
                eval_returns.append(mean_return)
                eval_timesteps.append(total_steps_taken)
                next_eval_step += n_eval_timesteps

        if step_count > 0:
            scheduler.step(metrics= episode_loss / step_count)
        else:
            scheduler.step(metrics=0.0)

        if episode_num % 100 == 0 and step_count > 0:
            time_1 = perf_counter()

            pbar.set_postfix(
                {
                    "Reward": f"{episode_reward:.1f}",
                    "Loss": f"{episode_loss / step_count:.4f}",
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            elapsed = time_1 - time_0

    env.close()
    return eval_returns, eval_timesteps


if __name__ == "__main__":
    import sys
    from exploration import EXPLORATIONS

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(sys.argv) < 2:
        print("Usage: python Cartpole.py <exp_id>")
        print("Available experiments:")
        for i, exp in enumerate(EXPLORATIONS):
            print(f"  {i}: {exp.name}")
        sys.exit(1)

    exp_id = int(sys.argv[1])
    exp = EXPLORATIONS[exp_id]

    print(f"\nRunning experiment [{exp_id}]: {exp.name}")

    # ------------------ Environment ------------------
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()
    env = RecordVideo(
        env,
        video_folder=f"cartpole-videos/{exp.name}",
        name_prefix="eval",
        episode_trigger=lambda x: x % 400 == 0,
    )
    env = RecordEpisodeStatistics(
        env, buffer_length=exp.n_eval_episodes
    )

    eval_env = gym.make("CartPole-v1")


    # ------------------ Agent ------------------
    experience_replay = ExperienceReplay(
        buffer=exp.buffer,
        buffer_size=exp.buffer_size,
        min_buffer_size=exp.min_buffer_size,
        batch_size=exp.batch_size,
        device=device
    )

    agent = DQNAgent(
        hidden_layers=exp.layers,
        width=exp.width,
        learning_rate=exp.lr,
        policy=exp.policy,
        epsilon=exp.epsilon,
        temp=exp.temperature,
        target=exp.target_network,
        device=device
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer,
        mode="min",
        factor=exp.reduce_factor,
        patience=exp.patience,
    )

    print(f"Batch size: {exp.batch_size}")
    print(
        f"Policy: {exp.policy}, epsilon: {exp.epsilon}, temp: {exp.temperature}"
    )

    # ------------------ Run ------------------
    eval_returns, eval_timesteps = cartpole(
        agent=agent,
        env=env,
        scheduler=scheduler,
        eval_env=eval_env,
        maximum_steps=exp.maximum_steps,
        batch_size=exp.batch_size,
        n_eval_timesteps=exp.n_eval_timesteps,
        n_eval_episodes=exp.n_eval_episodes,
        buffer=experience_replay,
    )

    eval_env.close()

    # ------------------ Metrics ------------------
    avg_reward = np.mean(env.return_queue)
    avg_length = np.mean(env.length_queue)
    std_reward = np.std(env.return_queue)

    print(
        f"\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}"
    )
    print(f"Average episode length: {avg_length:.1f} steps")
    print(
        f"Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}"
    )

    # ------------------ Save results ------------------
    os.makedirs("results_test", exist_ok=True)
    np.save(
        f"results_test/{exp.name}_eval_returns.npy",
        np.array(eval_returns),
    )
    np.save(
        f"results_test/{exp.name}_eval_timesteps.npy",
        np.array(eval_timesteps),
    )
    print(f"Results saved to results/{exp.name}_*.npy")
