import gymnasium as gym
import numpy as np
from time import perf_counter

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from DQN import DQNAgent
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    RecordVideo,
)


def cartpole(
    agent: DQNAgent,
    env: gym.Env,
    scheduler: ReduceLROnPlateau,
    maximum_steps: int = 10**6,
    batch_size: int = 5,
):
    """Main function for running cartpole

    Parameters:
    exp_id (int): Index of the experiment chosen from experiments.py

    Returns:
    Evaluation metrics for each of the runs

    """

    # ------------------ Main simulation loop ------------------

    pbar = tqdm(
        total=maximum_steps,
        desc="Training Steps",
        unit="step",
    )
    total_steps_taken = 0
    episode_num = 0

    while total_steps_taken <= maximum_steps:
        # Initial state
        episode_num += 1
        obs, _ = env.reset()
        Q_s = agent.eval_Q(state=obs)
        time_0 = perf_counter()

        episode_reward = 0
        step_count = 0
        episode_over = False
        total_loss = 0
        obs_prev = obs

        batch_loss = []
        batch_counter = 0

        while not episode_over:
            # Replace this with your trained agent's policy
            action, _ = agent.action(Q_s)

            obs_prev = obs
            obs, reward, terminated, truncated, _ = (
                env.step(action)
            )

            Q_s_next = agent.eval_Q(obs)

            current_state = obs_prev  # you'll need to keep the previous state
            next_state = obs
            done = terminated or truncated

            l = agent.loss(
                state=current_state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                done=done,
            )

            episode_over = terminated or truncated
            if l is not None:
                # Accumulate loss for current batch
                batch_loss.append(l)
                batch_counter += 1

                # Update step count and total loss (use .item() to detach from graph)
                step_count += 1
                total_loss += l.item()

                if (
                    batch_counter == batch_size
                    or episode_over
                ):
                    # Zero gradients before backward pass
                    agent.optimizer.zero_grad()

                    sum(batch_loss).backward()
                    agent.optimizer.step()

                    # Reset batch accumulators
                    batch_loss = []
                    batch_counter = 0

            Q_s = Q_s_next.detach()
            episode_reward += float(reward)
            total_steps_taken += 1
            pbar.update(1)

        if step_count > 0:
            scheduler.step(metrics=total_loss / step_count)
        else:
            scheduler.step(metrics=0.0)

        if episode_num % 100 == 0:
            time_1 = perf_counter()
            pbar.set_postfix(
                {
                    "Reward": f"{episode_reward:.1f}",
                    "Loss": f"{total_loss / step_count:.4f}",
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            time_0 = time_1

    env.close()


if __name__ == "__main__":
    # buffer_size = 200
    batch_size = 20
    num_eval_episodes: int = 4000
    maximum_steps: int = 10**6

    # Policy
    policy: str = (
        "epsilon-greedy"  # ['epsilon-greedy', 'softmax']
    )
    epsilon: float = 0.01
    temperature: float = 1.0

    # NN
    layers: int = 2
    width: int = 64
    output_len: int = 2
    input_len: int = 4
    lr: float = 5e-3
    batch_size: int = 5

    # ------------ LR Scheduler ------------
    reduce_factor: float = 0.5
    patience: int = 400

    # ------------------ Environment ------------------
    env_name = (
        "CartPole-v1"  # Replace with your environment
    )
    env = gym.make(
        env_name, render_mode="rgb_array"
    )  # rgb_array needed for video recording
    # Add video recording for every episode
    env = RecordVideo(
        env,
        video_folder="cartpole-agent",  # Folder to save videos
        name_prefix="eval",  # Prefix for video filenames
        episode_trigger=lambda x: x % 400
        == 0,  # Record every episode
    )
    # Add episode statistics tracking
    env = RecordEpisodeStatistics(
        env, buffer_length=num_eval_episodes
    )

    print(f"Batch size has been set to {batch_size}")
    print(
        f"Starting evaluation for {num_eval_episodes} episodes..."
    )

    # ------------------ Agent ------------------
    agent = DQNAgent(
        hidden_layers=layers,
        width=width,
        learning_rate=lr,
        batch_size=batch_size,
        policy=policy,
        epsilon=epsilon,
        temp=temperature,
    )

    # Learning rate scheduler, might be improved aswell
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer,
        mode="min",
        factor=reduce_factor,
        patience=patience,
    )

    cartpole(agent, env, scheduler)

    # Calculate some useful metrics
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
