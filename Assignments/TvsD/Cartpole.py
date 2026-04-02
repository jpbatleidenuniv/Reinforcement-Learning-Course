import gymnasium as gym
import numpy as np
import time
import torch

from arguments import args
from torch import optim
from tqdm import tqdm
from Assignments.TvsD.DQN_Target import DQNAgent
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

# ------------------ Environment Variables ------------------
""" Here you can change argument variables. Look at the arguments.py file to see which arguments you want to set differently or add other arguments. Pls keep it organized. 
Beneath is an example of how to change an argument."""

# args.buffer_size = 200
args.batch_size = 20

print(f"Batch size has been set to {args.batch_size}")

# ------------------ Environment ------------------

env_name = "CartPole-v1"  # Replace with your environment

# Create environment with recording capabilities
env = gym.make(env_name, render_mode="rgb_array")  # rgb_array needed for video recording

# Add video recording for every episode
env = RecordVideo(
    env,
    video_folder="cartpole-agent",    # Folder to save videos
    name_prefix="eval",               # Prefix for video filenames
    episode_trigger=lambda x: x%400 == 0   # Record every episode
)

# Add episode statistics tracking
env = RecordEpisodeStatistics(env, buffer_length=args.num_eval_episodes)

print(f"Starting evaluation for {args.num_eval_episodes} episodes...")
print(f"Videos will be saved to: cartpole-agent/")


# ------------------ Agent ------------------

naive_agent = DQNAgent(hidden_layers=args.layers,
                       width=args.width,
                       output_len=args.output_len,
                       input_len=args.input_len,
                       learning_rate=args.lr,
                       policy=args.policy,
                       epsilon=args.epsilon,
                       temp=args.temperature,
                       target_network=args.target_network,
                       update_count=args.update_target,
                       batch_size=args.batch_size
)

# Learning rate scheduler, might be improved aswell
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    naive_agent.optimizer, mode='min', factor=args.reduce_factor, patience=args.patience
)


# ------------------ Main simulation loop ------------------

pbar = tqdm(total=args.maximum_steps, desc="Training Steps", unit="step")
total_steps_taken = 0
episode_num = 0
while total_steps_taken <= args.maximum_steps:

    # Initial state
    episode_num += 1
    obs, info = env.reset()
    Q_s = naive_agent.eval_Q(state=obs)
    time_0 = time.time()

    episode_reward = 0
    step_count = 0
    episode_over = False
    total_loss = 0
    obs_prev = obs

    batch_loss = torch.tensor(0.0, dtype=torch.float32)   # accumulates losses in current batch
    batch_counter = 0

    while not episode_over:
        naive_agent.optimizer.zero_grad()

        # Replace this with your trained agent's policy
        action, _ =  naive_agent.action(Q_s)
        Q_sa = Q_s[action]
        obs, reward, terminated, truncated, info = env.step(action)

        Q_s_next = naive_agent.eval_Q(obs)
        _, optimal_Q_s_next = naive_agent.action(Q_s_next, optimal=True)

        current_state = obs_prev   # you'll need to keep the previous state
        next_state = obs
        done = terminated or truncated

        l = naive_agent.loss(state=current_state, action=action, reward=float(reward),
                                next_state=next_state, done=done,
                                count=episode_num * step_count)
        
        episode_over = terminated or truncated
        if l is not None:
            # Accumulate loss for current batch
            batch_loss = batch_loss + l
            batch_counter += 1

            # Update step count and total loss (use .item() to detach from graph)
            step_count += 1
            total_loss += l.item()

            if batch_counter == args.batch_size or episode_over:
                # Zero gradients before backward pass
                naive_agent.optimizer.zero_grad()
                batch_loss.backward()
                naive_agent.optimizer.step()

                # Reset batch accumulators
                batch_loss = torch.tensor(0.0, dtype=torch.float32)
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
        time_1 = time.time()
        pbar.set_postfix({
            'Reward': f'{episode_reward:.1f}',
            'Loss': f'{total_loss / step_count:.4f}',
            'lr': scheduler.get_last_lr()[0],
            })
        time_0 = time_1

if args.buffer and hasattr(naive_agent, 'buffer'):
    print("Clearing buffer...")
    cleared_count = 0
    while True:
        item = naive_agent.buffer.clear()
        if item is None:
            break
        cleared_count += 1
    print(f"Cleared {cleared_count} experiences from buffer.")

env.close()

# Calculate some useful metrics
avg_reward = np.mean(env.return_queue)
avg_length = np.mean(env.length_queue)
std_reward = np.std(env.return_queue)

print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
print(f'Average episode length: {avg_length:.1f} steps')
print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')
print(f"Total number of environment steps: {total_environment_steps}")