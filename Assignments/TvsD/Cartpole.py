import gymnasium as gym
import numpy as np

from torch import optim
from tqdm import tqdm
from DQN import DQNAgent
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

# ------------------ Environment Variables ------------------
USE_BUFFER = False


naive_agent = DQNAgent(hidden_layers=2,
                       width=64,
                       output_len=2,
                       input_len=4,
                       learning_rate=0.000005,
                       policy='epsilon-greedy',
                       epsilon=0.01,
                       target_network=False,
                       update_count=20,
                       buffer=USE_BUFFER, 
                       buffer_size=100
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    naive_agent.optimizer, mode='min', factor=0.5, patience=400
)

# Configuration
num_eval_episodes = 4000
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
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

print(f"Starting evaluation for {num_eval_episodes} episodes...")
print(f"Videos will be saved to: cartpole-agent/")


pbar = tqdm(total=num_eval_episodes, desc="Training Episodes", unit="episode")

for episode_num in range(num_eval_episodes):
    # Initial state
    obs, info = env.reset()
    Q_s = naive_agent.eval_Q(state=obs)

    episode_reward = 0
    step_count = 0
    episode_over = False
    total_loss = 0
    obs_prev = obs
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

        loss = naive_agent.loss(state=current_state, action=action, reward=float(reward),
                                next_state=next_state, done=done,
                                count=episode_num * step_count)
        if loss is not None:
            loss.backward()
            total_loss += loss

        naive_agent.optimizer.step()

        Q_s = Q_s_next.detach()

        episode_reward += float(reward)
        step_count += 1
        episode_over = terminated or truncated
    scheduler.step(metrics=total_loss/step_count)


    if episode_num % 100 == 0:
        pbar.set_postfix({
            'Reward': f'{episode_reward:.1f}',
            'Loss': f'{total_loss / step_count:.4f}',
            'Steps': step_count,
            'lr': scheduler.get_last_lr()
            })

    pbar.update(1)

if USE_BUFFER and hasattr(naive_agent, 'buffer'):
    print("Clearing buffer...")
    cleared_count = 0
    while True:
        item = naive_agent.buffer.clear()
        if item is None:
            break
        cleared_count += 1
    print(f"Cleared {cleared_count} experiences from buffer.")

env.close()

# # Print summary statistics
# print(f'\nEvaluation Summary:')
# print(f'Episode durations: {list(env.time_queue)}')
# print(f'Episode rewards: {list(env.return_queue)}')
# print(f'Episode lengths: {list(env.length_queue)}')

# Calculate some useful metrics
avg_reward = np.mean(env.return_queue)
avg_length = np.mean(env.length_queue)
std_reward = np.std(env.return_queue)

print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
print(f'Average episode length: {avg_length:.1f} steps')
print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')
