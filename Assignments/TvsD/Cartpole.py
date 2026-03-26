import gymnasium as gym
import numpy as np

from tqdm import tqdm
from DQN import DQNAgent
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


naive_agent = DQNAgent(hidden_layers=2,
                       width=64,
                       output_len=2,
                       input_len=4,
                       learning_rate=0.001,
                       policy='epsilon-greedy',
                       epsilon=0.01
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
    episode_trigger=lambda x: False    # Record every episode
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
    while not episode_over:
        naive_agent.optimizer.zero_grad()

        # Replace this with your trained agent's policy
        action, _ =  naive_agent.action(Q_s)
        Q_sa = Q_s[action]
        obs, reward, terminated, truncated, info = env.step(action)

        Q_s_next = naive_agent.eval_Q(obs)
        _, optimal_Q_s_next = naive_agent.action(Q_s_next, optimal=True)
        loss = naive_agent.loss(Q_sa=Q_sa, optimal_Q_sa_next=optimal_Q_s_next, r=float(reward))
        loss.backward()
        naive_agent.optimizer.step()

        Q_s = Q_s_next.detach()


        total_loss += loss
        episode_reward += float(reward)
        step_count += 1
        episode_over = terminated or truncated

    if episode_num % 100 == 0:
        pbar.set_postfix({
            'Reward': f'{episode_reward:.1f}',
            'Loss': f'{total_loss / step_count:.4f}',
            'Steps': step_count
            })
    #     print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}, average loss = {total_loss / step_count}")


    pbar.update(1)

env.close()

# Print summary statistics
print(f'\nEvaluation Summary:')
print(f'Episode durations: {list(env.time_queue)}')
print(f'Episode rewards: {list(env.return_queue)}')
print(f'Episode lengths: {list(env.length_queue)}')

# Calculate some useful metrics
avg_reward = np.sum(env.return_queue)
avg_length = np.sum(env.length_queue)
std_reward = np.std(env.return_queue)

print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
print(f'Average episode length: {avg_length:.1f} steps')
print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')
