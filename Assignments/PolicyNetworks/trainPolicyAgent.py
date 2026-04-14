import numpy as np
import torch
from cartpole import CartPoleEnv
from REINFORCE import REINFORCEAgent


env = CartPoleEnv(num_envs=1, seed=69)
agent = REINFORCEAgent(obs_dim=env.obs_dim, 
                       action_dim=env.action_dim, 
                       hidden_layers=[64, 64], 
                       learning_rate=1e-3, 
                       gamma=0.99, 
                       device='cpu')


max_env_steps = 10e6
done = False
for step in range(int(max_env_steps)):
    obs, info = env.reset()
    rewards = []
    actions_probs = []
    while not done:
        a, a_p = agent.select_actions(obs=obs, greedy=False) # For now assume single environment. Might adjust to multiple later-on
        actions_probs.append(a_p)
        n_obs, r, done, _ = env.step(actions=a)
        rewards.append(r)

        obs = n_obs
    arr_rewards = np.array(rewards)
    tensor_actions_probs = torch.tensor(actions_probs, device=agent.device)
    agent.train_step(rewards=arr_rewards, 
                     action_probs=tensor_actions_probs)
