#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent


class NstepQLearningAgent(BaseAgent):
    def update(self, states, actions, rewards, done, n):
        """states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state """

        T = len(actions)

        # Precompute n-step targets for all time steps
        targets = np.zeros(T)
        for t in range(T):
            end = min(t + n, T)  
            G = 0.0
            for i in range(t, end):
                G += (self.gamma ** (i - t)) * rewards[i]
            if not done:                              # bootstrap if episode did not terminate
                G += (self.gamma ** (end - t)) * np.max(self.Q_sa[states[end]])
            targets[t] = G

        # Apply updates
        for t in range(T):
            s, a = states[t], actions[t]
            self.Q_sa[s, a] += self.learning_rate * (targets[t] - self.Q_sa[s, a])


def n_step_Q(
    n_timesteps,
    max_episode_length,
    learning_rate,
    gamma,
    policy="egreedy",
    epsilon=None,
    temp=None,
    plot=False,
    n=5,
    eval_interval=500,
):
    """runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep"""

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(
        initialize_model=False
    )
    pi = NstepQLearningAgent(
        env.n_states, env.n_actions, learning_rate, gamma
    )
    eval_timesteps = []
    eval_returns = []

    steps = 0
    s = env.reset()
    while steps < n_timesteps:
        # Start a new episode
        episode_states = [s]
        episode_actions = []
        episode_rewards = []
        done = False

        for _ in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)

            episode_actions.append(a)
            episode_rewards.append(r)
            episode_states.append(s_next)
            steps += 1

            if plot:
                env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1)

            if steps % eval_interval == 0:
                mean_return = pi.evaluate(eval_env)
                eval_returns.append(mean_return)
                eval_timesteps.append(steps)

            s = s_next
            if done or steps >= n_timesteps:
                break

        # Episode finished – update using collected data
        pi.update(episode_states, episode_actions, episode_rewards, done, n)

        # Reset for next episode
        s = env.reset()

    return np.array(eval_returns), np.array(eval_timesteps)

        
def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5

    # Exploration
    policy = "egreedy"  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True
    n_step_Q(
        n_timesteps,
        max_episode_length,
        learning_rate,
        gamma,
        policy,
        epsilon,
        temp,
        plot,
        n=n,
    )


if __name__ == "__main__":
    test()
