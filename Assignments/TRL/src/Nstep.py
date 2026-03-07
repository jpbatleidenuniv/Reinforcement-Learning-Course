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

        G = sum([(self.gamma**i) * r for i, r in enumerate(rewards)])

        if not done:
            last_state = states[-1]
            G += (self.gamma**len(rewards)) * np.max(self.Q_sa[last_state])

        s_target = states[0]
        a_target = actions[0]
    
        self.Q_sa[s_target, a_target] += self.learning_rate * (G - self.Q_sa[s_target, a_target])


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

    s = env.reset()
    states, actions, rewards = [s], [], []
    
    for t in range(n_timesteps):
        a = pi.select_action(s, policy, epsilon, temp)
        s_next, r, done = env.step(a)
        
        states.append(s_next)
        actions.append(a)
        rewards.append(r)
        eval_timesteps.append(r)
        s = s_next

        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1)

        # Once we have at least n steps, we can start updating
        if len(rewards) >= n:
            pi.update(states[-n-1:], actions[-n:], rewards[-n:], done, n)
            states, actions, rewards = [states[-1]], [], []
        if done:
            pi.update(states[-n-1:], actions[-n:], rewards[-n:], done, n)
            s = env.reset()
            states, actions, rewards = [s], [], []
        
        if t % eval_interval == 0:
            mean_return = pi.evaluate(eval_env)
            eval_returns.append(mean_return)

        
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
