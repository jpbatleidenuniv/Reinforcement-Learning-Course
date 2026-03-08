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


class SarsaAgent(BaseAgent):
    def update(self, s, a, r, s_next, done, a_next=None):
        if a_next is None:
            raise ValueError(
                "The next action cannot be None."
            )

        if done:
            G_t = r
        else:
            G_t = r + self.gamma * self.Q_sa[s_next, a_next]

        new_val = self.Q_sa[s, a] + self.learning_rate * (
            G_t - self.Q_sa[s, a]
        )
        self.Q_sa[s, a] = new_val


def sarsa(
    n_timesteps,
    learning_rate,
    gamma,
    policy="egreedy",
    epsilon=None,
    temp=None,
    plot=True,
    eval_interval=500,
):
    """runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep"""

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(
        initialize_model=False
    )
    pi = SarsaAgent(
        env.n_states, env.n_actions, learning_rate, gamma
    )
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your SARSA algorithm here!

    s = (
        env.reset()
    )  # Let the agent start at the start position
    a = pi.select_action(
        s, policy, epsilon, temp
    )  # Select initial action

    for i in range(n_timesteps):
        s_next, r, done = env.step(a)
        a_next = pi.select_action(
            s_next, policy, epsilon, temp
        )
        pi.update(s, a, r, s_next, done, a_next)

        if done:
            s = env.reset()  # Reset the board
            a = pi.select_action(s, policy, epsilon, temp)
        else:
            s = s_next
            a = a_next


        if i % eval_interval == 0:
            mean_return = pi.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(i)

    if plot:
        env.render(
            Q_sa=pi.Q_sa,
            plot_optimal_policy=True,
            step_pause=0.1,
        )  # Plot the Q-value estimates during SARSA execution

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = "egreedy"  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True
    sarsa(
        n_timesteps,
        learning_rate,
        gamma,
        policy,
        epsilon,
        temp,
        plot,
    )


if __name__ == "__main__":
    test()
