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
    def update(self, s, a, r, s_next, a_next, done):
        target = (
            r + self.gamma * self.Q_sa[s_next, a_next]
            if not done
            else r
        )
        self.Q_sa[s, a] += self.learning_rate * (
            target - self.Q_sa[s, a]
        )


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

    s0 = env.reset()
    s = s0
    a = pi.select_action(
        s0, policy=policy, epsilon=epsilon, temp=temp
    )
    for i in range(n_timesteps):
        if i % eval_interval == 0:
            eval_timesteps.append(i)
            eval_returns.append(pi.evaluate(eval_env))

        s_next, r, done = env.step(a)
        a_next = pi.select_action(
            s_next,
            policy=policy,
            epsilon=epsilon,
            temp=temp,
        )
        pi.update(s, a, r, s_next, a_next, done)
        if done:
            s = env.reset()
            a = pi.select_action(
                s,
                policy=policy,
                epsilon=epsilon,
                temp=temp,
            )
        else:
            s = s_next
            a = a_next

        if plot:
            env.render(
                Q_sa=pi.Q_sa,
                plot_optimal_policy=True,
                step_pause=0.005,
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
