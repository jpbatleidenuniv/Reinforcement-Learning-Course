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
from Helper import LearningCurvePlot, smooth



class MonteCarloAgent(BaseAgent):
    def update(self, states, actions, rewards):
        """states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state"""

        T_ep = len(actions)

        target = 0
        for t in reversed(range(T_ep)):
            target = rewards[t] + self.gamma * target

            s = states[t]
            a = actions[t]

            self.Q_sa[s, a] += self.learning_rate * (
                target - self.Q_sa[s, a]
            )


def monte_carlo(
    n_timesteps,
    max_episode_length,
    learning_rate,
    gamma,
    policy="egreedy",
    epsilon=None,
    temp=None,
    plot=True,
    eval_interval=500,
):
    """runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep"""

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(
        initialize_model=False
    )
    pi = MonteCarloAgent(
        env.n_states, env.n_actions, learning_rate, gamma
    )
    eval_timesteps = []
    eval_returns = []

    timestep = 0
    while timestep < n_timesteps:
        states, actions, rewards = [], [], []
        s = env.reset()
        states.append(s)

        for _ in range(max_episode_length):
            a = pi.select_action(
                s, policy=policy, epsilon=epsilon, temp=temp
            )
            s_next, r, done = env.step(a)

            actions.append(a)
            rewards.append(r)
            states.append(s_next)

            timestep += 1

            if timestep % eval_interval == 0:
                eval_timesteps.append(timestep)
                eval_returns.append(pi.evaluate(eval_env))

            if done or timestep >= n_timesteps:
                break

            s = s_next

        # print(timestep)

        pi.update(states, actions, rewards)

        if plot:
            env.render(
                Q_sa=pi.Q_sa,
                plot_optimal_policy=True,
                step_pause=0.001,
            )  # Plot the Q-value estimates during Monte Carlo RL execution

    print(len(eval_returns))

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 100001
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.04
    eval_interval = 500

    # Exploration
    policy = "egreedy"  # 'egreedy' or 'softmax'
    epsilon = 0.08
    temp = 1.0

    # Plotting parameters
    plot = False

    rewards, t = monte_carlo(
        n_timesteps,
        max_episode_length,
        learning_rate,
        gamma,
        policy,
        epsilon,
        temp,
        plot,
        eval_interval,
    )
    print(len(rewards))
    print(len(t))

    lcp = LearningCurvePlot("Monte carlo")
    lcp.add_curve(t, rewards, "Monte Carlo")
    # lcp.add_curve(t, smooth(rewards, window=35))
    lcp.save("MonteCarloTest.png")


if __name__ == "__main__":
    test()
