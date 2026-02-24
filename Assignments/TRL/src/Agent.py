#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax


class BaseAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states: int = n_states
        self.n_actions: int = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(
        self, s, policy="egreedy", epsilon: float | None = None, temp=None
    ):
        possible_actions = np.arange(self.n_actions)

        if policy == "greedy":
            # We greedily take the action with the highest Q.
            a = argmax(self.Q_sa[s])

        elif policy == "egreedy":
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            pi = np.ones(self.n_actions) * epsilon / self.n_actions  # Will have small
            best_action = argmax(self.Q_sa[s])
            pi[best_action] = 1.0 - epsilon * (self.n_actions - 1) / self.n_actions

            assert np.isclose(np.sum(pi), 1.0)
            a = np.random.choice(possible_actions, p=pi)

        elif policy == "softmax":
            if temp is None:
                raise KeyError("Provide a temperature")
            pi = softmax(self.Q_sa[s], temp=temp)
            a = np.random.choice(possible_actions, p=pi)

        return a

    def update(self):
        raise NotImplementedError(
            "For each agent you need to implement its specific back-up method"
        )  # Leave this and overwrite in subclasses in other files

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, "greedy")
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
