#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax


class QValueIterationAgent:
    """Class to store the Q-value iteration solution, perform updates, and select the greedy action"""

    def __init__(
        self, n_states, n_actions, gamma, threshold=0.01
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s):
        """Returns the greedy best action in state s"""
        a = argmax(self.Q_sa[s])
        return a

    def update(self, s, a, p_sas, r_sas):
        """Function updates Q(s,a) using p_sas and r_sas"""
        Vs = np.max(
            self.Q_sa, axis=1
        )  # This is indicates the highest Q per state
        Q_upd = p_sas @ (r_sas + self.gamma * Vs)
        self.Q_sa[s, a] = Q_upd
        pass


def Q_value_iteration(
    env: StochasticWindyGridworld,
    gamma=1.0,
    threshold=0.001,
):
    """Runs Q-value iteration. Returns a converged QValueIterationAgent object"""

    QIagent = QValueIterationAgent(
        env.n_states, env.n_actions, gamma
    )
    s_0 = env.reset()

    i = 0
    delta = threshold * 10
    while delta > threshold:
        i += 1
        for s in range(env.n_states):
            for a in range(env.n_actions):
                old_Q_sa = QIagent.Q_sa.copy()[s, a]
                p_sas, r_sas = env.model(s, a)
                QIagent.update(s, a, p_sas, r_sas)
                delta = np.max(
                    np.abs(old_Q_sa - QIagent.Q_sa[s, a]),
                )

        # Plot current Q-value estimates & print max error
        env.render(
            Q_sa=QIagent.Q_sa,
            plot_optimal_policy=True,
            step_pause=0.2,
        )
        print(
            "Q-value iteration, iteration {}, max error {}".format(
                i, delta
            )
        )

    pi = [argmax(Q_s) for Q_s in QIagent.Q_sa]
    print(pi)

    return QIagent, QIagent.Q_sa, pi



def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent, Q_sa, pi = Q_value_iteration(
        env, gamma, threshold
    )

    # view optimal policy
    done = False
    Return=[]
    s = env.reset()
    rewards = []
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        rewards.append(r)
        env.render(
            Q_sa=Q_sa,
            plot_optimal_policy=True,
            step_pause=0.5,
        )
        s = s_next

    # TO DO: Compute mean reward per timestep under the optimal policy
    mean_reward_per_timestep = np.mean(rewards)
    print(
        "Mean reward per timestep under optimal policy: {}".format(
            mean_reward_per_timestep
        )
    )


if __name__ == "__main__":
    experiment()
