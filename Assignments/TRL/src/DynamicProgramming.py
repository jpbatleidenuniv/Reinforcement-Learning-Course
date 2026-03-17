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

        # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        # TO DO: Add own code

        """#WORKFLOW
        we have the Q(s,a) 2D table, we just need the row corresponding to the correct state
        from the row we search for the highest possible value
        that is our action
        """
        # we search for max_a Q(s,a)
        Q_s0_a = self.Q_sa[s]
        a = argmax(Q_s0_a)

        return a  # this action is our greedy policy for a state

    def update(self, s, a, p_sas, r_sas):
        """Function updates Q(s,a) using p_sas and r_sas"""
        # TO DO: Add own code
        V_s = np.max(self.Q_sa, axis=1)
        self.Q_sa[s, a] = p_sas @ (r_sas + self.gamma * V_s)

        pass


def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    """Runs Q-value iteration. Returns a converged QValueIterationAgent object"""

    QIagent = QValueIterationAgent(
        env.n_states, env.n_actions, gamma
    )

    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    """"WORKFLOW:
    To implement properly this, we need to break it down: we have a 
    Q table (s,a), we now need to open it and update every value following
    the dynamic programming algorithm. 
    To do that: for each tuple (s,a) we need to update the Q value until we reach convergence
    """

    error = 10 * threshold
    errors = []
    errors.append(error)
    count = 0

    while error > threshold:
        # save previsious Q(s,a)
        x = QIagent.Q_sa.copy()

        # update actual Q(s,a)
        for s in range(env.n_states):
            for a in range(env.n_actions):
                psas, rsas = env.model(s, a)
                QIagent.update(s, a, psas, rsas)

        # update of the error
        error = np.max(np.abs(x - QIagent.Q_sa))

        # print(QIagent.Q_sa)

        # Plot current Q-value estimates & print max error
        env.render(
            Q_sa=QIagent.Q_sa,
            plot_optimal_policy=True,
            step_pause=0.1,
        )
        """if error<threshold:
                env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=5)"""
        print(
            "Q-value iteration, iteration {}, max error {}".format(
                count, error
            )
        )
        count += 1

    return QIagent


def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env, gamma, threshold)

    # view optimal policy
    done = False
    Return = []
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(
            Q_sa=QIagent.Q_sa,
            plot_optimal_policy=True,
            step_pause=0.1,
        )
        s = s_next
        Return.append(r)
        # print(f"Reward:{r}")

    # TO DO: Compute mean reward per timestep under the optimal policy
    mean_reward_per_timestep = np.mean(Return)
    num_step = len(Return)
    scalar_return = np.sum(Return)
    print(
        "Mean reward per timestep under optimal policy: {}".format(
            mean_reward_per_timestep
        )
    )
    return mean_reward_per_timestep, num_step, scalar_return


if __name__ == "__main__":
    a_lot = False  # useful switch, False= just one experiment run, true= more than one right to have statistics

    if a_lot:
        mean_rewards, steps, returns = [], [], []

        for i in range(10):
            mean_reward, num_step, scal_return = (
                experiment()
            )
            mean_rewards.append(mean_reward)
            steps.append(num_step)
            returns.append(scal_return)

        final_r_mean = np.mean(mean_rewards)
        final_r_std = np.std(mean_rewards)

        final_R_mean = np.mean(returns)
        final_R_std = np.std(returns)

        final_step_mean = np.mean(steps)
        final_step_std = np.std(steps)

        print(
            f"The mean reward per timestep averaged on 10 simulation is: {final_r_mean} and error: {final_r_std}"
        )
        with open(
            "Dynamic_programming_avg_reward_per_timestep.txt",
            "w",
        ) as file:
            file.write(
                f"Mean reward per timestep on 10 simulations: {final_r_mean}"
            )
            file.write(f"\nStd deviation: {final_r_std}")
            file.write(
                f"\n\n Mean Return on 10 simulations: {final_R_mean}"
            )
            file.write(f"\nStd deviation: {final_R_std}")
            file.write(
                f"\n\n Mean steps on 10 simulations: {final_step_mean}"
            )
            file.write(f"\nStd deviation: {final_step_std}")
    else:
        experiment()

