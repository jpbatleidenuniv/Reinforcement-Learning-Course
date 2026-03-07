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


class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        G_t = 0
        states_r = states[:-1][::-1]
        actions_r = actions[::-1]
        rewards_r = rewards[::-1]
        for s, a, r in zip(states_r, actions_r, rewards_r): 
            G_t = G_t*self.gamma + r
            self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate*(G_t - self.Q_sa[s, a])
        


def monte_carlo(
    n_timesteps,
    max_episode_length,
    learning_rate,
    gamma,
    policy="egreedy",
    epsilon=None,
    temp=None,
    plot=False,
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
    
    steps = 0
    s = env.reset()
    while steps < n_timesteps:
        states, actions, rewards = [], [], []
        states.append(s)
        done = False
        for _ in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)
            states.append(s_next)
            actions.append(a)
            rewards.append(r)
            eval_timesteps.append(r)
            steps += 1
            if plot:
                env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution
            if steps % eval_interval == 0:
                mean_return = pi.evaluate(eval_env)
                eval_returns.append(mean_return)
            s = s_next
            if done: 
                break
            elif steps >= n_timesteps:
                break


        pi.update(states, actions, rewards, done)

        s = env.reset()

            


    return np.array(eval_returns), np.array(eval_timesteps)   

    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = "egreedy"  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    monte_carlo(
        n_timesteps,
        max_episode_length,
        learning_rate,
        gamma,
        policy,
        epsilon,
        temp,
        plot,
    )


if __name__ == "__main__":
    test()
