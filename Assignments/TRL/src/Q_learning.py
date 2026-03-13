"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent
from Helper import LearningCurvePlot, smooth


class QLearningAgent(BaseAgent):
    def update(self, s, a, r, s_next, done):
        target = (
            r
            if done
            else r + self.gamma * np.max(self.Q_sa[s_next])
        )
        self.Q_sa[s, a] += self.learning_rate * (
            target - self.Q_sa[s, a]
        )


def q_learning(
    n_timesteps,
    learning_rate,
    gamma,
    policy="egreedy",
    epsilon=None,
    temp=None,
    plot=True,
    eval_interval=500,
    add_extra_goal=False,
    zero_reward = False
):
    """runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep"""

    env = StochasticWindyGridworld(
        initialize_model=False,
    )
    eval_env = StochasticWindyGridworld(
        initialize_model=False
    )
    agent = QLearningAgent(
        env.n_states, env.n_actions, learning_rate, gamma
    )
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your Q-learning algorithm here!
    s = env.reset()
    if add_extra_goal:
        env.goal_locations = [[7, 3], [3, 2]]
        env.goal_rewards = [100, 5]
        eval_env.goal_locations = [[7, 3], [3, 2]]
        eval_env.goal_rewards = [100, 5]

    if zero_reward:
        env.reward_per_step = 0 
        eval_env.reward_per_step = 0 
    for i in range(n_timesteps):
        a = agent.select_action(
            s=s, policy=policy, epsilon=epsilon, temp=temp
        )
        s_next, r, done = env.step(a)
        agent.update(s, a, r, s_next, done)
        if done:
            s = env.reset()
        else:
            s = s_next

        if i % eval_interval == 0:
            mean_return = agent.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(i)

        if plot:
            env.render(
                Q_sa=agent.Q_sa,
                plot_optimal_policy=True,
                step_pause=0.05,
            )  # Plot the Q-value estimates during Q-learning execution

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 50001
    eval_interval = 500
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = "egreedy"  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = False

    eval_returns, eval_timesteps = q_learning(
        n_timesteps,
        learning_rate,
        gamma,
        policy,
        epsilon,
        temp,
        plot,
        eval_interval,
    )
    lcp = LearningCurvePlot("Q learning")
    lcp.add_curve(
        eval_timesteps,
        smooth(eval_returns, window=35),
        "Q Learning 1 goal",
    )

    # Multiple goals
    eval_returns_extra_goal, eval_timesteps_extra_goal = (
        q_learning(
            n_timesteps,
            learning_rate,
            gamma,
            policy,
            epsilon,
            temp,
            plot,
            eval_interval,
            add_extra_goal=True,
        )
    )
    lcp.add_curve(
        eval_timesteps_extra_goal,
        smooth(eval_returns_extra_goal, window=35),
        "Q learning extra goal",
    )

    lcp.save("QLearningExtraGoal.png")

    zero_reward_plot = LearningCurvePlot("Zero reward / step Q-learning.")
    eval_returns, eval_timesteps = q_learning(
        n_timesteps,
        learning_rate,
        gamma,
        policy,
        epsilon,
        temp,
        plot,
        eval_interval,
        zero_reward=True
    )
    zero_reward_plot.add_curve(eval_timesteps, smooth(eval_returns, window = 9), label="Zero reward / step Q-learning")
    zero_reward_plot.save("ZeroRewardQLearning.pdf")
if __name__ == "__main__":
    test()
