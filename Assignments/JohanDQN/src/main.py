import gymnasium as gym
from environment import run_episode
from agent import Agent
from run_config import RunConfig


def init_seed():
    pass


def run(run_config: RunConfig):
    agent = Agent()
    step = 0

    env = gym.make("CartPole-v1")
    obs, _ = env.reset()

    eval_returns = []
    eval_timesteps = []

    while step < n_steps:
        step += 1


if __name__ == "__main__":
    n_steps = 1000
    run(n_steps)
