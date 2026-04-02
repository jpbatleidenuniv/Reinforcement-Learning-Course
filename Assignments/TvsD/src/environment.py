import gymnasium as gym
from time import sleep


def run_episode(env: gym.Env):
    total_reward = 0
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(
            action
        )
        print(type(reward))
        total_reward += reward.__float__()
        sleep(0.2)

        if terminated or truncated:
            print(
                f"Episode ended at step {step + 1} because of {'terminated' if terminated else 'truncated'}, total reward: {total_reward}"
            )
            break
    env.close()


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    env.reset()
    run_episode(env)
