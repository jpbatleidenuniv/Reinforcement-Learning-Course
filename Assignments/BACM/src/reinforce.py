import gymnasium as gym
import yaml
from tqdm import tqdm
from pathlib import Path
from agent import Agent
from config import RunConfig, AgentConfig, NNConfig, Config
import numpy as np
import matplotlib.pyplot as plt


def load_config(name: str) -> Config:
    config_dir = Path("configs")
    config_path = config_dir / f"{name}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        nn_cfg = NNConfig(**config["nn"])
        agent_cfg = AgentConfig(nn_cfg=nn_cfg, **config["agent"])
        run_cfg = RunConfig(**config["run"])
        print(f"Using config: \n{config}")
    return Config(run=run_cfg, nn=nn_cfg, agent=agent_cfg)


def sample_monte_carlo(env: gym.Env, agent: Agent) -> Agent:
    obs, _ = env.reset()
    truncated, terminated = False, False

    while not (truncated or terminated):
        # We let the agent explore
        action, pred = agent.select_action(obs)
        (obs, r, terminated, truncated, _) = env.step(action)

        # For each t we save pi_at_st and r_t
        agent.rewards.append(float(r))
        agent.log_probs.append(pred)

    return agent


def reinforce(config: Config):
    agent = Agent(config.agent, config.nn)
    returns_history = []
    loss_history = []
    with tqdm(range(config.run.n_episodes)) as pbar:
        for episode in pbar:
            agent = sample_monte_carlo(env=env, agent=agent)
            info = agent.update()  # Contains loss, step, rewards
            returns_history.append(info["episode_return"])
            loss_history.append(info["loss"])
            pbar.set_postfix(
                loss=f"{info['loss']:.3f}",
                ret=f"{info['episode_return']:.1f}",
                steps=info["step"],
            )
    plot_training(returns_history, loss_history)


def plot_training(returns: list, losses: list, window: int = 20):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Smooth with rolling average
    def smooth(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    ax1.plot(returns, alpha=0.3, color="blue", label="raw")
    ax1.plot(smooth(returns, window), color="blue", label=f"smoothed (w={window})")
    ax1.set_ylabel("Episode Return")
    ax1.set_xlabel("Episode")
    ax1.legend()

    ax2.plot(losses, alpha=0.3, color="red", label="raw")
    ax2.plot(smooth(losses, window), color="red", label=f"smoothed (w={window})")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Episode")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training.png")
    plt.show()


if __name__ == "__main__":
    name = "base"
    cfg = load_config(name)
    env = gym.make("CartPole-v1")

    reinforce(config=cfg)
