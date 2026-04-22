import gymnasium as gym
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from agent import PolicyAgent, ValueAgent
from config import RunConfig, AgentConfig, NNConfig, Config
from plots import plot_training


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


def sample_monte_carlo(env: gym.Env, policy_agent: PolicyAgent, value_agent: ValueAgent) -> tuple[PolicyAgent, ValueAgent]:
    obs, _ = env.reset()
    truncated, terminated = False, False
    rewards = []
    log_probs = []
    V_ss = []

    while not (truncated or terminated):
        # We let the agent explore
        action, pred, pi_s = policy_agent.select_action(obs)
        Q_s = value_agent.values(obs)
        (obs, r, terminated, truncated, _) = env.step(action)
        V_s = torch.sum(Q_s * pi_s.detach(), dtype=torch.float32)

        # For each t we save pi_at_st and r_t
        rewards.append(float(r))
        log_probs.append(pred)
        V_ss.append(V_s)

    policy_agent.rewards = rewards
    policy_agent.log_probs = log_probs
    value_agent.rewards = rewards
    value_agent.V_s = V_ss

    return policy_agent, value_agent


def A2C(config: Config, env: gym.Env, save_plot: Path | None = None):

    policy_agent = PolicyAgent(config.agent, config.nn)
    value_agent = ValueAgent(config.agent, config.nn, advantage=True)

    returns_history = []
    loss_history = {"Policy": [], "Value": []}
    
    total_steps = 0
    max_steps = 10**6
    episode = 0

    with tqdm(total=max_steps, unit="steps") as pbar:
        while total_steps < max_steps:
            policy_agent, value_agent = sample_monte_carlo(env=env, policy_agent=policy_agent, value_agent=value_agent)
            G_t = value_agent.G_t
            policy_info = policy_agent.update(objectives=G_t)
            value_info = value_agent.update()

            episode_steps = policy_info["step"]
            total_steps += episode_steps
            episode += 1

            returns_history.append(policy_info["episode_return"])
            loss_history["Policy"].append(policy_info["loss"])
            loss_history["Value"].append(value_info["loss"])

            mean_return = np.mean(returns_history[-100:])

            pbar.set_postfix(
                policy_loss=f"{policy_info['loss']:.3f}",
                value_loss=f"{value_info['loss']:.3f}",
                episode=episode,
                mean_return=mean_return,
            )
            pbar.update(episode_steps)

            if mean_return >= 475:
                print("\n")
                print(f"Solved at episode {episode} with a mean return over the last 100 episodes of {mean_return}")
                break

    training_info = {"Returns": returns_history, "Policy Loss": loss_history["Policy"], "Value_Loss": loss_history["Value"]}
    fig = plot_training(training_info, window=20, poly=2)
    if save_plot:
        fig.savefig(save_plot / f"{config.run.name}_training.png")


if __name__ == "__main__":
    name = "A2C"
    save_path = Path("results/")
    save_path.mkdir(exist_ok=True)
    cfg = load_config(name)
    env = gym.make("CartPole-v1")

    A2C(config=cfg, env=env, save_plot=save_path)
