import gymnasium as gym
import yaml
from tqdm import tqdm
from pathlib import Path
from agent import Agent
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


def reinforce(config: Config, env: gym.Env, save_plot: Path | None = None):
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
    training_info = {"Returns": returns_history, "Loss": loss_history}
    fig = plot_training(training_info, window=20, poly=2)
    if save_plot:
        fig.savefig(save_plot / f"{config.run.name}_training.png")


if __name__ == "__main__":
    name = "base"
    save_path = Path("results/")
    save_path.mkdir(exist_ok=True)
    cfg = load_config(name)
    env = gym.make("CartPole-v1")

    reinforce(config=cfg, env=env, save_plot=save_path)
