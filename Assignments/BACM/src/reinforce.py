import gymnasium as gym
import yaml
from tqdm import tqdm
from pathlib import Path
from agent import PolicyAgent
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


def sample_monte_carlo(env: gym.Env, agent: PolicyAgent, seed: int) -> PolicyAgent:
    obs, _ = env.reset(seed=seed)
    truncated, terminated = False, False

    while not (truncated or terminated):
        # We let the agent explore
        action, pred, _ = agent.select_action(obs)
        (obs, r, terminated, truncated, _) = env.step(action)

        # For each t we save pi_at_st and r_t
        agent.rewards.append(float(r))
        agent.log_probs.append(pred)

    return agent


def reinforce(
    config: Config,
    env: gym.Env,
    save_plot: Path | None = None,
    plot: bool = True,
    iteration: int | None = None,
):
    agent = PolicyAgent(config.agent, config.nn)
    returns_history = []
    loss_history = []

    total_steps = 0
    max_steps = int(config.run.n_steps)
    episode = 0
    with tqdm(total=max_steps, unit="steps") as pbar:
        while total_steps < max_steps:

            if iteration is not None:
                seed = iteration + len(returns_history)
            else:
                seed = len(returns_history)

            agent = sample_monte_carlo(
                env=env, agent=agent, seed=seed
            )  # Sample trajectory and store it in agent
            info = agent.update()  # Contains loss, step, rewards

            returns_history.append(info["episode_return"])
            loss_history.append(info["loss"])
            pbar.set_postfix(
                loss=f"{info['loss']:.3f}",
                ret=f"{info['episode_return']:.1f}",
                steps=info["step"],
            )

            episode_steps = info["step"]
            total_steps += episode_steps
            episode += 1

    training_info = {"Returns": returns_history, "Loss": loss_history}
    fig = plot_training(training_info, window=20, poly=2, plot=plot)
    if save_plot:
        if iteration is None:
            fig.savefig(save_plot / f"{config.run.name}_training.png")
        else:
            fig.savefig(save_plot / f"{config.run.name}_{iteration}_training.png")

    return training_info


if __name__ == "__main__":
    name = "reinforce"
    save_path = Path("results/")
    save_path.mkdir(exist_ok=True)
    cfg = load_config(name)
    env = gym.make("CartPole-v1")

    reinforce(config=cfg, env=env, save_plot=save_path)
