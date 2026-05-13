import gymnasium as gym
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from reference_agent import PolicyAgent
from config import RunConfig, AgentConfig, NNConfig, Config
from plots import plot_training


def load_config(name: str) -> Config:
    config_path = f"{name}.yaml"
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


def evaluate(
    env: gym.Env,
    policy_agent: PolicyAgent,
    n_episodes: int = 10,
    seed_offset: int = 0,
) -> dict:
    """
    Run `n_episodes` evaluation episodes using a greedy policy.
    The agent is not updated and no gradients are tracked.

    Returns a dict with per-episode returns and their mean/std.
    """
    policy_agent.policy.eval()
    ep_returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        truncated, terminated = False, False
        ep_reward = 0.0

        while not (truncated or terminated):
            action = policy_agent.select_greedy_action(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_reward += float(r)

        ep_returns.append(ep_reward)


    policy_agent.policy.train()

    return {
        "returns": ep_returns,
        "mean": float(np.mean(ep_returns)),
        "std": float(np.std(ep_returns)),
    }


def reinforce(config: Config, env: gym.Env, save_plot: Path | None = None, plot: bool = True, iteration: int | None = None, eval_interval: int = 5000, n_eval_episodes: int = 10):

    agent = PolicyAgent(config.agent, config.nn)
    returns_history = []
    loss_history = []
    eval_history: list[dict] = []   # {"step": int, "mean": float, "std": float}


    total_steps = 0
    max_steps = int(config.run.n_steps)
    episode = 0
    next_eval_at = eval_interval

    with tqdm(total=max_steps, unit="steps") as pbar:
        while total_steps < max_steps:
            
            if iteration is not None:
                seed = iteration * len(returns_history) + iteration
            else:
                seed = len(returns_history)
            
            agent = sample_monte_carlo(env=env, agent=agent, seed=seed) # Sample trajectory and store it in agent
            info = agent.update()  # Contains loss, step, rewards

            episode_steps = info["step"]
            total_steps += episode_steps
            episode += 1

            #  Evaluation round
            if total_steps >= next_eval_at:
                eval_info = evaluate(
                    env=env,
                    policy_agent=agent,
                    n_episodes=n_eval_episodes,
                    seed_offset=total_steps,   
                )
                eval_info["step"] = total_steps
                eval_history.append(eval_info)

                tqdm.write(
                    f"[Eval @ {total_steps:,} steps]  "
                    f"mean return = {eval_info['mean']:.1f} ± {eval_info['std']:.1f}  "
                    f"(over {n_eval_episodes} greedy episodes)"
                )
                next_eval_at += eval_interval

            returns_history.append(info["episode_return"])
            loss_history.append(info["loss"])
            pbar.set_postfix(
                loss=f"{info['loss']:.3f}",
                ret=f"{info['episode_return']:.1f}",
                steps=info["step"],
            )
            pbar.update(episode_steps)



    training_info = {"Returns": returns_history, "Loss": loss_history}
    fig = plot_training(training_info, window=20, poly=2, plot=plot)
    if save_plot:
        if iteration is None:
            fig.savefig(save_plot / f"{config.run.name}_training.png")
        else:
            fig.savefig(save_plot / f"{config.run.name}_{iteration}_training.png")

    return eval_history


if __name__ == "__main__":
    name = "REINFORCE"
    save_path = Path("results/")
    save_path.mkdir(exist_ok=True)
    cfg = load_config(name)
    env = gym.make("CartPole-v1")

    reinforce(config=cfg, env=env, save_plot=save_path)
