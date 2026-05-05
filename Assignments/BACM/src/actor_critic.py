from pathlib import Path
from tqdm import tqdm
import gymnasium as gym
from agent import A2CAgent, ACAgent
from config import Config
from reinforce import sample_monte_carlo
from plots import plot_training
from utils import load_config


def actor_critic(
    config: Config,
    agent: ACAgent | A2CAgent,
    save_plot: Path | None = None,
):
    # Creating the environment and the score placeholders
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    returns_history = []
    loss_history = []
    eval_history = []
    eval_steps = []  # x-axis for eval plot
    total_steps = 0  # cumulative environment steps

    with tqdm(range(config.run.n_episodes)) as pbar:
        for episode in pbar:
            agent = sample_monte_carlo(env=env, agent=agent)
            info = agent.update()  # Contains loss, step, rewards

            total_steps += info["step"]
            returns_history.append((total_steps, info["episode_return"]))
            loss_history.append((total_steps, info["loss"]))

            if episode % config.run.evaluation_interval == 0:
                eval_return = agent.evaluate(
                    eval_env,
                    n_episodes=config.run.n_eval_episodes,
                )
                eval_history.append(eval_return)
                eval_steps.append(total_steps)

            pbar.set_postfix(
                loss=f"{info['loss']:.3f}",
                value_loss=f"{info['value_loss']:.3f}",
                ret=f"{info['episode_return']:.1f}",
                steps=info["step"],
            )
    training_info = {"Returns": returns_history, "Loss": loss_history}
    eval_info = {"Timesteps": eval_steps, "Evaluation Returns": eval_history}
    fig = plot_training(training_info, window=20, poly=2)
    if save_plot:
        fig.savefig(save_plot / f"{config.run.name}_training.png")

    return training_info, eval_info


if __name__ == "__main__":
    name = "base_ac"
    save_path = Path("results/")
    save_path.mkdir(exist_ok=True)
    cfg = load_config(name)

    ac_agent = A2CAgent(cfg.agent, cfg.pi_nn, value_network_cfg=cfg.v_nn)
    actor_critic(cfg, ac_agent, save_plot=save_path)
