import gymnasium as gym
import yaml
import numpy as np
import torch

from tqdm import tqdm
from pathlib import Path
from agent import PolicyAgent, ValueAgent
from config import RunConfig, AgentConfig, NNConfig, Config
from plots import plot_training


def load_config(name: str) -> Config:
    config_path = f"configs/{name}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        nn_cfg    = NNConfig(**config["nn"])
        agent_cfg = AgentConfig(nn_cfg=nn_cfg, **config["agent"])
        run_cfg   = RunConfig(**config["run"])
        print(f"Using config: \n{config}")
    return Config(run=run_cfg, nn=nn_cfg, agent=agent_cfg)


# Bufffer
class Buffer:
    def __init__(self):
        self.states:         list[torch.Tensor] = []
        self.actions:        list[int]          = []
        self.rewards:        list[float]        = []
        self.dones:          list[bool]         = []
        self.probs:          list[torch.Tensor] = []
        self.terminal_state: torch.Tensor | None = None
        self.terminated:     bool               = False

    def add(self, state, action, reward, done, old_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.probs.append(old_prob)

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.states         = []
        self.actions        = []
        self.rewards        = []
        self.dones          = []
        self.probs          = []
        self.terminal_state = None
        self.terminated     = False


# ── Trajectory sampling ───────────────────────────────────────────────────────

def sample_trajectory(env: gym.Env, policy_agent: PolicyAgent, seed: int,) -> Buffer:
    """Rolls out one episode under the current old policy and returns a Buffer."""

    buf = Buffer()
    obs, _ = env.reset(seed=seed)
    truncated, terminated = False, False

    while not (truncated or terminated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)   # s_t (before step)
        action, old_pi_sa = policy_agent.select_action(obs)
        obs, r, terminated, truncated, _ = env.step(action)
        buf.add(state=obs_tensor, action=action, reward=r,
                done=(terminated or truncated), old_prob=old_pi_sa)

    buf.terminal_state = torch.tensor(obs, dtype=torch.float32)
    buf.terminated     = bool(terminated)
    return buf


# GAE 
def GAE(rewards, values, dones, gamma, lambda_):
    """Computes Generalised Advantage Estimates for one trajectory."""
    advantages    = []
    last_advantage = 0
    for t in reversed(range(len(rewards))):
        mask           = 1 - dones[t]
        delta          = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last_advantage = delta + gamma * lambda_ * mask * last_advantage
        advantages.insert(0, last_advantage)
    return advantages


#  Evaluation 
def evaluate(env: gym.Env, policy_agent: PolicyAgent,
             n_episodes: int = 10, seed_offset: int = 0) -> dict:
    """Greedy evaluation over n_episodes. No gradient tracking."""
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
        "mean":    float(np.mean(ep_returns)),
        "std":     float(np.std(ep_returns)),
    }


# PPO 
def PPO(
        config:           Config,                # Configuration file
        env:              gym.Env,               # Instance of carpole environment
        save_plot:        Path | None = None,    # Path to save the plot
        plot:             bool        = True,    # Plot the results or not
        iteration:        int | None  = None,    # Iteraion number, used for seeding when running an experiment
        eval_interval:    int         = 5000,    # How many steps between evaluation episodes
        n_eval_episodes:  int         = 10,      # How many evaluation episodes
        n_trajectories:   int         = 4,       # How many trajectories to sample before updating
        ):
    """
    PPO training loop.

    Each outer iteration:
      1. Collect `n_trajectories` full episodes under the old policy.
      2. Run `config.agent.k` gradient-update epochs over the combined batch.
      3. Copy new to old policy.
    """
    policy_agent = PolicyAgent(config.agent, config.nn)
    value_agent  = ValueAgent(config.agent, config.nn)

    returns_history: list[int]   = []
    loss_history                 = {"Policy": [], "Value": []}
    eval_history:    list[dict]  = []

    total_steps  = 0
    max_steps    = int(config.run.n_steps)
    update_count = 0
    next_eval_at = eval_interval

    with tqdm(total=max_steps, unit="steps") as pbar:
        while total_steps < max_steps:

            # Collect n_trajectories episodes
            buffers: list[Buffer] = []
            batch_steps = 0

            for traj_idx in range(n_trajectories):
                # Seeding
                if iteration is not None:
                    seed = iteration * 10_000 + update_count * n_trajectories + traj_idx
                else:
                    seed = update_count * n_trajectories + traj_idx
                
                # Sample trajectory and store in buffer
                buf = sample_trajectory(env, policy_agent, seed)
                buffers.append(buf)
                batch_steps += len(buf)

            # Freeze old-policy probabilities for the whole batch before any update
            old_probs_per_traj = [torch.stack(buf.probs).detach() for buf in buffers]

            # K epochs of updates over the combined batch 
            for epoch in range(config.agent.k):

                batch_advantages: list[torch.Tensor] = []
                batch_ratios:     list[torch.Tensor] = []
                value_losses:     list[float]        = []

                for buf, old_probs in zip(buffers, old_probs_per_traj):

                    # Fresh value estimates with current network weights
                    states_tensor  = torch.stack(buf.states)                    # [T, dim]
                    v_s_current    = value_agent.value(states_tensor).squeeze(-1)  # [T]

                    v_terminal_val = value_agent.value(
                        buf.terminal_state.unsqueeze(0)
                    ).squeeze()   
                    
                    # Zero terminal bootstrap if episode ended naturally
                    v_terminal = v_terminal_val * float(not buf.terminated)
                    v_s_full   = torch.cat([v_s_current, v_terminal.unsqueeze(0)])  # [T+1]

                    # GAE (
                    gae_list = GAE(
                        rewards  = buf.rewards,
                        values   = v_s_full.detach(),
                        dones    = buf.dones,
                        gamma    = config.agent.gamma,
                        lambda_  = config.nn.lamb,
                    )
                    gae = torch.stack(gae_list).to(dtype=torch.float32)   # [T]
                    batch_advantages.append(gae)

                    # New-policy action probabilities and importance ratios
                    new_probs = policy_agent.probability_new_policy(buf.states, buf.actions)
                    batch_ratios.append(new_probs / old_probs)

                    # Value network update
                    v_info = value_agent.update(v_s=v_s_full, r=buf.rewards, done=buf.dones)
                    value_losses.append(v_info["loss"])

                # Policy update over the whole batch (all trajectories concatenated)
                all_advantages = torch.cat(batch_advantages)   
                all_ratios     = torch.cat(batch_ratios)       
                policy_info    = policy_agent.update(advantage=all_advantages, ratios=all_ratios)

            # Bookkeeping 
            total_steps  += batch_steps
            update_count += 1

            # Update old policy with new weights after K epochs of updates
            policy_agent.update_old_policy()

            mean_value_loss = float(np.mean(value_losses))

            #  Evaluation
            if total_steps >= next_eval_at:
                eval_info        = evaluate(env, policy_agent, n_eval_episodes, total_steps)
                eval_info["step"] = total_steps
                eval_history.append(eval_info)
                tqdm.write(
                           f"[Eval @ {total_steps:,} steps]  "
                           f"mean return = {eval_info['mean']:.1f} ± {eval_info['std']:.1f}  "
                           f"(over {n_eval_episodes} greedy episodes)"
                           )
                next_eval_at += eval_interval

            returns_history.append(batch_steps)
            loss_history["Policy"].append(policy_info["loss"])
            loss_history["Value"].append(mean_value_loss)

            pbar.set_postfix(
                             policy_loss = f"{policy_info['loss']:.3f}",
                             value_loss  = f"{mean_value_loss:.3f}",
                             update      = update_count,
                             steps       = batch_steps,
                            )
            pbar.update(batch_steps)

    training_info = {
                    "Returns":     returns_history,
                    "Policy Loss": loss_history["Policy"],
                    "Value_Loss":  loss_history["Value"],
                    "Eval":        eval_history,
                    }
    
    plot_data = {k: v for k, v in training_info.items() if k != "Eval"}
    fig = plot_training(plot_data, window=20, poly=2, plot=plot)
    if save_plot:
        suffix = f"_{iteration}" if iteration is not None else ""
        fig.savefig(save_plot / f"{config.run.name}{suffix}_training.png")  # type: ignore

    return eval_history


if __name__ == "__main__":
    name      = "PPO"
    save_path = Path("results/")
    save_path.mkdir(exist_ok=True)
    cfg = load_config(name)
    env = gym.make("CartPole-v1")

    PPO(config=cfg, env=env, save_plot=save_path, iteration=1, n_trajectories=4)