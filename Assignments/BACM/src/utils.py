import yaml
import numpy as np
from gymnasium import Env
from pathlib import Path
from scipy.signal import savgol_filter
from config import Config, RunConfig, NNConfig, AgentConfig


def load_config(name: str) -> Config:
    """Loads the Config dataclass from named configuration given in the configs/ directory"""
    config_dir = Path("configs")
    config_path = config_dir / f"{name}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        policy_network_cfg = NNConfig(**config["nn"]["policy_network"])
        value_network_cfg = NNConfig(**config["nn"]["value_network"])
        agent_cfg = AgentConfig(nn_cfg=policy_network_cfg, **config["agent"])
        run_cfg = RunConfig(**config["run"])
        print(f"Using config: \n{config}")
    return Config(
        run=run_cfg, pi_nn=policy_network_cfg, v_nn=value_network_cfg, agent=agent_cfg
    )


def sample_monte_carlo(env: Env, agent):
    """Sample monte carlo episode for a given agent, returns the experienced agent"""
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


def smooth(y, window, poly):
    """Savgol smoothing filter"""
    y = np.asarray(y)
    # Ensure valid window
    if len(y) < window:
        return y

    if window % 2 == 0:
        window += 1  # must be odd
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)

    if window < 3:
        return y

    return np.array(savgol_filter(y, window, poly))
