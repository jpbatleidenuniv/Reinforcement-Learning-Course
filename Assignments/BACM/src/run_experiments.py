import gymnasium as gym
import yaml

from A2C import A2C
from actor_critic import AC
from reinforce import reinforce
from pathlib import Path
from pathlib import Path
from config import RunConfig, AgentConfig, NNConfig, Config


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

N_REPETITIONS = 5

functions = {A2C: "A2C", AC: "AC", reinforce: "REINFORCE"}

results = {"A2C": [], "AC": [], "REINFORCE": []}
for func, name in functions.items():
    for repetition in range(N_REPETITIONS):
        save_path = Path("results/")
        save_path.mkdir(exist_ok=True)
        cfg = load_config(name)
        env = gym.make("CartPole-v1")
        r = func(config=cfg, env=env, save_plot=save_path, plot=False, iteration=repetition)

        results[name].append(r["Returns"])

