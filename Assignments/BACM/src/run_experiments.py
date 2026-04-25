import gymnasium as gym
import yaml
import pickle

from datetime import datetime
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
save_path = Path("results/")
save_path.mkdir(exist_ok=True)
results = {"A2C": [], "AC": [], "REINFORCE": []}
for func, name in functions.items():
    for repetition in range(N_REPETITIONS):
        cfg = load_config(name)
        env = gym.make("CartPole-v1")
        r = func(config=cfg, env=env, save_plot=save_path, plot=False, iteration=repetition)

        results[name].append(r)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(save_path / f"results_{timestamp}.pkl", "wb") as f:
    pickle.dump(results, f)

print(f"Saved results to results/results_{timestamp}.pkl")


#### EXAMPLE ON HOW TO LOAD ###
# with open("results/results_20240101_120000.pkl", "rb") as f:
#     results = pickle.load(f)

# results["A2C"][0]  →  eval_history from first A2C repetition
# results["A2C"][0][3]["mean"]  →  mean return of 4th eval checkpoint

