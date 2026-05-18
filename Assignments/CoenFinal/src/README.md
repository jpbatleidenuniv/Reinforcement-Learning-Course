# Assignment 4: PPO (Proximal Policy Optimization)

Implementation of PPO-clipped for CartPole-v1, with hyperparameter optimization and comparison against DQN, REINFORCE, and A2C from previous assignments.

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

### Quick-Start

to run the experiments run: 

```bash
python run_hp_tuning.py
```
For the hp tuning, and

```bash
python run_comparison.py
```

For the comparison to DQN, REINFORCE, and A2C. The results for these algorithms were collected from the previous assignments. 


### Hyperparameter tuning

```bash
python run_hp_tuning.py
```

Sweeps over `n_trajectories` ∈ {1, 5, 15}, λ ∈ {0.7, 0.9, 0.99}, and ε ∈ {0.05, 0.1, 0.2}, with 5 repetitions each. Results are saved to `results/ablation_independent_<timestamp>.pkl` and a figure is saved to `hp_tuning.png`.


### Comparison run 

```bash
python run_comparison.py
```
Runs 5 iterations for tuned PPO configurations and plots the results, and saves the results to `results/results_<timestamp>.pkl` and the figure to `comparison.png`

### Single PPO run

```bash
python PPO.py
```

Runs one training session using `configs/PPO.yaml` and saves a training plot to `results/`.

### Plot tuning results

```bash
python plot_hp_tuning.py
```
Can plot the figure again if needed. Does require manually entering the path to the results from the hyperparameter tuning 

### Comparison plot (PPO vs A2C vs REINFORCE vs DQN)

```bash
python plot_comparison.py
```

Requires result files from previous assignments to be present:
- `results/results_<timestamp>.pkl` — PPO results (update path in `plot_comparison.py`)
- `results_20260427_204003.pkl` — A2C / REINFORCE results from Assignment 3
- `Naive_all_returns.npy` / `Naive_all_timesteps.npy` — DQN results from Assignment 2

## File Overview

| File | Description |
|---|---|
| `PPO.py` | Main training loop |
| `agent.py` | Policy and value network definitions and update logic |
| `config.py` | Dataclasses for run / network / agent configuration |
| `run_hp_tuning.py` | HP tuning runner |
| `plot_hp_tuning.py` | Plots HP tuning results |
| `plot_comparison.py` | Plots PPO vs baselines from prior assignments |
| `plots.py` | Shared plotting utilities |
| `utils.py` | Savitzky-Golay smoothing helper |