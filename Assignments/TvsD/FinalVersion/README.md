# DQN on CartPole

A modular Deep Q-Network (DQN) implementation for the CartPole-v1 environment, supporting vectorized environments, experience replay, and target networks. Includes hyperparameter search and ablation study pipelines with automated result plotting.

---

## File Overview

| File | Description |
|---|---|
| `DQN.py` | Q-network architecture and `DQNAgent` (action selection, training, evaluation) |
| `cartopole.py` | CartPole-v1 wrapper with optional vectorization |
| `replay_buffer.py` | Fixed-capacity FIFO experience replay buffer |
| `run_dqn.py` | Core training loop (`train_dqn_naive`) |
| `run_experiments.py` | Experiment configs, parallel repetitions, result saving |
| `OptimalHyperparameter.py` | Hyperparameter sweep analysis and ranking |
| `AblationAnalysis.py` | Ablation study plotting (Naive / Buffer / Target / Both) |
| `Helper.py` | Smoothing, softmax, learning curve plot utilities |

---

## Setup

```bash
pip install numpy torch gymnasium matplotlib scipy statsmodels tqdm
```

---

## Usage

### Run a single experiment config (by index)
```bash
python run_experiments.py 0
```

This will not produce any statistical results or figures. 

### Run all experiment configs sequentially
```bash
python run_experiments.py
```

This will run all configs defined in `CONFIGS`, save results under `results/` and `results/combined/`, and then automatically generate the hyperparameter and ablation plots.

### Run only the analysis/plotting
```bash
python OptimalHyperparameter.py
python AblationAnalysis.py
```

---

## Experiment Configs

Configs are defined as `RunConfig` dataclasses in `run_experiments.py`. Each config controls:

| Parameter | Description |
|---|---|
| `n_envs` | Number of parallel environments |
| `policy` | Exploration strategy: `"softmax"` or `"epsilon_greedy"` |
| `temperature` / `epsilon` | Exploration parameter |
| `layers` / `width` | Network depth and width |
| `lr` | Learning rate |
| `batch_size` | Steps accumulated before each network update |
| `buffer` | Enable experience replay |
| `buffer_size` / `min_buffer_size` | Replay buffer capacity and minimum fill before training |
| `target_network` | Enable target network |
| `update_target` | Steps between target network syncs |
| `maximum_steps` | Total environment steps per run |

The predefined configs cover four study groups: exploration policies, network architectures, learning rates, and batch/data-to-update ratios, plus four ablation conditions.

## Results

After running, results are saved as `.npy` files:

```
results/
  {config_name}_returns_rep{i}.npy      # per-repetition eval returns
  {config_name}_timesteps_rep{i}.npy    # per-repetition eval timesteps
  combined/
    {config_name}_all_returns.npy       # shape (N_REPS, T)
    {config_name}_all_timesteps.npy     # shape (N_REPS, T)
```

Plots are saved as `HyperParameterResults.png` and `AblationResults.png`.

