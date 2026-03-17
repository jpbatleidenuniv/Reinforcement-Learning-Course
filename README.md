# Assignment 1: Tabular Reinforcement Learning

This repository contains the code for **Assignment 1: Tabular Reinforcement Learning** on the **Stochastic Windy Gridworld**. The assignment covers:

- Dynamic Programming (Q-value iteration)
- Q-learning
- SARSA
- n-step Q-learning
- Monte Carlo reinforcement learning
- Experiment scripts for comparing exploration strategies, backup targets, and backup depth

The README is written to satisfy the submission requirement that the experiments can be rerun on a Linux machine with a **single command per subtask**. The assignment explicitly asks for a README with instructions that allow the grader to rerun each experiment easily on Linux. fileciteturn1file1

## Environment

The environment is the **Stochastic Windy Gridworld**:

- Grid size: **10 x 7**
- Start state: **(0, 3)**
- Goal state: **(7, 3)**
- Actions: **up, right, down, left**
- Wind strength by column: `(0, 0, 0, 1, 1, 1, 2, 2, 1, 0)`
- Wind blows with probability **0.9** in the current implementation
- Step reward: **-1**
- Goal reward: **+100**

These settings are defined in `Environment.py`.
## Project structure

- `Environment.py` – Windy Gridworld environment and rendering
- `Helper.py` – plotting utilities, smoothing, `softmax`, and tie-breaking `argmax`
- `Agent.py` – base agent with greedy, epsilon-greedy, and softmax action selection
- `DynamicProgramming.py` – Q-value iteration for Dynamic Programming
- `Q_learning.py` – one-step Q-learning
- `SARSA.py` – one-step SARSA
- `Nstep.py` – n-step Q-learning
- `MonteCarlo.py` – Monte Carlo control
- `Experiment.py` – main experiment script for assignment comparisons
- `SaveResults.py` – saves experiment outputs to CSV files
- `plot_results.py` – helper script to plot saved CSV result files

This file organization matches the assignment components for DP, exploration, on-policy vs off-policy backup, and backup depth. 

## Requirements

Use **Python 3**.

- `numpy`
- `matplotlib`
- `scipy`
- `pandas`
- `tqdm`
- `statsmodels`

You can install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```
## Running the code

Run all commands from the project root directory.

### 1) Dynamic Programming

Runs tabular Q-value iteration and visualizes the value updates:

```bash
python3 DynamicProgramming.py
```

What it does:

- initializes a `QValueIterationAgent`
- repeatedly sweeps through all state-action pairs
- updates values with the environment model
- renders the Q-table and greedy policy during convergence
- writes summary statistics to `Dynamic_programming_avg_reward_per_timestep.txt` when `a_lot=True`

Implementation details are in `DynamicProgramming.py`.

### 2) Q-learning

Runs one-step Q-learning and saves example plots:

```bash
python3 Q_learning.py
```

What it does:

- trains a Q-learning agent
- evaluates the greedy policy periodically
- generates a comparison with an extra goal
- generates a zero-reward-per-step experiment
- saves plots such as `QLearningExtraGoal.png` and `ZeroRewardQLearning.pdf`

Q-learning uses the update
`Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]`.
The implementation is in `Q_learning.py`.

### 3) SARSA

Runs one-step SARSA:

```bash
python3 SARSA.py
```

What it does:

- trains a SARSA agent with online interaction
- evaluates the greedy policy periodically
- optionally renders the learned Q-values and greedy policy

The implementation follows the on-policy SARSA update in `SARSA.py`. 

### 4) n-step Q-learning

Runs n-step Q-learning with the default test settings:

```bash
python3 Nstep.py
```

What it does:

- collects complete episodes
- applies the n-step return update after each episode
- evaluates the greedy policy periodically

The default test uses `n = 5`. See `Nstep.py` for details. 

### 5) Monte Carlo

Runs Monte Carlo control:

```bash
python3 MonteCarlo.py
```

What it does:

- collects a full episode
- computes returns backward through the episode
- updates `Q(s,a)` from the Monte Carlo target
- evaluates periodically and saves a test plot

The implementation is in `MonteCarlo.py`. 

### 6) Full experiment suite

Runs the full set of experiments required for the RL comparisons:

```bash
python3 Experiment.py
```

This script averages over **20 repetitions** and reproduces the main comparison plots for:

- **Exploration:** epsilon-greedy with `epsilon = [0.03, 0.1, 0.3]` and softmax with `temp = [0.01, 0.1, 1.0]`
- **On-policy vs off-policy:** Q-learning vs SARSA with `learning_rate = [0.03, 0.1, 0.3]`
- **Backup depth:** n-step Q-learning with `n = [1, 3, 10]` plus Monte Carlo

The experiment script uses:

- `n_timesteps = 50001`
- `eval_interval = 1000`
- `n_repetitions = 20`
- `gamma = 1.0`

and saves plots:

- `exploration.png`
- `on_off_policy.png`
- `depth.png`

It also stores raw CSV results in the `results/` folder through `SaveResults.py`. 


## Notes on rendering

`Environment.py` uses a Matplotlib GUI backend:

```python
matplotlib.use('TkAgg')
```

If interactive rendering does not work on your machine, change the backend in `Environment.py`, as also suggested by the assignment instructions. The assignment notes that `Qt5Agg` or `TkAgg` may be needed depending on the local setup. 

For headless servers, disable plotting in the relevant test or experiment functions before running large batches.

## Saved outputs

Depending on which scripts are run, the code can generate:

- `exploration.png`
- `on_off_policy.png`
- `depth.png`
- `QLearningExtraGoal.png`
- `ZeroRewardQLearning.pdf`
- `MonteCarloTest.png`
- `Dynamic_programming_avg_reward_per_timestep.txt`
- CSV files in `results/`

The combined experiment script saves averaged CSV outputs using parameter-encoded filenames via `SaveResults.py`. 

