# Assignment 3: REINFORCE and Actor-Critic Methods

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

### Run all experiments (REINFORCE, AC, A2C) — 5 repetitions each

```bash
python run_experiments.py
```

Results are saved to `results/results_<timestamp>.pkl`.

### Run a single algorithm

```bash
python reinforce.py
python actor_critic.py
python A2C.py
```

Each script loads its config from `configs/<name>.yaml` and saves a training plot to `results/`.

## Reproducing the Comparison Plot

After running the experiments, update the `.pkl` filename in `plots.py` to match your results file, then run:

```bash
python plots.py
```

This saves `results/comparison.png` with the learning curves for all methods alongside the DQN baseline.

## Configuration

Each algorithm has a config file in `configs/`:

- `configs/REINFORCE.yaml`
- `configs/AC.yaml`
- `configs/A2C.yaml`