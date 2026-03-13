# Tabular Reinforcement Learning – Assignment 1

This repository contains the implementation for **Assignment 1: Tabular Reinforcement Learning**.  
The goal is to implement and compare several reinforcement learning algorithms in the **Stochastic Windy Gridworld** environment.

Implemented algorithms:

- Q-value Iteration (Dynamic Programming)
- Q-learning
- SARSA
- n-step Q-learning
- Monte Carlo learning


---

# Requirements

The code was developed with **Python 3**.

Install the required packages:

pip install numpy matplotlib scipy pandas tqdm statsmodels


---

# Repository Structure

Main files in the project:

Environment.py        # Windy Gridworld environment  
Agent.py              # Base RL agent with action selection  
DynamicProgramming.py # Q-value iteration implementation  
Q_learning.py         # Q-learning implementation  
SARSA.py              # SARSA implementation  
Nstep.py              # n-step Q-learning implementation  
MonteCarlo.py         # Monte Carlo RL implementation  
Experiment.py         # Runs all experiments  
Helper.py             # Utility functions  
plot_results.py       # Plot results from saved CSV files  
SaveResults.py        # Save experiment results  


---

# How to Run the Code

## 1. Dynamic Programming (Q-value Iteration)

Run:

python DynamicProgramming.py

This will:

- run Q-value iteration
- visualize the gridworld
- display the optimal policy and Q-values.


---

## 2. Run Reinforcement Learning Experiments

Run:

python Experiment.py

This script runs three experiments:

### Exploration comparison

ε-greedy with:
- ε = 0.03
- ε = 0.1
- ε = 0.3

Softmax with:
- τ = 0.01
- τ = 0.1
- τ = 1.0


### Q-learning vs SARSA

Learning rates tested:

α = 0.03  
α = 0.1  
α = 0.3  


### Backup depth comparison

Methods compared:

- 1-step Q-learning
- n-step Q-learning (n = 3, 10)
- Monte Carlo


Each experiment:

- runs **20 repetitions**
- trains for **50,001 timesteps**
- evaluates every **1000 steps**

Results are automatically saved in the **results/** folder.


---

# Plot Results

To plot the stored experiment results:

python plot_results.py

This script reads the saved CSV files and produces learning curve plots.


---

# Notes

- Rendering during training is slow, so plotting is disabled during repeated experiments.
- The environment supports both model-based interaction (for Dynamic Programming) and model-free interaction (for reinforcement learning algorithms).


---

# Author

Reinforcement Learning – Assignment 1  
Leiden University
