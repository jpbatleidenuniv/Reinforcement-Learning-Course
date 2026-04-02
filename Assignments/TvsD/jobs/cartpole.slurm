#!/bin/bash
#SBATCH --job-name=cartpole_dqn
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --mail-user="jbaars@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=2G                  # bumped up: torch + gymnasium needs more than 1G
#SBATCH --time=02:00:00                   # adjust based on your maximum_steps
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2                 # DQN on CartPole is single-threaded, 2 is enough
#SBATCH --array=0-17                      # one job per experiment: adjust upper bound to len(EXPLORATIONS)-1

# ------------------ Environment setup ------------------
module load Python/3.11.5-GCCcore-13.2.0
cd $HOME/data1/Reinforcement-Learning-Course/Assignments/TvsD/
source $HOME/data1/Reinforcement-Learning-Course/.venv/bin/activate

# ------------------ Logging ------------------
echo "[$SHELL] #### Starting CartPole DQN experiment"
echo "[$SHELL] ## User: $SLURM_JOB_USER on $HOSTNAME"
echo "[$SHELL] ## Job ID: $SLURM_JOB_ID | Array task: $SLURM_ARRAY_TASK_ID"
export CWD=$(pwd)
echo "[$SHELL] ## Working directory: $CWD"

# ------------------ Create output dirs ------------------
mkdir -p logs results cartpole-videos

# ------------------ Run ------------------
echo "[$SHELL] ## Running experiment $SLURM_ARRAY_TASK_ID"
uv run Cartpole.py $SLURM_ARRAY_TASK_ID
echo "[$SHELL] ## Experiment $SLURM_ARRAY_TASK_ID finished"
