#!/bin/bash
#SBATCH --job-name=cartpole_repeated
#SBATCH --output=logs/%x_%j_%a.out        # separate log per config
#SBATCH --mail-user="jbaars@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00                   # 20 reps x 10^6 steps each
#SBATCH --partition=cpu-zen4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-3                       # one job per config (4 configs: 0,1,2,3)

module load Python/3.11.5-GCCcore-13.2.0
cd $HOME/data1/Reinforcement-Learning-Course/Assignments/TvsD/
source $HOME/data1/Reinforcement-Learning-Course/.venv/bin/activate



echo "[$SHELL] #### CartPole repeated runs"
echo "[$SHELL] ## Job: $SLURM_JOB_ID | Array task: $SLURM_ARRAY_TASK_ID"
echo "[$SHELL] ## Host: $HOSTNAME | User: $SLURM_JOB_USER"

mkdir -p logs results

echo "[$SHELL] ## Running config $SLURM_ARRAY_TASK_ID"
uv run run_experiments.py $SLURM_ARRAY_TASK_ID
echo "[$SHELL] ## Config $SLURM_ARRAY_TASK_ID finished"
