#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCG --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=10gb
#SBATCH --job-name=minigrid-hyperparam-search

module load Python/3.8.16-GCCcore-11.2.0 
module load CUDA/12.1.1
source $HOME/venvs/cbet/bin/activate
OMP_NUM_THREADS=1 python main.py --model cbet --env MiniGrid-Unlock-v0 --intrinsic_reward_coef=0.001 --total_frames 1000000 --num_actors 8 --savedir ../logs/impala --xpid unlock-tabula-rasa-1M-coeff-0.001