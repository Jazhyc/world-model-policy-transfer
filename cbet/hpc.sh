#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCG --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=20gb
#SBATCH --job-name=minigrid-stress-impala
## SBATCH --array=0-2
#coeff_strength=(0.001 0.0025 0.005)

module load Python/3.8.16-GCCcore-11.2.0 
module load CUDA/12.1.1
source $HOME/venvs/cbet/bin/activate

# Regular
# OMP_NUM_THREADS=1 python main.py --model cbet --env CrafterReward-v0 --intrinsic_reward_coef=0.0 --total_frames 2000000 --num_actors 8 --savedir ../logs/impala --xpid crafter-base-2M

# Pre-train minigrid Doorkey
# OMP_NUM_THREADS=1 python main.py --model cbet --env MiniGrid-DoorKey-8x8-v0 --intrinsic_reward_coef=0.0025 --total_frames 1000000 --num_actors 8 --savedir ../logs/impala --xpid minigrid-pretrain-1M-Doorkey8x8-2 --no_reward

# Transfer minigrid
# OMP_NUM_THREADS=1 python main.py --model cbet --env MiniGrid-Unlock-v0 --intrinsic_reward_coef=0.0 --total_frames 1000000 --num_actors 8 --savedir ../logs/impala --xpid minigrid-transfer-1M-unlock --checkpoint=../logs/impala/minigrid-pretrain-1M-Doorkey8x8/model.tar

# Pre-train crafter cbet
OMP_NUM_THREADS=1 python main.py --model cbet --env CrafterReward-v0 --intrinsic_reward_coef=0.0025 --total_frames 1000000 --num_actors 8 --savedir ../logs/impala --xpid crafter-pretrain-1M-2 --seed 42 --no_reward

# Transfer crafter cbet
# OMP_NUM_THREADS=1 python main.py --model cbet --env CrafterReward-v0 --intrinsic_reward_coef=0.0 --total_frames 1000000 --num_actors 8 --savedir ../logs/impala --xpid crafter-transfer-1M --checkpoint=../logs/impala/crafter-pretrain-1M/model.tar