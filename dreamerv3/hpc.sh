#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCG --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32gb
#SBATCH --job-name=dreamer-test
## SBATCH --array=0-2

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
source $HOME/venvs/dreamerv3/bin/activate
cd ..
python -m dreamerv3 --configs minigrid-test --logdir ./logs/dreamerv3/minigrid-habrok-test --batch_size 16 --intrinsic True --hash_bits 128 --use_pseudocounts False --intr_reward_coeff 0.001