#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCG --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=48gb
#SBATCH --job-name=dreamer-test
#SBATCH --array=0-2
coeff_strength=(0.001 0.0025 0.005)

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
module load FFmpeg/6.0-GCCcore-12.3.0
source $HOME/venvs/dreamerv3/bin/activate
cd ..
python -m dreamerv3 --configs minigrid-unlock --logdir ./logs/dreamerv3/minigrid-coeff-${coeff_strength[${SLURM_ARRAY_TASK_ID}]} --batch_size 16 --intrinsic True --hash_bits 128 --use_pseudocounts False --intr_reward_coeff ${coeff_strength[${SLURM_ARRAY_TASK_ID}]}