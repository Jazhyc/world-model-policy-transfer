#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCG --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=48gb
#SBATCH --job-name=dreamer-test
##SBATCH --array=0-2
#coeff_strength=(0.001 0.0025 0.005)

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
# module load FFmpeg/6.0-GCCcore-12.3.0
source $HOME/venvs/dreamerv3/bin/activate
cd ..

# Crafter Pre-train
python -m dreamerv3 --configs crafter-pre-train --logdir ./logs/dreamerv3/crafter-pre-train-1M

# Minigrid Pre-train
# python -m dreamerv3 --configs minigrid-pre-train --logdir ./logs/dreamerv3/minigrid-pre-train-1M-Doorkey8x8