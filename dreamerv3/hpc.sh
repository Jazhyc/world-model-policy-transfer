#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCG --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32gb
#SBATCH --job-name=d-m-planning-base
#SBATCH --array=0-1
plan_ratio=(256 1024)

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
# module load FFmpeg/6.0-GCCcore-12.3.0
source $HOME/venvs/dreamerv3/bin/activate
cd ..

# Minigrid Pre-train
# python -m dreamerv3 --configs minigrid-pre-train --logdir ./logs/dreamerv3/minigrid-pre-train-1M-Doorkey8x8

# Minigrid Transfer
# python -m dreamerv3 --configs minigrid-transfer --logdir ./logs/dreamerv3/minigrid-transfer-1M-Unlock --run.from_checkpoint ./logs/dreamerv3/minigrid-pre-train-1M-Doorkey8x8/checkpoint.ckpt

# Crafter Tabula Rasa / Base (change intrinsic and path)
python -m dreamerv3 --configs minigrid-unlock --logdir ./logs/dreamerv3/minigrid-tabula-rasa-1M-planning-${plan_ratio[${SLURM_ARRAY_TASK_ID}]} --run.train_ratio ${plan_ratio[${SLURM_ARRAY_TASK_ID}]} --run.intrinsic True

# Crafter Pre-train
# python -m dreamerv3 --configs crafter-pre-train --logdir ./logs/dreamerv3/crafter-pre-train-1M

# Crafter Transfer
# python -m dreamerv3 --configs crafter-transfer --logdir ./logs/dreamerv3/crafter-transfer-1M --run.from_checkpoint ./logs/dreamerv3/crafter-pre-train-1M/checkpoint.ckpt