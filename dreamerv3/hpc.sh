#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCG --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32gb
#SBATCH --job-name=d-c-c-sweep
#SBATCH --array=0-4
# plan_ratio=(256 1024)

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
# module load FFmpeg/6.0-GCCcore-12.3.0
source $HOME/venvs/dreamerv3/bin/activate
cd ..

# Minigrid Pre-train
# python -m dreamerv3 --configs minigrid-pre-train --logdir ./logs/dreamerv3/minigrid-pre-train-1M-Doorkey8x8-2

# Minigrid Transfer
# python -m dreamerv3 --configs minigrid-transfer --logdir ./logs/dreamerv3/minigrid-transfer-1M-Unlock --run.from_checkpoint ./logs/dreamerv3/minigrid-pre-train-1M-Doorkey8x8/checkpoint.ckpt

# Crafter Tabula Rasa / Base (change intrinsic and path)
python -m dreamerv3 --configs minigrid-unlock --logdir ./logs/dreamerv3/minigrid-cbet-sweep-${SLURM_ARRAY_TASK_ID} --run.intrinsic True --run.intr_reward_coeff 0.001 --seed ${SLURM_ARRAY_TASK_ID}

# Crafter Pre-train
# python -m dreamerv3 --configs crafter-pre-train --logdir ./logs/dreamerv3/crafter-pre-train-1M-2

# Crafter Transfer
# python -m dreamerv3 --configs crafter-transfer --logdir ./logs/dreamerv3/crafter-transfer-1M --run.from_checkpoint ./logs/dreamerv3/crafter-pre-train-1M/checkpoint.ckpt

# Convergence Test
# python -m dreamerv3 --configs crafter-cbet --logdir ./logs/dreamerv3/crafter-base-2M --run.intrinsic False