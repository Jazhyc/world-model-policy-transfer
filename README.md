# Exploring Policy Transfer in World Model-Based Reinforcement Learning across Sparse Reward Environments

## DreamerV3

Usage:
Tabula Rasa: python -m dreamerv3 --configs minigrid --logdir ./logs/minigrid-8x8-xlarge --batch_size 16
Pre-train: python -m dreamerv3 --configs minigrid-test --logdir ./logs/dreamerv3/test --batch_size 16 --run.intrinsic True --run.ignore_extr_reward True --run.intr_reward_coeff 0.0025 --run.transfer True
Load from Checkpoint: python -m dreamerv3 --configs minigrid-test --logdir ./logs/dreamerv3/test-transfer --run.intrinsic False --run.ignore_extr_reward False --run.intr_reward_coeff 0.0025 --run.transfer True --run.from_checkpoint ./logs/dreamerv3/test/checkpoint.ckpt
Tensorboard: tensorboard --logdir [path]

## IMPALA (CBET)

In cbet directory
Usage: OMP_NUM_THREADS=1 python main.py --model cbet --env MiniGrid-Unlock-v0 --intrinsic_reward_coef=0.005 --total_frames 1000000 --num_actors 24 --savedir ../logs/impala --xpid unlock-tabula-rasa-1M-ego

OMP_NUM_THREADS=1 python main.py --model cbet --env CrafterReward-v1 --intrinsic_reward_coef=0.005 --total_frames 1000000 --num_actors 24 --savedir ../logs/impala --xpid unlock-tabula-rasa-1M-ego