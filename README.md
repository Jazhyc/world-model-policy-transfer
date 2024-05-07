# Exploring Policy Transfer in World Model-Based Reinforcement Learning across Sparse Reward Environments

## DreamerV3

Usage: python -m dreamerv3 --configs minigrid --logdir ./logs/minigrid-8x8-xlarge --batch_size 16
Tensorboard: tensorboard --logdir [path]

## IMPALA (CBET)

Usage: OMP_NUM_THREADS=1 python main.py --model vanilla --env MiniGrid-Unlock-v0 --intrinsic_reward_coef=0.005 --total_frames 500000 --num_actors 1 --savedir ../logs/impala
