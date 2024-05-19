# import gymnasium and minigrid
import gymnasium as gym
import matplotlib.pyplot as plt

for index in range(10):
    
    
    env = gym.make('MiniGrid-KeyCorridorS3R3-v0', render_mode='human')
    env.reset()

    # Create environment and save image
    env.step(env.action_space.sample())
