# import gymnasium and minigrid
import gymnasium as gym
import matplotlib.pyplot as plt

for index in range(1):
    
    env = gym.make('MiniGrid-DoorKey-8x8-v0', render_mode='human')
    env.reset()
    
    input("Press Enter to continue...")
