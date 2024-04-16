# import gymnasium and minigrid
import gymnasium as gym
import matplotlib.pyplot as plt

for index in range(1):

    # Create environment and save image
    env = gym.make('MiniGrid-ObstructedMaze-2Dlh-v0', render_mode='rgb_array')
    env.reset()
    img = env.render()
    env.close()

    # Save image using matplotlib
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'Maze_{index}.png', bbox_inches='tight')
