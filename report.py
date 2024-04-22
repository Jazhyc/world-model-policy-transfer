import gymnasium as gym

# Make the environment
env = gym.make('MiniGrid-Unlock-v0', render_mode='human')

# Reset the environment
obs = env.reset()

# Perform a loop and then pause to get a screenshot
for i in range(100):
  action = env.action_space.sample()
  env.step(action)
  env.render()
