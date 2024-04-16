import pandas as pd
import matplotlib.pyplot as plt
import json

path = 'dreamerv3/logs/'
filename = 'scores.jsonl'

# Read the JSONL file
with open(path + 'minigrid-8x8-small/' + filename, 'r') as file:
    data1 = [json.loads(line) for line in file]

with open(path + 'minigrid-8x8-xlarge/' + filename, 'r') as file:
    data2 = [json.loads(line) for line in file]

# Convert to DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Plot the data
plt.scatter(df1['step'], df1['episode/score'], label='S')
plt.scatter(df2['step'], df2['episode/score'], label='XL')
plt.xlabel('Step')
plt.ylabel('Episode/Score')
plt.title('Mean reward across workers at each step')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)


plt.savefig('score_per_step.png')
plt.show()

# Additional plot for score per episode
plt.bar(range(len(df1)), df1['episode/score'], label='S')
plt.bar(range(len(df2)), df2['episode/score'], label='XL')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Mean reward across workers per Episode')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.savefig('score_per_episode.png')
plt.show()
