import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import numpy as np

dreamer_logs = 'logs/'
dreamer_filename = 'scores.jsonl'

cbet_logs = 'logs/impala/'
cbet_filename = 'eval_results.csv'

num_eval_episodes = 8

# Read the JSONL file
with open(dreamer_logs + 'minigrid-unlock-eval8-500k-xlarge/' + dreamer_filename, 'r') as file:
    dreamer_data = [json.loads(line) for line in file]
    
# Read another dreamer file
with open(dreamer_logs + 'dreamerv3/unlock-tabula-rasa/' + dreamer_filename, 'r') as file:
    dreamer_cbet_data = [json.loads(line) for line in file]
    
# Read the CSV file
with open(cbet_logs + 'cbet-20240430-094041/' + cbet_filename, 'r') as file:
    cbet_data = list(csv.DictReader(file))
    
step_limit = 400000

def process_dreamer_scores(data, return_col, step_limit, num_eval_episodes):
    # Go over all rows and extract the return_col field and step if they exist
    scores = []
    for row in data:
        if return_col in row and 'step' in row:
            if row['step'] > step_limit:
                break
            score = float(row[return_col])
            step = int(row['step'])
            scores.append((step, score))
            
    # For all elements which have the same step, find mean and std
    scores.sort(key=lambda x: x[0])
    scores = [(step, np.mean([score for s, score in scores if s == step]), np.std([score for s, score in scores if s == step])) for step, _ in scores]

    # Divide std by sqrt of number of elements
    scores = [(step, mean, std / np.sqrt(num_eval_episodes)) for step, mean, std in scores]

    # remove duplicates and sort
    scores = list(set(scores))
    scores.sort(key=lambda x: x[0])
    
    return scores

dreamer_scores = process_dreamer_scores(dreamer_data, 'eval_episode/score', step_limit, num_eval_episodes)

def average_scores_within_window(scores, window=10000):
    scores.sort(key=lambda x: x[0])  # sort by step
    averaged_scores = []
    i = 0
    while i < len(scores):
        window_scores = [scores[i][1]]  # start with the score of the current step
        window_stds = [scores[i][2]]  # start with the std_dev of the current step
        min_step = scores[i][0]  # start with the current step
        j = i + 1
        while j < len(scores) and scores[j][0] - min_step <= window:
            window_scores.append(scores[j][1])  # add the score of the step within the window
            window_stds.append(scores[j][2])  # add the std_dev of the step within the window
            j += 1
        average = sum(window_scores) / len(window_scores)
        avg_std_dev = sum(window_stds) / len(window_stds)
        averaged_scores.append((min_step, average, avg_std_dev))  # append a tuple of (min_step, average_score, avg_std_dev)
        i = j
    return averaged_scores

# dreamer_scores = average_scores_within_window(dreamer_scores)

dreamer_cbet_scores = process_dreamer_scores(dreamer_cbet_data, 'episode/score', step_limit, num_eval_episodes)

dreamer_scores = average_scores_within_window(dreamer_scores)
dreamer_cbet_scores = average_scores_within_window(dreamer_cbet_scores)
        
impala_scores = []
# Sort the data by epoch
cbet_data.sort(key=lambda x: int(x['frame']))

# Get the mean return and std error
for row in cbet_data:
    
    step = int(row['frame'])
    mean = float(row['mean_reward'])
    std = float(row['std_reward'])
    impala_scores.append((step, mean, std))
        
# Calculate the standard error
impala_scores = [(step, mean, std / np.sqrt(num_eval_episodes)) for step, mean, std in impala_scores]
    
# impala_scores = average_scores_within_window(impala_scores)
    
def rolling_mean(data, window=10):
    return [sum(data[i:i+window]) / window for i in range(len(data) - window)]

# Create a pandas DataFrame
df = pd.DataFrame(dreamer_scores, columns=['step', 'mean_return', 'std_error_return'])

# Add label column and fill it with DreamerV3
df['label'] = 'DreamerV3'

df_cbet_dreamer = pd.DataFrame(dreamer_cbet_scores, columns=['step', 'mean_return', 'std_error_return'])
df_cbet_dreamer['label'] = 'DreamerV3 (CBET Tabula Rasa)'

# Create a DataFrame for the IMPALA scores
df_impala = pd.DataFrame(impala_scores, columns=['step', 'mean_return', 'std_error_return'])
df_impala['label'] = 'IMPALA'

# # Concatenate the DataFrames
df = pd.concat([df_cbet_dreamer])

# Plot the scores using groupby
plt.figure(figsize=(10, 5))
df.groupby('label').plot(x='step', y='mean_return', ax=plt.gca())

lines = []
labels = []

for label, group in df.groupby('label'):
    line, = plt.plot(group['step'], group['mean_return'])
    plt.fill_between(group['step'], group['mean_return'] - group['std_error_return'], group['mean_return'] + group['std_error_return'], alpha=0.2, color=line.get_color())
    lines.append(line)
    labels.append(label)

# Add legend with the lines and labels
plt.legend(lines, labels)

# Add title
plt.xlabel('Time step')
plt.ylabel('Mean Total Return')
plt.title('Mean Agent Return over Time')

# grid
plt.grid()

# Add ylim of 0 to 1
plt.ylim(0, 1)
plt.xlim(0, step_limit)

# Save the plot
plt.savefig('results.png', dpi=600)

plt.show()