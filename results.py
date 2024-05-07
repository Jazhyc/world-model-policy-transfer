import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import numpy as np

dreamer_logs = 'logs/'
dreamer_filename = 'scores.jsonl'

cbet_logs = 'logs/impala/'
cbet_filename = 'logs.csv'

# Read the JSONL file
with open(dreamer_logs + 'minigrid-unlock-eval8-500k-xlarge/' + dreamer_filename, 'r') as file:
    dreamer_data = [json.loads(line) for line in file]
    
# Read the CSV file
with open(cbet_logs + 'cbet-20240430-094041/' + cbet_filename, 'r') as file:
    cbet_data = list(csv.DictReader(file))
    
step_limit = 500000

# Go over all rows and extract the extr_return_mean_raw field and step if they exist
dreamer_scores = []
dreamer_return_col = 'eval_episode/score'
for row in dreamer_data:
    if dreamer_return_col in row and 'step' in row:
        
        if row['step'] > step_limit:
            break
        
        score = float(row[dreamer_return_col])
        step = int(row['step'])
        dreamer_scores.append((step, score))
        
# For all elements which have the same step, find mean and std
dreamer_scores = [(step, score) for step, score in dreamer_scores]
dreamer_scores.sort(key=lambda x: x[0])
dreamer_scores = [(step, np.mean([score for s, score in dreamer_scores if s == step]), np.std([score for s, score in dreamer_scores if s == step])) for step, _ in dreamer_scores]

dreamer_eval_len = 8

# Divide std by sqrt of number of elements
dreamer_scores = [(step, mean, std / np.sqrt(dreamer_eval_len)) for step, mean, std in dreamer_scores]

# remove duplicates and sort
dreamer_scores = list(set(dreamer_scores))
dreamer_scores.sort(key=lambda x: x[0])

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

dreamer_scores = average_scores_within_window(dreamer_scores)
        
impala_scores = []
for row in cbet_data:
        
    if int(row['frames']) > step_limit or int(row['frames']) == 0:
        continue
    
    mean_return = float(row['mean_episode_return'])
    std_return = float(row['episode_return_std'])
    step = int(row['frames'])
    impala_scores.append((step, mean_return, std_return))
    
# Calculate the standard error
impala_scores = [(step, mean, std / np.sqrt(11)) for step, mean, std in impala_scores]
    
impala_scores = average_scores_within_window(impala_scores)

print(len(dreamer_scores), len(impala_scores))
    
def rolling_mean(data, window=10):
    return [sum(data[i:i+window]) / window for i in range(len(data) - window)]

# Create a pandas DataFrame
df = pd.DataFrame(dreamer_scores, columns=['step', 'mean_return', 'std_error_return'])

# Add label column and fill it with DreamerV3
df['label'] = 'DreamerV3 (Eval)'

# Create a DataFrame for the IMPALA scores
df_impala = pd.DataFrame(impala_scores, columns=['step', 'mean_return', 'std_error_return'])
df_impala['label'] = 'IMPALA (Train)'

# # Concatenate the two DataFrames
df = pd.concat([df, df_impala])

# Plot the scores using groupby
plt.figure(figsize=(10, 5))
df.groupby('label').plot(x='step', y='mean_return', ax=plt.gca())

# Add standard deviation using different color
for label in df['label'].unique():
    data = df[df['label'] == label]
    plt.fill_between(data['step'], data['mean_return'] - data['std_error_return'], data['mean_return'] + data['std_error_return'], alpha=0.2)

# Add legend automatically
plt.legend(df['label'].unique())

# Add title
plt.xlabel('Time step')
plt.ylabel('Mean Return')
plt.title('Mean Agent Return over Time')

# Add ylim of 0 to 1
plt.ylim(0, 1)
plt.xlim(0, step_limit)

# Save the plot
plt.savefig('results.png', dpi=600)

plt.show()