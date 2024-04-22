import pandas as pd
import matplotlib.pyplot as plt
import json
import csv

dreamer_logs = 'logs/'
dreamer_filename = 'scores.jsonl'

cbet_logs = 'cbet/logs/vanilla/'
cbet_filename = 'logs.csv'

# Read the JSONL file
with open(dreamer_logs + 'minigrid-unlock-eval5-500k-xlarge/' + dreamer_filename, 'r') as file:
    dreamer_data = [json.loads(line) for line in file]
    
# Read the CSV file
with open(cbet_logs + 'vanilla-20240418-060918/' + cbet_filename, 'r') as file:
    cbet_data = list(csv.DictReader(file))
    
step_limit = 400000

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
        
# For scores within 1000 steps of each other, take the average of all scores in the window and then make a new list without the original scores

def average_scores_within_window(scores, window=10000):
    scores.sort(key=lambda x: x[0])  # sort by step
    averaged_scores = []
    i = 0
    while i < len(scores):
        window_scores = [scores[i][1]]  # start with the score of the current step
        min_step = scores[i][0]  # start with the current step
        j = i + 1
        while j < len(scores) and scores[j][0] - min_step <= window:
            window_scores.append(scores[j][1])  # add the score of the step within the window
            j += 1
        average = sum(window_scores) / len(window_scores)
        averaged_scores.append((min_step, average))  # append a tuple of (min_step, average_score)
        i = j
    return averaged_scores

# dreamer_scores = average_scores_within_window(dreamer_scores)
        
impala_scores = []
for row in cbet_data:
        
    if int(row['frames']) > step_limit or int(row['frames']) == 0:
        continue
    
    score = float(row['mean_episode_return'])
    step = int(row['frames'])
    impala_scores.append((step, score))
    
def rolling_mean(data, window=10):
    return [sum(data[i:i+window]) / window for i in range(len(data) - window)]

# Create a pandas DataFrame
df = pd.DataFrame(dreamer_scores, columns=['step', 'return'])

# Add label column and fill it with DreamerV3
df['label'] = 'DreamerV3 (Eval)'

# Create a DataFrame for the IMPALA scores
df_impala = pd.DataFrame(impala_scores, columns=['step', 'return'])
df_impala['label'] = 'IMPALA (Train)'

# # Concatenate the two DataFrames
# df = pd.concat([df, df_impala])

# Plot the scores using groupby
plt.figure(figsize=(10, 5))
df.groupby('label').plot(x='step', y='return', ax=plt.gca())

# Add legend automatically
plt.legend(df['label'].unique())

# Add title
plt.xlabel('Time step')
plt.ylabel('Mean Return')
plt.title('Mean Agent Return over Time')

# Save the plot
plt.savefig('results.png', dpi=300)

plt.show()