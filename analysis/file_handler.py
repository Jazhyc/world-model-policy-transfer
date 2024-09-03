import json
import csv
import numpy as np
import math
import pandas as pd

def process_dreamer_scores(filepath, return_col, step_limit, num_eval_episodes):
    
    with open(filepath, 'r') as file:
        data = [json.loads(line) for line in file]
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

def average_scores_within_window(scores, window=200000, step_limit=1e6):
    scores.sort(key=lambda x: x[0])  # sort by step
    averaged_scores = []
    
    # Dummy value
    averaged_scores.append((0, 0, 0))
    index = 0
    
    # Set step limit to min of the last step
    step_limit = min(step_limit, scores[-1][0])
    
    for step_width in range(0, int(step_limit), window):
        step_sum = 0
        mean_sum = 0
        count = 0
        
        while index < len(scores) and scores[index][0] < step_width + window:
            step, mean, _ = scores[index]
            step_sum += step
            mean_sum += mean
            count += 1
            index += 1
            
        # Calculate the std using mean
        std = np.std([mean for s, mean, _ in scores if step_width <= s < step_width + window])
        
        if count > 0:
            averaged_scores.append((step_width + window, mean_sum / count, std))
    
    return averaged_scores

def process_impala_scores(filepath, step_limit, num_eval_episodes):
    with open(filepath, 'r') as file:
        impala_data = list(csv.DictReader(file))
    impala_scores = []
    # Sort the data by epoch
    impala_data.sort(key=lambda x: int(x['frame']))

    # Get the mean return and std error
    for row in impala_data:
        step = int(row['frame'])
        
        if step > step_limit:
            break
        
        mean = float(row['mean_reward'])
        std = float(row['std_reward'])
        impala_scores.append((step, mean, std))

    # Calculate the standard error
    impala_scores = [(step, mean, std / np.sqrt(num_eval_episodes)) for step, mean, std in impala_scores]
    
    return impala_scores

def process_and_average_scores(base_path, sub_path, filename, process_func, score_key, step_limit=None, num_eval_episodes=None, window=200000):
    filepath = base_path + sub_path + filename
    scores = process_func(filepath, score_key, step_limit, num_eval_episodes) if score_key else process_func(filepath, step_limit, num_eval_episodes)
    return average_scores_within_window(scores, window, step_limit)

def process_intrinsic_dreamer_scores(filename, return_col, step_limit, num_eval_episodes):
    
    with open(filename, 'r') as file:
        data = [json.loads(line) for line in file]
    
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

    # remove duplicates and sort
    scores = list(set(scores))
    scores.sort(key=lambda x: x[0])
    
    return scores

def process_intrinsic_impala_scores(filepath, step_limit=1e6, num_eval_episodes=8):
    with open(filepath, 'r') as file:
        impala_data = list(csv.DictReader(file))
    impala_scores = []
    # Sort the data by epoch
    impala_data.sort(key=lambda x: int(x['frames']))

    # Get the mean return and std error
    for row in impala_data:
        step = int(row['frames'])
        
        if step > step_limit:
            break
        
        mean = float(row['mean_intrinsic_rewards']) * float(row['mean_episode_length'])
        
        if 'std_intrinsic_rewards' in row:
            std = float(row['std_intrinsic_rewards'])
        else:
            std = 0

        mean = 0 if math.isnan(mean) else mean
        std = 0 if math.isnan(std) else std
        impala_scores.append((step, mean, std))
    
    return impala_scores

def create_df(data, label):
    df = pd.DataFrame(data, columns=['step', 'mean_return', 'std_error_return'])
    df['label'] = label
    return df

def average_between_experiments(scores, window, step_limit, num_experiments):
    
    if num_experiments == 1:
        return scores
    
    mean_scores = []
    
    # Data of the form [(step, mean, std), ...] for each experiment
    # Iterate over the scores, calculate mean using individual means and std from these. Step is the same
    for i in range(len(scores[0])):
        step = scores[0][i][0]
        mean = np.mean([scores[j][i][1] for j in range(num_experiments)])
        std = np.std([scores[j][i][1] for j in range(num_experiments)])
        mean_scores.append((step, mean, std / np.sqrt(num_experiments)))
        
    return mean_scores
    

def get_dataframe(path, num_eval_episodes=8, step_limit=1e6, window=200000, mode='extrinsic', num_experiments=5):
    
    dreamer_logs = '../logs/'
    dreamer_filename = 'scores.jsonl'
    dreamer_score_key = 'eval_episode/score'
    dreamer_intrinsic_score_key = 'episode/intrinsic_return'

    cbet_logs = '../logs/'
    cbet_filename = 'eval_results.csv'
    cbet_intrinsic_filename = 'logs.csv'
    
    # Create empty object to store scores among all experiments
    scores = []
    
    for i in range(0, num_experiments):
        
        # Ternary, replace # with the experiment number otherwise leave it as is
        exp_path = path.replace('#', str(i)) if '#' in path else path
    
        if 'dreamerv3' in path:
            score = process_and_average_scores(
                base_path=dreamer_logs, 
                sub_path=exp_path, 
                filename=dreamer_filename, 
                process_func=process_dreamer_scores if mode == 'extrinsic' else process_intrinsic_dreamer_scores, 
                score_key=dreamer_score_key if mode == 'extrinsic' else dreamer_intrinsic_score_key, 
                step_limit=step_limit, 
                num_eval_episodes=num_eval_episodes,
                window=window
            )
        elif 'impala' in path:
            score = process_and_average_scores(
                base_path=cbet_logs, 
                sub_path=exp_path, 
                filename=cbet_filename if mode == 'extrinsic' else cbet_intrinsic_filename, 
                process_func=process_impala_scores if mode == 'extrinsic' else process_intrinsic_impala_scores, 
                score_key=None, 
                step_limit=step_limit, 
                num_eval_episodes=num_eval_episodes,
                window=window
            )
        else:
            raise ValueError('Invalid path', exp_path)
        
        scores.append(score)
        
    return average_between_experiments(scores, window, step_limit, num_experiments)
        
    
    
def generate_df(name, path, num_eval_episodes=8, step_limit=1e6, window=200000, mode='extrinsic', num_experiments=5):
    scores = get_dataframe(path, num_eval_episodes, step_limit, window, mode, num_experiments)
    return create_df(scores, name)

def generate_dfs(results, num_eval_episodes=8, step_limit=1e6, window=200000, mode='extrinsic', num_experiments=5):
    # Generate all and then concatenate
    dfs = []
    for name, path in results.items():
        dfs.append(generate_df(name, path, num_eval_episodes, step_limit, window, mode, num_experiments))
    return pd.concat(dfs)
