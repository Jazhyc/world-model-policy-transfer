import json
import csv
import numpy as np
import math

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
    i = 0
    
    for step_width in range(0, int(step_limit), window):
        step_sum = 0
        mean_sum = 0
        std_sum = 0
        count = 0
        
        while i < len(scores) and scores[i][0] < step_width + window:
            step, mean, std = scores[i]
            step_sum += step
            mean_sum += mean
            std_sum += std
            count += 1
            i += 1
        
        if count > 0:
            averaged_scores.append((step_width + window, mean_sum / count, std_sum / count))
    
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
    return average_scores_within_window(scores, window)

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
