import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import numpy as np

from file_handler import process_dreamer_scores, process_impala_scores, process_and_average_scores
from plot_func import create_df, plot_scores

dreamer_logs = '../logs/'
dreamer_filename = 'scores.jsonl'

cbet_logs = '../logs/impala/'
cbet_filename = 'eval_results.csv'

num_eval_episodes = 8
step_limit = 1e6

# Minigrid results
minigrid_window = 200000
dreamer_scores = process_and_average_scores(
    base_path=dreamer_logs, 
    sub_path='dreamerv3/unlock-base/', 
    filename=dreamer_filename, 
    process_func=process_dreamer_scores, 
    score_key='eval_episode/score', 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

dreamer_cbet_scores = process_and_average_scores(
    base_path=dreamer_logs, 
    sub_path='dreamerv3/unlock-tabula-rasa-1M-2/', 
    filename=dreamer_filename, 
    process_func=process_dreamer_scores, 
    score_key='eval_episode/score', 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

impala_scores = process_and_average_scores(
    base_path=cbet_logs, 
    sub_path='vanilla-20240512-000218/', 
    filename=cbet_filename, 
    process_func=process_impala_scores, 
    score_key=None, 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

impala_cbet_scores = process_and_average_scores(
    base_path=cbet_logs, 
    sub_path='cbet-20240512-120923/', 
    filename=cbet_filename, 
    process_func=process_impala_scores, 
    score_key=None, 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

# Create DataFrames
df_dreamer = create_df(dreamer_scores, 'DreamerV3')
df_cbet_dreamer = create_df(dreamer_cbet_scores, 'DreamerV3 (CBET)')
df_impala = create_df(impala_scores, 'IMPALA')
df_cbet_impala = create_df(impala_cbet_scores, 'IMPALA (CBET)')

# Concatenate the DataFrames
df = pd.concat([df_dreamer, df_cbet_dreamer, df_impala, df_cbet_impala])

# Plot the scores
plot_scores(df, step_limit, 'Extrinsic', 'MiniGrid', y_lim=1)

# Crafter results
crafter_window = 200000

dreamer_scores = process_and_average_scores(
    base_path=dreamer_logs, 
    sub_path='dreamerv3/crafter-base-1M/', 
    filename=dreamer_filename, 
    process_func=process_dreamer_scores, 
    score_key='eval_episode/score', 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=crafter_window
)

dreamer_cbet_scores = process_and_average_scores(
    base_path=dreamer_logs, 
    sub_path='dreamerv3/crafter-tabula-rasa-1M/', 
    filename=dreamer_filename, 
    process_func=process_dreamer_scores, 
    score_key='eval_episode/score', 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=crafter_window
)

df_dreamer = create_df(dreamer_scores, 'DreamerV3')
df_cbet_dreamer = create_df(dreamer_cbet_scores, 'DreamerV3 (CBET)')

df = pd.concat([df_dreamer, df_cbet_dreamer])

plot_scores(df, step_limit, 'Extrinsic', 'Crafter', y_lim=10)