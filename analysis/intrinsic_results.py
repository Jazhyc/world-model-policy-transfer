import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import math
import numpy as np

from file_handler import process_and_average_scores, process_intrinsic_dreamer_scores, process_intrinsic_impala_scores
from plot_func import plot_scores, create_df

dreamer_logs = '../logs/'
dreamer_filename = 'scores.jsonl'

cbet_logs = '../logs/impala/'
cbet_filename = 'logs.csv'

num_eval_episodes = 8
window_size = 20000    
step_limit = 1e6


dreamer_cbet_scores = process_and_average_scores(
    dreamer_logs,
    'dreamerv3/unlock-tabula-rasa-1M-2/',
    dreamer_filename,
    process_intrinsic_dreamer_scores,
    'episode/intrinsic_return',
    step_limit,
    num_eval_episodes,
    window_size
)

impala_cbet_scores = process_and_average_scores(
    cbet_logs,
    'unlock-tabula-rasa-1M-ego/',
    cbet_filename,
    process_intrinsic_impala_scores,
    None,
    step_limit,
    num_eval_episodes,
    window_size
)

# Create DataFrames
df_cbet_dreamer = create_df(dreamer_cbet_scores, 'DreamerV3 (CBET)')
df_cbet_impala = create_df(impala_cbet_scores, 'IMPALA (CBET)')

# Concatenate the DataFrames
df = pd.concat([df_cbet_dreamer, df_cbet_impala])

# Plot the scores
plot_scores(df, step_limit, 'intrinsic', 'Minigrid', y_lim=None)

dreamer_cbet_scores = process_and_average_scores(
    dreamer_logs,
    'dreamerv3/crafter-tabula-rasa-1M/',
    dreamer_filename,
    process_intrinsic_dreamer_scores,
    'episode/intrinsic_return',
    step_limit,
    num_eval_episodes,
    window_size
)

impala_cbet_scores = process_and_average_scores(
    cbet_logs,
    'crafter-tabula-rasa-1M/',
    cbet_filename,
    process_intrinsic_impala_scores,
    None,
    step_limit,
    num_eval_episodes,
    window_size
)

df_cbet_dreamer = create_df(dreamer_cbet_scores, 'DreamerV3 (CBET)')
df_cbet_impala = create_df(impala_cbet_scores, 'IMPALA (CBET)')
df = pd.concat([df_cbet_dreamer, df_cbet_impala])

plot_scores(df, step_limit, 'intrinsic', 'Crafter', y_lim=None)