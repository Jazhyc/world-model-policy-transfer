import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import numpy as np

from file_handler import process_dreamer_scores, process_impala_scores, process_and_average_scores, process_intrinsic_impala_scores
from plot_func import create_df, plot_scores

dreamer_logs = '../logs/'
dreamer_filename = 'scores.jsonl'

cbet_logs = '../logs/impala/'
cbet_filename = 'eval_results.csv'

num_eval_episodes = 8
step_limit = 1e6

minigrid_window = 200000

#! TODO: Make cleaner

cbet_scores_0_005 = process_and_average_scores(
    base_path=cbet_logs, 
    sub_path='unlock-tabula-rasa-1M-coeff-0.005/', 
    filename=cbet_filename, 
    process_func=process_impala_scores, 
    score_key=None, 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

cbet_scores_0_0025 = process_and_average_scores(
    base_path=cbet_logs, 
    sub_path='unlock-tabula-rasa-1M-coeff-0.0025/', 
    filename=cbet_filename, 
    process_func=process_impala_scores, 
    score_key=None, 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

cbet_scores_0_001 = process_and_average_scores(
    base_path=cbet_logs, 
    sub_path='unlock-tabula-rasa-1M-coeff-0.001/', 
    filename=cbet_filename, 
    process_func=process_impala_scores, 
    score_key=None, 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

cbet_scores_pano = process_and_average_scores(
    base_path=cbet_logs, 
    sub_path='cbet-20240512-120923/', 
    filename=cbet_filename, 
    process_func=process_impala_scores, 
    score_key=None, 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

# Create a dataframe
df_cbet_0_005 = create_df(cbet_scores_0_005, 'Ego 0.005')
df_cbet_0_0025 = create_df(cbet_scores_0_0025, 'Ego 0.0025')
df_cbet_0_001 = create_df(cbet_scores_0_001, 'Ego 0.001')
df_cbet_pano = create_df(cbet_scores_pano, 'Pano 0.005')

# concat
df_cbet = pd.concat([df_cbet_0_005, df_cbet_0_0025, df_cbet_0_001, df_cbet_pano])
plot_scores(df_cbet, step_limit, "Extrinsic", 'Impala Minigrid Hyperparameter tuning', y_lim=0.3)

minigrid_window = 20000
cbet_filename = 'logs.csv'

cbet_scores_0_005_intrinsic = process_and_average_scores(
    base_path=cbet_logs, 
    sub_path='unlock-tabula-rasa-1M-coeff-0.005/', 
    filename=cbet_filename, 
    process_func=process_intrinsic_impala_scores, 
    score_key=None, 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

cbet_scores_0_0025_intrinsic = process_and_average_scores(
    base_path=cbet_logs, 
    sub_path='unlock-tabula-rasa-1M-coeff-0.0025/', 
    filename=cbet_filename, 
    process_func=process_intrinsic_impala_scores, 
    score_key=None, 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

cbet_scores_0_001_intrinsic = process_and_average_scores(
    base_path=cbet_logs, 
    sub_path='unlock-tabula-rasa-1M-coeff-0.001/', 
    filename=cbet_filename, 
    process_func=process_intrinsic_impala_scores, 
    score_key=None, 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

cbet_scores_pano_intrinsic = process_and_average_scores(
    base_path=cbet_logs, 
    sub_path='cbet-20240512-120923/', 
    filename=cbet_filename, 
    process_func=process_intrinsic_impala_scores, 
    score_key=None, 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

# Create a dataframe
df_cbet_0_005_intrinsic = create_df(cbet_scores_0_005_intrinsic, 'Ego 0.005')
df_cbet_0_0025_intrinsic = create_df(cbet_scores_0_0025_intrinsic, 'Ego 0.0025')
df_cbet_0_001_intrinsic = create_df(cbet_scores_0_001_intrinsic, 'Ego 0.001')
df_cbet_pano_intrinsic = create_df(cbet_scores_pano_intrinsic, 'Pano 0.005')

# concat
df_cbet_intrinsic = pd.concat([df_cbet_0_005_intrinsic, df_cbet_0_0025_intrinsic, df_cbet_0_001_intrinsic, df_cbet_pano_intrinsic])
plot_scores(df_cbet_intrinsic, step_limit, "Intrinsic", 'Minigrid Hyperparameter Tuning', y_lim=0.3)

dreamer_scores_0_005 = process_and_average_scores(
    base_path=dreamer_logs, 
    sub_path='dreamerv3/minigrid-coeff-0.005/', 
    filename=dreamer_filename, 
    process_func=process_dreamer_scores, 
    score_key='episode/score', 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

dreamer_scores_0_0025 = process_and_average_scores(
    base_path=dreamer_logs, 
    sub_path='dreamerv3/minigrid-coeff-0.0025/', 
    filename=dreamer_filename, 
    process_func=process_dreamer_scores, 
    score_key='episode/score', 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

dreamer_scores_0_001 = process_and_average_scores(
    base_path=dreamer_logs, 
    sub_path='dreamerv3/minigrid-coeff-0.001/', 
    filename=dreamer_filename, 
    process_func=process_dreamer_scores, 
    score_key='episode/score', 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

df_dreamer_0_005 = create_df(dreamer_scores_0_005, 'Ego 0.005')
df_dreamer_0_0025 = create_df(dreamer_scores_0_0025, 'Ego 0.0025')
df_dreamer_0_001 = create_df(dreamer_scores_0_001, 'Ego 0.001')

df_dreamer = pd.concat([df_dreamer_0_005, df_dreamer_0_0025, df_dreamer_0_001])

plot_scores(df_dreamer, step_limit, "Extrinsic", 'Dreamer Minigrid Hyperparameter Tuning', y_lim=1)

# Do the same for episode/intrinsic_return
dreamer_scores_0_005_intrinsic = process_and_average_scores(
    base_path=dreamer_logs, 
    sub_path='dreamerv3/minigrid-coeff-0.005/', 
    filename=dreamer_filename, 
    process_func=process_dreamer_scores, 
    score_key='episode/intrinsic_return', 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

dreamer_scores_0_0025_intrinsic = process_and_average_scores(
    base_path=dreamer_logs, 
    sub_path='dreamerv3/minigrid-coeff-0.0025/', 
    filename=dreamer_filename, 
    process_func=process_dreamer_scores, 
    score_key='episode/intrinsic_return', 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

dreamer_scores_0_001_intrinsic = process_and_average_scores(
    base_path=dreamer_logs, 
    sub_path='dreamerv3/minigrid-coeff-0.001/', 
    filename=dreamer_filename, 
    process_func=process_dreamer_scores, 
    score_key='episode/intrinsic_return', 
    step_limit=step_limit, 
    num_eval_episodes=num_eval_episodes,
    window=minigrid_window
)

df_dreamer_0_005_intrinsic = create_df(dreamer_scores_0_005_intrinsic, 'Ego 0.005')
df_dreamer_0_0025_intrinsic = create_df(dreamer_scores_0_0025_intrinsic, 'Ego 0.0025')
df_dreamer_0_001_intrinsic = create_df(dreamer_scores_0_001_intrinsic, 'Ego 0.001')

df_dreamer_intrinsic = pd.concat([df_dreamer_0_005_intrinsic, df_dreamer_0_0025_intrinsic, df_dreamer_0_001_intrinsic])

plot_scores(df_dreamer_intrinsic, step_limit, "Intrinsic", 'Dreamer Minigrid Hyperparameter Tuning', y_lim=0.3)