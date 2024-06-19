import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import numpy as np

from file_handler import generate_dfs
from plot_func import plot_scores

window = 200000
step_limit = 1e6

minigrid_results = {
    'DreamerV3 (BASE)': 'dreamerv3/unlock-base/',
    'DreamerV3 (CBET)': 'dreamerv3/unlock-tabula-rasa-1M-2/',
    'IMPALA (BASE)': 'impala/unlock-base-1M/',
    'IMPALA (CBET)': 'impala/unlock-tabula-rasa-1M-ego/'
}

df = generate_dfs(minigrid_results, window=window, step_limit=step_limit)

# Plot the scores
plot_scores(df, 'Extrinsic', 'Tabula Rasa MiniGrid', y_lim=1)

crafter_results = {
    'DreamerV3 (BASE)': 'dreamerv3/crafter-base-1M/',
    'DreamerV3 (CBET)': 'dreamerv3/crafter-coeff-0.001/', # Better than base which uses 0.0025
    'IMPALA (BASE)': 'impala/crafter-base-1M/',
    'IMPALA (CBET)': 'impala/crafter-tabula-rasa-1M-coeff-0.005/' # same reason as above
}

df = generate_dfs(crafter_results, window=window, step_limit=step_limit)

plot_scores(df, 'Extrinsic', 'Tabula Rasa Crafter', y_lim=12)

transfer_minigrid_results = {
    'DreamerV3 (CBET)': 'dreamerv3/minigrid-transfer-1M-Unlock/',
    'IMPALA (CBET)': 'impala/minigrid-transfer-1M-unlock/'
}

df = generate_dfs(transfer_minigrid_results, window=window, step_limit=step_limit)

plot_scores(df, 'Extrinsic', 'Transfer MiniGrid', y_lim=1)

transfer_crafter_results = {
    'DreamerV3 (CBET)': 'dreamerv3/crafter-transfer-1M/',
    'IMPALA (CBET)': 'impala/crafter-transfer-1M/'
}

df = generate_dfs(transfer_crafter_results, window=window, step_limit=step_limit)

plot_scores(df, 'Extrinsic', 'Transfer Crafter', y_lim=12)