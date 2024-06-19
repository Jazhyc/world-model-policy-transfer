import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import numpy as np

from file_handler import generate_dfs
from plot_func import plot_scores

# minigrid_impala = {
#     'IMPALA (CBET) 0.001' : 'impala/unlock-tabula-rasa-1M-coeff-0.001/',
#     'IMPALA (CBET) 0.0025' : 'impala/unlock-tabula-rasa-1M-coeff-0.0025/',
#     'IMPALA (CBET) 0.005' : 'impala/unlock-tabula-rasa-1M-coeff-0.005/',
# }

# df = generate_dfs(minigrid_impala, window=200000, step_limit=1e6)

# plot_scores(df, 'Extrinsic', 'MiniGrid', y_lim=0.2)

# minigrid_dreamer = {
#     'DreamerV3 (CBET) 0.001' : 'dreamerv3/minigrid-coeff-0.001/',
#     'DreamerV3 (CBET) 0.0025' : 'dreamerv3/minigrid-coeff-0.0025/',
#     'DreamerV3 (CBET) 0.005' : 'dreamerv3/minigrid-coeff-0.005/',
# }

# df = generate_dfs(minigrid_dreamer, window=200000, step_limit=1e6)

# plot_scores(df, 'Extrinsic', 'MiniGrid', y_lim=1)

# crafter_impala = {
#     'IMPALA (CBET) 0.001' : 'impala/crafter-tabula-rasa-1M-coeff-0.001/',
#     'IMPALA (CBET) 0.0025' : 'impala/crafter-tabula-rasa-1M-coeff-0.0025/',
#     'IMPALA (CBET) 0.005' : 'impala/crafter-tabula-rasa-1M-coeff-0.005/',
# }

# df = generate_dfs(crafter_impala, window=200000, step_limit=1e6)

# plot_scores(df, 'Extrinsic', 'Crafter', y_lim=10)

# crafter_dreamer = {
#     'DreamerV3 (CBET) 0.001' : 'dreamerv3/crafter-coeff-0.001/',
#     'DreamerV3 (CBET) 0.0025' : 'dreamerv3/crafter-coeff-0.0025/',
#     'DreamerV3 (CBET) 0.005' : 'dreamerv3/crafter-coeff-0.005/',
#     'DreamerV3' : 'dreamerv3/crafter-base-1M/'
# }

# df = generate_dfs(crafter_dreamer, window=200000, step_limit=1e6)

# plot_scores(df, 'Extrinsic', 'Crafter', y_lim=10)

crafter_dreamer_plan = {
    'DreamerV3 (BASE) 256' : 'dreamerv3/crafter-base-1M-planning-256/',
    'DreamerV3 (BASE) 1024' : 'dreamerv3/crafter-base-1M-planning-1024/',
    'DreamerV3 (BASE) 64' : 'dreamerv3/crafter-base-1M/',
    # Same with cbet
    'DreamerV3 (CBET) 256' : 'dreamerv3/crafter-tabula-rasa-1M-planning-256/',
    'DreamerV3 (CBET) 1024' : 'dreamerv3/crafter-tabula-rasa-1M-planning-1024/',
    'DreamerV3 (CBET) 64' : 'dreamerv3/crafter-tabula-rasa-1M/'
}

df = generate_dfs(crafter_dreamer_plan, window=200000, step_limit=1e6)

plot_scores(df, 'Extrinsic', 'Crafter', y_lim=10)