import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import numpy as np

from file_handler import generate_dfs
from multi_plot import plot_scores

window = 200000
transfer_step_limit = crafter_step_limit = 1e6
tabula_minigrid_step_limit = 3e6

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.subplots_adjust(wspace=0.1)

minigrid_results = {
    'DreamerV3 (BASE)': 'dreamerv3/minigrid-base-sweep-#/',
    'DreamerV3 (CBET)': 'dreamerv3/minigrid-cbet-sweep-#/',
    'IMPALA (BASE)': 'impala/Minigrid-base-sweep-#/',
    'IMPALA (CBET)': 'impala/Minigrid-cbet-sweep-#/'
}

df = generate_dfs(minigrid_results, window=window, step_limit=tabula_minigrid_step_limit)
handles, labels = plot_scores(df, 'Extrinsic', 'Tabula Rasa MiniGrid', axs[0, 0], y_lim=1, step_limit=tabula_minigrid_step_limit)

transfer_minigrid_results = {
    'DreamerV3 (CBET)': 'dreamerv3/minigrid-transfer-1M-Unlock/',
    'IMPALA (CBET)': 'impala/minigrid-transfer-1M-unlock/'
}

df = generate_dfs(transfer_minigrid_results, window=window, step_limit=transfer_step_limit, num_experiments=1)
plot_scores(df, 'Extrinsic', 'Transfer MiniGrid', axs[0, 1], y_lim=1, hide_y_ticks=False)

crafter_results = {
    'DreamerV3 (BASE)': 'dreamerv3/crafter-base-sweep-#/',
    'DreamerV3 (CBET)': 'dreamerv3/crafter-cbet-sweep-#/',
    'IMPALA (BASE)': 'impala/Crafter-base-sweep-#/',
    'IMPALA (CBET)': 'impala/Crafter-cbet-sweep-#/'
}

df = generate_dfs(crafter_results, window=window, step_limit=crafter_step_limit)
plot_scores(df, 'Extrinsic', 'Tabula Rasa Crafter', axs[1, 0], y_lim=12, step_limit=crafter_step_limit)

transfer_crafter_results = {
    'DreamerV3 (CBET)': 'dreamerv3/crafter-transfer-1M/',
    'IMPALA (CBET)': 'impala/crafter-transfer-1M/'
}

df = generate_dfs(transfer_crafter_results, window=window, step_limit=transfer_step_limit, num_experiments=1)
plot_scores(df, 'Extrinsic', 'Transfer Crafter', axs[1, 1], y_lim=12, hide_y_ticks=False)

# Adjust these values as needed to better center the labels and set font size
x_label_x_position = 0.5125  # This is typically centered, but adjust if your figure's layout is unusual
y_label_y_position = 0.5  # Adjust this value to center the y-axis label, especially if the figure's height varies
label_font_size = 14 # Example font size, adjust as needed

fig.text(x_label_x_position, 0.05, 'Steps (Millions)', ha='center', fontsize=label_font_size)
fig.text(0.07, y_label_y_position, 'Mean Extrinsic Return', va='center', rotation='vertical', fontsize=label_font_size)

# After all plotting is done, but before plt.savefig and plt.show
handles, labels = axs[0, 0].get_legend_handles_labels()  # Get handles and labels from one of the subplots

# Filter out "mean_return" labels and their corresponding handles
filtered_handles = []
filtered_labels = []
for handle, label in zip(handles, labels):
    if label != "mean_return":
        filtered_handles.append(handle)
        filtered_labels.append(label)
        
FONT_SIZE = 14
LEGEND_SIZE = 16

# Add legend below the whole plot with filtered labels and handles
fig.legend(filtered_handles, filtered_labels, loc='lower center', ncol=4, fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, -0.01))

# make legend visible
for ax in axs.flat:
    ax.get_legend().remove()
    
# Increase font size of all text
for ax in axs.flat:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(FONT_SIZE)

#fig.legend(handles, labels, loc='lower center', ncol=4)
plt.savefig('images/combined_plots_poster.png', dpi=300)
plt.show()
