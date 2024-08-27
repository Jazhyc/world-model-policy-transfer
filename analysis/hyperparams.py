import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import numpy as np

from file_handler import generate_dfs
from multi_plot import plot_scores

from single_plot import plot_scores as single_plot_scores

# fig, axs = plt.subplots(2, 2, figsize=(15, 10))
# fig.subplots_adjust(wspace=0.1)

# minigrid_impala = {
#     'IMPALA (CBET) 0.001' : 'impala/unlock-tabula-rasa-1M-coeff-0.001/',
#     'IMPALA (CBET) 0.0025' : 'impala/unlock-tabula-rasa-1M-coeff-0.0025/',
#     'IMPALA (CBET) 0.005' : 'impala/unlock-tabula-rasa-1M-coeff-0.005/',
# }

# df = generate_dfs(minigrid_impala, window=200000, step_limit=1e6)

# plot_scores(df, 'Extrinsic', 'IMPALA Minigrid', axs[0,0], y_lim=1)

# minigrid_dreamer = {
#     'DreamerV3 (CBET) 0.001' : 'dreamerv3/minigrid-coeff-0.001/',
#     'DreamerV3 (CBET) 0.0025' : 'dreamerv3/minigrid-coeff-0.0025/',
#     'DreamerV3 (CBET) 0.005' : 'dreamerv3/minigrid-coeff-0.005/',
# }

# df = generate_dfs(minigrid_dreamer, window=200000, step_limit=1e6)

# plot_scores(df, 'Extrinsic', 'DreamerV3 Minigrid', axs[0,1], y_lim=1)

# crafter_impala = {
#     'IMPALA (CBET) 0.001' : 'impala/crafter-tabula-rasa-1M-coeff-0.001/',
#     'IMPALA (CBET) 0.0025' : 'impala/crafter-tabula-rasa-1M-coeff-0.0025/',
#     'IMPALA (CBET) 0.005' : 'impala/crafter-tabula-rasa-1M-coeff-0.005/',
# }

# df = generate_dfs(crafter_impala, window=200000, step_limit=1e6)

# plot_scores(df, 'Extrinsic', 'IMPALA Crafter', axs[1,0], y_lim=10)

# crafter_dreamer = {
#     'DreamerV3 (CBET) 0.001' : 'dreamerv3/crafter-coeff-0.001/',
#     'DreamerV3 (CBET) 0.0025' : 'dreamerv3/crafter-coeff-0.0025/',
#     'DreamerV3 (CBET) 0.005' : 'dreamerv3/crafter-coeff-0.005/',
# }

# df = generate_dfs(crafter_dreamer, window=200000, step_limit=1e6)

# plot_scores(df, 'Extrinsic', 'DreamerV3 Crafter', axs[1,1], y_lim=10)

crafter_dreamer_plan = {
    'BASE 256' : 'dreamerv3/crafter-base-1M-planning-256/',
    'BASE 1024' : 'dreamerv3/crafter-base-1M-planning-1024/',
    'BASE 64' : 'dreamerv3/crafter-base-1M/',
    # Same with cbet
    'CBET 256' : 'dreamerv3/crafter-tabula-rasa-1M-planning-256/',
    'CBET 1024' : 'dreamerv3/crafter-tabula-rasa-1M-planning-1024/',
    'CBET 64' : 'dreamerv3/crafter-coeff-0.001/'
}

df = generate_dfs(crafter_dreamer_plan, window=200000, step_limit=1e6)

single_plot_scores(df, 'Extrinsic', 'Crafter', y_lim=10)

# long_crafter_comparison = {
#     'DreamerV3 (CBET)' : 'dreamerv3/crafter-coeff-0.001/',
#     'IMPALA (CBET)' : 'impala/crafter-tabula-rasa-15H/',
# }

# df = generate_dfs(long_crafter_comparison, window=200000, step_limit=5e6)

# plot_scores(df, 'Extrinsic', 'Crafter', y_lim=13, step_limit=5e6)

# Adjust these values as needed to better center the labels and set font size
# x_label_x_position = 0.5125  # This is typically centered, but adjust if your figure's layout is unusual
# y_label_y_position = 0.5  # Adjust this value to center the y-axis label, especially if the figure's height varies
# label_font_size = 14 # Example font size, adjust as needed

# fig.text(x_label_x_position, 0.05, 'Steps (Millions)', ha='center', fontsize=label_font_size)
# fig.text(0.07, y_label_y_position, 'Mean Extrinsic Return', va='center', rotation='vertical', fontsize=label_font_size)

# # After all plotting is done, but before plt.savefig and plt.show
# handles, labels = axs[0, 0].get_legend_handles_labels()  # Get handles and labels from one of the subplots

# # Filter out "mean_return" labels and their corresponding handles
# filtered_handles = []
# filtered_labels = []
# for handle, label in zip(handles, labels):
#     if label != "mean_return":
        
#         if "IMPALA" in label:
            
#             # Remove first two words
#             label = "c = " + ' '.join(label.split(' ')[2:])
        
#         filtered_handles.append(handle)
#         filtered_labels.append(label)

# # Add legend below the whole plot with filtered labels and handles
# fig.legend(filtered_handles, filtered_labels, loc='lower center', ncol=4)

# # make legend visible
# for ax in axs.flat:
#     ax.get_legend().remove()

# plt.savefig('images/intrinsic_coeff_combined_plots.png', dpi=600)
# plt.show()