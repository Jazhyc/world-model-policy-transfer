import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import to_rgba, to_hex
import numpy as np

from single_plot import adjust_color_lightness

def plot_scores(df, reward_type, env_name, ax, y_lim=1.0, step_limit=1e6, add_legend=False, hide_y_ticks=False):
    df.groupby('label').plot(x='step', y='mean_return', ax=ax)

    color_dict = {
        'DreamerV3': '#4E79A7',  # Blue
        'IMPALA': '#F28E2B',  # Orange
        '0.001': '#4E79A7',  # Blue
        '0.0025': '#F28E2B',  # Orange
        '0.005': '#E15759',  # Red
    }
    
    # Triangle for BASE, Square for CBET
    shape_dict = {
        '(BASE)': 's',
        '(CBET)': '^'
    }

    for label, group in df.groupby('label'):
        words = label.split(' ')
        
        # split the label into the model and the type
        model, model_type = words[0], words[1]
        
        # get the color and shape
        color = color_dict[model]
        shape = shape_dict[model_type]
        
        # If model type is CBET, make color darker
        if model_type == '(CBET)':
            color = adjust_color_lightness(color, amount=0.5)  # Adjust the amount to control darkness
            
        if len(words) == 3:
            color = color_dict[words[2]]
        
        ax.plot(group['step'], group['mean_return'], color=color, marker=shape, label=label)
        ax.fill_between(group['step'], group['mean_return'] - group['std_error_return'], group['mean_return'] + group['std_error_return'], alpha=0.2, color=color)

    # ax.set_xlabel('Step (Millions)')
    # ax.set_ylabel(f'Mean {reward_type.capitalize()} Return')
    
    # remove labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    ax.grid()
    ax.set_ylim(0, y_lim)
    ax.set_xlim(0, step_limit)
    
    # add title
    ax.set_title(env_name)
    
    # hide y ticks, grid should still be visible but text should be hidden
    if hide_y_ticks:
        ax.yaxis.set_ticklabels([])
    
    # Format x axis
    formatter = FuncFormatter(lambda x, pos: f'{round(x * 1e-6, 2)}')
    ax.xaxis.set_major_formatter(formatter)

    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        return handles, labels

    return None, None
