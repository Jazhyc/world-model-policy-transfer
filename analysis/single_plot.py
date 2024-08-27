# Only used for plotting planning coefficients

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import to_rgba, to_hex
import numpy as np

def adjust_color_lightness(color, amount=0.5):
    """
    Adjusts a color's brightness.
    amount: < 1.0 will make the color darker, > 1.0 will make it brighter.
    """
    try:
        c = to_rgba(color)
        # Calculate brightness adjustment
        c = (c[0] * amount, c[1] * amount, c[2] * amount, c[3])
        return to_hex(c)
    except ValueError:
        # If color format is invalid, return the original color
        return color


def plot_scores(df, reward_type, env_name, y_lim=1.0, step_limit=1e6):
    plt.figure(figsize=(10, 5))
    df.groupby('label').plot(x='step', y='mean_return', ax=plt.gca())

    lines = []
    labels = []

    color_dict = {
        'DreamerV3': '#4E79A7',  # Blue
        'IMPALA': '#F28E2B',  # Orange,
        '64': '#4E79A7',  # Blue
        '256': '#F28E2B',  # Orange
        '1024': '#E15759',  # Red
    }
    
    # Triangle for BASE, Square for CBET
    shape_dict = {
        '(BASE)': 's',
        '(CBET)': '^'
    }

    for label, group in df.groupby('label'):
        
        words = label.split(' ')
        
        model, model_type = words[1], words[0]
        model_type = f'({model_type})'
        
        # get the color and shape
        color = color_dict[model]
        shape = shape_dict[model_type]
        
        # If model type is CBET, make color darker
        if model_type == '(CBET)':
            color = adjust_color_lightness(color, amount=0.5)  # Adjust the amount to control darkness
        
        if len(words) == 3:
            color = color_dict[words[2]]
            if model_type == '(CBET)':
                color = adjust_color_lightness(color, amount=0.5)
        
        line, = plt.plot(group['step'], group['mean_return'], color=color, marker=shape, label=label)
        plt.fill_between(group['step'], group['mean_return'] - group['std_error_return'], group['mean_return'] + group['std_error_return'], alpha=0.2, color=line.get_color())
        lines.append(line)
        labels.append(label)

    # Fix legend to top left
    plt.legend(lines, labels, loc='upper left')
    plt.xlabel('Step (Millions)')
    plt.ylabel(f'Mean {reward_type.capitalize()} Return')
    # plt.title(f"{env_name} {reward_type.capitalize()} Return")
    plt.grid()
    plt.ylim(0, y_lim)
    plt.xlim(0, step_limit)
    
    # Format x axis
    formatter = FuncFormatter(lambda x, pos: f'{round(x * 1e-6, 2)}')
    plt.gca().xaxis.set_major_formatter(formatter)
    
    plt.savefig(f"images/{env_name.capitalize()}_{reward_type}.png", dpi=600)
    plt.show()