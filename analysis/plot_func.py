import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def create_df(data, label):
    df = pd.DataFrame(data, columns=['step', 'mean_return', 'std_error_return'])
    df['label'] = label
    return df

def plot_scores(df, step_limit, reward_type, env_name, y_lim=1.0):
    plt.figure(figsize=(10, 5))
    df.groupby('label').plot(x='step', y='mean_return', ax=plt.gca())

    lines = []
    labels = []

    color_dict = {
        'DreamerV3': '#1f77b4',  # muted blue
        'DreamerV3 (CBET)': '#ff7f0e',  # safety orange
        'IMPALA': '#2ca02c',  # cooked asparagus green
        'IMPALA (CBET)': '#d62728',  # brick red
    }

    for label, group in df.groupby('label'):
        
        if label in color_dict:
            color = color_dict[label]
        else:
            # Generate a random color in hex
            color = '#{:06x}'.format(np.random.randint(0, 256**3))
        
        line, = plt.plot(group['step'], group['mean_return'], color)
        plt.fill_between(group['step'], group['mean_return'] - group['std_error_return'], group['mean_return'] + group['std_error_return'], alpha=0.2, color=line.get_color())
        lines.append(line)
        labels.append(label)

    plt.legend(lines, labels)
    plt.xlabel('Time step (in Millions)')
    plt.ylabel(f'Mean {reward_type.capitalize()} Return')
    plt.title(f"{env_name.capitalize()} {reward_type.capitalize()} Return over Time")
    plt.grid()
    plt.ylim(0, y_lim)
    plt.xlim(0, step_limit)
    
    # Format x axis
    formatter = FuncFormatter(lambda x, pos: f'{round(x * 1e-6, 2)}')
    plt.gca().xaxis.set_major_formatter(formatter)
    
    plt.savefig(f"{env_name.capitalize()}_{reward_type}.png", dpi=600)
    plt.show()