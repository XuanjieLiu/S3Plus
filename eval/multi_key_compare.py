from typing import List
import matplotlib.pyplot as plt
import numpy as np

class MultiKeyCompareGroup:
    def __init__(
            self,
            title: str,
            y_name: str,
            keys: List[str],
            values: List[List[float]]
    ):
        self.title = title
        self.y_name = y_name
        self.keys = keys
        self.values = values
        assert len(keys) == len(values[0]), "The number of keys should be equal to the number of values."


def plot_sub_graph(ax: plt.Axes, compare_group: MultiKeyCompareGroup):
    x = list(range(1, len(compare_group.keys)+1))
    y_list = compare_group.values
    y0_mean = float(np.mean([y[0] for y in y_list]))
    y1_mean = float(np.mean([y[1] for y in y_list]))
    ax.axhline(y=y0_mean, c='red', label=f'{compare_group.keys[0]} mean', linestyle='dashed', linewidth=1.5)
    ax.axhline(y=y1_mean, c='red', label=f'{compare_group.keys[1]} mean', linestyle='solid', linewidth=1.5)
    i = 1
    for y in y_list:
        ax.plot(x, y, marker='o', linestyle='dashed', markerfacecolor='none', linewidth=1, markersize=12, label=f'exp {i}')
        i += 1
    ax.grid(True)
    ax.set(ylabel=compare_group.y_name)
    ax.set_yticks([i * 0.1 for i in range(0, 11)])
    ax.set_xticks(x, compare_group.keys)
    ax.set_title(compare_group.title)


def plot_graph(groups: List[MultiKeyCompareGroup], save_path):
    group_num = len(groups)
    fig, axs = plt.subplots(1, group_num, sharey="all", figsize=(group_num * 3+1, 5))
    if group_num == 1:
        axs = [axs]
    for i in range(group_num):
        plot_sub_graph(axs[i], groups[i])
    for ax in axs.flat:
        ax.label_outer()

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path)
    plt.cla()
    plt.clf()
    plt.close()
