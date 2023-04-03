from typing import List
import matplotlib.pyplot as plt


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
    i = 1
    for y in y_list:
        ax.plot(x, y, marker='o', linestyle='dashed', markerfacecolor='none', linewidth=1, markersize=12, label=f'exp {i}')
        i += 1
    ax.grid(True)
    ax.set(ylabel=compare_group.y_name)
    ax.set_xticks(x, compare_group.keys)
    ax.set_title(compare_group.title)


def plot_graph(groups: List[MultiKeyCompareGroup], save_path):
    group_num = len(groups)
    fig, axs = plt.subplots(1, group_num, sharey="all", figsize=(group_num * 3, 5))
    if group_num == 1:
        axs = [axs]
    for i in range(group_num):
        plot_sub_graph(axs[i], groups[i])
    for ax in axs.flat:
        ax.label_outer()
    # plt.legend()
    plt.savefig(save_path)
    plt.cla()
    plt.clf()
    plt.close()
