from typing import List
import matplotlib.pyplot as plt

colors_order = ['blue', 'gold', 'chocolate', 'red', 'green', 'grey']
class EpochCompareGroup:
    def __init__(
            self,
            title: str,
            y_name: str,
            sub_record: List[dict],
            keys: List[str],
            key_labels: List[str],
            keys_linestyle: List[str],
    ):
        self.title = title
        self.y_name = y_name
        self.sub_record = sub_record
        self.keys = keys
        self.key_labels=key_labels
        self.keys_linestyle = keys_linestyle
        assert len(keys) == len(keys_linestyle), "The number of keys should be equal to the number of keys_linestyle."


def plot_sub_graph(ax: plt.Axes, compare_group: EpochCompareGroup):
    for i in range(len(compare_group.sub_record)):
        color = colors_order[i]
        for j in range(len(compare_group.keys)):
            key = compare_group.keys[j]
            key_label = compare_group.key_labels[j]
            line_style = compare_group.keys_linestyle[j]
            x = compare_group.sub_record[i][key].X
            y = compare_group.sub_record[i][key].Y
            ax.plot(x, y, marker='o', linestyle=line_style, linewidth=1, markersize=1, label=f'exp {i} {key_label}', color=color)
            ax.grid(True)
            ax.set(ylabel=compare_group.y_name)
            ax.set(xlabel="Epoch Number")
            ax.set_title(compare_group.title)


def plot_graph(groups: List[EpochCompareGroup], save_path):
    group_num = len(groups)
    fig, axs = plt.subplots(1, group_num, sharey="all", figsize=(group_num * 5, 5))
    if group_num == 1:
        axs = [axs]
    for i in range(group_num):
        plot_sub_graph(axs[i], groups[i])
    for ax in axs.flat:
        ax.label_outer()
    plt.legend()
    plt.savefig(save_path)
    plt.cla()
    plt.clf()
    plt.close()