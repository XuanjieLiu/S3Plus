import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from brokenaxes import brokenaxes


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "grid.alpha": 0.25,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "lines.dashed_pattern": [3, 3],
        "figure.dpi": 120,
    }
)



def plot_certain_val(ax, paths_to_csv, val=6):
    assert len(paths_to_csv) == 2
    assert "val" in paths_to_csv[0]
    assert "ood" in paths_to_csv[1]

    df_val_x = pd.read_csv(paths_to_csv[0], index_col=0, usecols=[0] + list(range(1, 8)))
    df_all_x = pd.read_csv(paths_to_csv[1], index_col=0, usecols=[0] + list(range(1, 8)))

    df_val_z = pd.read_csv(paths_to_csv[0], index_col=0, usecols=[0] + list(range(8, 15)))
    df_all_z = pd.read_csv(paths_to_csv[1], index_col=0, usecols=[0] + list(range(8, 15)))

    x = np.arange(7)

    symm_x_val = get_rows(df_val_x, "symm0.2") * 100
    nosymm_x_val = get_rows(df_val_x, "nosymm") * 100
    symm_z_val = get_rows(df_val_z, "symm0.2") * 100
    nosymm_z_val = get_rows(df_val_z, "nosymm") * 100

    symm_x_all = get_rows(df_all_x, "symm0.2") * 100
    nosymm_x_all = get_rows(df_all_x, "nosymm") * 100
    symm_z_all = get_rows(df_all_z, "symm0.2") * 100
    nosymm_z_all = get_rows(df_all_z, "nosymm") * 100

    data_groups = [
        ("w/ symm on X", symm_x_val, symm_x_all, "forestgreen"),
        ("w/o symm on X", nosymm_x_val, nosymm_x_all, "Sienna"),
        ("w/ symm on Z", symm_z_val, symm_z_all, "limegreen"),
        ("w/o symm on Z", nosymm_z_val, nosymm_z_all, "Chocolate"),
    ]
    bar_width = 0.2
    offsets = np.linspace(-1.5, 1.5, len(data_groups)) * bar_width

    # for offset, (label, data_1, data_2, color) in zip(offsets, data_groups):
    #     if data_1 is not None and not data_1.empty:
    #         mean_vals = data_1.mean(axis=0).values
    #         std_vals = data_1.std(axis=0).values
    #         ax.bar(x + offset, mean_vals, width=bar_width,
    #                 label=label, fill=False, edgecolor=color, alpha=0.5, linestyle="-", linewidth=1.2,)
    #     if data_2 is not None and not data_2.empty:
    #         mean_vals = data_2.mean(axis=0).values
    #         std_vals = data_2.std(axis=0).values
    #         ax.bar(x + offset, mean_vals, yerr=std_vals, width=bar_width,
    #                 label=label, color=color, alpha=1, edgecolor=color, linestyle="-", linewidth=1.2,)
    for offset, (label, data_1, data_2, color) in zip(offsets, data_groups):
        if data_1 is not None and not data_1.empty:
            mean_vals = data_1.mean(axis=0).values
            std_vals = data_1.std(axis=0).values
            ax.bar(x + offset, mean_vals, width=bar_width,
                    label="_nolegend_", color=color, alpha=0.2)
        if data_2 is not None and not data_2.empty:
            mean_vals = data_2.mean(axis=0).values
            std_vals = data_2.std(axis=0).values
            ax.bar(x + offset, mean_vals, yerr=std_vals, width=bar_width,
                    label=label, color=color, alpha=0.8, capsize=2, error_kw={"elinewidth":1.2})

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(0, 7)])
    ax.set_xlabel("Number of Predicted Step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(50, 100)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    # ax.legend(frameon=False)


def plot_series_belt(ax, x, data_series, label, colors, linestyle):
    """
    Plot a series with a shaded error band.
    """
    mean_diff_series = data_series.mean(axis=0).values
    std_diff_series = data_series.std(axis=0).values

    ax.plot(
        x,
        mean_diff_series,
        label=label,
        color=colors[0],
        marker="o",
        markeredgewidth=1,
        linestyle=linestyle,
    )
    ax.fill_between(
        x,
        mean_diff_series - std_diff_series,
        mean_diff_series + std_diff_series,
        color=colors[1],
        alpha=0.05,
    )


def get_rows(df, keyword):
    """
    Get a subset of the DataFrame that contains rows with a specific keyword.
    """
    subset = df[df.index.str.contains(keyword)]
    return subset


def get_rows_mean_std(df, keyword):
    """
    Get the mean and standard deviation of rows in a DataFrame that contain a specific keyword.
    """
    subset = df[df.index.str.contains(keyword)]
    mean = subset.mean(axis=0).values
    std = subset.std(axis=0).values
    return mean, std


def plot_main_exp():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # fig.suptitle("Preservation of Prediction Accuracies on OOD Inclusion", fontsize=18)

    # row_vars = ["X Domain", "Z Domain"]
    # for i in range(2):
    #     fig.text(
    #         0.11,  # x coordinate, adjust position
    #         0.78 - i * 0.4,  # y coordinate
    #         row_vars[i],
    #         va="center",
    #         ha="right",
    #         fontsize=16,
    #         rotation=0,
    #     )

    col_vars = [
        "Train & Val: 3 Keys",
        "Train & Val: 6 Keys",
    ]
    for j in range(2):
        fig.text(
            0.35
            + j
            * 0.43,  # x coordinate, change with column, need to adjust based on actual
            0.95,  # y coordinate, top
            col_vars[j],
            ha="center",
            va="bottom",
            fontsize=16,
        )

    plot_certain_val(
        ax=axs[0],
        paths_to_csv=[
            "experiments/transition/results/final_1119/source_csv/major_sax_val3_transition_val_1102.csv",
            "experiments/transition/results/final_1119/source_csv/major_sax_val3_transition_ood_1102.csv",
        ],
        val=3,
    )
    plot_certain_val(
        ax=axs[1],
        paths_to_csv=[
            "experiments/transition/results/final_1119/source_csv/major_sax_val6_transition_val_1031.csv",
            "experiments/transition/results/final_1119/source_csv/major_sax_val6_transition_ood_1031.csv",
        ],
        val=6,
    )

    handles, labels = axs[0].get_legend_handles_labels()
    # reorder = [0, 2, 4, 1, 3, 5]
    # handles = [handles[i] for i in reorder]
    # labels = [labels[i] for i in reorder]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        handlelength=4,
        ncol=2,
        bbox_to_anchor=(0.5, 0),  # (x, y) coordinates
        fontsize=14,
    )

    plt.tight_layout(
        rect=[0.1, 0.15, 1, 0.95]
    )  # the subplots will be put between left, bottom, right, top
    # plt.savefig("experiments/transition/results/archive/figures/transition_performance_plot_1105.pdf", dpi=500)
    plt.savefig("experiments/transition/results/archive/figures/transition_performance_plot_1105.png", dpi=500, transparent=True)



if __name__ == "__main__":
    plot_main_exp()
