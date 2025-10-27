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


def plot_certain_val_domain(ax, paths_to_csv, val=6, domain="x"):
    assert len(paths_to_csv) == 2
    assert "val" in paths_to_csv[0]
    assert "all" in paths_to_csv[1]
    if domain == "x":
        df_val = pd.read_csv(
            paths_to_csv[0], index_col=0, usecols=[0] + list(range(1, 8))
        )
        df_all = pd.read_csv(
            paths_to_csv[1], index_col=0, usecols=[0] + list(range(1, 8))
        )
        x = np.arange(0, 7)
    elif domain == "z":
        df_val = pd.read_csv(
            paths_to_csv[0], index_col=0, usecols=[0] + list(range(8, 15))
        )
        df_all = pd.read_csv(
            paths_to_csv[1], index_col=0, usecols=[0] + list(range(8, 15))
        )
        x = np.arange(7, 14)

    try:
        df_val_series = get_rows(df_val, "symm0.3") * 100
        df_all_series = get_rows(df_all, "symm0.3") * 100
        common_rows = df_val_series.index.intersection(df_all_series.index)
        plot_series_belt(
            ax,
            x,
            df_val_series,
            "w/ symmetry on val set",
            ["forestgreen", "forestgreen"],
            linestyle="--",
        )
        plot_series_belt(
            ax,
            x,
            df_all_series,
            "w/ symmetry on all set",
            ["forestgreen", "forestgreen"],
            linestyle="-",
        )
    except Exception as e:
        print(f"Error processing {domain}: {e}")

    # try:
    #     df_val_series = get_rows(df_val, "symm0.3k1") * 100
    #     df_all_series = get_rows(df_all, "symm0.3k1") * 100
    #     common_rows = df_val_series.index.intersection(df_all_series.index)
    #     plot_series_belt(
    #         ax,
    #         x,
    #         df_val_series,
    #         "w/ symmetry (K=1) on val set",
    #         ["yellowgreen", "yellowgreen"],
    #         linestyle="--",
    #     )
    #     plot_series_belt(
    #         ax,
    #         x,
    #         df_all_series,
    #         "w/ symmetry (K=1) on all set",
    #         ["yellowgreen", "yellowgreen"],
    #         linestyle="-",
    #     )
    # except Exception as e:
    #     print(f"Error processing {domain}: {e}")

    try:
        df_val_series = get_rows(df_val, "nosymm") * 100
        df_all_series = get_rows(df_all, "nosymm") * 100
        common_rows = df_val_series.index.intersection(df_all_series.index)
        plot_series_belt(
            ax,
            x,
            df_val_series,
            "w/o symmetry on val set",
            ["sienna", "sienna"],
            linestyle="--",
        )
        plot_series_belt(
            ax,
            x,
            df_all_series,
            "w/o symmetry on all set",
            ["sienna", "sienna"],
            linestyle="-",
        )
    except Exception as e:
        print(f"Error processing {domain}: {e}")

    tests = [str(i) for i in range(0, 7)]

    ax.set_xticks(x)
    ax.set_xticklabels(tests)
    ax.set_xlabel("Number of Predicted Step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)

    # ax.set_title('')
    # ax.legend(frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3)

    # plt.tight_layout()


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
        alpha=0.1,
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
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    # fig.suptitle("Preservation of Prediction Accuracies on OOD Inclusion", fontsize=18)

    row_vars = ["X Domain", "Z Domain"]
    for i in range(2):
        fig.text(
            0.11,  # x coordinate, adjust position
            0.78 - i * 0.4,  # y coordinate
            row_vars[i],
            va="center",
            ha="right",
            fontsize=16,
            rotation=0,
        )

    col_vars = [
        # "Train & Val: 1 Key",
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

    plot_certain_val_domain(
        ax=axs[0, 1],
        paths_to_csv=[
            "major_sax_val6_transition_val_1025.csv",
            "major_sax_val6_transition_all_1025.csv",
        ],
        val=6,
        domain="x",
    )
    plot_certain_val_domain(
        ax=axs[1, 1],
        paths_to_csv=[
            "major_sax_val6_transition_val_1025.csv",
            "major_sax_val6_transition_all_1025.csv",
        ],
        val=6,
        domain="z",
    )
    # plot_certain_val_domain(
    #     ax=axs[0, 0],
    #     paths_to_csv=[
    #         "major_sax_val3_induced_val_0827.csv",
    #         "major_sax_val3_induced_all_0827.csv",
    #     ],
    #     val=3,
    #     domain="x",
    # )
    # plot_certain_val_domain(
    #     ax=axs[1, 0],
    #     paths_to_csv=[
    #         "major_sax_val3_induced_val_0827.csv",
    #         "major_sax_val3_induced_all_0827.csv",
    #     ],
    #     val=3,
    #     domain="z",
    # )

    handles, labels = axs[0, 0].get_legend_handles_labels()
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
    # plt.savefig("performance_plot.svg", transparent=True, dpi=500)

    plt.savefig("transition_performance_plot.pdf", dpi=500)


def plot_downstream_exp():
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # fig.suptitle("Preservation of Prediction Accuracies on OOD Inclusion", fontsize=18)

    row_vars = ["On X Domain", "On Z Domain"]
    for i in range(2):
        fig.text(
            0.1,  # x coordinate, adjust position
            0.7 - i * 0.4,  # y coordinate, change with row
            row_vars[i],
            va="center",
            ha="right",
            fontsize=16,
            rotation=0,
        )

    col_vars = [
        "Train & Val: 3 Keys",
        "Train & Val: 6 Keys",
    ]
    for j in range(2):
        fig.text(
            0.33
            + j
            * 0.4,  # x coordinate, change with column, need to adjust based on actual
            0.95,  # y coordinate, top
            col_vars[j],
            ha="center",
            va="bottom",
            fontsize=16,
        )

    plot_certain_val_domain(
        ax=axs[0, 1],
        paths_to_csv=[
            "major_sax_val6_induced_minordown_val_0906.csv",
            "major_sax_val6_induced_minordown_all_0906.csv",
        ],
        val=6,
        domain="x",
    )
    plot_certain_val_domain(
        ax=axs[1, 1],
        paths_to_csv=[
            "major_sax_val6_induced_minordown_val_0906.csv",
            "major_sax_val6_induced_minordown_all_0906.csv",
        ],
        val=6,
        domain="z",
    )
    plot_certain_val_domain(
        ax=axs[0, 0],
        paths_to_csv=[
            "major_sax_val3_induced_minordown_val_0906.csv",
            "major_sax_val3_induced_minordown_all_0906.csv",
        ],
        val=3,
        domain="x",
    )
    plot_certain_val_domain(
        ax=axs[1, 0],
        paths_to_csv=[
            "major_sax_val3_induced_minordown_val_0906.csv",
            "major_sax_val3_induced_minordown_all_0906.csv",
        ],
        val=3,
        domain="z",
    )

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        handlelength=4,
        ncol=3,
        bbox_to_anchor=(0.55, 0),  # (x, y) coordinates
    )

    plt.tight_layout(
        rect=[0.1, 0.05, 1, 0.95]
    )  # the subplots will be put between left, bottom, right, top
    plt.savefig("performance_downstream_plot.pdf")


def plot_interval_probing_exp():
    """
    two subplots for val3 and val6, three pillars for k4, k1, nosymm
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    col_vars = ["Train & Val: 3 Keys", "Train & Val: 6 Keys"]
    for j in range(2):
        fig.text(
            0.28 + j * 0.45,
            0.92,
            col_vars[j],
            ha="center",
            va="bottom",
            fontsize=15,
        )

    csv_files = [
        "major_sav_val3_induced_probe_0914.csv",
        "major_sav_val6_induced_probe_0914.csv",
    ]
    bar_labels = [
        "w/ symmetry",
        # "w/ symmetry (K=1)",
        "w/o symmetry",
    ]
    keywords = ["symm1", "nosymm"]
    colors = ["forestgreen", "sienna"]

    for idx, ax in enumerate(axs):
        means = []
        stds = []
        for k, keyword in enumerate(keywords):
            df = pd.read_csv(csv_files[idx], index_col=0)
            mean, std = get_rows_mean_std(df, keyword)
            means.append(mean[0])
            stds.append(std[0])

        means = np.array(means)
        stds = np.array(stds)
        x = np.arange(len(bar_labels))

        bars = ax.bar(x, means, color=colors, yerr=stds, capsize=5, alpha=0.7)
        ax.set_ylabel("Interval Prober Val Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=15)

    plt.tight_layout(
        rect=[0.0, 0.05, 1, 0.95]
    )  # the subplots will be put between left, bottom, right, top

    plt.savefig("interval_probing_plot.svg", transparent=True, dpi=500)


if __name__ == "__main__":
    plot_main_exp()
    # plot_downstream_exp()
    # plot_interval_probing_exp()
