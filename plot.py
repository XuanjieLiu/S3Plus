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
        df_val_series = get_rows(df_val, "symm0.3k4") * 100
        df_all_series = get_rows(df_all, "symm0.3k4") * 100
        common_rows = df_val_series.index.intersection(df_all_series.index)
        plot_series_belt(
            ax,
            x,
            df_val_series,
            "With Intrinsic Symmetry (K=4) on Val Set",
            ["forestgreen", "forestgreen"],
            linestyle="--",
        )
        plot_series_belt(
            ax,
            x,
            df_all_series,
            "With Intrinsic Symmetry (K=4) on All Set",
            ["forestgreen", "forestgreen"],
            linestyle="-",
        )
    except Exception as e:
        print(f"Error processing {domain}: {e}")

    try:
        df_val_series = get_rows(df_val, "symm0.3k1") * 100
        df_all_series = get_rows(df_all, "symm0.3k1") * 100
        common_rows = df_val_series.index.intersection(df_all_series.index)
        plot_series_belt(
            ax,
            x,
            df_val_series,
            "With Intrinsic Symmetry (K=1) on Val Set",
            ["yellowgreen", "yellowgreen"],
            linestyle="--",
        )
        plot_series_belt(
            ax,
            x,
            df_all_series,
            "With Intrinsic Symmetry (K=1) on All Set",
            ["yellowgreen", "yellowgreen"],
            linestyle="-",
        )
    except Exception as e:
        print(f"Error processing {domain}: {e}")

    try:
        df_val_series = get_rows(df_val, "nosymm") * 100
        df_all_series = get_rows(df_all, "nosymm") * 100
        common_rows = df_val_series.index.intersection(df_all_series.index)
        plot_series_belt(
            ax,
            x,
            df_val_series,
            "Without Intrinsic Symmetry on Val Set",
            ["sienna", "sienna"],
            linestyle="--",
        )
        plot_series_belt(
            ax,
            x,
            df_all_series,
            "Without Intrinsic Symmetry on All Set",
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
    subset = df[df.index.str.contains(keyword)]
    return subset


def get_rows_mean_std(df, keyword):
    subset = df[df.index.str.contains(keyword)]
    mean = subset.mean(axis=0).values
    std = subset.std(axis=0).values
    return mean, std


def plot_main_exp():
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # fig.suptitle("Preservation of Prediction Accuracies on OOD Inclusion", fontsize=18)

    row_vars = ["On X Domain", "On Z Domain"]
    for i in range(2):
        fig.text(
            0.1,  # x coordinate, adjust position
            0.7 - i * 0.4,  # y coordinate
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
            "major_sax_val6_induced_val_0823.csv",
            "major_sax_val6_induced_all_0823.csv",
        ],
        val=6,
        domain="x",
    )
    plot_certain_val_domain(
        ax=axs[1, 1],
        paths_to_csv=[
            "major_sax_val6_induced_val_0823.csv",
            "major_sax_val6_induced_all_0823.csv",
        ],
        val=6,
        domain="z",
    )
    plot_certain_val_domain(
        ax=axs[0, 0],
        paths_to_csv=[
            "major_sax_val3_induced_val_0827.csv",
            "major_sax_val3_induced_all_0827.csv",
        ],
        val=3,
        domain="x",
    )
    plot_certain_val_domain(
        ax=axs[1, 0],
        paths_to_csv=[
            "major_sax_val3_induced_val_0827.csv",
            "major_sax_val3_induced_all_0827.csv",
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
    plt.savefig("performance_plot.pdf")


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


if __name__ == "__main__":
    plot_downstream_exp()
