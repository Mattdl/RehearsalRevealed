import pandas
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import rc
from matplotlib.lines import Line2D
import os

rc('text', usetex=True)
rc('font', size=16)

_BASE_PATH = ".."


def df_columns_to_numpy(frame, columns, dtype=np.int):
    col_arrays = []
    for col in columns:
        col_arrays.append(frame[col].to_numpy(dtype=dtype))
    return np.concatenate(col_arrays)


def plot_std_mean_curve(idx, path, ax, color="firebrick", ls="-", fs="full", zorder=0):
    selected_path = np.stack(np.array([path[i] for i in idx]))
    mean = np.mean(selected_path, axis=0)
    std = np.std(selected_path, axis=0)
    x = np.arange(0, len(mean))
    markersize = 16 if "+" in ls else 7
    ax.plot(x, mean, ls, c=color, markersize=markersize, fillstyle=fs, markeredgewidth=1.5, zorder=zorder)
    ax.fill_between(x, mean, mean+std, color=color, alpha=0.2)
    ax.fill_between(x, mean, mean-std, color=color, alpha=0.2)


def plot_all_curves(idx, path, ax, color="firebrick", ls="-", fs="full", zorder=0):
    selected_path = np.stack(np.array([path[i] for i in idx]))
    x = np.arange(0, len(selected_path[0]))
    for line in selected_path:
        ax.plot(x, line, alpha=0.3, c=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default="mnist")
    args = parser.parse_args()

    # Read in dataframe with indices
    df_idx = pandas.read_csv(f"{_BASE_PATH}/results/{args.data}/linear_path_idx_m.csv")

    # load paths and transpose (to make idx the key)
    paths = pandas.read_csv(f"{_BASE_PATH}/results/{args.data}/linear_path_raw_m.csv")
    paths = paths.T
    paths.columns = [int(i) for i in paths.iloc[0]]
    paths = paths.drop(paths.index[0])

    base_2_test_1_idx = df_columns_to_numpy(df_idx, ["w1_w2_test_1"])
    base_2_train_1_idx = df_columns_to_numpy(df_idx, ["w1_w2_train_1"])
    base_2_buffer_idx = df_columns_to_numpy(df_idx, ["w1_w2_mem_1", "w1_w2_mem_2"])

    inter_2_test_1_idx = df_columns_to_numpy(df_idx, ["w2_w2_test_1"])
    inter_2_test_12_idx = df_columns_to_numpy(df_idx, ["w2_w2_test_12"])

    base_5_test_1_idx = df_columns_to_numpy(df_idx, ["w1_w5_test_1"])
    base_5_train_1_idx = df_columns_to_numpy(df_idx, ["w1_w5_train_1"])
    base_5_buffer_idx = df_columns_to_numpy(df_idx, ["w1_w5_mem_1", "w1_w5_mem_2"])

    inter_5_test_1_idx = df_columns_to_numpy(df_idx, ["w5_w5_test_1"])
    inter_5_test_full_idx = df_columns_to_numpy(df_idx, ["w5_w5_test_15"])

    fig, axes = plt.subplots(1, 4, figsize=(15, 3))

    plot_std_mean_curve(base_2_test_1_idx, paths, axes[0], color="brown", ls="-o", zorder=1)
    plot_std_mean_curve(base_2_train_1_idx, paths, axes[0], color="black", ls="-+", zorder=0)
    plot_std_mean_curve(base_2_buffer_idx, paths, axes[0], color="blue", ls="-o")

    plot_std_mean_curve(base_5_test_1_idx, paths, axes[1], ls="-o", zorder=1)
    plot_std_mean_curve(base_5_train_1_idx, paths, axes[1], color="black", ls="-+", zorder=0)
    plot_std_mean_curve(base_5_buffer_idx, paths, axes[1], color="blue", ls="-o")

    plot_std_mean_curve(inter_2_test_1_idx, paths, axes[2], ls="-o")
    plot_std_mean_curve(inter_2_test_12_idx, paths, axes[2], color="green", ls="-o")

    plot_std_mean_curve(inter_5_test_1_idx, paths, axes[3], ls="-o")
    plot_std_mean_curve(inter_5_test_full_idx, paths, axes[3], color="green", ls="-o")

    if args.data == "mnist":
        ylim = (-0.1, 1.0)
    elif args.data == "cifar":
        ylim = (-0.5, 10)
    elif args.data == "min":
        ylim = (-1, 16)
    else:
        raise ValueError

    for ax in axes:
        ax.hlines(0, -0.01, 9.01, linestyles="dashed", zorder=-10, colors="gray")
        ax.set_ylim(ylim)
        ax.set_xlim((-0.01, 9.01))
        ax.set_xticks([0, 9])

    axes[0].set_xticklabels(["$w_1$", "$w_2$"])
    axes[1].set_xticklabels(["$w_1$", "$w_5$"])
    axes[2].set_xticklabels(["$w_2$", "$w_2'$"])
    axes[3].set_xticklabels(["$w_5$", "$w_5'$"])

    axes[0].set_ylabel("Loss")
    axes[0].set_title("$w_1$ to $w_2$ (a)")
    axes[1].set_title("$w_1$ to $w_5$ (b)")
    axes[2].set_title("$w_2$ to $w_2'$ (c)")
    axes[3].set_title("$w_5$ to $w_5'$ (d)")
    custom_lines_1 = [Line2D([0], [0], color="firebrick", lw=4),
                      Line2D([0], [0], color="black", lw=4),
                      Line2D([0], [0], color="blue", lw=4)]
    custom_lines_2 = [Line2D([0], [0], color="firebrick", lw=4),
                      Line2D([0], [0], color="green", lw=4)]

    axes[0].legend(custom_lines_1, ["T1 test", "T1 Train", "Reh. Memory"], fontsize="x-small", loc="upper left")
    axes[1].legend(custom_lines_1, ["T1 test", "T1 Train", "Reh. Memory"], fontsize="x-small", loc="upper left")
    axes[2].legend(custom_lines_2, ["T1 test", r"T\{1, 2\} test"], fontsize="x-small", loc="upper left")
    axes[3].legend(custom_lines_2, ["T1 test", r"T\{1 $\cdots$ 5\} test"], fontsize="x-small", loc="upper left")

    if not os.path.exists(f"{_BASE_PATH}/graphics/{args.data}"):
        os.makedirs(f"{_BASE_PATH}/graphics/{args.data}")

    plt.tight_layout()
    plt.savefig(f"{_BASE_PATH}/graphics/{args.data}/lin_paths.pdf")
    plt.savefig(f"{_BASE_PATH}/graphics/{args.data}/lin_paths.svg")
    plt.show()


if __name__ == '__main__':
    main()
