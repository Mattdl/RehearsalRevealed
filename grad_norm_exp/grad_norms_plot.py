import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

_BASE_PATH = ".."


def merge_paths(data):
    try:
        arrays = [np.load(f"{_BASE_PATH}/results/{data}/grad_norms_merged.npy")]
    except FileNotFoundError:
        arrays = []

    files = os.listdir(f"{_BASE_PATH}/results/{data}/")
    for file in files:
        if file.startswith("grad_norms") and not file.endswith("merged.npy"):
            arrays.append(np.load(f'{_BASE_PATH}/results/{data}/{file}'))
            os.remove(f'{_BASE_PATH}/results/{data}/{file}')

    arrays = np.concatenate(arrays, axis=1)
    print(f"{data} samples: {arrays.shape[1]}")
    np.save(f"{_BASE_PATH}/results/{data}/grad_norms_merged.npy", arrays)
    return arrays


results_mnist = merge_paths("mnist")
results_mnist /= np.max(results_mnist)
# results_cifar = merge_paths("cifar")
# results_cifar /= np.max(results_cifar)
# results_min = merge_paths("min")
# results_min /= np.max(results_min)

x = np.array([0, 1, 2, 3, 4, 5])
labels = ["$w_1$", "$w_2$", "$w_5$", "$w_1$", "$w_2$", "$w_5$"]

mnist_color = "firebrick"
cifar_color = "navy"
min_color = "forestgreen"
props = {"color": mnist_color, "markeredgecolor": mnist_color}
c_props = {"color": cifar_color, "markeredgecolor": cifar_color}
m_props = {"color": min_color, "markeredgecolor": min_color}
whis = (0, 100)
plt.figure(figsize=(7, 4))
plt.boxplot(results_mnist.T, positions=x-0.2, widths=0.1, boxprops=props, whiskerprops=props, flierprops=props,
            capprops=props, whis=whis)
# plt.boxplot(results_cifar.T, positions=x, widths=0.1, boxprops=c_props, whiskerprops=c_props, flierprops=c_props,
#             capprops=c_props, whis=whis)
# plt.boxplot(results_min.T, positions=x+0.2, widths=0.1, boxprops=m_props, whiskerprops=m_props, flierprops=m_props,
#             capprops=m_props, whis=whis)
plt.ylabel("Rescaled gradient norm")
plt.xticks(x, labels)
plt.tight_layout()

custom_lines = [Line2D([0], [0], color=mnist_color, lw=4),
                Line2D([0], [0], color=cifar_color, lw=4),
                Line2D([0], [0], color=min_color, lw=4)]
plt.legend(custom_lines, ["MNIST", "CIFAR", "MINI-IN"], fontsize="x-small", loc="upper left")

if not os.path.exists(f"{_BASE_PATH}/graphics"):
    os.makedirs(f"{_BASE_PATH}/graphics")

plt.savefig(f"{_BASE_PATH}/graphics/grad_norms.svg")
plt.savefig(f"{_BASE_PATH}/graphics/grad_norms.pdf")
plt.show()
