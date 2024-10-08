import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams

# rcParams["text.usetex"] = True
rcParams["font.family"] = "sans"

method = "random_search"
search_space = "small"
model = "bert-base-cased"
# model = 'roberta-base'
epochs = 5
df = pd.read_csv(f"all_results.csv")
df = df.query(
    f"search_space == '{search_space}' & epoch == {epochs} "
    f"& method == '{method}' & model == '{model}'"
)
config = {
    "standard": {"label": "standard", "random_sub_nets": 1},
    "random": {"label": "random", "random_sub_nets": 1},
    "random_linear": {"label": "linear", "random_sub_nets": 1},
    "sandwich": {"label": "sandwich", "random_sub_nets": 2},
    "kd": {"label": "inplace-kd", "random_sub_nets": 2},
    "full": {"label": "full", "random_sub_nets": 2},
}
marker = ["o", "x", "s", "d", "p", "P", "^", "v", "<", ">"]
checkpoint_names = [
    "standard",
    "random",
    "random_linear",
    "sandwich",
    "full",
    "kd",
]
for dataset, df_benchmark in df.groupby("dataset"):
    plt.figure(dpi=200)
    vals, xs = [], []
    for mi, checkpoint in enumerate(checkpoint_names):
        df_checkpoint = df_benchmark.query(
            f"checkpoint == '{checkpoint}'"
        )
        max_runtime = df_checkpoint.runtime.max()
        y = df_checkpoint[df_checkpoint["runtime"] == max_runtime]["hv"]
        # plt.boxplot(y, positions=[mi])
        vals.append(y)
        # checkpoint_names.append(checkpoint)
        xs.append(np.random.normal(mi + 1, 0.04, y.shape[0]))
    labels = [config[checkpoint]["label"] for checkpoint in checkpoint_names]
    plt.boxplot(vals)
    palette = [f"C{i}" for i in range(len(checkpoint_names))]
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)
    plt.xticks(np.arange(1, 1 + len(checkpoint_names)), labels, fontsize=15)
    plt.ylabel("hypervolume", fontsize=20)

    plt.title(f"{dataset.upper()}", fontsize=20)
    plt.xlabel("super-network training strategy", fontsize=20)
    plt.grid(linewidth="1", alpha=0.4)
    plt.savefig(
        f"./figures/hypervolume_checkpoints_{dataset}_{model}.pdf", bbox_inches="tight"
    )
    plt.show()
