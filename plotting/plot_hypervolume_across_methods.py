import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
from compute_ranks import compute_ranks


rcParams["text.usetex"] = False
rcParams["font.family"] = "sans"

df = pd.read_csv(f"all_results.csv")

checkpoint = "full"
search_space = "small"
epochs = 5
df = df.query(
    f"search_space == '{search_space}' & checkpoint == '{checkpoint}' & epoch == {epochs} "
)
method_info = {
    "morea": {"label": "MO-REA", "color": "C6"},
    "random_search": {"label": "RS", "color": "C0"},
    "local_search": {"label": "LS", "color": "C1"},
    "nsga2": {"label": "NSGA-2", "color": "C3"},
    "moasha": {"label": "MO-ASHA", "color": "C6"},
    "lsbo": {"label": "LS-BO", "color": "C2"},
    "rsbo": {"label": "RS-BO", "color": "C5"},
    "ehvi": {"label": "EHVI", "color": "black"},
}
ylims = {}
ylims["bert-base-cased"] = {
    "rte": (0.2, 0.4),
    "stsb": (0.3, 0.7),
    "cola": (0.3, 0.6),
    "mrpc": (0.2, 0.6),
    "swag": (0.3, 0.6),
    "sst2": (0.2, 0.4),
    "imdb": (0.4, 0.6),
    "qnli": (0.3, 0.6),
    "mnli": (0.0, 0.7),
    "qqp": (0.0, 1.7),
}
ylims["roberta-base"] = {
    "rte": (0.6, 0.8),
    "stsb": (0.75, 1.0),
    "cola": (0.8, 0.925),
    "mrpc": (0.8, 1.0),
    "swag": (0.75, 0.91),
    "sst2": (0.85, 1.0),
    "imdb": (0.7, 0.9),
    "qnli": (0.7, 0.9),
    "mnli": (0.0, 0.7),
    "qqp": (0.0, 1.7),
}
marker = ["o", "x", "s", "d", "p", "P", "^", "v", "<", ">"]
methods = []

for model, df_model in df.groupby("model"):
    n_runs = 1
    n_iters = 100
    n_methods = len(df["method"].unique())
    n_tasks = len(df["dataset"].unique())

    error = np.empty((n_methods, n_tasks, n_runs, n_iters))

    for di, (dataset, df_benchmark) in enumerate(df_model.groupby("dataset")):
        plt.figure(dpi=200)
        for mi, (method, df_method) in enumerate(df_benchmark.groupby("method")):
            if method not in method_info:
                continue

            methods.append(method)
            traj = []

            for checkpoint_seed, df_seeds in df_method.groupby("seed"):
                traj.append(list(df_seeds.sort_values(by="runtime")["hv"]))

            runtimes = df_method.runtime.unique()

            traj = 4 - np.array(traj)
            error[mi, di] = traj[:n_runs, :]
            print(dataset, model, method, traj.shape)
            mean_prediction = np.mean(traj, axis=0)
            variance_prediction = np.mean(
                (traj - mean_prediction) ** 2 + np.var(mean_prediction, axis=0), axis=0
            )
            plt.errorbar(
                runtimes,
                mean_prediction,
                yerr=variance_prediction,
                color=method_info[method]["color"],
                marker=marker[mi],
                fillstyle="none",
                label=method_info[method]["label"],
                linestyle="-",
                markersize=1,
                markeredgewidth=1.5,
            )

        plt.legend()
        plt.ylabel("regret hypervolume", fontsize=20)
        plt.xlabel("runtime (seconds)", fontsize=20)
        plt.title(f"{dataset.upper()}", fontsize=20)
        plt.grid(linewidth="1", alpha=0.4)
        # plt.ylim(ylims[model][dataset])
        plt.savefig(
            f"./figures/hypervolume_search_{dataset}_{model}_{checkpoint}.pdf",
            bbox_inches="tight",
        )
        plt.show()

    ranks = compute_ranks(error)
    for i in range(ranks.shape[0]):
        method = methods[i]
        plt.plot(
            ranks[i],
            marker="o",
            label=method_info[method]["label"],
            color=method_info[method]["color"],
            linestyle="--",
        )

    plot_label = False

    plt.grid(linewidth="1", alpha=0.4)
    plt.legend(loc=1)
    plt.title(model.replace("_", "-").upper())
    plt.xlabel("time steps", fontsize=20)
    plt.ylabel("average rank", fontsize=15)
    plt.show()
