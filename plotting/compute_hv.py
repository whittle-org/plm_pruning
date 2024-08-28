import os
import pandas as pd
import numpy as np
import json

from pathlib import Path
from collections import defaultdict
from itertools import product

from sklearn.preprocessing import QuantileTransformer
from pygmo import hypervolume


class HyperVolume:
    def __init__(self, ref_point):
        self.ref_point = ref_point

    def __call__(self, points):
        return hypervolume(points).compute(self.ref_point)


base_path = Path.home() / Path(f"experiments/plm_pruning/results")
methods = [
    "random_search",
    "local_search",
    "ehvi",
    "morea"
]

checkpoints = ["linear_random", "sandwich", "full", "standard", "random", "kd"]
epochs = [5]
models = ["bert-base-cased", "roberta-base"]
labels = checkpoints

ref_point = [2, 2]
seeds = np.arange(10)
random_sub_nets = [1, 2]
runs = np.arange(1)
datasets = ["rte", "mrpc", "cola", "stsb", "sst2", "qnli", 'imdb', 'swag']
search_spaces = [
    "layer",
    "small",
    "medium",
    "large",
]

runtimes = {
    "rte": np.linspace(150, 400, 100),
    "mrpc": np.linspace(150, 500, 100),
    "cola": np.linspace(205, 1000, 100),
    "stsb": np.linspace(200, 1000, 100),
    "sst2": np.linspace(150, 1000, 100),
    "imdb": np.linspace(500, 10000, 100),
    "swag": np.linspace(500, 5000, 100),
    "qnli": np.linspace(200, 2000, 100),
}
data = defaultdict(list)
hv = HyperVolume(ref_point=ref_point)
oracle_performance = dict()

print("collect all data")

for (
    dataset,
    search_space,
    model,
    epoch,
    checkpoint,
    random_sub_net,
    method,
    seed,
    run,
) in product(
    datasets,
    search_spaces,
    models,
    epochs,
    checkpoints,
    random_sub_nets,
    methods,
    seeds,
    runs,
):
    path = (
        base_path
        / dataset
        / model
        / search_space
        / checkpoint
        / method
        / f"seed_{seed}"
    )

    if not os.path.exists(path):
        # print(f"{path} is missing")
        continue
    d = json.load(open(path / f"results_{dataset}.json"))
    N = len(d["params"])
    print(
        f"Results {dataset} {method} {checkpoint} {seed}: N={N}, T-min={np.min(d['runtime'])} T-Max={np.max(d['runtime'])}"
    )
    for row in range(N):
        data["runtime"].append(d["runtime"][row])
        data["dataset"].append(dataset)
        data["method"].append(method)
        data["checkpoint"].append(checkpoint)
        data["seed"].append(seed)
        data["model"].append(model)
        data["epoch"].append(epoch)
        data["search_space"].append(search_space)
        data[f"objective_0"].append(d["error"][row])
        data[f"objective_1"].append(d["params"][row])

data = pd.DataFrame(data)
print("start Quantile normalization")
for dataset, df in data.groupby("dataset"):
    print(f"normalize benchmark: {dataset}")

    for i in range(2):
        qt = QuantileTransformer()
        transformed_objective = qt.fit_transform(
            df.loc[:, f"objective_{i}"].to_numpy().reshape(-1, 1)
        )
        mask = data["dataset"] == dataset
        data.loc[mask, f"objective_{i}"] = transformed_objective

print("finished data normalization")
final_results = defaultdict(list)

for dataset, df in data.groupby("dataset"):
    print(f"process dataset: {dataset}")

    # compute hypervolume
    for keys, sub_df in df.groupby(
        [
            "model",
            "search_space",
            "method",
            "epoch",
            "checkpoint",
            "seed",
        ]
    ):
        model = keys[0]
        search_space = keys[1]
        method = keys[2]
        epoch = keys[3]
        checkpoint = keys[4]
        seed = keys[5]

        for runtime in runtimes[dataset]:
            split = sub_df[sub_df["runtime"] <= runtime]
            points = np.empty((len(split), 2))
            for i in range(2):
                points[:, i] = split[f"objective_{i}"]

            y = hv(points)
            # runtime = list(split['runtime'])[-1]
            final_results["method"].append(method)
            final_results["model"].append(model)
            final_results["checkpoint"].append(checkpoint)
            final_results["seed"].append(seed)
            final_results["epoch"].append(epoch)
            final_results["dataset"].append(dataset)
            final_results["runtime"].append(runtime)
            final_results["hv"].append(y)
            final_results["search_space"].append(search_space)

final_results = pd.DataFrame(final_results)
final_results.to_csv(f"all_results.csv")
