import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path


search_space = "small"
dataset = 'rte'
checkpoint = "full"
seed = 0
model = 'bert-base-cased'

marker = ['x', 'o', 'D', 's', 'v',  '^', '<', '>', 'p', 'S']
base_path = Path.home() / Path(f"experiments/plm_pruning/results")
methods = ['local_search', 'random_search', 'morea']

plt.figure()

for i, method in enumerate(methods):
    path = (
        base_path
        / dataset
        / model
        / search_space
        / checkpoint
        / method
        / f"seed_{seed}"
    )

    data = json.load(open(path / f"results_{dataset}.json"))
    plt.scatter(data['params'], np.array(data['error']),
                color=f'C{i}', marker=marker[i], label=method)

plt.xlabel(r'parameter count')
plt.ylabel(r'downstream_error')
plt.grid(linewidth="1", alpha=0.9)
plt.legend()

plt.show()
