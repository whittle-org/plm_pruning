from pathlib import Path


def create_checkpoint_dir(params):
    path = (
        Path("plm_pruning")
        / "checkpoints"
        / params["tasks"]
        / params["model_types"]
        / params["search_space"]
        / params["sampling_strategy"]
        / f"seed_{params['seed']}"
    )
    return str(path)


def create_output_dir(params):
    path = (
        Path("plm_pruning")
        / "results"
        / params["tasks"]
        / params["model_types"]
        / params["search_space"]
        / params["sampling_strategy"]
        / params["search_methods"]
        / f"seed_{params['seed']}"
    )
    return str(path)


experiment_config = {
    "tasks": ["rte", 'mrpc', 'cola', 'stsb', 'swag', 'imdb', 'sst2', 'qnli'],
    "model_types": ["bert-base-cased", 'roberta-base'],
    "sampling_strategy": ["standard",  'random', 'random_linear', 'sandwich', 'full'],
    "search_methods": ["random_search", 'local_search', 'ehvi', 'morea'],
    "search_space": ["small", 'medium', 'large', 'layer'],
    "seed": [0, 1, 2, 3, 4],
}

search_hyperparameters = {"num_samples": 100}

training_hyperparameters = {
    "temperature": 10,
    "random_sub_nets": 2,
    "learning_rate": 2e-05,
    "num_train_epochs": 5,
    "eval_batch_size": 8,
    "train_batch_size": 4,
}
