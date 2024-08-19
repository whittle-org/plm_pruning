# Structural Pruning of Pre-trained Language Models via Neural Architecture Search

This package provides code for multi-objective Neural Architecture Search (NAS)
to prune pre-trained language models by searching for sub-networks that 
minimize both validation error and parameter count. For more details see the following paper:

```
@article{klein-tmlr24,
  title={Structural Pruning of Pre-trained Language Models via Neural Architecture Search},
  author={Aaron Klein, Jacek Golebiowski, Xingchen Ma, Valerio Perrone, Cedric Archambeau},
  journal={Submitted to Transactions on Machine Learning Research},
  year={2024},
  url={https://openreview.net/forum?id=XiK8tHDQNX},
  note={Under review}
}
```
   
   
   https://openreview.net/forum?id=XiK8tHDQNX&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DTMLR%2FAuthors%23your-submissions)

We distinguish between standard NAS, which fine-tunes each sub-network in isolation and weight-sharing 
based NAS. Our weight-sharing based NAS approach consists of two stages:
1. We fine-tune the pre-trained network (dubbed super-network) via weight-sharing based NAS strategies. 
   In a nutshell, in each update steps, we only update parts of the network to train different sub-networks.
2. In the second stage, we run multi-objective search to find the Pareto set of sub-networks 
   of the super-network. To evaluate each sub-network we use the shared weights of the super-networks, without
   any further training. This is relatively cheap compared to standard NAS, since we only do a single pass
   over the validation data without computing gradients. 

## Install

To get started, first install [whittle](https://github.com/whittle-org/whittle) following 
the installation instructions [here](https://github.com/whittle-org/whittle?tab=readme-ov-file#setup)

Afterwards, we install the dependencies via:

```bash
pip install -r requirements.txt
```

## Benchmarking Details

At the moment we support models from the BERT and RoBERTa family and the following datasets: 
```[rte', 'mrpc', 'cola', 'stsb', 'sst2', 'qnli', 'imdb', 'swag', 'mnli', 'qqp']```

Also you can use the following multi-objective methods from Syne Tune both for standard NAS and weight-sharing based NAS: 
```['random_search', 'morea', 'local_search', 'nsga2', 'moasha', 'ehvi']```

## Standard NAS

To run standard NAS, use the following script. This will run NAS using random search for 3600 seconds on the RTE dataset.

```python src/run_nas.py --output_dir=./output_standard_nas --model_name bert-base-cased --dataset rte --runtime 3600 --method random_search --num_train_epochs 5 --seed 0 --dataset_seed 0```

## Weight-sharing NAS

As described above, weight-sharing NAS runs in two phases. We first fine-tune the super-network and store the checkpoint on disk.
Afterwards, we can run our multi-objective search via Syne-Tune. 

### Super-Network Training

To run the training of the super-network, execute the following script:

```python src/train_supernet.py --learning_rate 2e-05 --model_name_or_path bert-base-cased --num_train_epochs 5 --output_dir ./supernet_model_checkpoint --save_strategy "epoch" --per_device_eval_batch_size 8 --per_device_train_batch_size 4 --sampling_strategy one_shot --save_strategy epoch --search_space small --seed 0 --task_name rte --num_random_sub_nets 2 --temperature 10 ``` 

This runs the super-network training ('one_shot') on the RTE dataset for 5 epochs. Checkpoints are saved in the
`output_dir`, such that we can load it later for the multi-objective search. 
Most hyperparameters follow the HuggingFace training arguments. At this point we
support the following super-network training strategies: ```['standard', 'random', 'linear_random', 'one_shot', 'sandwich', 'kd']```.
See the paper for a detailed description.


### Multi-Objective Search

Next, we use the model checkpoint from the previous step to perform the multi-objective search:

```python src/run_offline_search.py --model_name_or_path bert-base-cased --num_samples 100 --output_dir ./results_nas  --checkpoint_dir_model ./supernet_model_checkpoint --search_space small --search_strategy random_search --seed 0 --task_name rte``` 

Make sure that `checkpoint_dir_model` points to the directory with the model checkpoint from the previous step. 
Results will be saved as a json file in `output_dir`.