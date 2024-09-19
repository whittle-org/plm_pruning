import os
import time
import json

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import evaluate

from tqdm.auto import tqdm

from torch.optim import AdamW

from transformers import (
    AutoConfig,
    get_scheduler,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from whittle.sampling import RandomSampler
from whittle.training_strategies import (
    RandomLinearStrategy,
    RandomStrategy,
    SandwichStrategy,
    StandardStrategy,
)

from search_spaces import (
    FullSearchSpace,
    SmallSearchSpace,
    LayerSearchSpace,
    MediumSearchSpace,
)
from data_wrapper.task_data import GLUE_TASK_INFO
from hf_args import DataTrainingArguments, ModelArguments, parse_model_name
from data_wrapper import Glue, IMDB, SWAG
from bert import (
    SuperNetBertForMultipleChoiceSMALL,
    SuperNetBertForMultipleChoiceMEDIUM,
    SuperNetBertForMultipleChoiceLAYER,
    SuperNetBertForMultipleChoiceLARGE,
    SuperNetBertForSequenceClassificationSMALL,
    SuperNetBertForSequenceClassificationMEDIUM,
    SuperNetBertForSequenceClassificationLAYER,
    SuperNetBertForSequenceClassificationLARGE,
)
from roberta import (
    SuperNetRobertaForMultipleChoiceSMALL,
    SuperNetRobertaForMultipleChoiceMEDIUM,
    SuperNetRobertaForMultipleChoiceLAYER,
    SuperNetRobertaForMultipleChoiceLARGE,
    SuperNetRobertaForSequenceClassificationSMALL,
    SuperNetRobertaForSequenceClassificationMEDIUM,
    SuperNetRobertaForSequenceClassificationLAYER,
    SuperNetRobertaForSequenceClassificationLARGE,
)


def kd_loss(
    student_output, targets, teacher_output, temperature=1, is_regression=False
):
    teacher_logits = teacher_output.logits.detach()
    student_logits = student_output.logits
    if is_regression:
        return F.mse_loss(student_logits, teacher_logits)
    else:
        kd_loss = F.cross_entropy(
            student_logits / temperature, F.softmax(teacher_logits / temperature, dim=1)
        )
        predictive_loss = F.cross_entropy(student_logits, targets)
        return temperature ** 2 * kd_loss + predictive_loss


search_spaces = {
    "small": SmallSearchSpace,
    "medium": MediumSearchSpace,
    "layer": LayerSearchSpace,
    "large": FullSearchSpace,
}

model_types = dict()
model_types["bert"] = {
    "seq_classification": {
        "small": SuperNetBertForSequenceClassificationSMALL,
        "medium": SuperNetBertForSequenceClassificationMEDIUM,
        "layer": SuperNetBertForSequenceClassificationLAYER,
        "large": SuperNetBertForSequenceClassificationLARGE,
    },
    "multiple_choice": {
        "small": SuperNetBertForMultipleChoiceSMALL,
        "medium": SuperNetBertForMultipleChoiceMEDIUM,
        "layer": SuperNetBertForMultipleChoiceLAYER,
        "large": SuperNetBertForMultipleChoiceLARGE,
    },
}
model_types["roberta"] = {
    "seq_classification": {
        "small": SuperNetRobertaForSequenceClassificationSMALL,
        "medium": SuperNetRobertaForSequenceClassificationMEDIUM,
        "layer": SuperNetRobertaForSequenceClassificationLAYER,
        "large": SuperNetRobertaForSequenceClassificationLARGE,
    },
    "multiple_choice": {
        "small": SuperNetRobertaForMultipleChoiceSMALL,
        "medium": SuperNetRobertaForMultipleChoiceMEDIUM,
        "layer": SuperNetRobertaForMultipleChoiceLAYER,
        "large": SuperNetRobertaForMultipleChoiceLARGE,
    },
}


@dataclass
class NASArguments:
    search_space: str = field(metadata={"help": ""}, default="small")
    sampling_strategy: str = field(metadata={"help": ""}, default=None)
    num_random_sub_nets: int = field(metadata={"help": ""}, default=1)
    temperature: float = field(metadata={"help": ""}, default=1)


def main():
    start_time = time.time()
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, NASArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        nas_args,
    ) = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    if int(training_args.seed) == -1:
        training_args.seed = np.random.randint(2 ** 32 - 1)

    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)

    model_type = parse_model_name(model_args)

    # Load data
    if data_args.task_name in GLUE_TASK_INFO:
        data = Glue(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = evaluate.load("glue", data_args.task_name)
        metric_name = GLUE_TASK_INFO[data_args.task_name]["metric"]
    elif data_args.task_name == "imdb":
        data = IMDB(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = evaluate.load("accuracy")
        metric_name = "accuracy"
    elif data_args.task_name == "swag":
        data = SWAG(
            training_args=training_args, model_args=model_args, data_args=data_args
        )
        metric = evaluate.load("accuracy")
        metric_name = "accuracy"

    train_dataloader, eval_dataloader, test_dataloader = data.get_data_loaders()
    num_labels = data.num_labels

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_type,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_type.startswith("bert"):
        model_family = "bert"
    elif model_type.startswith("roberta"):
        model_family = "roberta"
    else:
        print(
            f"Model type {model_type} are not supported. "
            f"We only support models of the BERT or RoBERTa family."
        )
        raise NotImplementedError

    if data_args.task_name in ["swag"]:
        model_cls = model_types[model_family]["multiple_choice"][nas_args.search_space]
    else:
        model_cls = model_types[model_family]["seq_classification"][
            nas_args.search_space
        ]

    search_space = search_spaces[nas_args.search_space](config, seed=training_args.seed)

    model = model_cls.from_pretrained(
        model_type,
        from_tf=bool(".ckpt" in model_type),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    num_training_steps = int(training_args.num_train_epochs * len(train_dataloader))
    warmup_steps = int(training_args.warmup_ratio * num_training_steps)

    if training_args.lr_scheduler_type == "linear":
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif training_args.lr_scheduler_type == "cosine_with_restarts":
        lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=training_args.num_train_epochs,
        )

    progress_bar = tqdm(range(num_training_steps))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    step = 0
    print(f"Use {nas_args.sampling_strategy} to update super-network training")

    is_regression = True if data_args.task_name == "stsb" else False

    def loss_function(predictions, labels):
        return predictions.loss

    sampler = RandomSampler(search_space.config_space, seed=training_args.seed)
    training_strategies = {
        "standard": StandardStrategy(
            sampler=sampler,
            loss_function=loss_function,
        ),
        "sandwich": SandwichStrategy(
            sampler=sampler,
            loss_function=loss_function,
        ),
        "random": RandomStrategy(
            sampler=sampler,
            loss_function=loss_function,
        ),
        "random_linear": RandomLinearStrategy(
            sampler=sampler,
            loss_function=loss_function,
            total_number_of_steps=num_training_steps,
        ),
        "kd": RandomStrategy(
            sampler=sampler,
            kd_loss=kd_loss,
            loss_function=loss_function,
        ),
        "full": SandwichStrategy(
            sampler=sampler,
            kd_loss=kd_loss,
            loss_function=loss_function,
        ),
    }

    update_op = training_strategies[nas_args.sampling_strategy]

    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            loss = update_op(model, batch, batch["labels"])

            step += 1

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            train_loss += loss

        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(batch)

            logits = outputs.logits
            predictions = (
                torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)
            )

            metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = metric.compute()

        runtime = time.time() - start_time
        print(
            f"epoch {epoch}: training loss = {train_loss / len(train_dataloader)}, "
            f"evaluation metrics = {eval_metric}, "
            f"runtime = {runtime}"
        )
        print(f"epoch={epoch};")
        print(f"training loss={train_loss / len(train_dataloader)};")
        print(f"evaluation metrics={eval_metric[metric_name]};")
        print(f"runtime={runtime};")

        if training_args.save_strategy == "epoch":
            os.makedirs(training_args.output_dir, exist_ok=True)
            print(f"Store checkpoint in: {training_args.output_dir}")
            model.save_pretrained(training_args.output_dir)

    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(batch)

        logits = outputs.logits
        predictions = (
            torch.squeeze(logits) if is_regression else torch.argmax(logits, dim=-1)
        )

        metric.add_batch(predictions=predictions, references=batch["labels"])

    test_metric = metric.compute()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {}
    results["dataset"] = data_args.task_name
    results["params"] = n_params
    results["search_space"] = nas_args.search_space
    results["runtime"] = time.time() - start_time
    results[metric_name] = float(eval_metric[metric_name])
    results["test_" + metric_name] = float(test_metric[metric_name])
    fname = os.path.join(
        training_args.output_dir, f"results_{data_args.task_name}.json"
    )
    json.dump(results, open(fname, "w"))


if __name__ == "__main__":
    main()
