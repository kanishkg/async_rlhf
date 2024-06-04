import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import ModelConfig, RewardTrainer

from src.utils import TRLParser


tqdm.pandas()


@dataclass
class RewardScriptArguments:
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_eval_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    sanity_check: bool = field(default=False, metadata={"help": "only train on 1000 samples"})
    max_length: Optional[int] = None


def get_peft_config(model_config: ModelConfig):
    if model_config.use_peft is False:
        return None

    target_modules = model_config.lora_target_modules if model_config.lora_target_modules is not None else "all-linear"

    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        task_type=model_config.lora_task_type,
        target_modules=target_modules,
        modules_to_save=model_config.lora_modules_to_save,
    )

    return peft_config


def tldr_preprocess_function(examples, max_length):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for query, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(query + chosen, max_length=max_length, truncation=True)
        tokenized_rejected = tokenizer(query + rejected, max_length=max_length, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


if __name__ == "__main__":
    parser = TRLParser((RewardScriptArguments, TrainingArguments, ModelConfig))
    script_args, reward_config, model_config = parser.parse_args_and_config()
    # reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer_name = (
        script_args.tokenizer_name if script_args.tokenizer_name is not None else model_config.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    )

    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.config.pad_token_id = tokenizer.pad_token_id

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(script_args.dataset_name)

    if script_args.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(100))

        reward_config.report_to = ""
        reward_config.push_to_hub = False

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    raw_datasets = raw_datasets.map(
        tldr_preprocess_function,
        batched=True,
        fn_kwargs={"max_length": script_args.max_length},
    )
    train_dataset = raw_datasets[script_args.dataset_train_split]
    eval_dataset = raw_datasets[script_args.dataset_eval_split]

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
        max_length=script_args.max_length,
    )
    trainer.train()
    trainer.save_model(reward_config.output_dir)
