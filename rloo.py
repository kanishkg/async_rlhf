import multiprocessing
import torch
import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from accelerate import Accelerator
from trl import ModelConfig
from trl.trainer.rloo_trainer import RLOOConfig
from vllm import LLM

from src.online_bok_trainer import OnlineBoKTrainer
from src.rloo_trainer import MyRLOOTrainer as RLOOTrainer
from src.rloo_trainer_vllm import RLOOTrainer as VLLMRLOOTrainer
from src.utils import TRLParser, WandbLogModelConfig


@dataclass
class ScriptArguments:
    output_global_parent_dir: str = field(default=None)
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    # dataset_text_field: str = field(default=None, metadata={"help": "the text field of the dataset"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_test_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    # output_model_name: str = field(default="", metadata={"help": "model name to upload"})
    max_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    vllm: bool = field(default=False)
    bok: bool = field(default=False)
    reward_fn: str = field(default=None, metadata={"help": "The reward function to use"})
    wandb_run_id: Optional[str] = field(default=None)


def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        input_ids = tokenizer(
            element["query"],
            padding=False,
        )["input_ids"]
        return {"input_ids": input_ids, "lengths": [len(ids) for ids in input_ids]}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=multiprocessing.cpu_count(),
    )


if __name__ == "__main__":
    parser = TRLParser((ScriptArguments, RLOOConfig, ModelConfig))
    args, config, model_config = parser.parse_args_and_config()

    accelerate = Accelerator()
    if accelerate.is_main_process:
        print(f"🔥🔥🔥 vllm loading...{config.sft_model_path}")
        llm = LLM(
            model=config.sft_model_path,
            max_num_seqs=16,
            swap_space=64,
            dtype="bfloat16",
            max_model_len=2048,
            tensor_parallel_size=1,
            device="cuda:0",
        )
        print("🔥🔥🔥 vllm loaded")
    else:
        print("waiting for vllm to spin up...")
    accelerate.wait_for_everyone()
    torch.cuda.synchronize()

    if args.output_global_parent_dir is not None:
        run_id = os.path.basename(os.getcwd())
        config.output_dir = os.path.join(args.output_global_parent_dir, run_id, config.output_dir)

    if args.wandb_run_id == "slurm":
        run_id = os.environ["SLURM_JOB_ID"]
        config_name = os.path.basename(config.output_dir)
        # save to parent / slurm id / output_dir
        if args.output_global_parent_dir is not None:
            config.output_dir = os.path.join(args.output_global_parent_dir, run_id, config.output_dir)
        os.environ["WANDB_RUN_ID"] = run_id + "_" + config_name
    else:
        os.environ["WANDB_RUN_ID"] = args.wandb_run_id

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )

    reward_fn = None
    if args.reward_fn is None:
        reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    else:
        print("No reward model provided, setting to None")
        reward_model = None
        assert args.reward_fn is not None, "Reward function must be provided if no reward model is provided"
        if args.reward_fn == "countdown":
            from tasks.countdown import CountDown
            reward_fn = CountDown.verify_answer

    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path,
                                                    torch_dtype=torch.bfloat16,
                                                    trust_remote_code=True,
                                                    attn_implementation="flash_attention_2")
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path,
                                                  torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True,
                                                  attn_implementation="flash_attention_2")
    ################
    # Dataset
    ################
    raw_datasets = load_dataset("json", data_files={"train": f"./{args.dataset_name}_train.jsonl", "val":f"./{args.dataset_name}_val.jsonl", "test": f"./{args.dataset_name}_test.jsonl"})
    if config.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(1024))
        config.push_to_hub = False
        config.report_to = ""
        config.save_strategy = "no"
        config.num_sample_generations = 0

    train_dataset = raw_datasets[args.dataset_train_split]
    eval_dataset = raw_datasets[args.dataset_test_split]

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    # filtering
    train_dataset = train_dataset.filter(lambda x: x["lengths"] <= args.max_length)
    eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= args.max_length)
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    ################
    # Training
    ################

    if args.bok:
        TrainerCls = OnlineBoKTrainer
    else:
        if args.vllm:
            TrainerCls = VLLMRLOOTrainer
        else:
            TrainerCls = RLOOTrainer

    trainer = TrainerCls(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_fn=reward_fn,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbLogModelConfig(model_config)],
        llm=llm,
    )
    trainer.train()

    if not config.sanity_check:
        trainer.save_model(config.output_dir)
        if config.push_to_hub:
            trainer.push_to_hub()
        trainer.generate_completions()

        if trainer.accelerator.is_main_process:
            try:
                os.remove("output_dir")
            except OSError:
                pass

            os.symlink(config.output_dir, "output_dir")
