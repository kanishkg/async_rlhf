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

from vllm import LLM
from accelerate import Accelerator

accelerator = Accelerator()
if accelerator.is_main_process:
    print("Hello from main process")
    llm = LLM(
                model="meta-llama/Llama-3.1-8B-Instruct",
                max_num_seqs=16,
                swap_space=64,
                dtype="bfloat16",
                max_model_len=2048,
                tensor_parallel_size=1,
                device="cuda",
    )
else:
    print("Hello from subprocess")

accelerator.wait_for_everyone()
print("All processes are ready")