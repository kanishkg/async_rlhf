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
else:
    print("Hello from subprocess")

accelerator.wait_for_everyone()
print("All processes are ready")