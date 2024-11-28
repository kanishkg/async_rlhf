import multiprocessing
import threading
import torch
import os
from dataclasses import dataclass, field
from typing import Optional
import requests

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from vllm import LLM
from src.vllm_utils import vllm_single_gpu_patch
import sglang as sgl
from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)
from accelerate import Accelerator


import asyncio

if __name__ == "__main__":
    llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda", dtype="bfloat16", max_total_tokens=2048, base_gpu_id=0)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]*16

    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 2048}

    outputs = llm.generate(prompts, sampling_params)
    # for prompt, output in zip(prompts, outputs):
    #     print("===============================")
    #     print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
    #     print("===============================")

