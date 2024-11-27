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
from src.vllm_utils import vllm_single_gpu_patch
import sglang as sgl
from accelerate import Accelerator

accelerator = Accelerator()
if accelerator.is_main_process:
    print("Hello from main process")
    vllm_single_gpu_patch()
    llm = LLM(
                model="meta-llama/Llama-3.1-8B-Instruct",
                max_num_seqs=16,
                swap_space=64,
                dtype="bfloat16",
                max_model_len=2048,
                tensor_parallel_size=1,
                device="cuda",
    )
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
    # sampling_params = {"temperature": 0.8, "top_p": 0.95}
    # llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # outputs = llm.generate(prompts, sampling_params)
    # for prompt, output in zip(prompts, outputs):
    #     print("===============================")
    #     print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
else:
    print("Hello from subprocess")

accelerator.wait_for_everyone()
print("All processes are ready")