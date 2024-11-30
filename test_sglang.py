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

from vllm import LLM, SamplingParams
from src.vllm_utils import vllm_single_gpu_patch
import sglang as sgl
from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)
from accelerate import Accelerator
import time


import asyncio

if __name__ == "__main__":
    llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda", dtype="bfloat16", max_total_tokens=2048, base_gpu_id=0, max_num_requests=32, enable_torch_compile=True)
    # llm = LLM(
    #     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     tensor_parallel_size=1,
    #     enable_prefix_caching=True,
    #     enforce_eager=True,
    #     max_num_seqs=32,
    #     swap_space=64,
    #     dtype="bfloat16",
    #     max_model_len=2048,
    # )


    prompts = [
        "The president of the United States is",
    ]*16


    
    # sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 2048}

    start = time.time()
    print("starting gen")
    sampling_params = SamplingParams(
        n=1,
        temperature=1,
        max_tokens=2048
    )
    
    outputs = llm.generate(prompts, sampling_params)
    print(f"end: {time.time()-start}")
    # for prompt, output in zip(prompts, outputs):
    #     print("===============================")
    #     print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
    #     print("===============================")

