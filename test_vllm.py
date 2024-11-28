import multiprocessing
import threading
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
from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)
from accelerate import Accelerator

def vllm_generate(model_name_or_path: str, vllm_device: str, vllm_dtype: str, vllm_gpu_memory_utilization: float):
    llm = LLM(
        model=model_name_or_path,
        revision="main",
        tokenizer_revision="main",
        tensor_parallel_size=1,
        device=vllm_device,
        dtype=vllm_dtype,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    print(f"🔥🔥🔥 vllm loaded")
    print(f"🔥🔥🔥 vllm loaded in {vllm_dtype}")
    llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model

accelerator = Accelerator()
if accelerator.is_main_process:
    print("Hello from main process")
    server_process = execute_shell_command(f"python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port=30010 --device=cuda:3 --dtype=bfloat16 context_length=2048")
    wait_for_server("http://localhost:30010")
    print("Server is ready")
    # vllm_single_gpu_patch()
    # thread = threading.Thread(
    #             target=vllm_generate,
    #             args=(
    #                 "meta-llama/Llama-3.1-8B-Instruct",
    #                 "cuda:3",
    #                 "bfloat16",
    #                 0.95,
    #             ),
    #         )
    # thread.start()

else:
    print("Hello from subprocess")

accelerator.wait_for_everyone()
print("All processes are ready")