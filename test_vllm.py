import queue
import multiprocessing
import threading
import torch
import os
from dataclasses import dataclass, field
from typing import Optional
import requests
import time

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

def vllm_generate(model_name_or_path: str, vllm_device: str, vllm_dtype: str, vllm_gpu_memory_utilization: float, param_prompt_Q: queue.Queue):
    llm = LLM(
        model=model_name_or_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        device=vllm_device,
        dtype=vllm_dtype,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    print(f"🔥🔥🔥 vllm loaded")
    print(f"🔥🔥🔥 vllm loaded in {vllm_dtype}")
    llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model

    while True:
        i += 1
        model_named_parameters = param_prompt_Q.get()
        if i > 2:
            vllm_start_time = time.time()
            print("🔥🔥🔥 Loading weights using shared memory;" "we expect the generations to be completely different")
            llmp.load_weights(model_named_parameters)
            print(f"load weights took: {time.time() - vllm_start_time:.2f} seconds")
        time.sleep(5)  # Check every 10 seconds



def main():
    vllm_single_gpu_patch()
    param_prompt_Q = queue.Queue(maxsize=1)
    thread = threading.Thread(
                target=vllm_generate,
                args=(
                    "/scr/kanishkg/rloo_temp",
                    "cuda:0",
                    "bfloat16",
                    0.95,
                ),
            )
    thread.start()

    time.sleep(5)
    model = AutoModelForCausalLM.from_pretrained("/scr/kanishkg/rloo_temp")
    print("saving model")
    model.save_pretrained("/scr/kanishkg/rloo_temp")
    param_prompt_Q.put(model.named_parameters.items())
    print("model saved")

if __name__ == "__main__":
    main()