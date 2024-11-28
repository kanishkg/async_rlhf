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

from vllm import LLM, SamplingParams
from src.vllm_utils import vllm_single_gpu_patch
import sglang as sgl
from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)

def vllm_generate(model_name_or_path: str, vllm_device: str, vllm_dtype: str, vllm_gpu_memory_utilization: float, param_prompt_Q: queue.Queue, response_ids_Q: queue.Queue):
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
    sampling_params = SamplingParams(
        n=1,
        temperature=1,
        max_tokens=2048
    )

    i = 0
    while True:
        i += 1
        print(f"🔥🔥🔥 Waiting for weights to be loaded")
        model_named_parameters, queries_list = param_prompt_Q.get()
        print(f"🔥🔥🔥 Weights are loaded")
        if i > 0:
            vllm_start_time = time.time()
            print("🔥🔥🔥 Loading weights using shared memory;" "we expect the generations to be completely different")

            llmp.load_weights(model_named_parameters)
            print(f"load weights took: {time.time() - vllm_start_time:.2f} seconds")
        outputs = llm.generate(queries_list, sampling_params=sampling_params, use_tqdm=True)
        response_ids_Q.put(outputs)

def main():
    vllm_single_gpu_patch()
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    response_ids_Q = queue.Queue(maxsize=1)
    param_prompt_Q = queue.Queue(maxsize=1)
    thread = threading.Thread(
                target=vllm_generate,
                args=(
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "cuda:0",
                    "bfloat16",
                    0.95,
                    param_prompt_Q,
                    response_ids_Q,
                ),
            )
    thread.start()
    prompts = [
        "Hello, my name is",
    ]*16
    
    print("🔥🔥🔥 Putting weights in memory")
    param_prompt_Q.put((model.named_parameters(), prompts))
    print("🔥🔥🔥 Weights are in memory")
    respones = response_ids_Q.get()
    print(respones)

if __name__ == "__main__":
    main()