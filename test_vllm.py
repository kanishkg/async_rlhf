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

def vllm_generate(model_name_or_path: str, vllm_device: str, vllm_dtype: str, vllm_gpu_memory_utilization: float, param_Q: queue.Queue, prompt_Q: queue.Queue, response_ids_Q: queue.Queue):
    llm = LLM(
        model=model_name_or_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        device=vllm_device,
        dtype=vllm_dtype,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    print(f"🔥🔥🔥 vllm loaded")

    llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model
    sampling_params = SamplingParams(
        n=1,
        temperature=1,
        max_tokens=2048
    )

    i = 0
    while True:
        if not param_Q.empty():
            print(f"🔥🔥🔥 Waiting for weights to be loaded")
            model_named_parameters = param_Q.get()
            print(f"🔥🔥🔥 Weights are loaded")
            if i > 0:
                vllm_start_time = time.time()
                print("🔥🔥🔥 Loading weights using shared memory;" "we expect the generations to be completely different")

                llmp.load_weights(model_named_parameters)
                print(f"load weights took: {time.time() - vllm_start_time:.2f} seconds")

        # before populating the queue, make sure to sync processes so that the weights are loaded
        if not prompt_Q.empty():
            queries_list = []
            while not prompt_Q.empty():
                print(f"🔥🔥🔥 getting prompts")
                queries_list += prompt_Q.get()
            print(f"🔥🔥🔥 prompts are loaded {len(queries_list)}")
            outputs = llm.generate(queries_list, sampling_params=sampling_params, 
                                 use_tqdm=True)
            response_ids_Q.put(outputs)

def main():
    vllm_single_gpu_patch()
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    response_ids_Q = queue.Queue(maxsize=1)
    param_Q = queue.Queue(maxsize=1)
    prompt_Q = queue.Queue(maxsize=4)
    thread = threading.Thread(
                target=vllm_generate,
                args=(
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "cuda:0",
                    "bfloat16",
                    0.95,
                    param_Q,
                    prompt_Q,
                    response_ids_Q,
                ),
            )
    thread.start()
    prompts = [
        "Hello, my name is",
    ]
    
    print("🔥🔥🔥 Putting weights in memory")
    param_Q.put(model.named_parameters())
    print("🔥🔥🔥 Weights are in memory")
    prompt_Q.put(prompts)
    respones = response_ids_Q.get()
    import pdb; pdb.set_trace()
    # print(respones)

if __name__ == "__main__":
    main()