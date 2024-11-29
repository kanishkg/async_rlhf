import queue
import time
from typing import List

from vllm import LLM, SamplingParams

from src.vllm_utils import vllm_single_gpu_patch

def vllm_generate(
        model_name_or_path: str,
        sampling_params: SamplingParams,
        vllm_device: str,
        vllm_dtype: str,
        vllm_gpu_memory_utilization: float,
        param_Q: queue.Queue,
        prompt_Q: queue.Queue,
        response_ids_Q: queue.Queue
    ):

    vllm_single_gpu_patch()
    llm = LLM(
        model=model_name_or_path,
        enforce_eager=True,
        enable_prefix_caching=True,
        tensor_parallel_size=1,
        device=vllm_device,
        dtype=vllm_dtype,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        max_num_seqs=64,
        swap_space=64,
        dtype=vllm_dtype,
        max_model_len=2048,
    )
    print(f"🔥🔥🔥 vllm loaded")

    llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model

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
                print(f"🔥🔥🔥 load weights took: {time.time() - vllm_start_time:.2f} seconds")

        # before populating the queue, make sure to sync processes so that the weights are loaded
        if not prompt_Q.empty():
            print(f"🔥🔥🔥 getting prompts")
            queries_list = prompt_Q.get()
            print(f"🔥🔥🔥 prompts are loaded {len(queries_list)}")
            print(f"🔥🔥🔥 generating responses")
            start = time.time()
            outputs = llm.generate(prompt_token_ids=queries_list, sampling_params=sampling_params, 
                                 use_tqdm=True)
            print(f"🔥🔥🔥 generation took {time.time() - start:.2f} seconds")
            response_ids_Q.put(outputs)