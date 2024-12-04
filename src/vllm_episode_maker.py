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
        param_prompt_Q: object,
        response_ids_Q: object,
    ):

    print("+++ patching vllm")
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
        max_model_len=2048,
    )
    print(f"🔥🔥🔥 vllm loaded")
    print(f"🔥🔥🔥 samppling params {sampling_params}")

    llmp = llm.llm_engine.model_executor.driver_worker.model_runner.model
    i=0
    while True:
        i += 1
        model_named_parameters, g_queries_list = param_prompt_Q.get()
        print("got queries==================")
        if model_named_parameters is None and g_queries_list is None:
            print("model params and queries are None, exiting")
            break

        vllm_start_time = time.time()
        if i > 0:
            # print("🔥🔥🔥 Loading weights using shared memory;" "we expect the generations to be completely different")
            llmp.load_weights(model_named_parameters)
            print(f"load weights took: {time.time() - vllm_start_time:.2f} seconds")

        outputs = llm.generate(prompt_token_ids=g_queries_list, sampling_params=sampling_params, use_tqdm=False)
        print(
            f"🏃🏃🏃 load and gen of {len(g_queries_list)} prompts took: {time.time() - vllm_start_time:.2f} seconds"
        )
        # response_token_ids = []
        # for output in outputs:
        #     response_token_ids.append(output.outputs[0].token_ids)

        response_ids_Q.put(outputs)

    # i = 0
    # while True:
    #     if not param_Q.empty():
    #         print(f"🔥🔥🔥 Waiting for weights to be loaded")
    #         model_named_parameters = param_Q.get()
    #         print(f"🔥🔥🔥 Weights are loaded")
    #         if i > 0:
    #             vllm_start_time = time.time()
    #             print("🔥🔥🔥 Loading weights using shared memory;" "we expect the generations to be completely different")

    #             llmp.load_weights(model_named_parameters)
    #             print(f"🔥🔥🔥 load weights took: {time.time() - vllm_start_time:.2f} seconds")

    #     # before populating the queue, make sure to sync processes so that the weights are loaded
    #     if not prompt_Q.empty():
    #         print(f"🔥🔥🔥 getting prompts")
    #         queries_list = prompt_Q.get()
    #         print(f"🔥🔥🔥 prompts are loaded {len(queries_list)}")
    #         print(f"🔥🔥🔥 generating responses")
    #         start = time.time()
    #         outputs = llm.generate(prompt_token_ids=queries_list, sampling_params=sampling_params, 
    #                              use_tqdm=True)
    #         print(f"🔥🔥🔥 generation took {time.time() - start:.2f} seconds")
    #         response_ids_Q.put(outputs)