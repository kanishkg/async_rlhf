import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union, Callable
from multiprocessing import Manager
import threading
import requests
import queue
import gc

import numpy as np
import tqdm
import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import broadcast, gather_object, broadcast_object_list
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.rloo_config import RLOOConfig
from trl.trainer.utils import (
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    print_rich_table,
    truncate_response,
)
from vllm import SamplingParams, LLM

from src.utils import prepare_deepspeed
from src.vllm_utils import vllm_single_gpu_patch
from src.vllm_episode_maker import vllm_generate


INVALID_LOGPROB = 1.0

# from multiprocessing.managers import BaseManager

# Define the same Manager class
# class QueueManager(BaseManager):
#     pass

# QueueManager.register('get_response_ids_Q')
# QueueManager.register('get_prompt_Q')

# Combine operations to reduce memory overhead
@torch.jit.script
def fused_loss_computation(new_logprobs, ref_logprobs, advantages, kl_coef):
    kl = 0.5 * (new_logprobs - ref_logprobs).pow(2).sum(1)
    pg_loss = (-advantages * new_logprobs.sum(1)).mean()
    return pg_loss + kl_coef * kl.mean(), pg_loss, kl

class RLOOTrainer(Trainer):
    def __init__(
        self,
        config: RLOOConfig,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        # model_init: Optional[Callable[[torch.nn.Module], None]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        reward_fn: Optional[Callable] = None,
    ) -> None:
        vllm_single_gpu_patch()
        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.policy = policy

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.callbacks = callbacks
        self.reward_fn = reward_fn
        if reward_model is None:
            assert reward_fn is not None, "reward_fn must be provided if reward_model is None"

        #########
        # calculate various batch sizes
        #########
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        

        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(args.batch_size, args.num_mini_batches)
        args.local_mini_batch_size = exact_div(args.local_batch_size, args.num_mini_batches)
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_updates = args.total_episodes // args.batch_size
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_updates // args.num_sample_generations)

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.response_length,
            n=args.rloo_k,
        )        

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy, reward_model]:
            if module is not None:
                disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(num_training_steps=args.num_updates)

        #########
        ### trainer specifics
        #########
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        DEFAULT_CALLBACKS = [DefaultFlowCallback]
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        if self.callbacks is None:
            self.callbacks = default_callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.local_batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer),
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)
        if self.is_deepspeed_enabled:  # need to use for Trainer.save_model / push_to_hub
            self.deepspeed = self.model


        if self.is_deepspeed_enabled:

            if self.reward_model is not None:
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, config.fp16, config.bf16
                )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, config.fp16, config.bf16
            )
        else:
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)
            self.ref_policy = self.ref_policy.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        tokenizer = self.tokenizer
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())

        accelerator.print("===training policy===")
        global_step = 0
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        # Pre-allocate tensors
        # advantage_device = torch.zeros(args.per_device_train_batch_size, device=device)
        # logprob_shape = (args.per_device_train_batch_size, max_sequence_length)
        # new_logprobs = torch.zeros(logprob_shape, device=device)
        # ref_logprobs = torch.zeros(logprob_shape, device=device)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        self.state.max_steps = args.total_episodes
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()
        model.train()
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)


        # Connect to the manager
        # manager = QueueManager(address=('localhost', 50000), authkey=b'secret')
        # manager.connect()

        # Access the shared queues
        # response_ids_Q = manager.get_response_ids_Q()
        # prompt_Q = manager.get_prompt_Q()
        if accelerator.is_main_process:
            vllm_device = f"cuda:{accelerator.num_processes}"
            print(f"🔥🔥🔥 vllm device: {vllm_device}")

            response_ids_Q = queue.Queue(maxsize=1)
            param_Q = queue.Queue(maxsize=1)
            prompt_Q = queue.Queue(maxsize=1)

            thread = threading.Thread(
                target=vllm_generate,
                args=(
                    args.sft_model_path,
                    self.sampling_params,
                    vllm_device,
                    "bfloat16",
                    0.95,
                    param_Q,
                    prompt_Q,
                    response_ids_Q,
                ),
            )
            thread.start()

        accelerator.wait_for_everyone()



        for update in range(1, args.num_updates + 1):
            global_step += 1 * args.batch_size
            self.lr_scheduler.step()
            data = next(iter_dataloader)
            vllm_responses = torch.zeros(
                (args.batch_size * args.rloo_k, args.response_length),
                device=accelerator.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                repeated_queries = queries.repeat_interleave(args.rloo_k, dim=0)
                context_length = queries.shape[1]
                query_responses = []
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                # save model parameters

                g_queries_list = gather_object(queries.tolist())
                if self.accelerator.is_main_process:
                    # update sglang model
                    print("🔥🔥🔥 Updating weights")
                    start_time = time.time()
                    # param_Q.put(unwrapped_model.named_parameters())
                    model_named_parameters = accelerator._get_named_parameters(model)
                    param_Q.put(model_named_parameters)
                    g_queries_list = [
                        [inneritem for inneritem in item if inneritem != tokenizer.pad_token_id]
                        for item in g_queries_list
                    ] 
                    
                    print(f"🔥🔥🔥 Sending requests to vllm {len(g_queries_list)}")
                    prompt_Q.put(g_queries_list)
                    g_response_ids = response_ids_Q.get()

                    output_token_ids = [[list(output.token_ids) for output in response.outputs] for response in g_response_ids]
                    # flatten the list
                    output_token_ids = [item for sublist in output_token_ids for item in sublist]
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                    DUMMY_PAD_TOKEN = tokenizer.pad_token_id # we can't use tokenizer.pad_token_id because it's outside vocab and `torch.gather(all_logprob, 2, response.unsqueeze(-1))` will error out
                    g_padded_response_ids = [
                        list(response) + [DUMMY_PAD_TOKEN] * (args.response_length - len(response))
                        for response in output_token_ids 
                    ]
                    g_padded_response_ids = torch.tensor(g_padded_response_ids, device=device)
                    vllm_responses[:] = g_padded_response_ids


                accelerator.wait_for_everyone()
                broadcast(vllm_responses, 0)
                accelerator.wait_for_everyone()

                local_vllm_responses = vllm_responses[
                    accelerator.local_process_index * repeated_queries.shape[0] : (accelerator.local_process_index + 1)
                    * repeated_queries.shape[0]
                ]
                context_length = repeated_queries.shape[1]
                query_responses = torch.cat((repeated_queries, local_vllm_responses), 1)

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = repeated_queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]

                    # Response Processing 1. truncate response after the first occurrence of `truncate_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                    if reward_model is not None:
                        _, score, _ = get_reward(
                            reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                        )
                    else:
                        # use reward function
                        # decode the responses and queries
                        response_text = tokenizer.batch_decode(postprocessed_response, skip_special_tokens=True)
                        query_text = tokenizer.batch_decode(query, skip_special_tokens=True)
                        score = []
                        for q, r in zip(query_text, response_text):
                            s = self.reward_fn(q, r)
                            score.append(s)
                    del postprocessed_query_response
                        

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    sequence_lengths.append(sequence_length)
                    scores.append(torch.tensor(score))

                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                torch.cuda.empty_cache()

                # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_eos_token = torch.any(postprocessed_responses == tokenizer.eos_token_id, dim=-1)
                if args.non_eos_penalty:
                    scores = torch.where(contain_eos_token, scores, torch.full_like(scores, args.penalty_reward_value))
                accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                # logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                # ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                # 4. compute rewards
                # kl = logprobs - ref_logprobs
                # non_score_reward = (-args.kl_coef * kl).sum(1)
                # rlhf_reward = scores + non_score_reward

                # KG: I think the above is wrong; we should not add KL divergence to the reward
                scores = scores.to(device)
                rlhf_reward = scores

                # we generated `self.args.rloo_k` many responses per prompt
                # now we can implement the RLOO loss by subtracting the reward of
                # a response by the average rewards of other `rloo_k - 1` responses

                # KG: vectorized RLOO advantages implementation
                # print(f"shape of rlhf_reward: {rlhf_reward.shape}")
                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
                advantages = rlhf_reward - baseline
                advantages = advantages.flatten()
                # move to device
                advantages = advantages.to(device)
                gc.collect()
                torch.cuda.empty_cache()

            print(f"===training policy===")
            start_time = time.time()
            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0

                    for micro_batch_start in tqdm.tqdm(range(0, args.local_mini_batch_size, args.per_device_train_batch_size)):
                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]

                    
                        with torch.no_grad():
                            ref_output = forward(ref_policy, mb_query_responses, tokenizer.pad_token_id)
                            ref_logits = ref_output.logits[:, context_length - 1 : -1]
                            ref_logits /= args.temperature + 1e-7
                            ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
                            ref_logprobs = torch.gather(ref_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)

                        with accelerator.accumulate(model):
                            output = forward(model, mb_query_responses, tokenizer.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            # KG: compute approx kl
                            # kl = 0.5 * (new_logprobs - ref_logprobs)**2
                            # kl = kl.sum(1)


                            loss, pg_loss, kl = fused_loss_computation(new_logprobs, ref_logprobs, mb_advantage, args.kl_coef)
                            new_logprobs = new_logprobs.sum(1)

                            # KG: We should add kl directly to the loss
                            # pg_loss = -mb_advantage * new_logprobs
                            # pg_loss = pg_loss.mean() 
                            # loss = pg_loss + args.kl_coef * kl.mean()

                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = kl.mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_all_logprobs, new_logprobs,
                        pg_loss, loss, prob_dist, approxkl,
                        mb_advantage, mb_responses, mb_query_responses,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
                # accelerator.print(
                #     f"ppo_epoch_idx: {ppo_epoch_idx}",
                #     f"approxkl: {approxkl_stats[:ppo_epoch_idx + 1].mean().item():.4f}",
                #     f"pg_loss: {pg_loss_stats[:ppo_epoch_idx + 1].mean().item():.4f}",
                #     f"pg_clipfrac: {pg_clipfrac_stats[:ppo_epoch_idx + 1].mean().item():.4f}",
                #     f"ratio: {ratio_stats[:ppo_epoch_idx + 1].mean().item():.4f}",
                # )
            print(f"===training policy time = {time.time()-start_time}===")
            with torch.no_grad():
                rlhf_reward_mean = self.accelerator.gather(rlhf_reward).mean().item()
                accelerator.print(f"{rlhf_reward_mean=}")
                mean_kl = kl.mean()
                mean_entropy = entropy.mean()
                # mean_non_score_reward = non_score_reward.mean()
                eps = int(global_step / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                # metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                # metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
                # metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                # metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                # metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == tokenizer.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = global_step
                self.state.epoch = global_step / self.train_dataset_len  # used by self.log
                self.log(metrics)
            del kl, mean_kl, mean_entropy, scores, entropy
            torch.cuda.empty_cache()

            # KG: Skip eval loop for now
            # if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
            #     self.generate_completions(sampling=True)
            
            wandb.log({"completions": wandb.Table(dataframe=df)})
            self.state.global_step = global_step
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        accelerator = self.accelerator
        tokenizer = self.tokenizer
        device = accelerator.device
        generation_config = SamplingParams(
            temperature=(0.01 + 1e-7),
            top_p=1.0,
            max_tokens=args.response_length,
            include_stop_str_in_output=True,
        )

        table = defaultdict(list)
        g_responses = torch.zeros(
            (args.per_device_eval_batch_size * args.world_size, args.response_length),
            device=device,
            dtype=torch.long,
        )
        for batch in self.eval_dataloader:
            queries = batch["input_ids"]
            with torch.no_grad():
                context_length = queries.shape[1]
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    g_queries_list = gather_object(queries.tolist())

                    if accelerator.is_main_process:
                        print(
                            "🔥🔥🔥 Loading weights using shared memory;"
                            "we expect the generations to be completely different"
                        )
                        start_time = time.time()
                        self.llmp.load_weights(unwrapped_model.named_parameters())
                        print(f"Time to load weights: {time.time() - start_time:.2f} seconds")
                        g_queries_list = [
                            [inneritem for inneritem in item if inneritem != tokenizer.pad_token_id]
                            for item in g_queries_list
                        ]
                        outputs = self.llm.generate(prompt_token_ids=g_queries_list, sampling_params=generation_config)
                        padded_response_token_ids = []
                        for output in outputs:
                            token_ids = output.outputs[0].token_ids
                            padded_token_ids = token_ids + [tokenizer.pad_token_id] * (
                                args.response_length - len(token_ids)
                            )
                            padded_response_token_ids.append(padded_token_ids)

                        padded_response_token_ids = torch.tensor(padded_response_token_ids, device=device)

                        g_responses[:] = padded_response_token_ids

                    broadcast(g_responses, 0)
                    queries_responses = torch.cat(
                        (
                            queries,
                            g_responses[
                                accelerator.local_process_index
                                * queries.shape[0] : (accelerator.local_process_index + 1)
                                * queries.shape[0]
                            ],
                        ),
                        1,
                    )

                response = queries_responses[:, context_length:]
                postprocessed_response = response
                if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(args.stop_token_id, tokenizer.pad_token_id, response)
                table["query"].extend(gather_object(self.tokenizer.batch_decode(queries, skip_special_tokens=True)))
                table["model response"].extend(gather_object(self.tokenizer.batch_decode(postprocessed_response)))

                postprocessed_query_response = torch.cat((queries, postprocessed_response), 1)
                if self.reward_model is not None:
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, self.tokenizer.pad_token_id, context_length
                    )
                else:
                    # use reward function
                    # decode the responses and queries
                    response_text = tokenizer.batch_decode(postprocessed_response, skip_special_tokens=True)
                    query_response_text = tokenizer.batch_decode(queries, skip_special_tokens=True)
                    score = self.reward_fn(query_response_text, response_text)
                table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

            if sampling:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df.iloc[0 : 0 + 5])
        if "wandb" in args.report_to:
            import wandb

            if wandb.run is not None:
                wandb.log({"completions": wandb.Table(dataframe=df)})

if __name__ == "__main__":

    def test_rloo_reward():
        local_batch_size = 3
        # fmt: off
        rlhf_reward = torch.tensor([
            1, 2, 3, # first rlhf reward for three prompts
            2, 3, 4, # second rlhf reward for three prompts
            5, 6, 7, # third rlhf reward for three prompts
            8, 9, 10, # fourth rlhf reward for three prompts
        ]).float()
        # fmt: on

        advantages = torch.zeros_like(rlhf_reward)
        for i in range(0, len(advantages), local_batch_size):
            other_response_rlhf_rewards = []
            for j in range(0, len(advantages), local_batch_size):
                if i != j:
                    other_response_rlhf_rewards.append(rlhf_reward[j : j + local_batch_size])
            advantages[i : i + local_batch_size] = rlhf_reward[i : i + local_batch_size] - torch.stack(
                other_response_rlhf_rewards
            ).mean(0)
        assert (1 - (2 + 5 + 8) / 3 - advantages[0].item()) < 1e-6
        assert (6 - (3 + 2 + 9) / 3 - advantages[7].item()) < 1e-6
