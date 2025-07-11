# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """
# Metrics related to the PPO trainer.
# """

# from collections import defaultdict
# from functools import partial
# from typing import Any, Callable, Dict, List

# import numpy as np
# import torch

# from verl import DataProto
# from verl.utils.import_utils import deprecated


# @deprecated("verl.utils.metric.reduce_metrics")
# def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
#     """
#     Reduces a dictionary of metric lists by computing the mean of each list.

#     Args:
#         metrics: A dictionary mapping metric names to lists of metric values.

#     Returns:
#         A dictionary with the same keys but with each list replaced by its mean value.

#     Example:
#         >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
#         >>> reduce_metrics(metrics)
#         {"loss": 2.0, "accuracy": 0.8}
#     """
#     from verl.utils.metric import reduce_metrics

#     return reduce_metrics(metrics)


# def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
#     """
#     Computes information about prompts and responses from a batch.
    
#     This is an internal helper function that extracts masks and lengths for prompts and responses.
    
#     Args:
#         batch: A DataProto object containing batch data with responses and attention masks.
        
#     Returns:
#         A dictionary containing:
#             - response_mask: Attention mask for the response tokens
#             - prompt_length: Tensor of prompt lengths for each item in the batch
#             - response_length: Tensor of response lengths for each item in the batch
#     """
#     response_length = batch.batch["responses"].shape[-1]

#     prompt_mask = batch.batch["attention_mask"][:, :-response_length]
#     response_mask = batch.batch["attention_mask"][:, -response_length:]

#     prompt_length = prompt_mask.sum(-1).float()
#     response_length = response_mask.sum(-1).float()  # (batch_size,)

#     return dict(
#         response_mask=response_mask,
#         prompt_length=prompt_length,
#         response_length=response_length,
#     )


# def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
#     """
#     Computes various metrics from a batch of data for PPO training.

#     This function calculates metrics related to scores, rewards, advantages, returns, values,
#     and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
#     for each metric category.

#     Args:
#         batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
#         use_critic: Whether to include critic-specific metrics. Defaults to True.

#     Returns:
#         A dictionary of metrics including:
#             - critic/score/mean, max, min: Statistics about sequence scores
#             - critic/rewards/mean, max, min: Statistics about sequence rewards
#             - critic/advantages/mean, max, min: Statistics about advantages
#             - critic/returns/mean, max, min: Statistics about returns
#             - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
#             - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
#             - response_length/mean, max, min, clip_ratio: Statistics about response lengths
#             - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
#     """
#     sequence_score = batch.batch["token_level_scores"].sum(-1)
#     sequence_reward = batch.batch["token_level_rewards"].sum(-1)

#     advantages = batch.batch["advantages"]
#     returns = batch.batch["returns"]

#     max_response_length = batch.batch["responses"].shape[-1]

#     prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
#     response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

#     max_prompt_length = prompt_mask.size(-1)

#     response_info = _compute_response_info(batch)
#     prompt_length = response_info["prompt_length"]
#     response_length = response_info["response_length"]

#     valid_adv = torch.masked_select(advantages, response_mask)
#     valid_returns = torch.masked_select(returns, response_mask)

#     if use_critic:
#         values = batch.batch["values"]
#         valid_values = torch.masked_select(values, response_mask)
#         return_diff_var = torch.var(valid_returns - valid_values)
#         return_var = torch.var(valid_returns)

#     metrics = {
#         # score
#         "critic/score/mean": torch.mean(sequence_score).detach().item(),
#         "critic/score/max": torch.max(sequence_score).detach().item(),
#         "critic/score/min": torch.min(sequence_score).detach().item(),
#         # reward
#         "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
#         "critic/rewards/max": torch.max(sequence_reward).detach().item(),
#         "critic/rewards/min": torch.min(sequence_reward).detach().item(),
#         # adv
#         "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
#         "critic/advantages/max": torch.max(valid_adv).detach().item(),
#         "critic/advantages/min": torch.min(valid_adv).detach().item(),
#         # returns
#         "critic/returns/mean": torch.mean(valid_returns).detach().item(),
#         "critic/returns/max": torch.max(valid_returns).detach().item(),
#         "critic/returns/min": torch.min(valid_returns).detach().item(),
#         **(
#             {
#                 # values
#                 "critic/values/mean": torch.mean(valid_values).detach().item(),
#                 "critic/values/max": torch.max(valid_values).detach().item(),
#                 "critic/values/min": torch.min(valid_values).detach().item(),
#                 # vf explained var
#                 "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
#             }
#             if use_critic
#             else {}
#         ),
#         # response length
#         "response_length/mean": torch.mean(response_length).detach().item(),
#         "response_length/max": torch.max(response_length).detach().item(),
#         "response_length/min": torch.min(response_length).detach().item(),
#         "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
#         # prompt length
#         "prompt_length/mean": torch.mean(prompt_length).detach().item(),
#         "prompt_length/max": torch.max(prompt_length).detach().item(),
#         "prompt_length/min": torch.min(prompt_length).detach().item(),
#         "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
#     }
#     return metrics


# def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
#     """
#     Computes timing metrics for different processing stages in PPO training.
    
#     This function calculates both raw timing metrics (in seconds) and per-token timing metrics 
#     (in milliseconds) for various processing stages like generation, reference computation, 
#     value computation, advantage computation, and model updates.

#     Args:
#         batch: A DataProto object containing batch data with responses and attention masks.
#         timing_raw: A dictionary mapping stage names to their execution times in seconds.

#     Returns:
#         A dictionary containing:
#             - timing_s/{name}: Raw timing in seconds for each stage
#             - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

#     Note:
#         Different stages use different token counts for normalization:
#         - "gen" uses only response tokens
#         - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
#           (prompt + response)
#     """
#     response_info = _compute_response_info(batch)
#     num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
#     num_response_tokens = torch.sum(response_info["response_length"]).item()
#     num_overall_tokens = num_prompt_tokens + num_response_tokens

#     num_tokens_of_section = {
#         "gen": num_response_tokens,
#         **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
#     }

#     return {
#         **{f"timing_s/{name}": value for name, value in timing_raw.items()},
#         **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
#     }


# def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
#     """
#     Computes throughput metrics for PPO training.
    
#     This function calculates performance metrics related to token processing speed,
#     including the total number of tokens processed, time per step, and throughput
#     (tokens per second per GPU).
    
#     Args:
#         batch: A DataProto object containing batch data with meta information about token counts.
#         timing_raw: A dictionary mapping stage names to their execution times in seconds.
#                    Must contain a "step" key with the total step time.
#         n_gpus: Number of GPUs used for training.
        
#     Returns:
#         A dictionary containing:
#             - perf/total_num_tokens: Total number of tokens processed in the batch
#             - perf/time_per_step: Time taken for the step in seconds
#             - perf/throughput: Tokens processed per second per GPU
            
#     Note:
#         The throughput is calculated as total_tokens / (time * n_gpus) to normalize
#         across different GPU counts.
#     """
#     total_num_tokens = sum(batch.meta_info["global_token_num"])
#     time = timing_raw["step"]
#     # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
#     # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
#     # f'Theoretical TFLOPs/s/GPU​': promised_flops,
#     return {
#         "perf/total_num_tokens": total_num_tokens,
#         "perf/time_per_step": time,
#         "perf/throughput": total_num_tokens / (time * n_gpus),
#     }


# def bootstrap_metric(
#     data: list[Any],
#     subset_size: int,
#     reduce_fns: list[Callable[[np.ndarray], float]],
#     n_bootstrap: int = 1000,
#     seed: int = 42,
# ) -> list[tuple[float, float]]:
#     """
#     Performs bootstrap resampling to estimate statistics of metrics.

#     This function uses bootstrap resampling to estimate the mean and standard deviation
#     of metrics computed by the provided reduction functions on random subsets of the data.

#     Args:
#         data: List of data points to bootstrap from.
#         subset_size: Size of each bootstrap sample.
#         reduce_fns: List of functions that compute a metric from a subset of data.
#         n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
#         seed: Random seed for reproducibility. Defaults to 42.

#     Returns:
#         A list of tuples, where each tuple contains (mean, std) for a metric
#         corresponding to each reduction function in reduce_fns.

#     Example:
#         >>> data = [1, 2, 3, 4, 5]
#         >>> reduce_fns = [np.mean, np.max]
#         >>> bootstrap_metric(data, 3, reduce_fns)
#         [(3.0, 0.5), (4.5, 0.3)]  # Example values
#     """
#     np.random.seed(seed)

#     bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
#     for _ in range(n_bootstrap):
#         bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
#         bootstrap_data = [data[i] for i in bootstrap_idxs]
#         for i, reduce_fn in enumerate(reduce_fns):
#             bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
#     return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


# def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
#     """
#     Calculate a value based on majority voting.

#     This function identifies the most common value for a specified vote key
#     in the data, then returns the corresponding value for that majority vote.

#     Args:
#         data: List of dictionaries, where each dictionary contains both vote_key and val_key.
#         vote_key: The key in each dictionary used for voting/counting.
#         val_key: The key in each dictionary whose value will be returned for the majority vote.

#     Returns:
#         The value associated with the most common vote.

#     Example:
#         >>> data = [
#         ...     {"pred": "A", "val": 0.9},
#         ...     {"pred": "B", "val": 0.8},
#         ...     {"pred": "A", "val": 0.7}
#         ... ]
#         >>> calc_maj_val(data, vote_key="pred", val_key="val")
#         0.9  # Returns the first "val" for the majority vote "A"
#     """
#     vote2vals = defaultdict(list)
#     for d in data:
#         vote2vals[d[vote_key]].append(d[val_key])

#     vote2cnt = {k: len(v) for k, v in vote2vals.items()}
#     maj_vote = max(vote2cnt, key=vote2cnt.get)

#     maj_val = vote2vals[maj_vote][0]

#     return maj_val


# def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
#     """
#     Process validation metrics into a structured format with statistical analysis.
    
#     This function organizes validation metrics by data source and prompt, then computes
#     various statistical measures including means, standard deviations, best/worst values,
#     and majority voting results. It also performs bootstrap sampling to estimate statistics
#     for different sample sizes.
    
#     Args:
#         data_sources: List of data source identifiers for each sample.
#         sample_inputs: List of input prompts corresponding to each sample.
#         infos_dict: Dictionary mapping variable names to lists of values for each sample.
#         seed: Random seed for bootstrap sampling. Defaults to 42.

#     Returns:
#         A nested dictionary with the structure:
#         {
#             data_source: {
#                 variable_name: {
#                     metric_name: value
#                 }
#             }
#         }
        
#         Where metric_name includes:
#         - "mean@N": Mean value across N samples
#         - "std@N": Standard deviation across N samples
#         - "best@N/mean": Mean of the best values in bootstrap samples of size N
#         - "best@N/std": Standard deviation of the best values in bootstrap samples
#         - "worst@N/mean": Mean of the worst values in bootstrap samples
#         - "worst@N/std": Standard deviation of the worst values in bootstrap samples
#         - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
#         - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)
        
#     Example:
#         >>> data_sources = ["source1", "source1", "source2"]
#         >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
#         >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
#         >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
#         >>> # result will contain statistics for each data source and variable
#     """
#     # Group metrics by data source, prompt and variable
#     data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#     for sample_idx, data_source in enumerate(data_sources):
#         prompt = sample_inputs[sample_idx]
#         var2vals = data_src2prompt2var2vals[data_source][prompt]
#         for var_name, var_vals in infos_dict.items():
#             var2vals[var_name].append(var_vals[sample_idx])

#     # Calculate metrics for each group
#     data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
#     for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
#         for prompt, var2vals in prompt2var2vals.items():
#             for var_name, var_vals in var2vals.items():
#                 if isinstance(var_vals[0], str):
#                     continue

#                 metric = {}
#                 n_resps = len(var_vals)
#                 metric[f"mean@{n_resps}"] = np.mean(var_vals)

#                 if n_resps > 1:
#                     metric[f"std@{n_resps}"] = np.std(var_vals)

#                     ns = []
#                     n = 2
#                     while n < n_resps:
#                         ns.append(n)
#                         n *= 2
#                     ns.append(n_resps)

#                     for n in ns:
#                         [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
#                         metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
#                         metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
#                         if var2vals.get("pred", None) is not None:
#                             vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
#                             [(maj_n_mean, maj_n_std)] = bootstrap_metric(
#                                 data=vote_data,
#                                 subset_size=n,
#                                 reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
#                                 seed=seed,
#                             )
#                             metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

#                 data_src2prompt2var2metric[data_source][prompt][var_name] = metric

#     # Aggregate metrics across prompts
#     data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#     for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
#         for prompt, var2metric in prompt2var2metric.items():
#             for var_name, metric in var2metric.items():
#                 for metric_name, metric_val in metric.items():
#                     data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

#     data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
#     for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
#         for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
#             for metric_name, prompt_vals in metric2prompt_vals.items():
#                 data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

#     return data_src2var2metric2val
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# """
# Metrics related to the PPO trainer.
# """

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List
import os

import numpy as np
import torch
import pandas as pd

from verl import DataProto
from verl.utils.import_utils import deprecated

# Add wandb import with error handling since it's already configured in your setup
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.
    
    This is an internal helper function that extracts masks and lengths for prompts and responses.
    
    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        
    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def log_individual_lengths_to_wandb_and_csv(
    batch: DataProto,
    step: int,
    output_dir: str = "outputs",
    universal_id_key: str = "universal_id"
) -> None:
    print(f"[DEBUG] Logging individual lengths to wandb/csv at step {step} (output_dir={output_dir})")
    # Debug: Print batch keys to help trace missing universal_id
    print(f"[DEBUG] batch.batch keys: {list(batch.batch.keys())}")
    if not WANDB_AVAILABLE or not wandb.run:
        print("[DEBUG] Wandb not available or not initialized, skipping logging")
        return
        
    try:
        response_info = _compute_response_info(batch)
        prompt_lengths = response_info["prompt_length"].cpu().numpy()
        response_lengths = response_info["response_length"].cpu().numpy()
        
        print(f"[DEBUG] Processing {len(prompt_lengths)} samples")
        
        # Try to get prompt/response text if available
        prompts = batch.batch.get("prompts", None)
        responses = batch.batch.get("responses_text", None)
        if prompts is None:
            prompts = [None] * len(prompt_lengths)
            print("[DEBUG] No prompts found in batch")
        if responses is None:
            responses = [None] * len(response_lengths)
            print("[DEBUG] No responses found in batch")

        # --- UPDATED: Try to get universal_id from non_tensor_batch if not in batch.batch ---
        universal_ids = batch.batch.get(universal_id_key, None)
        if universal_ids is None:
            universal_ids = batch.non_tensor_batch.get(universal_id_key, None)
        if universal_ids is not None:
            print(f"[DEBUG] Found universal_ids of type: {type(universal_ids)} (key: {universal_id_key})")
            if hasattr(universal_ids, 'tolist'):
                universal_ids = universal_ids.tolist()
            else:
                universal_ids = list(universal_ids)
            universal_ids = [str(uid) if uid is not None else None for uid in universal_ids]
            print(f"[DEBUG] Converted to {len(universal_ids)} string UIDs")
        else:
            # Try fallback: check for "universal_id" if key is different
            if universal_id_key != "universal_id":
                fallback_uids = batch.batch.get("universal_id", None)
                if fallback_uids is None:
                    fallback_uids = batch.non_tensor_batch.get("universal_id", None)
                if fallback_uids is not None:
                    universal_ids = fallback_uids
                    if hasattr(universal_ids, 'tolist'):
                        universal_ids = universal_ids.tolist()
                    else:
                        universal_ids = list(universal_ids)
                    universal_ids = [str(uid) if uid is not None else None for uid in universal_ids]
                    print(f"[DEBUG] Fallback: found 'universal_id' key, converted to {len(universal_ids)} string UIDs")
                else:
                    universal_ids = [f"missing_uid_{i}" for i in range(len(prompt_lengths))]
                    print(f"[DEBUG] No universal_ids found for key '{universal_id_key}', creating placeholder UIDs")
            else:
                universal_ids = [f"missing_uid_{i}" for i in range(len(prompt_lengths))]
                print(f"[DEBUG] No universal_ids found for key '{universal_id_key}', creating placeholder UIDs")
        
        # Get additional metrics if available
        sequence_scores = None
        sequence_rewards = None
        if "token_level_scores" in batch.batch:
            sequence_scores = batch.batch["token_level_scores"].sum(-1).cpu().numpy()
            print(f"[DEBUG] Found sequence scores: {len(sequence_scores)}")
        if "token_level_rewards" in batch.batch:
            sequence_rewards = batch.batch["token_level_rewards"].sum(-1).cpu().numpy()
            print(f"[DEBUG] Found sequence rewards: {len(sequence_rewards)}")
        
        # Create data for wandb table and CSV
        table_data = []
        csv_data = []
        
        for i, (uid, prompt, response, prompt_len, response_len) in enumerate(zip(universal_ids, prompts, responses, prompt_lengths, response_lengths)):
            row = {
                universal_id_key: uid,  # Use configurable key here
                "step": step,
                "sample_idx": i,
                "prompt": prompt,
                "response": response,
                "prompt_length": int(prompt_len),
                "response_length": int(response_len),
                "total_length": int(prompt_len + response_len)
            }
            if sequence_scores is not None:
                row["sequence_score"] = float(sequence_scores[i])
            if sequence_rewards is not None:
                row["sequence_reward"] = float(sequence_rewards[i])
            
            # Clean up row for CSV/JSON
            row_clean = {k: _safe_primitive(v) for k, v in row.items()}
            table_data.append(list(row_clean.values()))
            csv_data.append(row_clean)
        
        print(f"[DEBUG] Created {len(csv_data)} rows of data")
        
        # Define columns for table
        columns = [universal_id_key, "step", "sample_idx", "prompt", "response", "prompt_length", "response_length", "total_length"]
        if sequence_scores is not None:
            columns.append("sequence_score")
        if sequence_rewards is not None:
            columns.append("sequence_reward")
        
        # Create and log wandb table
        table = wandb.Table(columns=columns, data=table_data)
        
        # Handle string steps for wandb (it only accepts integers)
        wandb_step = step
        if isinstance(step, str) and step.startswith("test_step_"):
            wandb_step = int(step.replace("test_step_", ""))
        
        wandb.log({"individual_lengths/batch_details": table}, step=wandb_step)
        print(f"[DEBUG] Logged wandb table with {len(table_data)} rows (wandb_step={wandb_step})")
        
        # Save to CSV and log as artifact
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle test step naming to avoid redundant "step_" prefix
        if isinstance(step, str) and step.startswith("test_step_"):
            # For test validation runs, use cleaner filename
            step_suffix = step.replace("test_step_", "")
            csv_filename = os.path.join(output_dir, f"test_validation_step_{step_suffix}.csv")
        else:
            # For regular training steps
            csv_filename = os.path.join(output_dir, f"individual_lengths_step_{step}.csv")
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_filename, index=False)
        print(f"[DEBUG] Saved CSV with {len(csv_data)} rows to {csv_filename}")
        
        # Print first few rows for verification
        print("[DEBUG] First 3 rows of CSV data:")
        for i, row in enumerate(csv_data[:3]):
            print(f"  Row {i}: {universal_id_key}={row.get(universal_id_key)}, prompt_length={row.get('prompt_length')}, response_length={row.get('response_length')}")
        
        # Create artifact and log
        if isinstance(step, str) and step.startswith("test_step_"):
            # For test validation runs, use cleaner artifact name
            step_suffix = step.replace("test_step_", "")
            artifact = wandb.Artifact(f"test_validation_step_{step_suffix}", type="dataset")
        else:
            # For regular training steps
            artifact = wandb.Artifact(f"lengths_step_{step}", type="dataset")
        
        artifact.add_file(csv_filename)
        wandb.log_artifact(artifact)
        print(f"[DEBUG] Created and logged wandb artifact for step {step}")
        
        # Log summary statistics for easy plotting
        wandb.log({
            "individual_lengths/prompt_length_mean": float(prompt_lengths.mean()),
            "individual_lengths/prompt_length_std": float(prompt_lengths.std()),
            "individual_lengths/prompt_length_min": float(prompt_lengths.min()),
            "individual_lengths/prompt_length_max": float(prompt_lengths.max()),
            "individual_lengths/response_length_mean": float(response_lengths.mean()),
            "individual_lengths/response_length_std": float(response_lengths.std()),
            "individual_lengths/response_length_min": float(response_lengths.min()),
            "individual_lengths/response_length_max": float(response_lengths.max()),
        }, step=wandb_step)
        print("[DEBUG] Logged summary statistics")
        
    except Exception as e:
        print(f"Warning: Failed to log individual lengths to wandb: {e}")
        import traceback
        traceback.print_exc()


def compute_data_metrics(batch: DataProto, use_critic: bool = True, step: int = None, log_individual: bool = True, output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.
        step: Current training step for individual logging. Defaults to None.
        log_individual: Whether to log individual sample data to wandb and CSV. Defaults to True.
        output_dir: Directory to save CSV files.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Calculate training accuracy from scores
    training_accuracy = torch.mean((sequence_score > 0).float()).detach().item()
    
    # Print training accuracy to log file
    print(f"Training Accuracy: {training_accuracy:.4f} ({training_accuracy*100:.2f}%)")
    
    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # training accuracy
        "training/accuracy": training_accuracy,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    
    # Log individual lengths if requested and step is provided
    if log_individual and step is not None:
        print(f"[DEBUG] About to log individual lengths for step {step}")
        log_individual_lengths_to_wandb_and_csv(batch, step, output_dir=output_dir, universal_id_key="universal_id")
    
    return metrics


def _safe_primitive(val):
    """Convert values to JSON-serializable primitives"""
    # Only allow str, int, float, None
    if isinstance(val, (str, int, float)) or val is None:
        return val
    # Convert numpy types to python types
    if hasattr(val, "item") and hasattr(val, "shape") and val.shape == ():  # scalar tensor/array
        return val.item()
    # Convert torch.Tensor or np.ndarray to list if not scalar
    if hasattr(val, "tolist"):
        try:
            return val.tolist()
        except:
            return str(val)
    # Fallback: convert to string
    return str(val)

def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.
    
    This function calculates both raw timing metrics (in seconds) and per-token timing metrics 
    (in milliseconds) for various processing stages like generation, reference computation, 
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    """
    Computes throughput metrics for PPO training.
    
    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).
    
    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.
        
    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU
            
    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.
    
    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.
    
    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }
        
        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)
        
    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    # Calculate and log test accuracy from score metrics
    for data_source, var2metric2val in data_src2var2metric2val.items():
        if "score" in var2metric2val:
            test_accuracy_dict = {}
            for metric_name, metric_val in var2metric2val["score"].items():
                if "mean@" in metric_name:
                    test_accuracy = metric_val
                    print(f"Test Accuracy ({data_source}, {metric_name}): {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
                    # Add to return dict for wandb logging
                    test_accuracy_dict[metric_name] = test_accuracy
                    # Also add a simple test_accuracy metric for easy wandb tracking
                    test_accuracy_dict["current"] = test_accuracy
            if test_accuracy_dict:
                data_src2var2metric2val[data_source]["test_accuracy"] = test_accuracy_dict

    return data_src2var2metric2val


import collections.abc

# def _safe_primitive(val):
#     # Only allow str, int, float, None
#     if isinstance(val, (str, int, float)) or val is None:
#         return val
#     # Convert numpy types to python types
#     if hasattr(val, "item") and val.shape == ():  # scalar tensor/array
#         return val.item()
#     # Convert torch.Tensor or np.ndarray to list if not scalar
#     if hasattr(val, "tolist"):
#         return val.tolist()
#     # Fallback: convert to string
#     return str(val)