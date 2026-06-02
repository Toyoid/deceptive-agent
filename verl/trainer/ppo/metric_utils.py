# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Hanxiao Li, Beihang University
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
"""
Metrics related to the PPO trainer.
"""

import json
import os
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from agent_system.utils.metric_contract import EPISODE_METRIC_PREFIX
from verl import DataProto
from verl.utils.import_utils import deprecated


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


# ------------------------------------------------------------------------------
# Episode and PPO batch metrics
# ------------------------------------------------------------------------------

def compute_episode_metric_stats(
    non_tensor_batch: Dict[str, Any],
    unique_idx: np.ndarray,
    metric_prefix: str = "",
) -> Dict[str, Any]:
    def _key(name: str) -> str:
        return f"{metric_prefix}/{name}" if metric_prefix else name

    metrics: Dict[str, Any] = {}
    for key, values in non_tensor_batch.items():
        if not key.startswith(EPISODE_METRIC_PREFIX):
            continue
        metric_name = key[len(EPISODE_METRIC_PREFIX):]
        metric_values = np.asarray(values[unique_idx], dtype=np.float32)
        metrics[_key(f"episode/{metric_name}/mean")] = float(metric_values.mean())
        metrics[_key(f"episode/{metric_name}/max")] = float(metric_values.max())
        metrics[_key(f"episode/{metric_name}/min")] = float(metric_values.min())

    return metrics


def compute_data_metrics(
    batch: DataProto,
    use_critic: bool = True,
    metric_prefix: str = "",
    include_episode_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.
        metric_prefix: Optional prefix for metric keys (e.g., "monitor" -> "monitor/critic/score/mean").
                      If empty, no prefix is added (backward compatible).
        include_episode_metrics: Whether to log environment-level episode metrics such as
                      success rates or tool counts. Prompt-only auxiliary batches can share
                      the score/reward/advantage metrics while intentionally skipping the
                      env-only episode statistics.

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
    # Helper to add prefix to metric keys
    def _key(name: str) -> str:
        return f"{metric_prefix}/{name}" if metric_prefix else name
    
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

    include_episode_metrics = include_episode_metrics and metric_prefix != "monitor"
    unique_idx = None
    if include_episode_metrics:
        _, unique_idx = np.unique(batch.non_tensor_batch["traj_uid"], return_index=True)

    metrics = {
        # score
        _key("critic/score/mean"): torch.mean(sequence_score).detach().item(),
        _key("critic/score/max"): torch.max(sequence_score).detach().item(),
        _key("critic/score/min"): torch.min(sequence_score).detach().item(),
        # reward
        _key("critic/rewards/mean"): torch.mean(sequence_reward).detach().item(),
        _key("critic/rewards/max"): torch.max(sequence_reward).detach().item(),
        _key("critic/rewards/min"): torch.min(sequence_reward).detach().item(),
        # adv
        _key("critic/advantages/mean"): torch.mean(valid_adv).detach().item(),
        _key("critic/advantages/max"): torch.max(valid_adv).detach().item(),
        _key("critic/advantages/min"): torch.min(valid_adv).detach().item(),
        # returns
        _key("critic/returns/mean"): torch.mean(valid_returns).detach().item(),
        _key("critic/returns/max"): torch.max(valid_returns).detach().item(),
        _key("critic/returns/min"): torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                _key("critic/values/mean"): torch.mean(valid_values).detach().item(),
                _key("critic/values/max"): torch.max(valid_values).detach().item(),
                _key("critic/values/min"): torch.min(valid_values).detach().item(),
                # vf explained var
                _key("critic/vf_explained_var"): (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        _key("response_length/mean"): torch.mean(response_length).detach().item(),
        _key("response_length/max"): torch.max(response_length).detach().item(),
        _key("response_length/min"): torch.min(response_length).detach().item(),
        _key("response_length/clip_ratio"): torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        _key("prompt_length/mean"): torch.mean(prompt_length).detach().item(),
        _key("prompt_length/max"): torch.max(prompt_length).detach().item(),
        _key("prompt_length/min"): torch.min(prompt_length).detach().item(),
        _key("prompt_length/clip_ratio"): torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        # episode metrics - only for actor, not for monitor
        # NOTE: Monitor batches don't have episode_lengths, episode_rewards, tool_callings, etc.
        # because monitor currently evaluates actor's entire trajectory of responses rather than each step.
        **(
            {
                _key("episode/reward/mean"):
                    batch.non_tensor_batch["episode_rewards"][unique_idx].mean().item(),
                _key("episode/reward/max"):
                    batch.non_tensor_batch["episode_rewards"][unique_idx].max().item(),
                _key("episode/reward/min"):
                    batch.non_tensor_batch["episode_rewards"][unique_idx].min().item(),
                **({
                    _key("episode/trust_penalty/mean"): batch.non_tensor_batch["trust_penalties"][unique_idx].mean().item(),
                    _key("episode/trust_penalty/max"): batch.non_tensor_batch["trust_penalties"][unique_idx].max().item(),
                    _key("episode/trust_penalty/min"): batch.non_tensor_batch["trust_penalties"][unique_idx].min().item(),
                } if "trust_penalties" in batch.non_tensor_batch else {}),
                _key("episode/length/mean"):
                    batch.non_tensor_batch["episode_lengths"][unique_idx].mean().item(),
                _key("episode/length/max"):
                    batch.non_tensor_batch["episode_lengths"][unique_idx].max().item(),
                _key("episode/length/min"):
                    batch.non_tensor_batch["episode_lengths"][unique_idx].min().item(),
                _key("episode/tool_call_count/mean"): 
                    batch.non_tensor_batch["tool_callings"][unique_idx].mean().item(),
                _key("episode/tool_call_count/max"):
                    batch.non_tensor_batch["tool_callings"][unique_idx].max().item(),
                _key("episode/tool_call_count/min"):
                    batch.non_tensor_batch["tool_callings"][unique_idx].min().item(),
                **{_key(f"episode/{k}"): v[0].item() for k, v in batch.non_tensor_batch.items() if k.endswith('_rate')},
                **compute_episode_metric_stats(
                    non_tensor_batch=batch.non_tensor_batch,
                    unique_idx=unique_idx,
                    metric_prefix=metric_prefix,
                ),
            }
            if include_episode_metrics
            else {}
        ),
    }
    return metrics


# ------------------------------------------------------------------------------
# External monitor diagnostics
# ------------------------------------------------------------------------------

def get_actor_cost_threshold(config: Any, default: float = 0.5) -> float:
    monitor_cfg = getattr(config, "monitor_rollout_ref", None)
    if monitor_cfg is None:
        raise ValueError("monitor_rollout_ref config not found in the provided config object")
    return float(monitor_cfg.get("actor_cost_threshold", default))


def restricted_tool_used_from_batch(batch: DataProto) -> Optional[np.ndarray]:
    batch_size = len(batch.non_tensor_batch["traj_uid"])
    restricted = np.zeros(batch_size, dtype=bool)
    found = False
    for key in (
        f"{EPISODE_METRIC_PREFIX}used_insider",
        f"{EPISODE_METRIC_PREFIX}restricted_tool_calls",
    ):
        if key in batch.non_tensor_batch:
            restricted |= np.asarray(batch.non_tensor_batch[key], dtype=np.float32) > 0.0
            found = True
    return restricted if found else None


def compute_monitor_action_metrics(monitor_batch: DataProto) -> Dict[str, float]:
    if "monitor_action_type" not in monitor_batch.non_tensor_batch:
        return {}
    action_types = np.asarray(monitor_batch.non_tensor_batch["monitor_action_type"], dtype=object)
    n_total = len(action_types)
    if n_total == 0:
        return {}

    metrics = {
        "monitor/action/no_issue_rate": float(np.mean(action_types == "no_issue")),
        "monitor/action/issue_rate": float(np.mean(action_types == "issue")),
        "monitor/action/invalid_rate": float(np.mean(action_types == "invalid")),
    }
    if "correct_no_issue" in monitor_batch.non_tensor_batch:
        correctness = np.asarray(monitor_batch.non_tensor_batch["correct_no_issue"], dtype=np.float32)
        valid_no_issue = (action_types == "no_issue") & (correctness >= 0.0)
        metrics["monitor/action/correct_no_issue_rate"] = (
            float(np.mean(correctness[valid_no_issue])) if np.any(valid_no_issue) else 0.0
        )
    else:
        metrics["monitor/action/correct_no_issue_rate"] = 0.0
    if "monitor_anchor_valid" in monitor_batch.non_tensor_batch:
        anchor_valid = np.asarray(monitor_batch.non_tensor_batch["monitor_anchor_valid"], dtype=np.float32)
        issue_anchor_rows = (action_types == "issue") & (anchor_valid >= 0.0)
        metrics["monitor/action/anchor_pass_rate"] = (
            float(np.mean(anchor_valid[issue_anchor_rows])) if np.any(issue_anchor_rows) else 0.0
        )
    else:
        metrics["monitor/action/anchor_pass_rate"] = 0.0
    return metrics


def compute_restricted_tool_confusion_metrics(
    batch: DataProto,
    actor_cost_threshold: float,
) -> Dict[str, float]:
    if "trust_penalties" not in batch.non_tensor_batch:
        return {}
    restricted = restricted_tool_used_from_batch(batch)
    if restricted is None:
        return {}

    traj_uids = batch.non_tensor_batch["traj_uid"]
    _, unique_idx = np.unique(traj_uids, return_index=True)
    unique_idx = np.sort(unique_idx)
    restricted = restricted[unique_idx]
    costs = np.asarray(batch.non_tensor_batch["trust_penalties"], dtype=np.float32)[unique_idx]
    caught = costs > actor_cost_threshold

    tp = int(np.sum(restricted & caught))
    fp = int(np.sum(~restricted & caught))
    fn = int(np.sum(restricted & ~caught))
    tn = int(np.sum(~restricted & ~caught))
    restricted_total = tp + fn
    clean_total = fp + tn
    return {
        "episode/monitor_confusion/tp_count": float(tp),
        "episode/monitor_confusion/fp_count": float(fp),
        "episode/monitor_confusion/fn_count": float(fn),
        "episode/monitor_confusion/tn_count": float(tn),
        "episode/restricted_tool_caught_rate": float(tp / restricted_total) if restricted_total > 0 else 0.0,
        "episode/clean_false_positive_rate": float(fp / clean_total) if clean_total > 0 else 0.0,
    }


# ------------------------------------------------------------------------------
# JSONL rollout dumps
# ------------------------------------------------------------------------------

def dump_generations(
    system_infos,
    inputs,
    outputs,
    scores,
    reward_extra_infos_dict,
    dump_path: str,
    step: int,
) -> None:
    """Dump rollout/validation samples as JSONL."""
    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"{step}.jsonl")

    assert len(system_infos) == len(inputs) == len(outputs) == len(scores), (
        f"Mismatched lengths in generations to dump. len(system_infos)={len(system_infos)}, "
        f"len(inputs={len(inputs)}, len(outputs)={len(outputs)}, len(scores)={len(scores)}"
    )
    n = len(inputs)
    base_data = {
        "system_info": system_infos,
        "input": inputs,
        "output": outputs,
        "score": scores,
        "step": [step] * n,
    }

    for k, v in reward_extra_infos_dict.items():
        if len(v) == n:
            base_data[k] = v

    with open(filename, "w", encoding="utf-8") as f:
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Dumped {n} generations to {filename}")


def dump_monitor_false_positives(
    actor_batch: DataProto,
    monitor_batch: DataProto,
    dump_path: str,
    step: int,
    monitor_tokenizer,
    actor_cost_threshold: float,
) -> None:
    restricted = restricted_tool_used_from_batch(actor_batch)
    if restricted is None or "trust_penalties" not in actor_batch.non_tensor_batch:
        return

    actor_traj_uids = np.asarray(actor_batch.non_tensor_batch["traj_uid"], dtype=object)
    actor_costs = np.asarray(actor_batch.non_tensor_batch["trust_penalties"], dtype=np.float32)
    actor_rewards = np.asarray(actor_batch.non_tensor_batch["episode_rewards"], dtype=np.float32)
    actor_data_sources = np.asarray(actor_batch.non_tensor_batch["data_source"], dtype=object)

    _, unique_actor_idx = np.unique(actor_traj_uids, return_index=True)
    unique_actor_idx = np.sort(unique_actor_idx)

    monitor_outputs = monitor_tokenizer.batch_decode(
        monitor_batch.batch["responses"], skip_special_tokens=True
    )
    if len(unique_actor_idx) == 0 or len(monitor_outputs) % len(unique_actor_idx) != 0:
        return
    monitor_rollout_n = len(monitor_outputs) // len(unique_actor_idx)

    monitor_action_types = np.asarray(monitor_batch.non_tensor_batch.get("monitor_action_type", []), dtype=object)
    judge_score_tokens = np.asarray(monitor_batch.non_tensor_batch.get("judge_score_token", []), dtype=object)
    agent_trajectories = np.asarray(monitor_batch.non_tensor_batch.get("agent_trajectory", []), dtype=object)

    rows = []
    for monitor_idx in range(len(monitor_outputs)):
        actor_idx = unique_actor_idx[monitor_idx // monitor_rollout_n]
        is_restricted = bool(restricted[actor_idx])
        actor_cost = float(actor_costs[actor_idx])
        if is_restricted or actor_cost <= actor_cost_threshold:
            continue
        rows.append({
            "step": step,
            "data_source": str(actor_data_sources[actor_idx]),
            "traj_uid": str(actor_traj_uids[actor_idx]),
            "restricted_tool_used": False,
            "actor_trust_penalty": actor_cost,
            "actor_cost_threshold": actor_cost_threshold,
            "episode_reward": float(actor_rewards[actor_idx]),
            "agent_trajectory": str(agent_trajectories[monitor_idx]) if len(agent_trajectories) > monitor_idx else "",
            "monitor_output": monitor_outputs[monitor_idx],
            "monitor_action_type": str(monitor_action_types[monitor_idx]) if len(monitor_action_types) > monitor_idx else "",
            "judge_score_token": str(judge_score_tokens[monitor_idx]) if len(judge_score_tokens) > monitor_idx else "",
        })

    if not rows:
        return

    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"{step}.jsonl")
    with open(filename, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Dumped {len(rows)} monitor false positives to {filename}")


# ------------------------------------------------------------------------------
# Timing and throughput metrics
# ------------------------------------------------------------------------------

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


def compute_throughout_metrics(total_num_tokens: int, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    """
    Computes throughput metrics for PPO training.
    
    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).
    
    Args:
        total_num_tokens: Total number of tokens processed in the training step.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.
        
    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the training step
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU
            
    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
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

    return data_src2var2metric2val


# ==============================================================================
#              Distribution Logging (optional, wandb only)
# ==============================================================================

def extract_trajectory_distributions(batch: DataProto, use_lag: bool = False) -> Dict[str, np.ndarray]:
    """Extract per-trajectory/per-sequence distribution values from a training batch.

    Args:
        batch: Training batch after advantage computation (and Lagrangian update if applicable).
        use_lag: Whether Lagrangian RL is enabled. When True, also extracts
                 reward_advantages and cost_advantages.

    Returns:
        Dict mapping metric names to 1-D numpy arrays of per-trajectory/per-sequence values.
    """
    distributions: Dict[str, np.ndarray] = {}

    # reward_score: per-sequence sum of token_level_scores (combined RM + episode_rewards)
    if "token_level_scores" in batch.batch.keys():
        reward_scores = batch.batch["token_level_scores"].sum(-1).detach().float().cpu().numpy()
        distributions["reward_score"] = reward_scores

    # trust_penalties: per-trajectory scalar (deduped by traj_uid), from actor batch
    if "trust_penalties" in batch.non_tensor_batch:
        _, unique_idx = np.unique(batch.non_tensor_batch["traj_uid"], return_index=True)
        unique_idx = np.sort(unique_idx)
        distributions["trust_penalties"] = np.asarray(
            batch.non_tensor_batch["trust_penalties"][unique_idx], dtype=np.float32
        )

    # Compute response_mask for masking token-level tensors
    max_response_length = batch.batch["responses"].shape[-1]
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()
    # Per-sequence response lengths for mean computation
    response_lengths = response_mask.sum(-1).float().clamp(min=1)  # (batch_size,)

    # advantages: per-sequence mean (this is lag_advantages when use_lag, else reward advantages)
    if "advantages" in batch.batch.keys():
        adv = batch.batch["advantages"].detach().float().clone()
        adv[~response_mask] = 0.0
        per_seq_adv = adv.sum(-1) / response_lengths
        distributions["advantages"] = per_seq_adv.cpu().numpy()

    # reward_advantages and cost_advantages (only when use_lag is True)
    if use_lag:
        if "reward_advantages" in batch.batch.keys():
            r_adv = batch.batch["reward_advantages"].detach().float().clone()
            r_adv[~response_mask] = 0.0
            per_seq_r_adv = r_adv.sum(-1) / response_lengths
            distributions["reward_advantages"] = per_seq_r_adv.cpu().numpy()

        if "cost_advantages" in batch.batch.keys():
            c_adv = batch.batch["cost_advantages"].detach().float().clone()
            c_adv[~response_mask] = 0.0
            per_seq_c_adv = c_adv.sum(-1) / response_lengths
            distributions["cost_advantages"] = per_seq_c_adv.cpu().numpy()

    return distributions


def plot_distribution(
    current_values: np.ndarray,
    metric_name: str,
    current_step: int,
    initial_values: Optional[np.ndarray] = None,
    initial_step: Optional[int] = None,
    bins: int = 40,
) -> Any:
    """Generate a publication-quality histogram distribution plot.

    Style is modelled after ``plot_distribution_comparison`` in
    ``retroactive_eval/analysis/plotter.py``.  When *initial_values* is
    provided the plot overlays two distributions (initial vs current); otherwise
    a single histogram is drawn.

    Args:
        current_values: 1-D array of values for the current step.
        metric_name: Human-readable metric name (used in title/axis labels).
        current_step: Training step number for the current distribution.
        initial_values: Optional 1-D array for the initial (reference) step.
        initial_step: Step number of the initial distribution.
        bins: Number of histogram bins.

    Returns:
        A matplotlib Figure object.  Caller is responsible for closing it
        (``plt.close(fig)``) after use to avoid memory leaks.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INITIAL_COLOR = "#4878D0"  # muted blue
    CURRENT_COLOR = "#EE6677"  # coral red

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.set_facecolor("#F5F5F5")
    fig.patch.set_facecolor("white")

    # Compute common bin edges
    if initial_values is not None:
        all_values = np.concatenate([initial_values, current_values])
    else:
        all_values = current_values
    bin_edges = np.linspace(all_values.min(), all_values.max(), bins + 1)

    # Plot initial distribution if available
    if initial_values is not None:
        ax.hist(
            initial_values,
            bins=bin_edges,
            color=INITIAL_COLOR,
            alpha=0.65,
            label=f"Step {initial_step} (n={len(initial_values)})",
            edgecolor="none",
            rwidth=0.85,
        )

    # Plot current distribution
    ax.hist(
        current_values,
        bins=bin_edges,
        color=CURRENT_COLOR,
        alpha=0.65,
        label=f"Step {current_step} (n={len(current_values)})",
        edgecolor="none",
        rwidth=0.85,
    )

    # Vertical mean lines + annotations
    y_max = ax.get_ylim()[1]

    if initial_values is not None:
        init_mean = float(np.mean(initial_values))
        ax.axvline(init_mean, color=INITIAL_COLOR, linestyle="--", linewidth=2, alpha=0.9)
        ax.annotate(
            f"\u03bc={init_mean:.2f}",
            xy=(init_mean, y_max * 0.92),
            fontsize=9, color=INITIAL_COLOR, fontweight="bold", ha="center",
        )

    cur_mean = float(np.mean(current_values))
    ax.axvline(cur_mean, color=CURRENT_COLOR, linestyle="--", linewidth=2, alpha=0.9)
    y_annot = y_max * 0.82 if initial_values is not None else y_max * 0.92
    ax.annotate(
        f"\u03bc={cur_mean:.2f}",
        xy=(cur_mean, y_annot),
        fontsize=9, color=CURRENT_COLOR, fontweight="bold", ha="center",
    )

    # Styling
    if initial_values is not None:
        title = f"{metric_name}: Step {initial_step} vs Step {current_step}"
    else:
        title = f"{metric_name}: Step {current_step}"
    ax.set_xlabel(f"{metric_name}", fontsize=11, fontweight="medium")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="medium")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    legend = ax.legend(
        loc="upper right", fontsize=9, frameon=True, fancybox=False,
        edgecolor="#CCCCCC", framealpha=0.95,
    )
    legend.get_frame().set_linewidth(0.5)

    ax.grid(True, linestyle="-", alpha=0.5, color="white", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#888888")
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", labelsize=9, colors="#444444")
    ax.tick_params(axis="x", direction="out", length=4, width=0.6)
    ax.tick_params(axis="y", direction="out", length=4, width=0.6)

    plt.tight_layout()
    return fig


def compute_distribution_log_data(
    batch: DataProto,
    use_lag: bool,
    current_step: int,
    initial_distributions: Optional[Dict[str, np.ndarray]] = None,
    initial_step: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Build wandb-compatible log dict with Histogram and Image entries.

    Args:
        batch: Training batch (after advantage / Lagrangian computation).
        use_lag: Whether Lagrangian RL is enabled.
        current_step: Current global training step.
        initial_distributions: Distributions from the first logged step
            (``None`` on the very first call).
        initial_step: Step number of *initial_distributions*.

    Returns:
        A tuple ``(log_dict, current_distributions)`` where *log_dict*
        maps wandb-compatible keys (``distributions/...``) to
        ``wandb.Histogram`` / ``wandb.Image`` objects, and
        *current_distributions* is the raw dict returned by
        :func:`extract_trajectory_distributions` (to be stored as the
        initial reference on first call).
    """
    import wandb
    import matplotlib.pyplot as plt

    current_dists = extract_trajectory_distributions(batch, use_lag=use_lag)
    log_data: Dict[str, Any] = {}

    for metric_name, values in current_dists.items():
        if len(values) == 0:
            continue

        # Native wandb histogram
        log_data[f"distributions/{metric_name}"] = wandb.Histogram(values.tolist())

        # Matplotlib comparison plot
        init_vals = initial_distributions.get(metric_name) if initial_distributions else None
        fig = plot_distribution(
            current_values=values,
            metric_name=metric_name,
            current_step=current_step,
            initial_values=init_vals,
            initial_step=initial_step,
        )
        log_data[f"distributions/{metric_name}_plot"] = wandb.Image(fig)
        plt.close(fig)

    return log_data, current_dists
