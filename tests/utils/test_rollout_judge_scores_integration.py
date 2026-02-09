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
Integration test for `_compute_judge_scores` in `rollout_loop.py`.

The test runs the full judge-scoring path with a small LM (Qwen2.5-0.5B-Instruct),
prints intermediate tensors, and verifies that aggregation back to per-sample scores
matches the raw per-critique outputs from the judge worker.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_rollout_judge_scores_integration.py -s
"""

import os
import random
from typing import List

import numpy as np
import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from agent_system.environments.prompts.judge_prompt import (
    build_judge_prompt,
    extract_critiques,
)
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.device import is_cuda_available, is_npu_available
from verl.workers.fsdp_workers import JudgeModelWorker


# Minimal single-process distributed defaults
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29506")

# Skip early if no accelerator is available for the judge worker path
pytestmark = pytest.mark.skipif(
    not (is_cuda_available or is_npu_available),
    reason="Accelerator (CUDA or NPU) required for Qwen2.5-0.5B-Instruct integration test",
)


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed(seed)
    print(f"[seed] set to {seed}")


def print_separator(title: str, char: str = "="):
    line = char * 80
    print(f"\n{line}\n  {title}\n{line}")


def print_tensor_info(name: str, tensor: torch.Tensor, show_values: bool = False, max_show: int = 20):
    print(f"  [{name}] shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    if show_values:
        flat = tensor.flatten()
        values = flat[:max_show].tolist()
        print(f"    values (first {len(values)}): {values}")
    if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        print(f"    stats: min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")


def setup_distributed():
    backend = "nccl" if is_cuda_available else "hccl"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    print(f"[dist] initialized with backend={backend}, rank={dist.get_rank()}, world_size={dist.get_world_size()}")


def create_judge_worker_config(model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    return OmegaConf.create(
        {
            "model": {
                "path": model_path,
                "fsdp_config": {"fsdp_size": 1, "wrap_policy": None, "reshard_after_forward": True},
                "use_remove_padding": False,
                "trust_remote_code": False,
                "use_shm": False,
            },
            "strategy": "fsdp",
            "valid_tokens": ["0", "1", "2", "3"],
            "token_weights": [0.0, 0.33, 0.66, 1.0],
            "top_k": 4,
            "micro_batch_size": None,
            "micro_batch_size_per_gpu": 2,
            "use_dynamic_bsz": False,
            "ulysses_sequence_parallel_size": 1,
            "forward_max_token_len_per_gpu": 4096,
        }
    )


def create_collector_config():
    return OmegaConf.create(
        {
            "monitor_rollout_ref": {"enable": True},
            "judge_model": {
                "enable": True,
                "max_prompt_length": 256,
                "truncation": "right",
            },
        }
    )


def build_monitor_batch(monitor_outputs: List[str], tokenizer) -> DataProto:
    encoded = tokenizer(
        monitor_outputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    # Only responses are needed for _compute_judge_scores
    return DataProto.from_dict(tensors={"responses": encoded["input_ids"]})


def test_compute_judge_scores_end_to_end():
    set_random_seed(1234)
    setup_distributed()

    print_separator("Create JudgeModelWorker")
    worker_config = create_judge_worker_config()
    worker = JudgeModelWorker(worker_config)
    worker.init_model()
    tokenizer = worker.tokenizer
    print(f"[model] tokenizer={type(tokenizer).__name__}, vocab={tokenizer.vocab_size}")

    collector_config = create_collector_config()
    collector = TrajectoryCollector(
        config=collector_config,
        tokenizer=tokenizer,
        processor=None,
        monitor_tokenizer=tokenizer,
        monitor_processor=None,
        judge_tokenizer=tokenizer,
        judge_processor=None,
    )

    monitor_outputs = [
        "<critique>The answer should cite the study ID.</critique>"
        "<critique>The numeric range omits the upper bound.</critique>",
        "<critique>The response contradicts the second paragraph.</critique>",
    ]
    evidence = [
        "Paper states ID=42 with bounds [0, 10].",
        "Document paragraph two affirms success.",
    ]
    agent_answers = [
        "Final answer: uses ID but no citation, range stated as 0-9.",
        "Final answer: claims the process failed.",
    ]
    infos = [
        {"task_type": "qa_factoid"},
        {"task_type": "analysis"},
    ]

    monitor_batch = build_monitor_batch(monitor_outputs, tokenizer)
    decoded_monitor = tokenizer.batch_decode(monitor_batch.batch["responses"], skip_special_tokens=True)
    print_separator("Monitor outputs and extracted critiques")
    sample_critique_counts = []
    judge_prompts = []
    for idx, raw in enumerate(decoded_monitor):
        critiques = extract_critiques(raw)
        sample_critique_counts.append(len(critiques))
        print(f"[sample {idx}] monitor text: {raw}")
        print(f"[sample {idx}] critiques ({len(critiques)}): {critiques}")
        for critique in critiques:
            prompt = build_judge_prompt(
                task_type=infos[idx]["task_type"],
                evidence=evidence[idx],
                agent_answer=agent_answers[idx],
                critique=critique,
            )
            judge_prompts.append(prompt)
            preview = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            print(f"[sample {idx}] judge prompt preview:\n{preview}")

    print_separator("Manual preprocessing preview")
    processed_samples = []
    for i, prompt in enumerate(judge_prompts):
        processed = collector._process_chat_to_model_inputs(
            chat=prompt,
            obs_image=None,
            tokenizer=tokenizer,
            processor=None,
            max_prompt_length=collector_config.judge_model.max_prompt_length,
            truncation=collector_config.judge_model.truncation,
        )
        processed_samples.append(processed)
        print(f"[critique {i}] input_ids len={processed['input_ids'].shape[0]}, attention_mask sum={processed['attention_mask'].sum().item()}")

    judge_batch_preview = DataProto.from_single_dict(data=collate_fn(processed_samples))
    print_tensor_info("judge_batch input_ids", judge_batch_preview.batch["input_ids"], show_values=False)
    print_tensor_info("judge_batch attention_mask", judge_batch_preview.batch["attention_mask"], show_values=False)
    print_tensor_info("judge_batch position_ids", judge_batch_preview.batch["position_ids"], show_values=False)

    # Patch compute_judge_score to log tensors flowing through _compute_judge_scores
    captured = {}
    original_compute = worker.compute_judge_score

    def logging_compute_judge_score(data: DataProto):
        print_separator("judge_wg.compute_judge_score input")
        for key, tensor in data.batch.items():
            print_tensor_info(key, tensor, show_values=False)
        output = original_compute(data)
        print_separator("judge_wg.compute_judge_score output")
        print_tensor_info("judge_scores", output.batch["judge_scores"])
        print(f"  {output.batch['judge_scores']}")
        print_tensor_info("judge_token_probs", output.batch["judge_token_probs"], show_values=False)
        print(f"  {output.batch['judge_token_probs']}")
        captured["output"] = output
        return output

    worker.compute_judge_score = logging_compute_judge_score

    scores = collector._compute_judge_scores(
        monitor_batch=monitor_batch,
        obs={"monitor_text": evidence, "text": agent_answers},
        infos=infos,
        judge_wg=worker,
    )

    print_separator("Aggregated per-sample scores")
    print(f"trust penalties: {scores.tolist()}")
    assert scores.shape == (len(monitor_outputs),)
    assert scores.dtype == np.float32

    flat_scores = captured["output"].batch["judge_scores"].numpy()
    critique_counts = np.asarray(sample_critique_counts, dtype=np.int64)
    assert critique_counts.sum() == len(flat_scores)

    splits = np.cumsum(critique_counts)[:-1]
    expected = [chunk.mean() for chunk in np.split(flat_scores, splits)]
    np.testing.assert_allclose(scores, np.array(expected, dtype=np.float32))

    print("[aggregation] verified per-sample means match flattened judge scores")

    if dist.is_initialized():
        dist.destroy_process_group()
