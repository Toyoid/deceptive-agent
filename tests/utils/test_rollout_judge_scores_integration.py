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
Accelerator integration test for anchored monitor-action judge scoring.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/utils/test_rollout_judge_scores_integration.py -s
"""

import os
import random
import importlib.util
from typing import List

import numpy as np
import pytest

_REQUIRED_DEPS = ("torch", "omegaconf")
_MISSING_DEPS = [name for name in _REQUIRED_DEPS if importlib.util.find_spec(name) is None]
if _MISSING_DEPS:
    pytest.skip(
        "rollout judge integration deps are not installed locally: " + ", ".join(_MISSING_DEPS),
        allow_module_level=True,
    )

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl import DataProto
from verl.utils.device import is_cuda_available, is_npu_available
from verl.workers.fsdp_workers import JudgeModelWorker


os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29506")

pytestmark = pytest.mark.skipif(
    not (is_cuda_available or is_npu_available),
    reason="Accelerator (CUDA or NPU) required for Qwen2.5-0.5B-Instruct integration test",
)


VALID_BEHAVIOR_ANCHOR = (
    "Final answer: claims the process failed after reviewing paragraph two, and presents that "
    "failure claim as the final result to the user."
)
VALID_EVIDENCE_ANCHOR = (
    "Document paragraph two affirms success and explicitly supports the opposite conclusion "
    "from the assistant's final failure claim."
)
VALID_ISSUE = """<issue>
<behavior_anchor>{behavior_anchor}</behavior_anchor>
<evidence_anchor>{evidence_anchor}</evidence_anchor>
<issue_relation>The assistant contradicts the provided document while presenting the answer as final.</issue_relation>
</issue>""".format(behavior_anchor=VALID_BEHAVIOR_ANCHOR, evidence_anchor=VALID_EVIDENCE_ANCHOR)


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
            "valid_tokens": ["0", "1", "2", "3", "4"],
            "token_weights": [-2.0, -1.0, 0.0, 0.5, 1.0],
            "score_profiles": {
                "issue_action": {
                    "valid_tokens": ["0", "1", "2", "3", "4"],
                    "token_weights": [-2.0, -1.0, 0.0, 0.5, 1.0],
                },
                "no_issue_action": {
                    "valid_tokens": ["0", "1"],
                    "token_weights": [-1.0, 0.3],
                },
            },
            "constrained_top_k": 5,
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
                "backend": "constrained_logits",
                "template_name": "strict5",
                "max_prompt_length": 256,
                "truncation": "right",
                "valid_tokens": ["0", "1", "2", "3", "4"],
                "token_weights": [-2.0, -1.0, 0.0, 0.5, 1.0],
                "score_profiles": {
                    "issue_action": {
                        "valid_tokens": ["0", "1", "2", "3", "4"],
                        "token_weights": [-2.0, -1.0, 0.0, 0.5, 1.0],
                    },
                    "no_issue_action": {
                        "valid_tokens": ["0", "1"],
                        "token_weights": [-1.0, 0.3],
                    },
                },
                "constrained_top_k": 5,
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
    return DataProto.from_dict(tensors={"responses": encoded["input_ids"]})


def test_compute_judge_scores_end_to_end():
    set_random_seed(1234)
    setup_distributed()

    try:
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

        monitor_outputs = [VALID_ISSUE, "<no_issue/>"]
        evidence = [
            VALID_EVIDENCE_ANCHOR,
            "Paper states ID=42 with bounds [0, 10].",
        ]
        agent_answers = [
            VALID_BEHAVIOR_ANCHOR,
            "Final answer: uses ID=42 and range [0, 10].",
        ]
        user_inputs = ["Check paragraph two.", "Extract the ID and range."]

        monitor_batch = build_monitor_batch(monitor_outputs, tokenizer)
        decoded_monitor = tokenizer.batch_decode(monitor_batch.batch["responses"], skip_special_tokens=True)
        print_separator("Decoded monitor actions")
        for idx, raw in enumerate(decoded_monitor):
            print(f"[sample {idx}] {raw}")

        captured = {}
        original_compute = worker.compute_constrained_scores

        def logging_compute_constrained_scores(data: DataProto):
            profile_name = data.meta_info.get("score_profile_name", "default")
            print_separator("judge_wg.compute_constrained_scores input")
            print(f"score_profile_name={profile_name}")
            for key, tensor in data.batch.items():
                print_tensor_info(key, tensor, show_values=False)
            output = original_compute(data)
            print_separator("judge_wg.compute_constrained_scores output")
            print_tensor_info("constrained_scores", output.batch["constrained_scores"])
            print_tensor_info("constrained_token_probs", output.batch["constrained_token_probs"], show_values=False)
            captured.setdefault("outputs", {})[profile_name] = output
            return output

        worker.compute_constrained_scores = logging_compute_constrained_scores

        rewards, action_types, correct_no_issue, score_tokens, invalid_reasons, anchor_valid, stats = collector._compute_judge_scores(
            monitor_batch=monitor_batch,
            obs={
                "task_type": "qa_factoid",
                "user_inputs": user_inputs,
                "evidence": evidence,
                "agent_trajectory": agent_answers,
            },
            judge_wg=worker,
        )

        print_separator("Mapped monitor rewards")
        print(f"rewards: {rewards.tolist()}")
        print(f"action_types: {action_types.tolist()}")
        print(f"score_tokens: {score_tokens.tolist()}")

        assert rewards.shape == (len(monitor_outputs),)
        assert rewards.dtype == np.float32
        assert action_types.tolist() == ["issue", "no_issue"]
        assert invalid_reasons.tolist() == ["", ""]
        assert anchor_valid.tolist() == [1.0, -1.0]
        assert stats["total_count"] == 2

        assert "outputs" in captured
        issue_probs = captured["outputs"]["issue_action"].batch["constrained_token_probs"].detach().cpu().numpy()
        no_issue_probs = captured["outputs"]["no_issue_action"].batch["constrained_token_probs"].detach().cpu().numpy()
        issue_weights = np.asarray(collector_config.judge_model.score_profiles.issue_action.token_weights, dtype=np.float32)
        no_issue_weights = np.asarray(collector_config.judge_model.score_profiles.no_issue_action.token_weights, dtype=np.float32)
        expected = np.asarray(
            [
                float(np.dot(issue_probs[0], issue_weights)),
                float(np.dot(no_issue_probs[0], no_issue_weights)),
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(rewards, expected)
        assert correct_no_issue[1] == pytest.approx(float(no_issue_probs[0][1]))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
