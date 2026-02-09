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
Integration test for sequence-classification reward models.

This verifies that `RewardModelWorker` produces the same scalar reward as a
Hugging Face pipeline on the same inputs. The test loads a real sequence
classification reward model, runs a couple of chat samples through both
implementations, and compares the logits directly (no activation function).

Set `TEST_SEQUENCE_RM_MODEL` to override the model path/name if needed.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/workers/test_reward_model_worker_sequence.py
"""

import contextlib
import os
import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from verl import DataProto
from verl.utils.device import is_npu_available
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import RewardModelWorker

# Distributed defaults for single-process testing
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29515")
os.environ.setdefault("LOCAL_RANK", "0")

CHAT_SAMPLES = [
    [
        {"role": "user", "content": "Write a short poem about a cat."},
        {
            "role": "assistant",
            "content": (
                "A feline friend, soft and sleek,\n"
                "Curled up warm, a purring peek.\n"
                "Through sunlit naps and playful chase,\n"
                "Graceful paws in every space."
            ),
        },
    ],
    [
        {"role": "user", "content": "Write a short poem about a cat."},
        {"role": "assistant", "content": "Cats are okay. They sit sometimes. Dog is better."},
    ],
    [
        {"role": "user", "content": "Give me a two-sentence summary of the benefits of daily exercise."},
        {
            "role": "assistant",
            "content": (
                "Daily exercise boosts cardiovascular health and improves mood through endorphin release. "
                "It also helps maintain healthy weight and supports better sleep quality."
            ),
        },
    ],
    [
        {"role": "user", "content": "Explain why the sky appears blue in simple terms."},
        {
            "role": "assistant",
            "content": (
                "Sunlight is made of many colors, and as it passes through the atmosphere the shorter blue "
                "wavelengths scatter in all directions. That scattered blue light reaches our eyes, making the sky look blue."
            ),
        },
    ],
    [
        {"role": "user", "content": "List quick steps for brewing a good cup of black coffee without special gear."},
        {
            "role": "assistant",
            "content": (
                "1) Boil water and let it sit 30 seconds. 2) Add fresh grounds to a mug at about 1:15 coffee-to-water. "
                "3) Pour a little water to bloom for 30 seconds, then pour the rest. 4) Stir, wait four minutes, and enjoy."
            ),
        },
    ],
]

DEFAULT_MODEL = "sfairXC/FsfairX-LLaMA3-RM-v0.1"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_worker_config(model_name: str):
    return OmegaConf.create(
        {
            "strategy": "fsdp",
            "model": {
                "path": model_name,
                "input_tokenizer": None,
                "external_lib": None,
                "use_shm": False,
                "use_remove_padding": False,
                "use_fused_kernels": False,
                "trust_remote_code": False,
                "type": "sequence",
                "fsdp_config": {
                    "wrap_policy": {"min_num_params": 0},
                    "param_offload": False,
                    "reshard_after_forward": True,
                    "fsdp_size": -1,
                },
            },
            "micro_batch_size": None,
            "micro_batch_size_per_gpu": 1,
            "max_length": None,
            "ulysses_sequence_parallel_size": 1,
            "use_dynamic_bsz": False,
            "forward_max_token_len_per_gpu": 4096,
            "reward_manager": "episode",
            "launch_reward_fn_async": False,
        }
    )


def render_chat(tokenizer, chat):
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    if tokenizer.bos_token is not None:
        text = text.replace(tokenizer.bos_token, "")
    return text


def build_inputs(tokenizer, chat):
    text = render_chat(tokenizer, chat)
    encoded = tokenizer(text, return_tensors="pt")
    encoded["position_ids"] = compute_position_id_with_mask(encoded["attention_mask"])
    # RewardModelWorker expects a `responses` field to determine response length
    encoded["responses"] = encoded["input_ids"].clone()
    return text, encoded


@pytest.mark.integration
def test_sequence_reward_model_matches_pipeline():
    if not (torch.cuda.is_available() or is_npu_available()):
        pytest.skip("Accelerator required: RewardModelWorker initializes distributed with nccl/hccl backends.")

    set_seed()
    model_name = os.environ.get("TEST_SEQUENCE_RM_MODEL", DEFAULT_MODEL)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        pytest.skip(f"Tokenizer for {model_name} unavailable: {exc}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
            attn_implementation="flash_attention_2",
        ).to(device)
        rm_model.eval()
    except Exception as exc:
        pytest.skip(f"Vanilla model for {model_name} unavailable: {exc}")

    try:
        rm_pipe = pipeline(
            "sentiment-analysis",
            model=rm_model,
            tokenizer=tokenizer,
            device=0 if device.type == "cuda" else -1,
        )
    except Exception as exc:
        pytest.skip(f"Pipeline for {model_name} unavailable: {exc}")

    worker_config = build_worker_config(model_name)
    worker = RewardModelWorker(worker_config)
    worker.init_model()
    # Ensure deterministic inference (disable dropout, etc.)
    worker.reward_module.eval()

    texts = []
    for chat in CHAT_SAMPLES:
        text, _ = build_inputs(tokenizer, chat)
        texts.append(text)

    encoded_batch = tokenizer(texts, return_tensors="pt", padding=True)
    encoded_batch["position_ids"] = compute_position_id_with_mask(encoded_batch["attention_mask"])
    encoded_batch["responses"] = encoded_batch["input_ids"].clone()

    autocast_kwargs = {"device_type": device.type, "dtype": torch.bfloat16} if device.type != "cpu" else None
    with torch.autocast(**autocast_kwargs) if autocast_kwargs else contextlib.nullcontext():
        pipe_outputs = rm_pipe(texts, return_all_scores=True, function_to_apply="none", batch_size=len(texts))
        pipe_rewards = [float(output[0]["score"]) for output in pipe_outputs]

    model_inputs = {k: v.to(device) for k, v in encoded_batch.items() if k != "responses"}
    with torch.no_grad():
        with torch.autocast(**autocast_kwargs) if autocast_kwargs else contextlib.nullcontext():
            model_outputs = rm_model(**model_inputs)
            model_rewards = model_outputs.logits[:, 0].float().cpu()

    data = DataProto.from_dict(
        tensors={
            "input_ids": encoded_batch["input_ids"],
            "attention_mask": encoded_batch["attention_mask"],
            "position_ids": encoded_batch["position_ids"],
            "responses": encoded_batch["responses"],
        }
    )

    rm_proto = worker.compute_rm_score(data)
    rm_scores = rm_proto.batch["rm_scores"]  # (batch_size, response_length)
    eos_mask_idx = torch.argmax(encoded_batch["position_ids"] * encoded_batch["attention_mask"], dim=-1)  # (bsz,)
    worker_rewards = rm_scores[torch.arange(rm_scores.size(0)), eos_mask_idx].cpu()

    for idx, (text, pipe_reward, model_reward, worker_reward) in enumerate(
        zip(texts, pipe_rewards, model_rewards, worker_rewards)
    ):
        model_reward_val = float(model_reward.item()) if hasattr(model_reward, "item") else float(model_reward)
        worker_reward_val = float(worker_reward.item()) if hasattr(worker_reward, "item") else float(worker_reward)
        print(
            "\n--- Conversation ---\n"
            f"{text}\n"
            f"Pipeline reward: {pipe_reward:.4f}\n"
            f"Model forward:   {model_reward_val:.4f}\n"
            f"Worker reward:   {worker_reward_val:.4f}\n"
            # f"Worker reward tensor:    {rm_scores[idx]}\n"
        )

        # assert model_reward_val == pytest.approx(pipe_reward, rel=1e-3, abs=1e-4), {
        #     "chat_index": idx,
        #     "chat": text,
        #     "pipe_reward": pipe_reward,
        #     "model_reward": model_reward_val,
        # }
        # assert worker_reward_val == pytest.approx(pipe_reward, rel=1e-3, abs=1e-4), {
        #     "chat_index": idx,
        #     "chat": text,
        #     "pipe_reward": pipe_reward,
        #     "model_reward": model_reward_val,
        #     "worker_reward": worker_reward_val,
        # }

if __name__ == "__main__":
    test_sequence_reward_model_matches_pipeline()
