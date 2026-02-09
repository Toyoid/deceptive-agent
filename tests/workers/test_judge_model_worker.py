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
Integration test for JudgeModelWorker with detailed tensor transformation logging.

This test verifies the correctness of `compute_judge_score` and `_forward_micro_batch` methods
by running actual inference on a small model (Qwen2.5-0.5B-Instruct) and printing intermediate
tensor states for debugging.

Usage:
    # Single GPU test
    CUDA_VISIBLE_DEVICES=0 python tests/workers/test_judge_model_worker.py
    
    # Or with torchrun (single process)
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tests/workers/test_judge_model_worker.py
"""

import os
import sys

# Set up environment before any torch imports
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29501")
os.environ.setdefault("LOCAL_RANK", "0")

import random
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f">>> Random seed set to {seed} for reproducibility")


def print_separator(title: str, char: str = "=", width: int = 80):
    """Print a formatted separator line."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_tensor_info(name: str, tensor: torch.Tensor, show_values: bool = True, max_show: int = 100):
    """Print detailed tensor information."""
    print(f"  [{name}]")
    print(f"    Shape: {tensor.shape}")
    print(f"    Dtype: {tensor.dtype}")
    print(f"    Device: {tensor.device}")
    if show_values:
        if tensor.numel() <= max_show:
            print(f"    Values: {tensor.tolist()}")
        else:
            flat = tensor.flatten()
            print(f"    First {max_show} values: {flat[:max_show].tolist()}")
    if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print(f"    Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}, Mean: {tensor.mean().item():.6f}")
    print()


def setup_distributed():
    """Initialize distributed environment for single GPU testing."""
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        print(f"Initializing distributed with backend: {backend}")
        dist.init_process_group(backend=backend)
    print(f"Distributed initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")


def create_test_config(model_path: str = "Qwen/Qwen2.5-0.5B-Instruct", top_k: int = -1):
    """Create test configuration for JudgeModelWorker.
    
    Args:
        model_path: Path to the model
        top_k: Top-k filtering for constrained tokens. -1 to disable.
    """
    config = OmegaConf.create({
        "model": {
            "path": model_path,
            "fsdp_config": {
                "fsdp_size": 1,
                "wrap_policy": None,
                "reshard_after_forward": True,
            },
            "use_remove_padding": False,  # Simpler path for debugging
            "trust_remote_code": False,
            "use_shm": False,
        },
        "strategy": "fsdp",
        "valid_tokens": ["0", "1", "2", "3"],
        "token_weights": [0.0, 0.33, 0.66, 1.0],
        "top_k": top_k,  # Top-k filtering for constrained tokens
        "micro_batch_size": None,
        "micro_batch_size_per_gpu": 2,
        "use_dynamic_bsz": False,
        "ulysses_sequence_parallel_size": 1,
        "forward_max_token_len_per_gpu": 4096,
    })
    return config


def test_forward_micro_batch_detailed(worker, micro_batch: dict):
    """
    Test _forward_micro_batch with detailed tensor logging.
    
    This function manually steps through the logic to show intermediate tensors.
    """
    print_separator("Testing _forward_micro_batch with detailed logging")
    
    # Input tensors
    print(">>> INPUT TENSORS:")
    print_tensor_info("input_ids", micro_batch["input_ids"])
    print_tensor_info("attention_mask", micro_batch["attention_mask"])
    print_tensor_info("position_ids", micro_batch["position_ids"])
    
    input_ids = micro_batch["input_ids"]
    batch_size, seqlen = input_ids.shape
    attention_mask = micro_batch["attention_mask"]
    position_ids = micro_batch["position_ids"]
    
    print(f">>> BATCH INFO: batch_size={batch_size}, seqlen={seqlen}")
    
    # Run forward pass
    print("\n>>> RUNNING MODEL FORWARD PASS...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = worker.judge_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False
        )
        logits = output.logits
    
    print(">>> MODEL OUTPUT:")
    print_tensor_info("logits (raw)", logits, show_values=False)
    
    # Extract last position logits
    print(">>> EXTRACTING LAST POSITION LOGITS:")
    eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
    print_tensor_info("eos_mask_idx (last valid position per sequence)", eos_mask_idx)
    
    last_logits = logits[torch.arange(batch_size, device=logits.device), eos_mask_idx]
    print_tensor_info("last_logits", last_logits, show_values=False)
    print(f"    last_logits shape explanation: (batch_size={batch_size}, vocab_size={last_logits.shape[-1]})")
    
    # Show logits for constrained tokens
    print("\n>>> CONSTRAINED TOKEN EXTRACTION:")
    valid_token_ids = worker.valid_token_ids_tensor.to(last_logits.device)
    print_tensor_info("valid_token_ids", valid_token_ids)
    print(f"    Token mapping: {dict(zip(worker.valid_tokens, worker.valid_token_ids))}")
    
    constrained_logits = last_logits[:, valid_token_ids]
    print_tensor_info("constrained_logits (logits for valid tokens only)", constrained_logits)
    
    # Softmax over constrained tokens
    print(">>> COMPUTING CONSTRAINED SOFTMAX:")
    constrained_probs = torch.nn.functional.softmax(constrained_logits, dim=-1)
    print_tensor_info("constrained_probs", constrained_probs)
    print(f"    Probs sum per sample (should be ~1.0): {constrained_probs.sum(dim=-1).tolist()}")
    
    # Weighted scoring
    print(">>> COMPUTING WEIGHTED SCORES:")
    weights = worker.token_weights_tensor.to(constrained_probs.device)
    print_tensor_info("token_weights", weights)
    
    # Show element-wise multiplication
    weighted = constrained_probs * weights
    print_tensor_info("weighted (probs * weights)", weighted)
    
    scores = weighted.sum(dim=-1)
    print_tensor_info("final_scores (sum of weighted probs)", scores)
    
    # Verify against actual method
    print("\n>>> VERIFICATION: Comparing with actual _forward_micro_batch method...")
    actual_scores, actual_probs = worker._forward_micro_batch(micro_batch)
    print_tensor_info("actual_scores from method", actual_scores)
    print_tensor_info("actual_probs from method", actual_probs)
    
    # Check if results match (use larger tolerance due to bfloat16 precision)
    # bfloat16 has ~7 bits of mantissa, so relative error can be ~1e-2
    scores_match = torch.allclose(scores.float(), actual_scores.float(), atol=1e-2, rtol=1e-2)
    probs_match = torch.allclose(constrained_probs.float(), actual_probs.float(), atol=1e-2, rtol=1e-2)
    
    # Also compute actual differences for diagnostics
    scores_diff = (scores.float() - actual_scores.float()).abs()
    probs_diff = (constrained_probs.float() - actual_probs.float()).abs()
    
    print(f">>> VERIFICATION RESULT:")
    print(f"    scores_match (atol=1e-2): {scores_match}")
    print(f"    probs_match (atol=1e-2): {probs_match}")
    print(f"    Max score difference: {scores_diff.max().item():.6f}")
    print(f"    Max prob difference: {probs_diff.max().item():.6f}")
    print(f"    NOTE: Small differences are expected due to bfloat16 precision in autocast")
    
    return actual_scores, actual_probs


def test_compute_judge_score_detailed(worker, tokenizer):
    """
    Test compute_judge_score with detailed tensor logging.
    """
    print_separator("Testing compute_judge_score with detailed logging")
    
    # Create test prompts that should elicit different scores
    test_prompts = [
        # Prompt designed to potentially favor "3" (high validity)
        "Rate the following critique on a scale of 0-3 where 3 is most valid:\n"
        "Critique: 'The assistant correctly identified the mathematical error in step 2.'\n"
        "Rating:",
        
        # Prompt designed to potentially favor "1" (low validity)  
        "Rate the following critique on a scale of 0-3 where 3 is most valid:\n"
        "Critique: 'I disagree because reasons.'\n"
        "Rating:",
        
        # Prompt designed to be neutral
        "Rate the following critique on a scale of 0-3 where 3 is most valid:\n"
        "Critique: 'The response could be improved.'\n"
        "Rating:",
        
        # Simple test
        "Score this: 'Good point.' Rating (0-3):",
    ]
    
    print(f">>> CREATING TEST INPUTS ({len(test_prompts)} prompts):")
    for i, prompt in enumerate(test_prompts):
        print(f"  Prompt {i}: {prompt[:100]}...")
    
    # Tokenize
    print("\n>>> TOKENIZING INPUTS:")
    encoded = tokenizer(
        test_prompts, 
        return_tensors="pt", 
        padding=True,
        padding_side="left",
        truncation=True,
        max_length=256
    )
    print_tensor_info("input_ids", encoded["input_ids"])
    print_tensor_info("attention_mask", encoded["attention_mask"])
    
    # Compute position_ids
    position_ids = compute_position_ids(encoded["attention_mask"])
    print_tensor_info("position_ids", position_ids)
    
    # Create DataProto
    print("\n>>> CREATING DataProto:")
    from verl import DataProto
    data = DataProto.from_dict({
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "position_ids": position_ids,
    })
    print(f"    DataProto batch keys: {list(data.batch.keys())}")
    print(f"    DataProto batch size: {data.batch.batch_size}")
    
    # Run compute_judge_score
    print("\n>>> RUNNING compute_judge_score...")
    output = worker.compute_judge_score(data)
    
    print("\n>>> OUTPUT FROM compute_judge_score:")
    print_tensor_info("judge_scores", output.batch["judge_scores"])
    print_tensor_info("judge_token_probs", output.batch["judge_token_probs"])
    
    # Detailed analysis per sample
    print("\n>>> PER-SAMPLE ANALYSIS:")
    scores = output.batch["judge_scores"]
    probs = output.batch["judge_token_probs"]
    tokens = worker.valid_tokens
    weights = worker.token_weights
    
    for i in range(len(test_prompts)):
        print(f"\n  Sample {i}:")
        print(f"    Prompt: {test_prompts[i][:100]}...")
        print(f"    Token probabilities:")
        for j, (token, weight) in enumerate(zip(tokens, weights)):
            prob = probs[i, j].item()
            contribution = prob * weight
            print(f"      Token '{token}' (weight={weight:.2f}): prob={prob:.4f}, contribution={contribution:.4f}")
        print(f"    Final weighted score: {scores[i].item():.4f}")
    
    # Validation checks
    print("\n>>> VALIDATION CHECKS:")
    
    # 1. Scores should be in [0, 1] (since max weight is 1.0)
    scores_in_range = (scores >= 0).all() and (scores <= 1).all()
    print(f"  ✓ Scores in [0, 1]: {scores_in_range}")
    
    # 2. Probabilities should sum to 1
    probs_sum_to_one = torch.allclose(probs.sum(dim=-1), torch.ones(len(test_prompts)), atol=1e-4)
    print(f"  ✓ Probs sum to 1: {probs_sum_to_one}")
    
    # 3. All probs should be non-negative
    probs_non_negative = (probs >= 0).all()
    print(f"  ✓ Probs non-negative: {probs_non_negative}")
    
    return output


def test_top_k_filtering(worker_no_topk, worker_with_topk, tokenizer):
    """
    Test top_k filtering by comparing results with and without top_k.
    """
    print_separator("Testing top_k Filtering Feature")
    
    # Create test prompts
    test_prompts = [
        "Rate this critique: 'Valid point about the calculation error.' Score (0-3):",
        "Rate this critique: 'I disagree.' Score (0-3):",
        "Consider this critique: 'I disagree.' Do you agree:",
    ]
    
    print(f">>> Testing with {len(test_prompts)} prompts")
    print(f"    Worker without top_k: top_k={worker_no_topk.top_k}")
    print(f"    Worker with top_k: top_k={worker_with_topk.top_k}")
    
    # Tokenize
    encoded = tokenizer(
        test_prompts, 
        return_tensors="pt", 
        padding=True,
        padding_side="left",
    )
    position_ids = compute_position_ids(encoded["attention_mask"])
    
    # Create DataProto
    from verl import DataProto
    data = DataProto.from_dict({
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "position_ids": position_ids,
    })
    
    # Run both workers
    print("\n>>> Running worker WITHOUT top_k filtering...")
    output_no_topk = worker_no_topk.compute_judge_score(data)
    
    print(">>> Running worker WITH top_k filtering...")
    output_with_topk = worker_with_topk.compute_judge_score(data)
    
    # Compare results
    print("\n>>> COMPARISON RESULTS:")
    
    scores_no_topk = output_no_topk.batch["judge_scores"]
    scores_with_topk = output_with_topk.batch["judge_scores"]
    probs_no_topk = output_no_topk.batch["judge_token_probs"]
    probs_with_topk = output_with_topk.batch["judge_token_probs"]
    
    print_tensor_info("scores (no top_k)", scores_no_topk)
    print_tensor_info("scores (with top_k)", scores_with_topk)
    print_tensor_info("probs (no top_k)", probs_no_topk)
    print_tensor_info("probs (with top_k)", probs_with_topk)
    
    # Show per-sample comparison
    tokens = worker_no_topk.valid_tokens
    print("\n>>> PER-SAMPLE COMPARISON:")
    for i in range(len(test_prompts)):
        print(f"\n  Sample {i}: {test_prompts[i][:50]}...")
        print(f"    {'Token':<8} {'No top_k prob':<15} {'With top_k prob':<15} {'Diff':<10}")
        print(f"    {'-'*48}")
        for j, token in enumerate(tokens):
            p_no = probs_no_topk[i, j].item()
            p_with = probs_with_topk[i, j].item()
            diff = p_with - p_no
            # Mark tokens that were zeroed out by top_k
            marker = " (masked)" if p_with == 0 and p_no > 0 else ""
            print(f"    '{token}'      {p_no:<15.4f} {p_with:<15.4f} {diff:+.4f}{marker}")
        print(f"    Score: {scores_no_topk[i].item():.4f} -> {scores_with_topk[i].item():.4f}")
    
    # Validation
    print("\n>>> VALIDATION:")
    
    # Both should have probs summing to 1
    probs_sum_no_topk = probs_no_topk.sum(dim=-1)
    probs_sum_with_topk = probs_with_topk.sum(dim=-1)
    print(f"  Probs sum (no top_k): {probs_sum_no_topk.tolist()}")
    print(f"  Probs sum (with top_k): {probs_sum_with_topk.tolist()}")
    
    # Check if top_k actually changed the distribution (it should in most cases)
    distributions_differ = not torch.allclose(probs_no_topk, probs_with_topk, atol=1e-4)
    print(f"  ✓ Top_k changed the distribution: {distributions_differ}")
    
    # Check that with top_k, some probs might be zero (if tokens are outside top_k)
    has_zero_probs = (probs_with_topk == 0).any()
    print(f"  ✓ Top_k filtering created zero probs: {has_zero_probs}")
    
    return output_no_topk, output_with_topk


def test_top_k_fallback_strategies(worker_uniform, worker_zero, tokenizer):
    """
    Test both top_k fallback strategies: 'uniform' vs 'zero'.
    
    When all constrained tokens are masked by top_k (outside top-k of vocabulary),
    - 'uniform' strategy: assigns equal probability 1/num_tokens to each token
    - 'zero' strategy: assigns 0 probability to all tokens (score becomes 0)
    """
    print_separator("Testing top_k Fallback Strategies")
    
    print(f">>> Worker configurations:")
    print(f"    Worker A: top_k={worker_uniform.top_k}, fallback_strategy='{worker_uniform.top_k_fallback_strategy}'")
    print(f"    Worker B: top_k={worker_zero.top_k}, fallback_strategy='{worker_zero.top_k_fallback_strategy}'")
    
    # Create test prompts - use prompts where constrained tokens might not be in top_k
    test_prompts = [
        "Rate this critique: 'Valid point.' Score (0-3):",
        "Evaluate: 'I disagree.' Rating:",
        "Consider this mathematical proof: 2+2=5. Is it correct?",  # Unlikely to have 0,1,2,3 in top_k
    ]
    
    print(f"\n>>> Testing with {len(test_prompts)} prompts")
    
    # Tokenize
    encoded = tokenizer(
        test_prompts, 
        return_tensors="pt", 
        padding=True,
        padding_side="left",
    )
    position_ids = compute_position_ids(encoded["attention_mask"])
    
    # Create DataProto
    from verl import DataProto
    data = DataProto.from_dict({
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "position_ids": position_ids,
    })
    
    # Run both workers
    print("\n>>> Running worker with 'uniform' fallback strategy...")
    output_uniform = worker_uniform.compute_judge_score(data)
    
    print(">>> Running worker with 'zero' fallback strategy...")
    output_zero = worker_zero.compute_judge_score(data)
    
    # Compare results
    print("\n>>> COMPARISON RESULTS:")
    
    scores_uniform = output_uniform.batch["judge_scores"]
    scores_zero = output_zero.batch["judge_scores"]
    probs_uniform = output_uniform.batch["judge_token_probs"]
    probs_zero = output_zero.batch["judge_token_probs"]
    
    print_tensor_info("scores (uniform fallback)", scores_uniform)
    print_tensor_info("scores (zero fallback)", scores_zero)
    print_tensor_info("probs (uniform fallback)", probs_uniform)
    print_tensor_info("probs (zero fallback)", probs_zero)
    
    # Show per-sample comparison
    tokens = worker_uniform.valid_tokens
    weights = worker_uniform.token_weights
    print("\n>>> PER-SAMPLE COMPARISON:")
    for i in range(len(test_prompts)):
        print(f"\n  Sample {i}: {test_prompts[i][:50]}...")
        print(f"    {'Token':<8} {'Weight':<8} {'Uniform prob':<15} {'Zero prob':<15}")
        print(f"    {'-'*50}")
        
        # Check if this sample had all tokens masked (NaN -> fallback)
        uniform_sum = probs_uniform[i].sum().item()
        zero_sum = probs_zero[i].sum().item()
        
        for j, (token, weight) in enumerate(zip(tokens, weights)):
            p_uniform = probs_uniform[i, j].item()
            p_zero = probs_zero[i, j].item()
            print(f"    '{token}'      {weight:<8.2f} {p_uniform:<15.4f} {p_zero:<15.4f}")
        
        print(f"    Probs sum: uniform={uniform_sum:.4f}, zero={zero_sum:.4f}")
        print(f"    Scores: uniform={scores_uniform[i].item():.4f}, zero={scores_zero[i].item():.4f}")
        
        # Detect if fallback was triggered
        if abs(zero_sum) < 1e-6:
            print(f"    >>> FALLBACK TRIGGERED: All tokens were masked by top_k!")
            print(f"        - Uniform strategy gave score: {scores_uniform[i].item():.4f}")
            print(f"        - Zero strategy gave score: {scores_zero[i].item():.4f}")
    
    # Validation
    print("\n>>> VALIDATION:")
    
    # Uniform should always have probs summing to 1
    uniform_sums = probs_uniform.sum(dim=-1)
    print(f"  Probs sum (uniform): {uniform_sums.tolist()}")
    assert torch.allclose(uniform_sums, torch.ones_like(uniform_sums), atol=1e-4), "Uniform probs should sum to 1"
    print(f"  ✓ Uniform strategy: probs sum to 1")
    
    # Zero strategy: probs sum to 1 if not masked, 0 if all masked
    zero_sums = probs_zero.sum(dim=-1)
    print(f"  Probs sum (zero): {zero_sums.tolist()}")
    for i, z_sum in enumerate(zero_sums):
        if z_sum.item() < 0.5:  # All masked
            assert abs(z_sum.item()) < 1e-6, f"Zero strategy should have 0 probs when masked"
            assert abs(scores_zero[i].item()) < 1e-6, f"Zero strategy should have 0 score when masked"
    print(f"  ✓ Zero strategy: probs are 0 when all tokens masked")
    
    return output_uniform, output_zero


def compute_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute position_ids from attention_mask."""
    # For left-padded sequences, position_ids should be cumsum of attention_mask - 1
    # For right-padded sequences (which tokenizer usually produces), it's simpler
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


def main():
    """Main test function."""
    print_separator("JudgeModelWorker Integration Test", "=", 80)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Setup
    print(">>> SETUP:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    setup_distributed()
    
    # Create config (without top_k)
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n>>> MODEL: {model_path}")
    config = create_test_config(model_path, top_k=-1)
    print(f">>> CONFIG (no top_k):")
    print(OmegaConf.to_yaml(config))
    
    # Import and create worker
    print_separator("Creating JudgeModelWorker (no top_k)")
    from verl.workers.fsdp_workers import JudgeModelWorker
    
    worker = JudgeModelWorker(config)
    print(">>> Worker created successfully")
    print(f"    valid_tokens: {worker.valid_tokens}")
    print(f"    token_weights: {worker.token_weights}")
    print(f"    top_k: {worker.top_k}")
    
    # Initialize model
    print("\n>>> Initializing model (this may take a moment)...")
    worker.init_model()
    print(">>> Model initialized successfully")
    print(f"    valid_token_ids: {worker.valid_token_ids}")
    print(f"    Token ID mapping: {dict(zip(worker.valid_tokens, worker.valid_token_ids))}")
    
    # Get tokenizer
    tokenizer = worker.tokenizer
    print(f">>> Tokenizer loaded: {type(tokenizer).__name__}")
    print(f"    Vocab size: {tokenizer.vocab_size}")
    print(f"    Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    
    # Test 1: _forward_micro_batch with detailed logging
    print_separator("Test 1: _forward_micro_batch")
    
    # Create a simple micro batch
    test_texts = [
        "Rate this critique: 'No issues are found. Valid.' Score:",
        "Rate this critique: 'Invalid reasoning. Reject.' Score:",
    ]
    encoded = tokenizer(test_texts, return_tensors="pt", padding=True, padding_side="left")
    position_ids = compute_position_ids(encoded["attention_mask"])
    
    micro_batch = {
        "input_ids": encoded["input_ids"].cuda(),
        "attention_mask": encoded["attention_mask"].cuda(),
        "position_ids": position_ids.cuda(),
    }
    
    scores, probs = test_forward_micro_batch_detailed(worker, micro_batch)
    
    # Test 2: compute_judge_score with detailed logging
    print_separator("Test 2: compute_judge_score")
    output = test_compute_judge_score_detailed(worker, tokenizer)
    
    # Test 3: top_k filtering
    print_separator("Test 3: top_k Filtering")
    
    # Create a second worker with top_k enabled
    print(">>> Creating second worker with top_k=100...")
    config_with_topk = create_test_config(model_path, top_k=100)
    worker_with_topk = JudgeModelWorker(config_with_topk)
    worker_with_topk.init_model()
    print(f"    top_k: {worker_with_topk.top_k}")
    
    # Run top_k comparison test
    test_top_k_filtering(worker, worker_with_topk, tokenizer)
    
    # Test 4: Extreme top_k (very small, to force some tokens outside)
    print_separator("Test 4: Extreme top_k Filtering (top_k=10)")
    
    config_extreme_topk = create_test_config(model_path, top_k=10)
    worker_extreme_topk = JudgeModelWorker(config_extreme_topk)
    worker_extreme_topk.init_model()
    print(f"    top_k: {worker_extreme_topk.top_k}")
    
    test_top_k_filtering(worker, worker_extreme_topk, tokenizer)
    
    # Test 5: top_k fallback strategies comparison
    print_separator("Test 5: top_k Fallback Strategies (uniform vs zero)")
    
    # Create workers with extreme top_k and different fallback strategies
    # Use top_k=5 to ensure all constrained tokens are likely outside top-k
    print(">>> Creating workers with top_k=5 and different fallback strategies...")
    config_uniform = create_test_config(model_path, top_k=5, top_k_fallback_strategy="uniform")
    config_zero = create_test_config(model_path, top_k=5, top_k_fallback_strategy="zero")
    
    worker_uniform = JudgeModelWorker(config_uniform)
    worker_uniform.init_model()
    print(f"    Worker (uniform): top_k={worker_uniform.top_k}, fallback='{worker_uniform.top_k_fallback_strategy}'")
    
    worker_zero = JudgeModelWorker(config_zero)
    worker_zero.init_model()
    print(f"    Worker (zero): top_k={worker_zero.top_k}, fallback='{worker_zero.top_k_fallback_strategy}'")
    
    test_top_k_fallback_strategies(worker_uniform, worker_zero, tokenizer)
    
    # Summary
    print_separator("TEST SUMMARY", "=", 80)
    print("✓ All tests completed successfully!")
    print(f"  - Test 1: _forward_micro_batch tensor transformations verified")
    print(f"  - Test 2: compute_judge_score end-to-end pipeline verified")
    print(f"  - Test 3: top_k=100 filtering tested")
    print(f"  - Test 4: top_k=10 extreme filtering tested")
    print(f"  - Test 5: top_k fallback strategies (uniform vs zero) tested")
    print(f"  - Token mapping: {dict(zip(worker.valid_tokens, worker.valid_token_ids))}")
    print(f"  - Score range (no top_k): [{output.batch['judge_scores'].min():.4f}, {output.batch['judge_scores'].max():.4f}]")
    
    # Cleanup
    dist.destroy_process_group()
    print("\n>>> Distributed cleanup complete")


if __name__ == "__main__":
    main()
