"""
Device handling utilities for multi-GPU inference.

Avoids torch.distributed complexity for inference-only workloads by using
explicit device mapping or memory-based distribution.
"""

import os
import torch
from typing import Optional, Dict, Union


def get_device_map(device_arg: str = "auto", model_size_gb: Optional[float] = None) -> Union[str, Dict]:
    """
    Get an appropriate device_map for model loading.
    
    Args:
        device_arg: User-provided device argument (e.g., "auto", "cuda:0", "cuda")
        model_size_gb: Estimated model size in GB (for memory-based distribution)
        
    Returns:
        device_map suitable for transformers.from_pretrained()
    """
    
    # Explicit GPU specification (e.g., "cuda:0", "cuda:1")
    if device_arg.startswith("cuda:"):
        return device_arg
    
    # CPU-only
    if device_arg == "cpu":
        return "cpu"
    
    # For "auto" or "cuda", check available GPUs
    if device_arg in ("auto", "cuda"):
        if not torch.cuda.is_available():
            return "cpu"
        
        num_gpus = torch.cuda.device_count()
        
        # Single GPU: use standard device_map
        if num_gpus == 1:
            return "cuda:0"
        
        # Multiple GPUs: use max_memory strategy (avoids distributed setup)
        # This distributes the model across GPUs by available memory.
        gpu_memory = {}
        for i in range(num_gpus):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
            # Leave 1 GB headroom per GPU
            gpu_memory[i] = f"{max(1, int(total_memory - 1))}GB"
        
        print(f"Multi-GPU detected ({num_gpus} GPUs). Using memory-based distribution:")
        for gpu_id, mem in gpu_memory.items():
            print(f"  GPU {gpu_id}: {mem}")
        
        return {"": gpu_memory}  # Empty string captures CPU offloading
    
    # Fallback
    return device_arg


def initialize_distributed_if_needed():
    """
    Initialize torch.distributed if running in a distributed context.
    
    Safely checks for distributed environment variables (RANK, WORLD_SIZE, etc.)
    and initializes if present. This allows scripts to work in both single and
    multi-process contexts without forcing distributed initialization.
    """
    # Check if we're in a distributed launcher context
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    local_rank = os.environ.get("LOCAL_RANK")
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    
    # Only initialize if all distributed env vars are present
    if all([rank, world_size, local_rank, master_addr, master_port]):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
            print(f"Initialized distributed (rank {rank}/{world_size}, local_rank {local_rank})")
