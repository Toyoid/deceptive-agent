"""
Device handling utilities for multi-GPU inference.

Avoids torch.distributed complexity for inference-only workloads by using
explicit device_map="balanced" with per-GPU max_memory constraints.

Usage:
    device_kwargs = get_device_kwargs(args.device)
    model = AutoModelForCausalLM.from_pretrained(model_id, **device_kwargs, ...)
"""

import torch
from typing import Dict, Any


def get_device_kwargs(device_arg: str = "auto") -> Dict[str, Any]:
    """
    Return a dict of device-related kwargs to unpack into from_pretrained().

    Args:
        device_arg: User-provided device string. Accepted values:
            "auto"   — detect GPUs and distribute across all of them (default)
            "cuda"   — same as "auto"
            "cuda:N" — pin to a single GPU (e.g., "cuda:0", "cuda:1")
            "cpu"    — force CPU

    Returns:
        Dict with "device_map" and optionally "max_memory" keys.
        Unpack directly: from_pretrained(model_id, **get_device_kwargs(arg), ...)

    Notes:
        - Uses device_map="balanced" for multi-GPU to avoid torch.distributed.
        - device_map="auto" in new transformers versions may activate tensor
          parallelism which requires LOCAL_RANK env var — this function avoids that.
    """
    # Explicit single-GPU or CPU
    if device_arg == "cpu":
        return {"device_map": "cpu"}

    if device_arg.startswith("cuda:"):
        return {"device_map": device_arg}

    # "auto" or "cuda": inspect available GPUs
    if not torch.cuda.is_available():
        print("No CUDA GPUs detected, falling back to CPU.")
        return {"device_map": "cpu"}

    num_gpus = torch.cuda.device_count()

    if num_gpus == 1:
        print("Single GPU detected: using cuda:0")
        return {"device_map": "cuda:0"}

    # Multi-GPU: use "balanced" device_map + explicit max_memory per GPU.
    # "balanced" distributes layers evenly across GPUs without torch.distributed.
    # max_memory caps each GPU to leave ~1 GiB headroom for activations.
    max_memory: Dict[int, str] = {}
    for i in range(num_gpus):
        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        usable_gib = max(1, int(total_gib) - 1)
        max_memory[i] = f"{usable_gib}GiB"

    print(f"Multi-GPU detected ({num_gpus} GPUs). Using balanced distribution:")
    for gpu_id, mem in max_memory.items():
        name = torch.cuda.get_device_properties(gpu_id).name
        print(f"  GPU {gpu_id} ({name}): capped at {mem}")

    return {"device_map": "balanced", "max_memory": max_memory}
