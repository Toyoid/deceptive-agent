from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def constrained_probs_to_binary_penalties(
    constrained_token_probs: Any,
    valid_tokens: Sequence[str],
) -> np.ndarray:
    probs = constrained_token_probs.detach().cpu().numpy() if hasattr(constrained_token_probs, "detach") else np.asarray(constrained_token_probs)
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(f"constrained_token_probs must be rank-2, got shape {probs.shape}")

    tokens = list(valid_tokens)
    if len(tokens) != 2 or set(tokens) != {"0", "1"}:
        raise ValueError(f"valid_tokens must be exactly ['0', '1'] in any order, got {tokens}")

    argmax_idx = np.argmax(probs, axis=-1)
    penalties = np.asarray([1.0 if tokens[idx] == "1" else 0.0 for idx in argmax_idx], dtype=np.float32)
    return penalties
