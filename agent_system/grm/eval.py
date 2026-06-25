"""Metric helpers for GRM reward/evaluation outputs."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .reward import GrmRewardResult


def summarize_grm_results(
    results: Iterable[GrmRewardResult],
    *,
    prefix: str = "grm",
) -> dict[str, float]:
    rows = list(results)
    if not rows:
        return {}
    rewards = np.asarray([row.reward for row in rows], dtype=np.float32)
    return {
        f"{prefix}/reward_mean": float(np.mean(rewards)),
        f"{prefix}/exact_acc": float(np.mean([row.exact for row in rows])),
        f"{prefix}/invalid_format_rate": float(np.mean([row.invalid_format for row in rows])),
        f"{prefix}/wrong_profile_token_rate": float(np.mean([row.wrong_profile_token for row in rows])),
        f"{prefix}/adjacent_same_direction_rate": float(np.mean([row.adjacent_same_direction for row in rows])),
    }
