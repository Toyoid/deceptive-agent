"""Metric helpers for GRM reward/evaluation outputs."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .reward import GrmRewardResult


_STRICT5_TOKENS = ("0", "1", "2", "3", "4")
_NEGATIVE_TOKENS = {"0", "1", "2"}
_POSITIVE_TOKENS = {"3", "4"}


def _strict5_confusion_metrics(rows: list[GrmRewardResult], prefix: str) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for row in rows:
        if tuple(row.valid_tokens) != _STRICT5_TOKENS:
            continue
        if row.parsed_token is None:
            continue
        if row.label not in _NEGATIVE_TOKENS | _POSITIVE_TOKENS:
            continue
        if row.parsed_token not in _NEGATIVE_TOKENS | _POSITIVE_TOKENS:
            continue

        label_positive = row.label in _POSITIVE_TOKENS
        pred_positive = row.parsed_token in _POSITIVE_TOKENS
        if label_positive and pred_positive:
            tp += 1
        elif not label_positive and pred_positive:
            fp += 1
        elif label_positive and not pred_positive:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    if total == 0:
        return {
            f"{prefix}/confusion/tp_rate": 0.0,
            f"{prefix}/confusion/fp_rate": 0.0,
            f"{prefix}/confusion/fn_rate": 0.0,
            f"{prefix}/confusion/tn_rate": 0.0,
        }
    return {
        f"{prefix}/confusion/tp_rate": float(tp / total),
        f"{prefix}/confusion/fp_rate": float(fp / total),
        f"{prefix}/confusion/fn_rate": float(fn / total),
        f"{prefix}/confusion/tn_rate": float(tn / total),
    }


def summarize_grm_results(
    results: Iterable[GrmRewardResult],
    *,
    prefix: str = "grm",
) -> dict[str, float]:
    rows = list(results)
    if not rows:
        return {}
    rewards = np.asarray([row.reward for row in rows], dtype=np.float32)
    metrics = {
        f"{prefix}/reward_mean": float(np.mean(rewards)),
        f"{prefix}/exact_acc": float(np.mean([row.exact for row in rows])),
        f"{prefix}/invalid_format_rate": float(np.mean([row.invalid_format for row in rows])),
        f"{prefix}/wrong_profile_token_rate": float(np.mean([row.wrong_profile_token for row in rows])),
        f"{prefix}/adjacent_same_direction_rate": float(np.mean([row.adjacent_same_direction for row in rows])),
    }
    metrics.update(_strict5_confusion_metrics(rows, prefix))
    return metrics
