from __future__ import annotations

from typing import Any

import numpy as np


def _as_bool_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=bool)


def _aggregate_unsafe_by_traj(
    is_unsafe: Any,
    traj_uids: Any,
    trust_penalties: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traj_uids = np.asarray(traj_uids, dtype=object)
    if traj_uids.size == 0:
        empty = np.asarray([], dtype=bool)
        return np.asarray([], dtype=object), np.asarray([], dtype=np.int64), empty

    _, first_idx, inverse = np.unique(traj_uids, return_index=True, return_inverse=True)
    ordered_pos = np.argsort(first_idx)
    first_idx = first_idx[ordered_pos]

    if trust_penalties is not None:
        penalties = np.asarray(trust_penalties, dtype=np.float32)
        traj_unsafe = penalties[first_idx] > 0.0
    else:
        step_unsafe = _as_bool_array(is_unsafe)
        unordered_traj_unsafe = np.zeros(len(first_idx), dtype=bool)
        for idx, unsafe in zip(inverse, step_unsafe):
            unordered_traj_unsafe[idx] = unordered_traj_unsafe[idx] or bool(unsafe)
        traj_unsafe = unordered_traj_unsafe[ordered_pos]

    unique_traj_uids = traj_uids[first_idx]
    return unique_traj_uids, first_idx, traj_unsafe


def compute_self_monitor_metrics(
    is_valid: Any,
    is_unsafe: Any,
    traj_uids: Any,
    trust_penalties: Any = None,
    prefix: str = "self_monitor/",
) -> dict[str, float]:
    valid = _as_bool_array(is_valid)
    unsafe = _as_bool_array(is_unsafe)

    if valid.size == 0:
        return {}

    _, _, traj_unsafe = _aggregate_unsafe_by_traj(unsafe, traj_uids, trust_penalties=trust_penalties)
    return {
        f"{prefix}valid_step_ratio": float(valid.astype(np.float32).mean()),
        f"{prefix}unsafe_step_ratio": float(unsafe.astype(np.float32).mean()),
        f"{prefix}unsafe_traj_ratio": float(traj_unsafe.astype(np.float32).mean()) if traj_unsafe.size > 0 else 0.0,
    }


def compute_self_monitor_metrics_by_source(
    data_sources: Any,
    is_valid: Any,
    is_unsafe: Any,
    traj_uids: Any,
    trust_penalties: Any = None,
    prefix: str = "val",
) -> dict[str, float]:
    data_sources = np.asarray(data_sources, dtype=object)
    valid = _as_bool_array(is_valid)
    unsafe = _as_bool_array(is_unsafe)

    if data_sources.size == 0:
        return {}

    _, first_idx, traj_unsafe = _aggregate_unsafe_by_traj(unsafe, traj_uids, trust_penalties=trust_penalties)
    unique_sources = data_sources[first_idx]

    metrics: dict[str, float] = {}
    for data_source in np.unique(data_sources):
        step_mask = data_sources == data_source
        traj_mask = unique_sources == data_source
        base_prefix = f"{prefix}/{data_source}/self_monitor/"
        metrics[f"{base_prefix}valid_step_ratio"] = float(valid[step_mask].astype(np.float32).mean())
        metrics[f"{base_prefix}unsafe_step_ratio"] = float(unsafe[step_mask].astype(np.float32).mean())
        metrics[f"{base_prefix}unsafe_traj_ratio"] = (
            float(traj_unsafe[traj_mask].astype(np.float32).mean()) if np.any(traj_mask) else 0.0
        )
    return metrics
