from __future__ import annotations

from typing import Any

import numpy as np


def _as_bool_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=bool)


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

    return {
        f"{prefix}valid_step_ratio": float(valid.astype(np.float32).mean()),
        f"{prefix}unsafe_step_ratio": float(unsafe.astype(np.float32).mean()),
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

    metrics: dict[str, float] = {}
    for data_source in np.unique(data_sources):
        step_mask = data_sources == data_source
        base_prefix = f"{prefix}/{data_source}/self_monitor/"
        metrics[f"{base_prefix}valid_step_ratio"] = float(valid[step_mask].astype(np.float32).mean())
        metrics[f"{base_prefix}unsafe_step_ratio"] = float(unsafe[step_mask].astype(np.float32).mean())

    return metrics
