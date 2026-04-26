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

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from omegaconf import ListConfig


@dataclass
class EvalRow:
    data_source: str
    prompt: List[Dict[str, str]]
    env_kwargs: Optional[Dict[str, Any]]
    index: int
    ability: Optional[Any] = None
    reward_model: Optional[Any] = None
    extra_info: Optional[Any] = None
    metadata: Optional[Any] = None
    source_file: Optional[str] = None
    rollout_index: int = 0


def _as_file_list(files: Any) -> List[str]:
    if files is None:
        return []
    if isinstance(files, (str, Path)):
        return [str(files)]
    if isinstance(files, ListConfig):
        return [str(path) for path in files]
    if isinstance(files, Iterable):
        return [str(path) for path in files]
    raise TypeError(f"Unsupported data.files type: {type(files)}")


def _normalize_nested(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_normalize_nested(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_normalize_nested(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_nested(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_nested(item) for key, item in value.items()}
    if _is_missing_scalar(value):
        return None
    return value


def _is_missing_scalar(value: Any) -> bool:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(np.isscalar(value) and np.isnan(value))
    except TypeError:
        return False


def _normalize_prompt(value: Any) -> List[Dict[str, str]]:
    value = _normalize_nested(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [{"role": "user", "content": value}]
        value = parsed
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"prompt must be a chat-message list or string, got {type(value)}")
    messages: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"prompt message must be a dict, got {type(item)}")
        role = str(item.get("role", "user"))
        content = item.get("content", "")
        messages.append({"role": role, "content": "" if content is None else str(content)})
    return messages


def _normalize_env_kwargs(value: Any) -> Optional[Dict[str, Any]]:
    value = _normalize_nested(value)
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError("env_kwargs string must be valid JSON")
    if not isinstance(value, dict):
        raise ValueError(f"env_kwargs must be a dict or null, got {type(value)}")
    return value


def _normalize_metadata_field(value: Any) -> Any:
    value = _normalize_nested(value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _index_from_extra_info(extra_info: Any, fallback: int) -> int:
    if isinstance(extra_info, dict) and "index" in extra_info:
        try:
            return int(extra_info["index"])
        except (TypeError, ValueError):
            return fallback
    return fallback


def _row_to_eval_row(row: Dict[str, Any], *, local_idx: int, source_file: str) -> EvalRow:
    extra_info = _normalize_metadata_field(row.get("extra_info"))
    return EvalRow(
        data_source=str(row["data_source"]),
        prompt=_normalize_prompt(row["prompt"]),
        env_kwargs=_normalize_env_kwargs(row["env_kwargs"]),
        index=_index_from_extra_info(extra_info, local_idx),
        ability=_normalize_metadata_field(row.get("ability")),
        reward_model=_normalize_metadata_field(row.get("reward_model")),
        extra_info=extra_info,
        metadata=_normalize_metadata_field(row.get("metadata")),
        source_file=str(source_file),
    )


def _load_parquet_rows(config) -> List[EvalRow]:
    import datasets
    from verl.utils.fs import copy_to_local

    files = _as_file_list(config.data.get("files"))
    if not files:
        raise ValueError("data.files must be set for ReasonChat and deceptive_search API eval.")

    cache_dir = os.path.expanduser(config.data.get("cache_dir", "~/.cache/verl/rlhf"))
    use_shm = bool(config.data.get("use_shm", False))
    required = {"data_source", "prompt", "env_kwargs"}
    rows: List[EvalRow] = []
    for file_name in files:
        local_file = copy_to_local(src=file_name, cache_dir=cache_dir, use_shm=use_shm)
        dataset = datasets.load_dataset("parquet", data_files=local_file)["train"]
        missing = required - set(dataset.column_names)
        if missing:
            raise ValueError(f"{file_name} is missing required columns: {sorted(missing)}")
        for local_idx, row in enumerate(dataset):
            rows.append(_row_to_eval_row(row, local_idx=local_idx, source_file=file_name))
    return rows


def _load_cheatshop_rows(config) -> List[EvalRow]:
    num_episodes = config.data.get("num_episodes")
    if num_episodes is None:
        raise ValueError("data.num_episodes must be set for CheatShop API eval. data.files is ignored.")
    num_episodes = int(num_episodes)
    if num_episodes <= 0:
        raise ValueError("data.num_episodes must be positive for CheatShop API eval.")
    return [
        EvalRow(data_source="cheatshop", prompt=[], env_kwargs=None, index=i)
        for i in range(num_episodes)
    ]


def _shuffle_and_limit(rows: List[EvalRow], config) -> List[EvalRow]:
    rows = list(rows)
    if bool(config.data.get("shuffle", False)):
        rng = np.random.RandomState(int(config.data.get("seed", 42)))
        order = rng.permutation(len(rows))
        rows = [rows[int(i)] for i in order]
    max_samples = config.data.get("max_samples")
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    return rows


def _expand_rollouts(rows: List[EvalRow], config) -> List[EvalRow]:
    rollout_n = int(config.env.get("rollout", {}).get("n", 1))
    if rollout_n <= 1:
        return rows
    expanded: List[EvalRow] = []
    for row in rows:
        for rollout_index in range(rollout_n):
            expanded.append(
                replace(
                    row,
                    prompt=copy.deepcopy(row.prompt),
                    env_kwargs=copy.deepcopy(row.env_kwargs),
                    ability=copy.deepcopy(row.ability),
                    reward_model=copy.deepcopy(row.reward_model),
                    extra_info=copy.deepcopy(row.extra_info),
                    metadata=copy.deepcopy(row.metadata),
                    rollout_index=rollout_index,
                )
            )
    return expanded


def load_eval_rows(config) -> List[EvalRow]:
    env_name = str(config.env.env_name).lower()
    if "cheatshop" in env_name:
        rows = _load_cheatshop_rows(config)
    elif "reasonchat" in env_name or "reason_chat" in env_name or "deceptive_search" in env_name:
        rows = _load_parquet_rows(config)
    else:
        raise ValueError(
            f"Unsupported env.env_name={config.env.env_name!r}. "
            "Supported API eval envs: ReasonChat, deceptive_search, CheatShop."
        )
    return _expand_rollouts(_shuffle_and_limit(rows, config), config)
