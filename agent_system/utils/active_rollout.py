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

from dataclasses import dataclass
from typing import Any, Iterator, List, Sequence

import numpy as np


@dataclass(frozen=True)
class ActiveIndexMap:
    """Map compact active rows back to stable full-batch environment slots."""

    active_idx: np.ndarray
    batch_size: int

    @classmethod
    def from_done(cls, is_done: Sequence[bool]) -> "ActiveIndexMap":
        done = np.asarray(is_done, dtype=bool).reshape(-1)
        return cls(active_idx=np.flatnonzero(~done), batch_size=int(done.shape[0]))

    @property
    def n_active(self) -> int:
        return int(self.active_idx.shape[0])

    @property
    def has_active(self) -> bool:
        return self.n_active > 0

    def active_mask(self) -> np.ndarray:
        mask = np.zeros(self.batch_size, dtype=bool)
        mask[self.active_idx] = True
        return mask

    def active_flags(self, *, dtype=object) -> np.ndarray:
        return np.full(self.n_active, True, dtype=dtype)

    def select_array(self, values: Any, *, dtype=None) -> np.ndarray:
        array = np.asarray(values, dtype=dtype)
        return array[self.active_idx]

    def select_list(self, values: Sequence[Any]) -> List[Any]:
        return [values[int(index)] for index in self.active_idx]

    def select_info_values(self, infos: Sequence[dict], key: str, *, dtype=object) -> np.ndarray:
        return np.asarray([infos[int(index)][key] for index in self.active_idx], dtype=dtype)

    def iter_active(self) -> Iterator[tuple[int, int]]:
        for local_i, global_i in enumerate(self.active_idx):
            yield local_i, int(global_i)

    def scatter_actions(self, active_actions: Sequence[str], previous_actions: Sequence[str]) -> List[str]:
        if len(previous_actions) != self.batch_size:
            raise ValueError(
                f"previous_actions length {len(previous_actions)} does not match batch_size {self.batch_size}."
            )
        if len(active_actions) != self.n_active:
            raise ValueError(f"Got {len(active_actions)} active actions for {self.n_active} active slots.")

        full_actions = list(previous_actions)
        for local_i, global_i in self.iter_active():
            full_actions[global_i] = active_actions[local_i]
        return full_actions

    def append_active_records(
        self,
        *,
        total_batch_list: List[List[dict]],
        total_infos: List[List[dict]],
        active_records: Sequence[dict],
        infos: Sequence[dict],
    ) -> None:
        if len(active_records) != self.n_active:
            raise ValueError(f"Got {len(active_records)} active records for {self.n_active} active slots.")
        for local_i, env_i in self.iter_active():
            total_batch_list[env_i].append(active_records[local_i])
            total_infos[env_i].append(infos[env_i])
