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

import os
from typing import Any, Dict, Optional, Tuple

import math
import torch

from verl import DataProto


class RewardNormalizer:
    """
    Streaming z-score normalizer for reward model outputs.

    - update(): stream in raw scalars
    - finalize(): keep accumulated mean/std
    - normalize(): apply fixed stats (no running updates)

    NOTE: The class is not thread-safe. Concurrent calls to update() would corrupt the statistics.
    """

    def __init__(self, clip: Optional[float] = None, eps: float = 1e-6):
        self.clip = clip
        self.eps = eps
        self.reset()

    def reset(self):
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._finalized = False

    @property
    def count(self) -> int:
        return self._count

    def update(self, values: torch.Tensor):
        """Batch Welford streaming update.
        
        Keeps computation on the original device (GPU) for efficiency.
        Only scalar results are transferred to CPU via .item().
        """
        flat = values.detach().float().view(-1)
        batch_count = flat.numel()
        if batch_count == 0:
            return
        
        # only transfer 2 scalars to CPU
        batch_sum = flat.sum().item()
        batch_sumsq = torch.dot(flat, flat).item()
        batch_mean = batch_sum / batch_count
        # sum of squared deviations for the batch
        batch_m2 = batch_sumsq - batch_sum * batch_sum / batch_count

        total_count = self._count + batch_count
        delta = batch_mean - self._mean
        new_mean = self._mean + delta * batch_count / total_count
        # Parallel variance merge
        self._m2 = self._m2 + batch_m2 + delta * delta * self._count * batch_count / total_count
        self._mean = new_mean
        self._count = total_count
        self._finalized = False

    def _compute_std(self) -> float:
        if self._count <= 1:
            return 1.0
        return math.sqrt(self._m2 / max(self._count - 1, 1))

    def finalize(self):
        """Mark stats as frozen."""
        self._finalized = True

    def stats(self):
        std = self._compute_std()
        return {"mean": self._mean, "std": std, "count": self._count}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self._mean,
            "m2": self._m2,
            "count": self._count,
            "clip": self.clip,
            "eps": self.eps,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self._mean = state["mean"]
        self._m2 = state["m2"]
        self._count = state["count"]
        self.clip = state.get("clip", None)
        self.eps = state.get("eps", 1e-6)
        self._finalized = True

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Apply fixed z-score using current stats (assumes finalized)."""
        std = self._compute_std()
        mean_t = torch.tensor(self._mean, device=values.device, dtype=values.dtype)
        std_t = torch.tensor(std, device=values.device, dtype=values.dtype)
        normed = (values.float() - mean_t) / (std_t + self.eps)
        if self.clip is not None:
            normed = torch.clamp(normed, -self.clip, self.clip)
        return normed


def get_dataset_name(data_config) -> str:
    train_files = data_config.train_files
    if isinstance(train_files, (list, tuple)):
        primary = train_files[0]
    else:
        primary = str(train_files)
        if "," in primary:
            primary = primary.split(",")[0]
    return primary


def get_rm_norm_stats_path(norm_cfg, trainer_cfg) -> str:
    if norm_cfg.get("stats_path"):
        return norm_cfg.stats_path
    return os.path.join(trainer_cfg.default_local_dir, "reward_normalizer.pt")


def load_rm_normalizer_if_available(stats_path: str, default_clip: Optional[float], default_eps: float) -> Tuple[Optional[RewardNormalizer], Optional[Dict[str, Any]]]:
    if stats_path is None or not os.path.exists(stats_path):
        return None, None
    payload = torch.load(stats_path, weights_only=False)
    state = payload.get("state", payload)
    meta = payload.get("meta", {})
    normalizer = RewardNormalizer(clip=state.get("clip", default_clip), eps=state.get("eps", default_eps))
    normalizer.load_state_dict(state)
    return normalizer, meta


def save_rm_normalizer(normalizer: RewardNormalizer, meta: Dict[str, Any], stats_path: str):
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    payload = {"state": normalizer.state_dict(), "meta": meta}
    torch.save(payload, stats_path)


def apply_rm_normalization(
    rm_scores_proto: DataProto, 
    response_mask: torch.Tensor,
    normalizer: Optional[RewardNormalizer]
) -> Tuple[DataProto, Optional[torch.Tensor], Optional[torch.Tensor]]:
    rm_scores = rm_scores_proto.batch["rm_scores"]
    valid_len = response_mask.sum(dim=1).long()
    last_idx = torch.clamp(valid_len - 1, min=0)
    raw_scalar = (rm_scores * response_mask).sum(dim=-1)
    if normalizer is None:
        return rm_scores_proto, raw_scalar, None

    normed_scalar = normalizer.normalize(raw_scalar)

    new_scores = torch.zeros_like(rm_scores)
    batch_idx = torch.arange(new_scores.size(0), device=new_scores.device)
    # Cast normed_scalar back to original dtype (e.g., bfloat16) to match new_scores
    new_scores[batch_idx, last_idx] = normed_scalar.to(rm_scores.dtype)
    rm_scores_proto.batch["rm_scores"] = new_scores
    return rm_scores_proto, raw_scalar, normed_scalar


