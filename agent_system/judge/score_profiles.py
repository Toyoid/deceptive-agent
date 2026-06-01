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
from typing import Any, Dict, Optional


DEFAULT_SCORE_PROFILE_NAME = "default"
ISSUE_ACTION_SCORE_PROFILE = "issue_action"
NO_ISSUE_ACTION_SCORE_PROFILE = "no_issue_action"


@dataclass(frozen=True)
class ScoreProfile:
    name: str
    valid_tokens: tuple[str, ...]
    token_weights: tuple[float, ...]


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def _items(mapping: Any):
    if mapping is None:
        return []
    items = getattr(mapping, "items", None)
    if callable(items):
        return list(items())
    raise TypeError(f"score_profiles must be a mapping, got {type(mapping)}")


def _list(value: Any) -> list:
    if value is None:
        return []
    return list(value)


def _build_profile(name: str, profile_config: Any) -> ScoreProfile:
    valid_tokens = [str(token) for token in _list(cfg_get(profile_config, "valid_tokens", None))]
    token_weights = [float(weight) for weight in _list(cfg_get(profile_config, "token_weights", None))]
    if len(valid_tokens) == 0:
        raise ValueError(f"judge_model.score_profiles.{name}.valid_tokens must be non-empty.")
    if len(valid_tokens) != len(token_weights):
        raise ValueError(
            f"judge_model.score_profiles.{name}.valid_tokens and token_weights must have the same length, "
            f"got {len(valid_tokens)} and {len(token_weights)}."
        )
    return ScoreProfile(
        name=str(name),
        valid_tokens=tuple(valid_tokens),
        token_weights=tuple(token_weights),
    )


def load_score_profiles(judge_config: Any) -> Dict[str, ScoreProfile]:
    raw_profiles = cfg_get(judge_config, "score_profiles", None)
    if raw_profiles is not None:
        raw_profile_items = _items(raw_profiles)
        if len(raw_profile_items) == 0:
            raise ValueError("judge_model.score_profiles must contain at least one profile.")
        profiles = {
            str(name): _build_profile(str(name), profile_config)
            for name, profile_config in raw_profile_items
        }
        return profiles

    # Backward-compatible fallback for older judge configs.
    valid_tokens = cfg_get(judge_config, "valid_tokens", None)
    token_weights = cfg_get(judge_config, "token_weights", None)
    if valid_tokens is None or token_weights is None:
        raise ValueError("judge_model.score_profiles must be set, or legacy valid_tokens/token_weights must be set.")
    return {
        DEFAULT_SCORE_PROFILE_NAME: _build_profile(
            DEFAULT_SCORE_PROFILE_NAME,
            {"valid_tokens": valid_tokens, "token_weights": token_weights},
        )
    }


def default_score_profile_name(profiles: Dict[str, ScoreProfile]) -> str:
    if DEFAULT_SCORE_PROFILE_NAME in profiles:
        return DEFAULT_SCORE_PROFILE_NAME
    if ISSUE_ACTION_SCORE_PROFILE in profiles:
        return ISSUE_ACTION_SCORE_PROFILE
    return next(iter(profiles))


def resolve_score_profile(profiles: Dict[str, ScoreProfile], profile_name: Optional[str]) -> ScoreProfile:
    selected = profile_name or default_score_profile_name(profiles)
    if selected not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown judge score profile {selected!r}. Available profiles: {available}")
    return profiles[selected]
