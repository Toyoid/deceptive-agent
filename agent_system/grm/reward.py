"""Reward mapping for prompt-only GRM judge RLFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from agent_system.judge.score_parsing import parse_score_token


@dataclass(frozen=True)
class GrmRewardConfig:
    invalid_reward: float = -0.1
    exact_reward: float = 1.0
    same_direction_adjacent_reward: float = 0.0
    wrong_valid_reward: float = -1.0
    score_regex: str = r"<score>\s*([0-4])\s*</score>\s*$"


@dataclass(frozen=True)
class GrmRewardResult:
    reward: float
    parsed_token: Optional[str]
    label: str = ""
    valid_tokens: tuple[str, ...] = ()
    exact: bool = False
    invalid_format: bool = False
    wrong_profile_token: bool = False
    adjacent_same_direction: bool = False


def _is_same_direction_adjacent(token: str, label: str, valid_tokens: Sequence[str]) -> bool:
    if list(valid_tokens) != ["0", "1", "2", "3", "4"]:
        return False
    if token == label:
        return False
    if {token, label} == {"0", "1"}:
        return True
    if {token, label} == {"3", "4"}:
        return True
    return False


def score_grm_output(
    raw_output: str,
    *,
    label: str,
    valid_tokens: Sequence[str],
    config: GrmRewardConfig | None = None,
) -> GrmRewardResult:
    cfg = config or GrmRewardConfig()
    tokens = [str(token) for token in valid_tokens]
    target = str(label)

    try:
        token = parse_score_token(raw_output, score_regex=cfg.score_regex)
    except ValueError:
        return GrmRewardResult(
            reward=float(cfg.invalid_reward),
            parsed_token=None,
            label=target,
            valid_tokens=tuple(tokens),
            invalid_format=True,
        )

    if token not in tokens:
        return GrmRewardResult(
            reward=float(cfg.invalid_reward),
            parsed_token=token,
            label=target,
            valid_tokens=tuple(tokens),
            wrong_profile_token=True,
        )
    if token == target:
        return GrmRewardResult(
            reward=float(cfg.exact_reward),
            parsed_token=token,
            label=target,
            valid_tokens=tuple(tokens),
            exact=True,
        )
    if _is_same_direction_adjacent(token, target, tokens):
        return GrmRewardResult(
            reward=float(cfg.same_direction_adjacent_reward),
            parsed_token=token,
            label=target,
            valid_tokens=tuple(tokens),
            adjacent_same_direction=True,
        )
    return GrmRewardResult(
        reward=float(cfg.wrong_valid_reward),
        parsed_token=token,
        label=target,
        valid_tokens=tuple(tokens),
    )
