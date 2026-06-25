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

import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from agent_system.api_rollout_eval.clients import OpenAICompatibleChatClient
from agent_system.judge.score_parsing import parse_score_token
from agent_system.judge.score_profiles import (
    ScoreProfile,
    default_score_profile_name,
    load_score_profiles,
    resolve_score_profile,
)


@dataclass
class ApiCotJudgeScoreResult:
    scores: np.ndarray
    token_probs: np.ndarray
    parsed_tokens: List[Optional[str]]
    raw_outputs: List[str]
    errors: List[Optional[str]]


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
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


class ApiCotJudgeScorer:
    """OpenAI-compatible CoT judge scorer.

    The scorer asks a judge model to reason in text and finish with a parseable
    final tag, then maps that final token through the selected judge score profile.
    """

    def __init__(self, judge_config: Any) -> None:
        self.judge_config = judge_config
        self.api_config = _cfg_get(judge_config, "api_cot", {})
        self.score_profiles = load_score_profiles(judge_config)
        self.default_score_profile_name = default_score_profile_name(self.score_profiles)
        default_profile = self.score_profiles[self.default_score_profile_name]
        self.valid_tokens = list(default_profile.valid_tokens)
        self.token_weights = list(default_profile.token_weights)

        provider = str(_cfg_get(self.api_config, "provider", "openai_compatible"))
        if provider != "openai_compatible":
            raise ValueError(f"Unsupported judge_model.api_cot.provider={provider!r}; only 'openai_compatible' is supported.")

        self.score_regex = str(_cfg_get(self.api_config, "score_regex", r"<score>\s*([0-4])\s*</score>\s*$"))
        self.parse_error = str(_cfg_get(self.api_config, "parse_error", "raise"))
        self.api_error = str(_cfg_get(self.api_config, "api_error", "raise"))
        if self.parse_error not in {"raise", "neutral"}:
            raise ValueError("judge_model.api_cot.parse_error must be one of ['raise', 'neutral'].")
        if self.api_error not in {"raise", "neutral"}:
            raise ValueError("judge_model.api_cot.api_error must be one of ['raise', 'neutral'].")

    def score_batch(
        self,
        batch_messages: List[List[dict]],
        score_profile_name: Optional[str] = None,
    ) -> ApiCotJudgeScoreResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.score_batch_async(
                batch_messages,
                score_profile_name=score_profile_name,
            ))
        raise RuntimeError("ApiCotJudgeScorer.score_batch() cannot be called from an active event loop; use score_batch_async().")

    async def score_batch_async(
        self,
        batch_messages: List[List[dict]],
        score_profile_name: Optional[str] = None,
    ) -> ApiCotJudgeScoreResult:
        client = self._build_client()
        try:
            try:
                responses = await client.generate_batch(batch_messages)
            except Exception as exc:
                if self.api_error == "raise":
                    raise
                return self.score_texts(
                    raw_outputs=[""] * len(batch_messages),
                    errors=[str(exc)] * len(batch_messages),
                    score_profile_name=score_profile_name,
                )
        finally:
            await client.close()

        raw_outputs = [response.text for response in responses]
        errors = [response.error for response in responses]
        return self.score_texts(
            raw_outputs=raw_outputs,
            errors=errors,
            score_profile_name=score_profile_name,
        )

    def score_texts(
        self,
        raw_outputs: Sequence[str],
        errors: Optional[Sequence[Optional[str]]] = None,
        score_profile_name: Optional[str] = None,
    ) -> ApiCotJudgeScoreResult:
        profile = resolve_score_profile(self.score_profiles, score_profile_name)
        if errors is None:
            errors = [None] * len(raw_outputs)
        if len(errors) != len(raw_outputs):
            raise ValueError(f"errors length {len(errors)} does not match raw_outputs length {len(raw_outputs)}.")

        scores: List[float] = []
        token_probs: List[np.ndarray] = []
        parsed_tokens: List[Optional[str]] = []
        normalized_errors: List[Optional[str]] = []

        for idx, (raw_output, error) in enumerate(zip(raw_outputs, errors)):
            if error:
                message = f"API CoT judge call failed for item {idx}: {error}"
                if self.api_error == "raise":
                    raise RuntimeError(message)
                score, probs = self._neutral_score(profile)
                scores.append(score)
                token_probs.append(probs)
                parsed_tokens.append(None)
                normalized_errors.append(message)
                continue

            try:
                token = self.parse_score_token(raw_output)
                score, probs, token_error = self._score_token(token, profile)
                scores.append(score)
                token_probs.append(probs)
                parsed_tokens.append(token)
                normalized_errors.append(token_error)
            except ValueError as exc:
                message = f"Failed to parse API CoT judge score for item {idx}: {exc}"
                score, probs = self._neutral_score(profile)
                scores.append(score)
                token_probs.append(probs)
                parsed_tokens.append(None)
                normalized_errors.append(message)

        return ApiCotJudgeScoreResult(
            scores=np.asarray(scores, dtype=np.float32),
            token_probs=np.stack(token_probs).astype(np.float32) if token_probs else np.zeros((0, len(profile.valid_tokens)), dtype=np.float32),
            parsed_tokens=parsed_tokens,
            raw_outputs=list(raw_outputs),
            errors=normalized_errors,
        )

    def parse_score_token(self, raw_output: str) -> str:
        return parse_score_token(raw_output, score_regex=self.score_regex)

    def _score_token(self, token: str, profile: ScoreProfile) -> tuple[float, np.ndarray, Optional[str]]:
        probs = np.zeros(len(profile.valid_tokens), dtype=np.float32)
        if token not in profile.valid_tokens:
            return 0.0, probs, (
                f"parsed score token {token!r} is invalid for score_profile={profile.name!r}; "
                f"expected one of {list(profile.valid_tokens)!r}"
            )
        token_idx = profile.valid_tokens.index(token)
        probs[token_idx] = 1.0
        return float(profile.token_weights[token_idx]), probs, None

    def _neutral_score(self, profile: ScoreProfile) -> tuple[float, np.ndarray]:
        probs = np.zeros(len(profile.valid_tokens), dtype=np.float32)
        return 0.0, probs

    def _build_client(self) -> OpenAICompatibleChatClient:
        model = _cfg_get(self.api_config, "model", None)
        if model is None:
            raise ValueError("judge_model.api_cot.model must be set when judge_model.backend=api_cot.")
        max_output_length = _cfg_get(
            self.api_config,
            "max_output_length",
            _cfg_get(self.api_config, "max_tokens", 2048),
        )
        return OpenAICompatibleChatClient(
            model=str(model),
            api_base=_cfg_get(self.api_config, "api_base", None),
            api_key=_cfg_get(self.api_config, "api_key", None),
            api_key_env=_cfg_get(self.api_config, "api_key_env", "OPENAI_API_KEY"),
            temperature=float(_cfg_get(self.api_config, "temperature", 0.0)),
            top_p=float(_cfg_get(self.api_config, "top_p", 1.0)),
            top_k=_cfg_get(self.api_config, "top_k", None),
            min_p=_cfg_get(self.api_config, "min_p", None),
            presence_penalty=float(_cfg_get(self.api_config, "presence_penalty", 0.0)),
            max_tokens=int(max_output_length),
            max_concurrent=int(_cfg_get(self.api_config, "max_concurrent", 64)),
            max_retries=int(_cfg_get(self.api_config, "max_retries", 3)),
            timeout=float(_cfg_get(self.api_config, "timeout", 120.0)),
            retry_delay=float(_cfg_get(self.api_config, "retry_delay", 1.0)),
        )
