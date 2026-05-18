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
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from agent_system.api_rollout_eval.clients import OpenAICompatibleChatClient


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
    final tag, then maps that final token through judge_model.token_weights.
    """

    def __init__(self, judge_config: Any) -> None:
        self.judge_config = judge_config
        self.api_config = _cfg_get(judge_config, "api_cot", {})
        self.valid_tokens = [str(token) for token in list(_cfg_get(judge_config, "valid_tokens", []))]
        self.token_weights = [float(weight) for weight in list(_cfg_get(judge_config, "token_weights", []))]
        if len(self.valid_tokens) == 0:
            raise ValueError("judge_model.valid_tokens must be set for api_cot judge scoring.")
        if len(self.valid_tokens) != len(self.token_weights):
            raise ValueError(
                "judge_model.valid_tokens and judge_model.token_weights must have the same length "
                f"for api_cot judge scoring, got {len(self.valid_tokens)} and {len(self.token_weights)}."
            )

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

    def score_batch(self, batch_messages: List[List[dict]]) -> ApiCotJudgeScoreResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.score_batch_async(batch_messages))
        raise RuntimeError("ApiCotJudgeScorer.score_batch() cannot be called from an active event loop; use score_batch_async().")

    async def score_batch_async(self, batch_messages: List[List[dict]]) -> ApiCotJudgeScoreResult:
        client = self._build_client()
        try:
            responses = await client.generate_batch(batch_messages)
        finally:
            await client.close()

        raw_outputs = [response.text for response in responses]
        errors = [response.error for response in responses]
        return self.score_texts(raw_outputs=raw_outputs, errors=errors)

    def score_texts(
        self,
        raw_outputs: Sequence[str],
        errors: Optional[Sequence[Optional[str]]] = None,
    ) -> ApiCotJudgeScoreResult:
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
                score, probs = self._neutral_score()
                scores.append(score)
                token_probs.append(probs)
                parsed_tokens.append(None)
                normalized_errors.append(message)
                continue

            try:
                token = self.parse_score_token(raw_output)
                score, probs = self._score_token(token)
                scores.append(score)
                token_probs.append(probs)
                parsed_tokens.append(token)
                normalized_errors.append(None)
            except ValueError as exc:
                message = f"Failed to parse API CoT judge score for item {idx}: {exc}"
                if self.parse_error == "raise":
                    raise ValueError(message) from exc
                score, probs = self._neutral_score()
                scores.append(score)
                token_probs.append(probs)
                parsed_tokens.append(None)
                normalized_errors.append(message)

        return ApiCotJudgeScoreResult(
            scores=np.asarray(scores, dtype=np.float32),
            token_probs=np.stack(token_probs).astype(np.float32) if token_probs else np.zeros((0, len(self.valid_tokens)), dtype=np.float32),
            parsed_tokens=parsed_tokens,
            raw_outputs=list(raw_outputs),
            errors=normalized_errors,
        )

    def parse_score_token(self, raw_output: str) -> str:
        text = raw_output or ""

        # 1) Preferred: user-configured regex (usually <score>...</score> at end).
        match = re.search(self.score_regex, text, flags=re.DOTALL)
        if match is not None:
            token = match.group(1).strip()
            if token in self.valid_tokens:
                return token

        # 2) Strip <think>...</think> blocks and anything before the last </think>.
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        if re.search(r"</think>", stripped, flags=re.IGNORECASE):
            stripped = re.split(r"</think>", stripped, flags=re.IGNORECASE)[-1]
        stripped = stripped.strip()

        # 2a) Accept <score> N </score> anywhere in the stripped content.
        match = re.search(r"<score>\s*([0-9]+)\s*</score>", stripped, flags=re.DOTALL | re.IGNORECASE)
        if match is not None:
            token = match.group(1).strip()
            if token in self.valid_tokens:
                return token

        # 2b) Accept a single-token output (exact match).
        if stripped in self.valid_tokens:
            return stripped

        # 2c) Accept labeled patterns like "score: N" or "final: N".
        label_pattern = r"(?:score|final|answer|verdict|result|output)\s*[:=]\s*([0-9]+)"
        matches = re.findall(label_pattern, stripped, flags=re.IGNORECASE)
        for token in reversed(matches):
            if token in self.valid_tokens:
                return token

        raise ValueError(
            f"no parseable score token found; expected one of {self.valid_tokens!r} "
            f"matching score_regex={self.score_regex!r}; output={raw_output!r}"
        )

    def _score_token(self, token: str) -> tuple[float, np.ndarray]:
        token_idx = self.valid_tokens.index(token)
        probs = np.zeros(len(self.valid_tokens), dtype=np.float32)
        probs[token_idx] = 1.0
        return float(self.token_weights[token_idx]), probs

    def _neutral_score(self) -> tuple[float, np.ndarray]:
        # The default 5-token judge scale treats "2" as neutral. Some older
        # 4-token templates use "0" as the no-issue baseline, so fall back to
        # "0" when a custom valid-token set has no explicit neutral "2".
        if "2" in self.valid_tokens:
            return self._score_token("2")
        if "0" in self.valid_tokens:
            return self._score_token("0")
        probs = np.zeros(len(self.valid_tokens), dtype=np.float32)
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
