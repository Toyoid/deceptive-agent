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
import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


Message = Dict[str, str]


@dataclass
class ChatResponse:
    text: str
    finish_reason: Optional[str] = None
    latency: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw_response: Any = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _is_local_api_base(api_base: Optional[str]) -> bool:
    if not api_base:
        return False
    hostname = urlparse(api_base).hostname
    return hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def resolve_api_key(
    *,
    api_key: Optional[str],
    api_key_env: Optional[str],
    api_base: Optional[str],
) -> str:
    if api_key:
        return api_key
    if api_key_env:
        env_value = os.getenv(api_key_env)
        if env_value:
            return env_value
    if _is_local_api_base(api_base):
        return "dummy"
    raise ValueError(
        "API key required. Set model.api_key, set model.api_key_env, or use a local "
        "OpenAI-compatible endpoint such as localhost/127.0.0.1."
    )


class OpenAICompatibleChatClient:
    """Small async chat client for OpenAI-compatible generation endpoints."""

    def __init__(
        self,
        *,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = "OPENAI_API_KEY",
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_tokens: int = 2048,
        max_concurrent: int = 32,
        max_retries: int = 3,
        timeout: float = 120.0,
        retry_delay: float = 1.0,
        async_client: Any = None,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = resolve_api_key(api_key=api_key, api_key_env=api_key_env, api_base=api_base)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = None if top_k is None else int(top_k)
        self.min_p = None if min_p is None else float(min_p)
        self.max_tokens = max_tokens
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

        if async_client is not None:
            self.client = async_client
            return

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("The openai package is required for API rollout eval. Install with: pip install openai") from exc

        client_kwargs = {"api_key": self.api_key, "timeout": timeout}
        if api_base:
            client_kwargs["base_url"] = api_base
        self.client = AsyncOpenAI(**client_kwargs)

    @classmethod
    def from_config(cls, config) -> "OpenAICompatibleChatClient":
        provider = str(config.model.get("provider", "openai_compatible"))
        if provider != "openai_compatible":
            raise ValueError(f"Unsupported model.provider={provider!r}; only 'openai_compatible' is supported.")
        return cls(
            model=config.model.model,
            api_base=config.model.get("api_base"),
            api_key=config.model.get("api_key"),
            api_key_env=config.model.get("api_key_env", "OPENAI_API_KEY"),
            temperature=float(config.model.get("temperature", 0.0)),
            top_p=float(config.model.get("top_p", 1.0)),
            top_k=config.model.get("top_k"),
            min_p=config.model.get("min_p"),
            max_tokens=int(config.model.get("max_tokens", 2048)),
            max_concurrent=int(config.model.get("max_concurrent", 32)),
            max_retries=int(config.model.get("max_retries", 3)),
            timeout=float(config.model.get("timeout", 120.0)),
            retry_delay=float(config.model.get("retry_delay", 1.0)),
        )

    async def generate_one(self, messages: List[Message]) -> ChatResponse:
        async with self.semaphore:
            return await self._generate_one_with_retries(messages)

    async def _generate_one_with_retries(self, messages: List[Message]) -> ChatResponse:
        last_error: Optional[BaseException] = None
        started = time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                request_started = time.perf_counter()
                request_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                }
                extra_body = {}
                if self.top_k is not None:
                    extra_body["top_k"] = self.top_k
                if self.min_p is not None:
                    extra_body["min_p"] = self.min_p
                if extra_body:
                    request_kwargs["extra_body"] = extra_body
                try:
                    response = await self.client.chat.completions.create(**request_kwargs)
                except TypeError as exc:
                    if extra_body and "extra_body" in str(exc):
                        request_kwargs.pop("extra_body", None)
                        response = await self.client.chat.completions.create(**request_kwargs)
                    else:
                        raise
                latency = time.perf_counter() - request_started
                choice = response.choices[0]
                message = getattr(choice, "message", None)
                text = getattr(message, "content", "") if message is not None else ""
                usage = getattr(response, "usage", None)
                return ChatResponse(
                    text=text or "",
                    finish_reason=getattr(choice, "finish_reason", None),
                    latency=latency,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                    raw_response=response,
                )
            except Exception as exc:  # pragma: no cover - exact API exceptions vary by provider
                last_error = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))

        return ChatResponse(text="", latency=time.perf_counter() - started, error=str(last_error))

    async def generate_batch(self, batch_messages: List[List[Message]]) -> List[ChatResponse]:
        if not batch_messages:
            return []
        return await asyncio.gather(*(self.generate_one(messages) for messages in batch_messages))

    async def close(self) -> None:
        closer = getattr(self.client, "close", None) or getattr(self.client, "aclose", None)
        if closer is None:
            return
        result = closer()
        if inspect.isawaitable(result):
            await result
