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
"""
OpenAI-compatible API client for extracting token probabilities.

This client supports:
- OpenAI API (gpt-4o, gpt-4o-mini, etc.)
- Third-party OpenAI-compatible APIs (DeepSeek, Together AI, etc.)
- Local models served via vLLM, SGLang, or other OpenAI-compatible servers

The client uses the `logprobs` feature to extract probabilities for
constrained token sets (e.g., "0" through "10" for scoring).

For local models:
    # Start vLLM server:
    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Llama-3.1-8B-Instruct --port 8000 --max-logprobs 100
    
    # Use the client:
    client = OpenAIClient(
        api_key="dummy",  # Not required for local servers
        model="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000/v1",
    )
"""

import asyncio
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Async OpenAI client with token probability extraction.
    
    Uses the `logprobs` parameter to extract probabilities for a constrained
    set of tokens, similar to how JudgeModelWorker computes scores.
    
    Example:
        >>> client = OpenAIClient(api_key="sk-...", model="gpt-4o-mini")
        >>> probs = await client.get_token_probs(
        ...     messages=[{"role": "user", "content": "Rate 1-10: ..."}],
        ...     valid_tokens=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        ... )
        >>> print(probs)  # {"0": 0.01, "1": 0.05, ..., "10": 0.02}
    
    Concurrency is controlled via asyncio.Semaphore to respect API rate limits.
    """
    
    # Default valid tokens for 0-5 scoring
    DEFAULT_VALID_TOKENS = ["0", "1", "2", "3", "4", "5"]
    # Default token weights (linear mapping: weight = i / 5)
    DEFAULT_TOKEN_WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_concurrent: int = 10,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model name to use for inference.
            base_url: Base URL for API. Use for third-party compatible APIs.
            max_concurrent: Maximum concurrent API calls (rate limiting).
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries on failure.
            retry_delay: Base delay between retries (exponential backoff).
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # Allow "dummy" or "none" API key for local servers (vLLM, SGLang)
        if not self.api_key:
            if base_url and "localhost" in base_url:
                self.api_key = "dummy"
                logger.info("Using dummy API key for local server")
            else:
                raise ValueError(
                    "API key required. Set OPENAI_API_KEY env var or pass api_key parameter. "
                    "For local servers (vLLM, SGLang), you can use api_key='dummy'."
                )
        
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Semaphore for rate limiting concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize async client
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = AsyncOpenAI(**client_kwargs)
        
        logger.info(
            f"Initialized OpenAIClient: model={model}, max_concurrent={max_concurrent}"
        )
    
    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        top_logprobs: int = 20,
        **kwargs,
    ) -> Any:
        """
        Make a single API request with retries.
        
        Args:
            messages: Chat messages to send.
            top_logprobs: Number of top logprobs to return (max 20 for OpenAI).
            **kwargs: Additional arguments to pass to the API.
        
        Returns:
            API response object.
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1,  # We only need the first token
                    logprobs=True,
                    top_logprobs=top_logprobs,
                    temperature=0,  # Deterministic for consistency
                    **kwargs,
                )
                return response
            
            except Exception as e:
                last_error = e
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"API request failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
        
        raise last_error
    
    def _extract_logprobs(
        self,
        response: Any,
        valid_tokens: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Extract token probabilities from API response.
        
        Tokens in top_logprobs get prob = exp(logprob).
        Tokens NOT in top_logprobs get prob = 0.
        Final probs are normalized over valid_tokens.
        
        This "sharp" approach makes scores more decisive by ignoring
        tokens outside the top-k returned by the API.
        
        Args:
            response: API response object.
            valid_tokens: List of tokens to extract probabilities for.
        
        Returns:
            Tuple of (token_probs, raw_logprobs):
            - token_probs: Dict mapping token -> probability (normalized over valid_tokens)
            - raw_logprobs: Dict mapping token -> raw logprob from API
        """
        # Extract logprobs from response
        try:
            content = response.choices[0].logprobs.content
            if not content:
                logger.warning("No logprobs in response, returning zero probabilities")
                return {t: 0.0 for t in valid_tokens}, {}
            
            top_logprobs_list = content[0].top_logprobs
        except (AttributeError, IndexError) as e:
            logger.warning(f"Failed to extract logprobs: {e}\nreturning zero probabilities")
            return {t: 0.0 for t in valid_tokens}, {}
        
        # Build mapping from token string to logprob
        raw_logprobs = {}
        for lp in top_logprobs_list:
            # OpenAI returns token as string, possibly with leading space
            token = lp.token.strip()
            raw_logprobs[token] = lp.logprob
        
        # Convert logprobs to probabilities for valid tokens
        # Tokens in top_logprobs get exp(logprob), others get 0
        valid_probs = {}
        for token in valid_tokens:
            if token in raw_logprobs:
                valid_probs[token] = math.exp(raw_logprobs[token])
            elif token.strip() in raw_logprobs:
                valid_probs[token] = math.exp(raw_logprobs[token.strip()])
            else:
                # Token not in top_logprobs - assign zero probability (sharp cutoff)
                valid_probs[token] = 0.0
        
        # Normalize over valid tokens
        total_prob = sum(valid_probs.values())
        if total_prob > 0:
            token_probs = {k: v / total_prob for k, v in valid_probs.items()}
        else:
            # No valid tokens found in top logprobs - return zero probabilities
            logger.warning(
                f"No valid tokens found in top logprobs. "
                f"Valid tokens: {valid_tokens}, Got: {list(raw_logprobs.keys())}"
            )
            token_probs = valid_probs  # All zeros
        
        return token_probs, raw_logprobs
    
    
    async def get_score(
        self,
        messages: List[Dict[str, str]],
        valid_tokens: Optional[List[str]] = None,
        token_weights: Optional[List[float]] = None,
        top_logprobs: int = 3,
        round_off: bool = False,
        **kwargs,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Get weighted score and token probabilities.
        
        Args:
            messages: Chat messages (system, user, assistant).
            valid_tokens: List of valid token strings. Defaults to ["0".."5"].
            token_weights: Weights for each token. Defaults to [0.0, 0.2, ..., 1.0].
            top_logprobs: Number of top logprobs to request from API.
            round_off: Whether to round off the final score.
            **kwargs: Additional API parameters.
        
        Returns:
            Tuple of (score, token_probs).
        """
        if valid_tokens is None:
            valid_tokens = self.DEFAULT_VALID_TOKENS
        if token_weights is None:
            token_weights = self.DEFAULT_TOKEN_WEIGHTS
        
        async with self.semaphore:
            response = await self._make_request(
                messages=messages,
                top_logprobs=top_logprobs,
                **kwargs,
            )
            token_probs, _ = self._extract_logprobs(response, valid_tokens)

            probs = list(token_probs.values())
            score = sum(p * w for p, w in zip(probs, token_weights))

            if round_off:
                score = round(score, 2)  # Round to 2 decimal places
    
            return score, token_probs
    
    async def get_token_probs(
        self,
        messages: List[Dict[str, str]],
        valid_tokens: Optional[List[str]] = None,
        top_logprobs: int = 3,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Get probabilities for constrained token set.
        
        Args:
            messages: Chat messages (system, user, assistant).
            valid_tokens: List of valid token strings. Defaults to ["0".."5"].
            top_logprobs: Number of top logprobs to request from API.
            **kwargs: Additional API parameters.
        
        Returns:
            Dict mapping token string -> probability (normalized over valid_tokens).
        """
        if valid_tokens is None:
            valid_tokens = self.DEFAULT_VALID_TOKENS
        
        async with self.semaphore:
            response = await self._make_request(
                messages=messages,
                top_logprobs=top_logprobs,
                **kwargs,
            )
            token_probs, _ = self._extract_logprobs(response, valid_tokens)
            return token_probs
    
    async def batch_get_scores(
        self,
        batch_messages: List[List[Dict[str, str]]],
        valid_tokens: Optional[List[str]] = None,
        token_weights: Optional[List[float]] = None,
        top_logprobs: int = 3,
        show_progress: bool = True,
        progress_desc: str = "API calls",
        round_off: bool = False,
        **kwargs,
    ) -> List[Tuple[float, Dict[str, float]]]:
        """
        Get scores and token probabilities for a batch of message sets.
        
        Uses asyncio.gather for concurrent execution, with rate limiting
        controlled by the semaphore.
        
        Args:
            batch_messages: List of messages.
            valid_tokens: Valid tokens for scoring.
            token_weights: Weights for each token.
            top_logprobs: Number of top logprobs to request.
            show_progress: Whether to show progress bar.
            progress_desc: Description for progress bar.
            round_off: Whether to round off the final score.
            **kwargs: Additional API parameters.
        
        Returns:
            List of (score, token_probs) tuples, one per input.
        """
        if valid_tokens is None:
            valid_tokens = self.DEFAULT_VALID_TOKENS
        if token_weights is None:
            token_weights = self.DEFAULT_TOKEN_WEIGHTS
        
        tasks = [
            self.get_score(
                messages=message,
                valid_tokens=valid_tokens,
                token_weights=token_weights,
                top_logprobs=top_logprobs,
                round_off=round_off,
                **kwargs,
            )
            for message in batch_messages
        ]
        
        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio
                results = await tqdm_asyncio.gather(
                    *tasks, desc=progress_desc, total=len(tasks)
                )
            except ImportError:
                results = await asyncio.gather(*tasks)
        else:
            results = await asyncio.gather(*tasks)
        
        return results
    
    async def close(self):
        """Close the client and release resources."""
        await self.client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def create_openai_client(
    provider: str = "openai",
    **kwargs,
) -> OpenAIClient:
    """
    Factory function to create OpenAI-compatible clients.
    
    Args:
        provider: Provider name ("openai", "deepseek", "together", etc.)
        **kwargs: Additional arguments passed to OpenAIClient.
    
    Returns:
        Configured OpenAIClient instance.
    """
    # Provider-specific configurations
    provider_configs = {
        "openai": {
            "base_url": None,
            "model": "gpt-4o-mini",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
        },
        "together": {
            "base_url": "https://api.together.xyz/v1",
            "model": "meta-llama/Llama-3-70b-chat-hf",
        },
    }
    
    config = provider_configs.get(provider.lower(), {})
    config.update(kwargs)
    
    return OpenAIClient(**config)
