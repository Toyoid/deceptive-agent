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
Base metric class for retroactive evaluation.

This module provides the BaseMetric class that:
1. Builds evaluation prompts from generation data
2. Calls the API client to get token probabilities and scores
3. Integrates with ResultCache for caching
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..data_reader.schemas import Generation
from ..cache import CachedResult, ResultCache
from ..clients.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Configuration for a single metric."""
    name: str
    system_prompt: str
    prompt_suffix: str
    valid_tokens: List[str]
    token_weights: List[float]
    top_logprobs: int
    enabled: bool = True
    round_off: bool = False
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> "MetricConfig":
        """
        Create MetricConfig from a dictionary (e.g., from YAML).
        
        Args:
            name: Name of the metric.
            data: Dictionary with metric configuration.
            defaults: Default values for missing fields.
        
        Returns:
            MetricConfig instance.
        """
        defaults = defaults or {}
        
        return cls(
            name=name,
            system_prompt=data.get("system_prompt", ""),
            prompt_suffix=data.get("prompt_suffix", ""),
            valid_tokens=data.get("valid_tokens", None),
            token_weights=data.get("token_weights", None),
            top_logprobs=data.get("top_logprobs", None),
            enabled=data.get("enabled", True),
            round_off=data.get("round_off", False),
        )


@dataclass
class MetricResult:
    """Result from evaluating a single generation with a metric."""
    generation_uid: str
    metric_name: str
    score: float
    token_probs: Dict[str, float]
    step: int
    cached: bool = False
    error: Optional[str] = None
    
    def to_cached_result(self, model: str = "") -> CachedResult:
        """Convert to CachedResult for caching."""
        return CachedResult(
            generation_uid=self.generation_uid,
            metric_name=self.metric_name,
            score=self.score,
            token_probs=self.token_probs,
            step=self.step,
            model=model,
        )
    
    @classmethod
    def from_cached_result(cls, cached: CachedResult) -> "MetricResult":
        """Create from a CachedResult."""
        return cls(
            generation_uid=cached.generation_uid,
            metric_name=cached.metric_name,
            score=cached.score,
            token_probs=cached.token_probs,
            step=cached.step,
            cached=True,
        )
    
    @classmethod
    def error_result(
        cls, 
        generation_uid: str, 
        metric_name: str, 
        step: int, 
        error: str
    ) -> "MetricResult":
        """Create an error result."""
        return cls(
            generation_uid=generation_uid,
            metric_name=metric_name,
            score=float("nan"),
            token_probs={},
            step=step,
            error=error,
        )


class BaseMetric:
    """
    Base class for metrics that evaluate generations using an LLM API.
    
    This class handles:
    1. Building evaluation prompts from Generation objects
    2. Making API calls to get token probabilities and scores
    3. Caching results to avoid re-computation
    """
    
    def __init__(
        self,
        config: MetricConfig,
        client: OpenAIClient,
        cache: Optional[ResultCache] = None,
    ):
        """
        Initialize the metric.
        
        Args:
            config: Metric configuration.
            client: OpenAI API client for making requests.
            cache: Optional result cache for storing/loading results.
        """
        self.config = config
        self.client = client
        self.cache = cache
        
        logger.info(f"Initialized metric: {config.name}")
    
    @property
    def name(self) -> str:
        """Get the metric name."""
        return self.config.name
    
    def build_message(self, generation: Generation) -> List[Dict[str, str]]:
        """
        Build the evaluation prompt messages for a generation.
        
        This creates a chat message format that presents the conversation
        to the evaluator model and asks for a rating.
        
        Args:
            generation: The generation to evaluate.
        
        Returns:
            List of chat messages in OpenAI format.
        """
        # Build the conversation content to evaluate
        # The input contains the prompt, output contains the response
        context_prompt = "The conversation history is:\n\n" + \
            f"[ENVIRONMENT INFORMATION]:\n{generation.system_info}\n\n" + \
            f"[USER]: {generation.input}\n\n[AGENT]: {generation.output}\n\n" + \
            self.config.prompt_suffix
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": context_prompt},
        ]
        
        return messages
    
    async def evaluate(
        self, 
        generation: Generation,
        use_cache: bool = True,
    ) -> MetricResult:
        """
        Evaluate a single generation.
        
        Args:
            generation: The generation to evaluate.
            use_cache: Whether to check/use the cache.
        
        Returns:
            MetricResult with score and token probabilities.
        """
        # Check cache first
        if use_cache and self.cache and self.cache.enabled:
            cached = self.cache.get_result(
                self.config.name, 
                generation.uid, 
                generation.step
            )
            if cached is not None:
                logger.debug(f"Cache hit for {self.config.name}/{generation.uid}")
                return MetricResult.from_cached_result(cached)
        
        try:
            # Build messages and call API
            messages = self.build_message(generation)
            score, token_probs = await self.client.get_score(
                messages=messages,
                valid_tokens=self.config.valid_tokens,
                token_weights=self.config.token_weights,
                top_logprobs=self.config.top_logprobs,
                round_off=self.config.round_off,
            )
            
            # Create result
            result = MetricResult(
                generation_uid=generation.uid,
                metric_name=self.config.name,
                score=score,
                token_probs=token_probs,
                step=generation.step,
                cached=False,
            )
            
            # Save to cache
            if use_cache and self.cache and self.cache.enabled:
                cached_result = result.to_cached_result(model=self.client.model)
                self.cache.save_result(cached_result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating {self.config.name}/{generation.uid}: {e}")
            return MetricResult.error_result(
                generation_uid=generation.uid,
                metric_name=self.config.name,
                step=generation.step,
                error=str(e),
            )
    
    async def evaluate_batch(
        self,
        generations: List[Generation],
        use_cache: bool = True,
        progress_callback: Optional[callable] = None,
        progress_desc: Optional[str] = None,
    ) -> List[MetricResult]:
        """
        Evaluate a batch of generations with caching support.
        
        This method efficiently handles caching by:
        1. Loading all cached results first
        2. Only making API calls for uncached generations
        3. Saving new results to cache
        
        Args:
            generations: List of generations to evaluate.
            use_cache: Whether to use caching.
            progress_callback: Optional callback(completed, total) for progress.
            progress_desc: Optional description for progress bar.
        
        Returns:
            List of MetricResults in the same order as input.
        """
        if not generations:
            return []
        
        results: Dict[str, MetricResult] = {}
        to_evaluate: List[Generation] = []
        
        # Step 1: Check cache for existing results
        if use_cache and self.cache and self.cache.enabled:
            step = generations[0].step  # Assume all same step
            cached_uids = self.cache.get_cached_uids(self.config.name, step)
            
            for gen in generations:
                if gen.uid in cached_uids:
                    cached = self.cache.get_result(self.config.name, gen.uid, gen.step)
                    if cached is not None:
                        results[gen.uid] = MetricResult.from_cached_result(cached)
                    else:
                        to_evaluate.append(gen)
                else:
                    to_evaluate.append(gen)
            
            logger.info(
                f"[{progress_desc}] Cache: {len(results)} hits, "
                f"{len(to_evaluate)} to evaluate"
            )
        else:
            to_evaluate = list(generations)
        
        # Report initial progress (cached items)
        if progress_callback:
            progress_callback(len(results), len(generations))
        
        # Step 2: Evaluate uncached generations
        if to_evaluate:
            # Use the client's batch method for efficient concurrent requests
            message_list = [self.build_message(gen) for gen in to_evaluate]
            
            try:
                results_list = await self.client.batch_get_scores(
                    batch_messages=message_list,
                    valid_tokens=self.config.valid_tokens,
                    token_weights=self.config.token_weights,
                    top_logprobs=self.config.top_logprobs,
                    progress_desc=progress_desc,
                    round_off=self.config.round_off,
                )
                # Process results
                for gen, (score, token_probs) in zip(to_evaluate, results_list):
                    if token_probs is None:
                        # API call failed
                        result = MetricResult.error_result(
                            generation_uid=gen.uid,
                            metric_name=self.config.name,
                            step=gen.step,
                            error="API call returned None",
                        )
                    else:
                        result = MetricResult(
                            generation_uid=gen.uid,
                            metric_name=self.config.name,
                            score=score,
                            token_probs=token_probs,
                            step=gen.step,
                            cached=False,
                        )
                        
                        # Save to cache
                        if use_cache and self.cache and self.cache.enabled:
                            cached_result = result.to_cached_result(model=self.client.model)
                            self.cache.save_result(cached_result)
                    
                    results[gen.uid] = result
                    
                    # Report progress
                    if progress_callback:
                        progress_callback(len(results), len(generations))
                        
            except Exception as e:
                logger.error(f"Batch evaluation failed: {e}")
                # Mark all remaining as errors
                for gen in to_evaluate:
                    if gen.uid not in results:
                        results[gen.uid] = MetricResult.error_result(
                            generation_uid=gen.uid,
                            metric_name=self.config.name,
                            step=gen.step,
                            error=str(e),
                        )
        
        # Step 3: Return results in original order
        return [results[gen.uid] for gen in generations]
    
    def __repr__(self) -> str:
        return f"BaseMetric(name={self.config.name!r}, enabled={self.config.enabled})"
